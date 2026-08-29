from __future__ import annotations

import copy
from typing import Any

from .testnet_entry_roundtrip_v1613 import (
    EXIT_STRESS_BPS,
    _normalize_buy,
    _preflight,
    _supported,
)
from .testnet_exit_recycle import (
    MODELED_ROUND_TRIP_COST_FLOOR_BPS,
)
from .testnet_roundtrip_candidate_router_v1616 import (
    _route_blocked_until,
)


# v1.60.35: restore broad bounded Testnet execution probing
# Market intelligence remains authoritative; this layer validates executability across a wider ranked cohort instead of starving the universe behind two probes and a five-minute positive cache.
MAX_NETWORK_PROBES_PER_CALL = 2
FAIL_CACHE_SECONDS = 12.0
PASS_CACHE_SECONDS = 15.0
MIN_FREE_QUOTE_RESERVE_USD = 0.01
SIGNAL_REFRESH_PIN_SECONDS = 6.0


def _n(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _free_usdt(
    engine: Any,
) -> float:
    return max(
        0.0,
        _n(
            (
                (
                    engine.state.get(
                        "account_balance"
                    )
                    or {}
                ).get("free")
                or {}
            ).get("USDT")
        ),
    )


def _safe_minimum_cost(
    engine: Any,
    symbol: str,
) -> dict[str, Any]:
    market = engine.exchange.market(
        symbol
    )

    limits = (
        market.get("limits")
        or {}
    )

    min_cost = max(
        0.0,
        _n(
            (
                limits.get("cost")
                or {}
            ).get("min")
        ),
    )

    stress_fraction = max(
        0.01,
        1.0
        - EXIT_STRESS_BPS
        / 10_000.0,
    )

    cost_fraction = max(
        0.01,
        1.0
        - MODELED_ROUND_TRIP_COST_FLOOR_BPS
        / 10_000.0,
    )

    safe_required = (
        min_cost
        / (
            stress_fraction
            * cost_fraction
        )
        if min_cost > 0.0
        else 0.0
    )

    return {
        "minimum_cost_usd": min_cost,
        "safe_required_usd": (
            safe_required
        ),
    }


def _probe_notional(
    lane: Any,
) -> float:
    free = _free_usdt(
        lane.testnet
    )

    available = max(
        0.0,
        free
        - MIN_FREE_QUOTE_RESERVE_USD,
    )

    local_cap = max(
        0.0,
        _n(
            getattr(
                lane,
                "maximum_order_usd",
                0.0,
            )
        ),
    )

    executor_cap = max(
        0.0,
        _n(
            getattr(
                lane.testnet,
                "max_order_usd",
                0.0,
            )
        ),
    )

    base = max(
        0.01,
        _n(
            getattr(
                lane,
                "order_usd",
                1.0,
            ),
            1.0,
        ),
    )

    caps = [
        value
        for value in (
            base,
            local_cap,
            executor_cap,
            available,
        )
        if value > 0.0
    ]

    return (
        min(caps)
        if caps
        else 0.0
    )


def _probe_candidate(
    lane: Any,
    symbol: str,
) -> dict[str, Any]:
    exchange = (
        lane.testnet.exchange
    )

    try:
        ticker = (
            exchange.fetch_ticker(
                symbol
            )
            or {}
        )
    except Exception as exc:
        return {
            "allowed": False,
            "reason": (
                "fresh_testnet_quote_error"
            ),
            "error_type": (
                type(exc).__name__
            ),
        }

    bid = max(
        0.0,
        _n(
            ticker.get("bid")
        ),
    )

    ask = max(
        0.0,
        _n(
            ticker.get("ask")
        ),
    )

    last = max(
        0.0,
        _n(
            ticker.get("last")
        ),
    )

    price = (
        ask
        if ask > 0.0
        else (
            last
            if last > 0.0
            else bid
        )
    )

    if price <= 0.0:
        return {
            "allowed": False,
            "reason": (
                "fresh_testnet_quote_unavailable"
            ),
        }

    notional = _probe_notional(
        lane
    )

    if notional <= 0.0:
        return {
            "allowed": False,
            "reason": (
                "no_free_quote_for_probe"
            ),
        }

    event = {
        "symbol": symbol,
        "side": "buy",
        "price": price,
        "quantity": (
            notional
            / price
        ),
    }

    try:
        normalized, normalization = (
            _normalize_buy(
                lane.testnet,
                event,
            )
        )

        normalization_block = str(
            normalization.get(
                "blocked_reason"
            )
            or ""
        )

        if normalization_block:
            return {
                "allowed": False,
                "reason": (
                    normalization_block
                ),
                "normalization": (
                    normalization
                ),
            }

        preflight = _preflight(
            lane.testnet,
            normalized,
        )

    except Exception as exc:
        return {
            "allowed": False,
            "reason": (
                "execution_first_probe_error"
            ),
            "error_type": (
                type(exc).__name__
            ),
        }

    return {
        "allowed": (
            preflight.get(
                "allowed"
            )
            is True
        ),
        "reason": str(
            preflight.get(
                "reason"
            )
            or "entry_round_trip_blocked"
        ),
        "preflight": (
            copy.deepcopy(
                preflight
            )
        ),
        "normalization": (
            copy.deepcopy(
                normalization
            )
        ),
    }


class _ExecutionFirstCandidateProxy:
    def __init__(
        self,
        service: Any,
        lane: Any,
        now: float,
    ) -> None:
        self._service = service
        self._lane = lane
        self._now = now

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self._service,
            name,
        )

    def _signal_ready(
        self,
        symbol: str,
    ) -> bool:
        signal_method = getattr(
            self._service,
            "collective_signal",
            None,
        )

        # Preserve older/non-swarm fixtures.
        if not callable(signal_method):
            return True

        reason = ""
        age_seconds = None

        try:
            signal = (
                signal_method(symbol)
                or {}
            )

            age_seconds = signal.get(
                "age_seconds"
            )

            if signal.get("fresh") is True:
                return True

            reason = "fast_signal_not_fresh"

        except Exception as exc:
            reason = (
                "fast_signal_refresh_error:"
                + type(exc).__name__
            )

        pinned = False

        pinner = getattr(
            self._service,
            "pin_execution_candidate_symbols",
            None,
        )

        if not callable(pinner):
            pinner = getattr(
                self._service,
                "pin_execution_symbols",
                None,
            )

        if callable(pinner):
            try:
                pinner(
                    {symbol},
                    ttl_seconds=(
                        SIGNAL_REFRESH_PIN_SECONDS
                    ),
                )
                pinned = True
            except Exception:
                pinned = False

        with self._lane._lock:
            self._lane.state[
                "v1625_signal_refresh_deferrals"
            ] = (
                int(
                    self._lane.state.get(
                        "v1625_signal_refresh_deferrals"
                    )
                    or 0
                )
                + 1
            )

            if pinned:
                self._lane.state[
                    "v1625_execution_candidate_pins"
                ] = (
                    int(
                        self._lane.state.get(
                            "v1625_execution_candidate_pins"
                        )
                        or 0
                    )
                    + 1
                )

            self._lane.state[
                "v1625_last_signal_refresh"
            ] = {
                "symbol": symbol,
                "reason": reason,
                "age_seconds": age_seconds,
                "microstream_pinned": pinned,
                "candidate_returned": False,
                "execution_preflight_bypassed": False,
                "live_authority": False,
                "observed_at": self._now,
            }

        return False

    def collective_candidates(
        self,
        limit: int = 8,
    ) -> list[str]:
        bounded = max(
            1,
            min(
                48,
                int(limit),
            ),
        )

        # Search deeper than the caller's
        # immediate candidate window, but
        # remain inside the service's existing
        # strategy-ranked universe.
        requested = max(
            bounded,
            min(
                48,
                bounded + 16,
            ),
        )

        raw = list(
            self._service.collective_candidates(
                limit=requested
            )
            or []
        )

        # v1.60.36: preserve the intelligence-ranked universe while
        # rotating the expensive execution-preflight starting point.
        # This prevents every fast pass from spending its network budget
        # on the same front-ranked symbols.
        normalized_raw = []
        raw_seen = set()

        for value in raw:
            symbol = str(value or "").upper()

            if symbol and symbol not in raw_seen:
                normalized_raw.append(symbol)
                raw_seen.add(symbol)

        raw = normalized_raw

        with self._lane._lock:
            cursor_start = int(
                self._lane.state.get(
                    "v1636_execution_probe_cursor"
                )
                or 0
            )

        if raw:
            cursor_start %= len(raw)

            raw = (
                raw[cursor_start:]
                + raw[:cursor_start]
            )
        else:
            cursor_start = 0

        visited = 0

        snapshot = (
            self._lane.testnet.safe_snapshot()
        )

        current_positions = {
            str(symbol).upper()
            for symbol, quantity in (
                snapshot.get(
                    "positions"
                )
                or {}
            ).items()
            if _n(quantity) > 0.0
        }

        eligible_method = getattr(
            self._lane.testnet,
            "eligible_symbols",
            None,
        )

        eligible = (
            {
                str(symbol).upper()
                for symbol in (
                    eligible_method("USDT")
                    or set()
                )
            }
            if callable(
                eligible_method
            )
            else {
                str(symbol).upper()
                for symbol in getattr(
                    self._lane.testnet,
                    "_eligible_symbols",
                    set(),
                )
            }
        )

        free_usdt = _free_usdt(
            self._lane.testnet
        )

        available_quote = max(
            0.0,
            free_usdt
            - MIN_FREE_QUOTE_RESERVE_USD,
        )

        max_order = max(
            0.0,
            _n(
                getattr(
                    self._lane.testnet,
                    "max_order_usd",
                    0.0,
                )
            ),
        )

        with self._lane._lock:
            cache = copy.deepcopy(
                self._lane.state.get(
                    "v1619_probe_cache"
                )
                or {}
            )

        selected: list[str] = []
        seen: set[str] = set()

        probe_checks = 0
        probe_passes = 0
        probe_blocks = 0
        metadata_filters = 0
        cache_hits = 0
        budget_deferrals = 0

        block_reasons: dict[
            str,
            int,
        ] = {}

        new_cache = copy.deepcopy(
            cache
        )

        last_probe: dict[
            str,
            Any,
        ] = {}

        for value in raw:
            symbol = str(
                value or ""
            ).upper()

            if (
                not symbol
                or symbol in seen
            ):
                continue

            seen.add(symbol)
            visited += 1

            if (
                symbol in current_positions
                or (
                    eligible
                    and symbol
                    not in eligible
                )
                or _route_blocked_until(
                    self._lane,
                    symbol,
                )
                > self._now
            ):
                continue

            try:
                floor = _safe_minimum_cost(
                    self._lane.testnet,
                    symbol,
                )
            except Exception:
                reason = (
                    "market_metadata_unavailable"
                )

                metadata_filters += 1

                block_reasons[
                    reason
                ] = (
                    block_reasons.get(
                        reason,
                        0,
                    )
                    + 1
                )

                continue

            safe_required = max(
                0.0,
                _n(
                    floor.get(
                        "safe_required_usd"
                    )
                ),
            )

            metadata_reason = ""

            if (
                max_order > 0.0
                and safe_required
                > max_order
                + 1e-12
            ):
                metadata_reason = (
                    "safe_minimum_exceeds_order_cap"
                )

            elif (
                safe_required
                > available_quote
                + 1e-12
            ):
                metadata_reason = (
                    "minimum_cost_exceeds_free_quote"
                )

            if metadata_reason:
                metadata_filters += 1

                block_reasons[
                    metadata_reason
                ] = (
                    block_reasons.get(
                        metadata_reason,
                        0,
                    )
                    + 1
                )

                new_cache[
                    symbol
                ] = {
                    "allowed": False,
                    "reason": (
                        metadata_reason
                    ),
                    "expires_at": (
                        self._now
                        + FAIL_CACHE_SECONDS
                    ),
                }

                continue

            cached = (
                cache.get(symbol)
                or {}
            )

            if (
                _n(
                    cached.get(
                        "expires_at"
                    )
                )
                > self._now
            ):
                cache_hits += 1

                if (
                    cached.get(
                        "allowed"
                    )
                    is True
                ):
                    if self._signal_ready(
                        symbol
                    ):
                        selected.append(
                            symbol
                        )

                        if (
                            len(selected)
                            >= bounded
                        ):
                            break

                continue

            if (
                probe_checks
                >= MAX_NETWORK_PROBES_PER_CALL
            ):
                # Leave this candidate at the head of the next rotating
                # execution-preflight window instead of skipping it.
                budget_deferrals += 1
                visited = max(0, visited - 1)
                break

            result = _probe_candidate(
                self._lane,
                symbol,
            )

            probe_checks += 1

            allowed = (
                result.get(
                    "allowed"
                )
                is True
            )

            reason = str(
                result.get("reason")
                or (
                    "round_trip_executable"
                    if allowed
                    else "entry_round_trip_blocked"
                )
            )

            if allowed:
                probe_passes += 1

                if self._signal_ready(
                    symbol
                ):
                    selected.append(
                        symbol
                    )

                ttl = (
                    PASS_CACHE_SECONDS
                )

            else:
                probe_blocks += 1

                block_reasons[
                    reason
                ] = (
                    block_reasons.get(
                        reason,
                        0,
                    )
                    + 1
                )

                ttl = (
                    FAIL_CACHE_SECONDS
                )

            new_cache[
                symbol
            ] = {
                "allowed": allowed,
                "reason": reason,
                "expires_at": (
                    self._now + ttl
                ),
                "detail": copy.deepcopy(
                    result
                ),
            }

            last_probe = {
                "symbol": symbol,
                "allowed": allowed,
                "reason": reason,
                "detail": copy.deepcopy(
                    result
                ),
                "observed_at": (
                    self._now
                ),
                "executor_order_created": (
                    False
                ),
                "live_authority": False,
            }

            if (
                len(selected)
                >= bounded
            ):
                break

        next_cursor = (
            (
                cursor_start
                + visited
            )
            % len(raw)
            if raw
            else 0
        )

        with self._lane._lock:
            self._lane.state[
                "v1636_execution_probe_cursor"
            ] = next_cursor

            self._lane.state[
                "v1636_execution_probe_rotation_calls"
            ] = (
                int(
                    self._lane.state.get(
                        "v1636_execution_probe_rotation_calls"
                    )
                    or 0
                )
                + 1
            )

            self._lane.state[
                "v1619_probe_cache"
            ] = new_cache

            self._lane.state[
                "v1619_candidate_probe_checks"
            ] = (
                int(
                    self._lane.state.get(
                        "v1619_candidate_probe_checks"
                    )
                    or 0
                )
                + probe_checks
            )

            self._lane.state[
                "v1619_candidate_probe_passes"
            ] = (
                int(
                    self._lane.state.get(
                        "v1619_candidate_probe_passes"
                    )
                    or 0
                )
                + probe_passes
            )

            self._lane.state[
                "v1619_candidate_probe_blocks"
            ] = (
                int(
                    self._lane.state.get(
                        "v1619_candidate_probe_blocks"
                    )
                    or 0
                )
                + probe_blocks
            )

            self._lane.state[
                "v1619_metadata_filters"
            ] = (
                int(
                    self._lane.state.get(
                        "v1619_metadata_filters"
                    )
                    or 0
                )
                + metadata_filters
            )

            self._lane.state[
                "v1619_probe_cache_hits"
            ] = (
                int(
                    self._lane.state.get(
                        "v1619_probe_cache_hits"
                    )
                    or 0
                )
                + cache_hits
            )

            self._lane.state[
                "v1619_probe_budget_deferrals"
            ] = (
                int(
                    self._lane.state.get(
                        "v1619_probe_budget_deferrals"
                    )
                    or 0
                )
                + budget_deferrals
            )

            reasons = (
                self._lane.state.setdefault(
                    "v1619_candidate_block_reasons",
                    {},
                )
            )

            for reason, count in (
                block_reasons.items()
            ):
                reasons[
                    reason
                ] = (
                    int(
                        reasons.get(
                            reason
                        )
                        or 0
                    )
                    + int(count)
                )

            if last_probe:
                self._lane.state[
                    "v1619_last_candidate_probe"
                ] = last_probe

            self._lane.state[
                "v1619_last_candidate_selection"
            ] = {
                "raw_count": len(
                    raw
                ),
                "requested_count": (
                    requested
                ),
                "selected": list(
                    selected
                ),
                "selected_count": len(
                    selected
                ),
                "network_probes": (
                    probe_checks
                ),
                "metadata_filters": (
                    metadata_filters
                ),
                "cache_hits": (
                    cache_hits
                ),
                "probe_budget_deferrals": (
                    budget_deferrals
                ),
                "probe_cursor_start": cursor_start,
                "probe_cursor_next": next_cursor,
                "persistent_rotating_probe_cursor": True,
                "cyclic_strategy_rank_order_preserved": True,
                "free_usdt": (
                    free_usdt
                ),
                "available_quote_usd": (
                    available_quote
                ),
                "arbitrary_market_injection": (
                    False
                ),
                "strategy_rank_order_preserved": (
                    True
                ),
                "live_authority": False,
                "observed_at": (
                    self._now
                ),
            }

            self._lane._save_locked()

        return selected[:bounded]


def install_testnet_execution_first_candidates_v1619() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1619_execution_first_candidates_installed",
        False,
    ):
        return

    original_step = (
        HyperSpeedCollectiveTestnetLane.step
    )

    original_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def step(
        self: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        import time

        current = (
            time.time()
            if now is None
            else float(now)
        )

        if not _supported(
            self.testnet
        ):
            return original_step(
                self,
                now=current,
            )

        provider = (
            self.service_provider
        )

        try:
            service = provider()
        except Exception:
            return original_step(
                self,
                now=current,
            )

        if service is None:
            return original_step(
                self,
                now=current,
            )

        proxy = (
            _ExecutionFirstCandidateProxy(
                service,
                self,
                current,
            )
        )

        self.service_provider = (
            lambda: proxy
        )

        try:
            return original_step(
                self,
                now=current,
            )
        finally:
            self.service_provider = (
                provider
            )

    def health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_health(
                self
            )
        )

        with self._lock:
            payload[
                "execution_first_candidate_substitution"
            ] = {
                "version": "1.60.36",
                "enabled": True,
                "maximum_network_probes_per_call": (
                    MAX_NETWORK_PROBES_PER_CALL
                ),
                "persistent_rotating_probe_cursor": True,
                "probe_cursor": int(
                    self.state.get(
                        "v1636_execution_probe_cursor"
                    )
                    or 0
                ),
                "failed_probe_cache_seconds": (
                    FAIL_CACHE_SECONDS
                ),
                "passed_probe_cache_seconds": (
                    PASS_CACHE_SECONDS
                ),
                "free_quote_aware": True,
                "exchange_minimum_aware": True,
                "strategy_rank_order_preserved": True,
                "arbitrary_market_injection": False,
                "executor_order_created_during_probe": False,
                "v1616_submit_preflight_preserved": True,
                "v1613_executor_recheck_preserved": True,
                "modeled_cost_floor_lowered": False,
                "price_limit_protection_lowered": False,
                "liquidity_protection_lowered": False,
                "checks": int(
                    self.state.get(
                        "v1619_candidate_probe_checks"
                    )
                    or 0
                ),
                "passes": int(
                    self.state.get(
                        "v1619_candidate_probe_passes"
                    )
                    or 0
                ),
                "blocks": int(
                    self.state.get(
                        "v1619_candidate_probe_blocks"
                    )
                    or 0
                ),
                "metadata_filters": int(
                    self.state.get(
                        "v1619_metadata_filters"
                    )
                    or 0
                ),
                "cache_hits": int(
                    self.state.get(
                        "v1619_probe_cache_hits"
                    )
                    or 0
                ),
                "probe_budget_deferrals": int(
                    self.state.get(
                        "v1619_probe_budget_deferrals"
                    )
                    or 0
                ),
                "block_reasons": (
                    copy.deepcopy(
                        self.state.get(
                            "v1619_candidate_block_reasons"
                        )
                        or {}
                    )
                ),
                "last_probe": (
                    copy.deepcopy(
                        self.state.get(
                            "v1619_last_candidate_probe"
                        )
                        or {}
                    )
                ),
                "last_selection": (
                    copy.deepcopy(
                        self.state.get(
                            "v1619_last_candidate_selection"
                        )
                        or {}
                    )
                ),
                "live_authority": False,
            }

        payload["version"] = (
            "1.60.19"
        )

        payload[
            "live_authority"
        ] = False

        return payload

    HyperSpeedCollectiveTestnetLane.step = (
        step
    )

    HyperSpeedCollectiveTestnetLane.health = (
        health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.19"
    )

    VelocitySniperTestnetLane.VERSION = (
        "1.60.19"
    )

    HyperSpeedCollectiveTestnetLane._v1619_execution_first_candidates_installed = (
        True
    )
