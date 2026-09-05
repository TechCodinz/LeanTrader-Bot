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
# v1.60.38: when the normal two-probe window finds no executable
# candidate, permit two additional rotating probes. Once at least
# one executable candidate exists, retain the low-latency two-probe
# ceiling. All execution preflight protections remain authoritative.
MAX_EMPTY_SELECTION_NETWORK_PROBES_PER_CALL = 4
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
        *,
        pin_on_miss: bool = True,
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

            micro_velocity = (
                signal.get("micro_velocity")
                or {}
            )

            age_seconds = _n(
                micro_velocity.get(
                    "age_seconds"
                ),
                1_000_000.0,
            )

            if (
                signal.get("fresh") is True
                and micro_velocity.get(
                    "fresh"
                ) is True
                and age_seconds <= 2.0
            ):
                return True

            if signal.get("fresh") is not True:
                reason = "fast_signal_not_fresh"
            elif micro_velocity.get("fresh") is not True:
                reason = "execution_micro_velocity_not_fresh"
            else:
                reason = "execution_micro_velocity_age_exceeded"

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

        if callable(pinner) and pin_on_miss:
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

    def _fresh_opportunity_priority(
        self,
        symbol: str,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Rank already-fresh execution signals without granting authority."""

        signal_method = getattr(
            self._service,
            "collective_signal",
            None,
        )

        # Preserve compatibility with older fixtures. They remain eligible
        # for the normal downstream gates but receive no artificial edge.
        if not callable(signal_method):
            return True, 0.0, {}

        try:
            signal = (
                signal_method(symbol)
                or {}
            )
        except Exception:
            return False, 0.0, {}

        micro_velocity = (
            signal.get("micro_velocity")
            or {}
        )

        age_seconds = _n(
            micro_velocity.get(
                "age_seconds"
            ),
            1_000_000.0,
        )

        fresh = (
            signal.get("fresh") is True
            and micro_velocity.get(
                "fresh"
            ) is True
            and age_seconds <= 2.0
        )

        if not fresh:
            return False, 0.0, signal

        # v1.60.55: collective_signal() returns the RAW micro-velocity
        # snapshot. projected_capture_bps_5s and qualified_long are derived
        # by the VelocitySniper lane and therefore are not guaranteed to
        # exist in that raw mapping.
        #
        # v1.60.54 incorrectly attempted to rank raw candidates using those
        # post-assessment fields, which could collapse every fresh candidate
        # to a 0-bps priority score before scarce authenticated preflight
        # probes were spent.
        #
        # Reuse the lane's exact read-only velocity derivation when available.
        # This grants no execution authority and changes no hard gate.
        velocity_state: dict[str, Any] = {}

        velocity_method = getattr(
            self._lane,
            "_velocity_state",
            None,
        )

        if callable(velocity_method):
            try:
                velocity_state = (
                    velocity_method(signal)
                    or {}
                )
            except Exception:
                velocity_state = {}

        if velocity_state:
            projected_capture = max(
                0.0,
                _n(
                    velocity_state.get(
                        "projected_capture_bps_5s"
                    )
                ),
            )

            velocity_qualified = (
                velocity_state.get(
                    "qualified_long"
                )
                is True
            )

        else:
            # Compatibility fallback for older/lightweight lane fixtures.
            trend_5s = _n(
                micro_velocity.get(
                    "recent_midpoint_trend_bps_5s"
                )
            )

            velocity_bps_s = _n(
                micro_velocity.get(
                    "midpoint_velocity_bps_per_second"
                )
            )

            acceleration_bps_s2 = _n(
                micro_velocity.get(
                    "midpoint_acceleration_bps_per_second2"
                )
            )

            projected_capture = max(
                0.0,
                trend_5s,
                velocity_bps_s * 5.0
                + max(
                    0.0,
                    acceleration_bps_s2,
                )
                * 4.0,
            )

            # Full qualification contains spread/depth/sample constraints.
            # Never recreate or relax it incompletely in the fallback.
            velocity_qualified = False

        # v1.60.54 also read micro_support/qualified_micro_proposals,
        # which are assessment-layer fields. At this point we still possess
        # the raw signal, so rank from the actual raw path assessments that
        # later feed micro_support.
        micro = (
            signal.get("microstructure")
            or {}
        )

        path_rows = [
            row
            for row in (
                micro.get("path_assessments")
                or []
            )
            if isinstance(row, dict)
        ]

        micro_edges: list[float] = []

        for row in path_rows:
            direction = str(
                row.get("direction")
                or ""
            ).lower()

            confidence = max(
                0.0,
                _n(
                    row.get("confidence")
                ),
            )

            edge = _n(
                row.get(
                    "expected_edge_bps"
                )
            )

            if (
                direction
                in {
                    "long",
                    "buy",
                    "bull",
                    "bullish",
                }
                and confidence >= 0.10
                and edge > 0.0
            ):
                micro_edges.append(edge)

        best_micro_edge = max(
            micro_edges
            or [0.0]
        )

        # Ranking only. This score does not satisfy or replace v1634.
        # The authenticated candidate still has to survive every existing
        # execution, profitability, cost, liquidity, freshness, sellability
        # and reconciliation gate.
        priority_score = max(
            projected_capture,
            best_micro_edge,
        )

        if velocity_qualified:
            priority_score += 0.001

        return (
            True,
            priority_score,
            signal,
        )

    def _warm_execution_candidate_cohort(
        self,
        symbols: list[str],
    ) -> list[str]:
        """Warm a bounded candidate cohort without granting execution authority."""

        normalized: list[str] = []

        for value in symbols:
            symbol = str(value or "").upper()

            if (
                symbol
                and symbol not in normalized
            ):
                normalized.append(symbol)

        adaptive_capacity = int(
            _n(
                getattr(
                    self._service,
                    "_precision_micro_capacity",
                    6,
                ),
                6.0,
            )
        )

        # v1.60.53: keep a small but meaningful execution-ready cohort hot.
        # This is sampling/warming authority only; it creates no order and
        # cannot bypass subsequent strategy or authenticated preflight gates.
        # Honor the precision service's actual adaptive capacity.
        # A capacity of one means warm one symbol, not four. This avoids
        # over-subscribing a constrained microstream while preserving the
        # existing upper bound and all execution gates.
        capacity = max(
            1,
            min(
                12,
                adaptive_capacity,
            ),
        )

        cohort = normalized[:capacity]

        if not cohort:
            return []

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

        if not callable(pinner):
            return []

        try:
            # One wake-up for the whole cohort prevents successive stale
            # candidates from thrashing the precision queue individually.
            pinner(
                list(reversed(cohort)),
                ttl_seconds=SIGNAL_REFRESH_PIN_SECONDS,
            )
        except Exception:
            return []

        with self._lane._lock:
            # v1.60.51: cohort warming is the component that actually
            # performs the pin. Reflect that in the v1625 diagnostic
            # rather than leaving microstream_pinned=False after a
            # successful cohort pin.
            last_refresh = dict(
                self._lane.state.get(
                    "v1625_last_signal_refresh"
                )
                or {}
            )

            if str(
                last_refresh.get("symbol")
                or ""
            ).upper() in set(cohort):
                last_refresh[
                    "microstream_pinned"
                ] = True
                last_refresh[
                    "pin_source"
                ] = "candidate_cohort_warm"

                self._lane.state[
                    "v1625_last_signal_refresh"
                ] = last_refresh

            self._lane.state[
                "v1625_execution_candidate_pins"
            ] = (
                int(
                    self._lane.state.get(
                        "v1625_execution_candidate_pins"
                    )
                    or 0
                )
                + len(cohort)
            )

            self._lane.state[
                "v1648_candidate_warm_calls"
            ] = (
                int(
                    self._lane.state.get(
                        "v1648_candidate_warm_calls"
                    )
                    or 0
                )
                + 1
            )

            self._lane.state[
                "v1648_candidate_warm_symbols"
            ] = (
                int(
                    self._lane.state.get(
                        "v1648_candidate_warm_symbols"
                    )
                    or 0
                )
                + len(cohort)
            )

            self._lane.state[
                "v1648_last_candidate_warm_cohort"
            ] = {
                "symbols": list(cohort),
                "count": len(cohort),
                "adaptive_capacity": capacity,
                "freshness_requirement_seconds": 2.0,
                "execution_authority": False,
                "testnet_order_created": False,
                "normal_execution_preflight_still_required": True,
                "network_probe_budget_changed": False,
                "live_authority": False,
                "observed_at": self._now,
            }

        return cohort

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

        # v1.60.54: capital limits entries, and Bybit probe budget limits
        # authenticated preflights, but neither should force the scarce probe
        # onto the first merely-fresh symbol encountered.
        #
        # Re-rank a bounded front window using only current <=2s micro evidence.
        # Stale rows remain in the universe for cohort warming. This is purely
        # read-only prioritization and grants no execution authority.
        priority_window = min(
            12,
            len(raw),
        )

        priority_rows: list[
            tuple[str, bool, float, int]
        ] = []

        for rank_index, symbol in enumerate(
            raw[:priority_window]
        ):
            fresh, score, _signal = (
                self._fresh_opportunity_priority(
                    symbol
                )
            )

            priority_rows.append(
                (
                    symbol,
                    fresh,
                    score,
                    rank_index,
                )
            )

        priority_rows.sort(
            key=lambda row: (
                0 if row[1] else 1,
                -row[2],
                row[3],
            )
        )

        prioritized_front = [
            row[0]
            for row in priority_rows
        ]

        raw = (
            prioritized_front
            + raw[priority_window:]
        )

        fresh_priority_count = sum(
            1
            for row in priority_rows
            if row[1]
        )

        highest_priority_score = max(
            [
                row[2]
                for row in priority_rows
                if row[1]
            ]
            or [0.0]
        )

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

        # v1.60.50: stop expensive candidate probing once the current
        # authenticated quote balance can no longer fund another minimum
        # fast-lane ticket. Previously the wrapper kept searching toward the
        # caller's broad limit (often 48) after finding a fresh executable
        # candidate, allowing that candidate to age beyond the 2-second
        # execution freshness gate before the entry assessor consumed it.
        #
        # This does not increase authority or relax any gate. As capital grows,
        # the target expands again up to the lane's existing adaptive entry cap.
        minimum_ticket = max(
            1.0,
            _n(
                getattr(
                    self._lane,
                    "order_usd",
                    1.0,
                ),
                1.0,
            ),
        )

        # v1.60.51: authenticated quote capital constrains how many
        # positions may be OPENED, not how much existing intelligence
        # may be assessed. The hyper lane still derives the final
        # entry_limit from authenticated capital and executor capacity.
        capital_funded_candidates = max(
            1,
            int(
                available_quote
                // minimum_ticket
            ),
        )

        lane_entry_cap = max(
            1,
            int(
                getattr(
                    self._lane,
                    "maximum_adaptive_entries_per_cycle",
                    1,
                )
                or 1
            ),
        )

        selection_target = max(
            1,
            min(
                bounded,
                lane_entry_cap,
            ),
        )

        selected: list[str] = []
        seen: set[str] = set()
        metadata_warm_candidates: list[str] = []
        execution_clean_stale: list[str] = []

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
                        symbol,
                        pin_on_miss=False,
                    ):
                        selected.append(
                            symbol
                        )

                        if (
                            len(selected)
                            >= selection_target
                        ):
                            break
                    else:
                        execution_clean_stale.append(
                            symbol
                        )

                continue

            # v1.60.53: freshness comes before expensive authenticated
            # execution probing. A stale micro/velocity snapshot cannot
            # authorize an entry anyway, so spending one of the small Bybit
            # preflight slots on it only starves fresher ranked candidates.
            #
            # The candidate is warmed without execution authority and can be
            # reconsidered on the next fast pass. The existing <=2s execution
            # freshness rule remains unchanged.
            if not self._signal_ready(
                symbol,
                pin_on_miss=False,
            ):
                execution_clean_stale.append(
                    symbol
                )
                metadata_warm_candidates.append(
                    symbol
                )
                continue

            metadata_warm_candidates.append(
                symbol
            )

            probe_budget = (
                MAX_NETWORK_PROBES_PER_CALL
                if selected
                else MAX_EMPTY_SELECTION_NETWORK_PROBES_PER_CALL
            )

            if (
                probe_checks
                >= probe_budget
            ):
                # Leave this candidate at the head of the next rotating
                # execution-preflight window instead of skipping it.
                # v1.60.38 only expands the budget while no executable
                # candidate has been found during this fast pass.
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
                    symbol,
                    pin_on_miss=False,
                ):
                    selected.append(
                        symbol
                    )
                else:
                    execution_clean_stale.append(
                        symbol
                    )

                ttl = (
                    PASS_CACHE_SECONDS
                )

            else:
                probe_blocks += 1

                if symbol in metadata_warm_candidates:
                    metadata_warm_candidates.remove(
                        symbol
                    )

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
                >= selection_target
            ):
                break

        selected_set = set(selected)

        warm_candidates = [
            *execution_clean_stale,
            *[
                symbol
                for symbol in metadata_warm_candidates
                if symbol not in selected_set
            ],
        ]

        warmed_candidates = (
            self._warm_execution_candidate_cohort(
                warm_candidates
            )
        )

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
                "selection_target": selection_target,
                "capital_funded_candidate_target": (
                    capital_funded_candidates
                ),
                "minimum_candidate_ticket_usd": (
                    minimum_ticket
                ),
                "fresh_selection_budgeting": True,
                "capital_limits_entries_not_assessment": True,
                "assessment_target": selection_target,
                "authenticated_entry_capacity_target": (
                    capital_funded_candidates
                ),
                "candidate_warm_cohort": list(
                    warmed_candidates
                ),
                "candidate_warm_count": len(
                    warmed_candidates
                ),
                "adaptive_candidate_cohort_warming": True,
                "fresh_first_execution_probing": True,
                "stale_candidates_consume_network_probe": False,
                "opportunity_prioritized_fresh_routing": True,
                "raw_signal_priority_alignment": True,
                "velocity_state_reused_for_priority": True,
                "raw_micro_path_priority": True,
                "opportunity_priority_window": priority_window,
                "fresh_priority_candidates": fresh_priority_count,
                "highest_fresh_priority_score_bps": highest_priority_score,
                "priority_is_ranking_only": True,
                "priority_bypasses_profit_gate": False,
                "priority_bypasses_execution_preflight": False,
                "freshness_gate_seconds": 2.0,
                "warmed_candidates_require_normal_preflight": True,
                "network_probe_budget_changed": False,
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
