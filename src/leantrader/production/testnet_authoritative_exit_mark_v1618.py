from __future__ import annotations

import copy
from typing import Any

from .testnet_exit_price_guard_v1611 import (
    _fresh_bid,
)


EXIT_MARK_CACHE_SECONDS = 1.0


def _n(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _spread_bps(
    bid: float,
    ask: float,
) -> float:
    if (
        bid <= 0.0
        or ask <= 0.0
        or ask < bid
    ):
        return 0.0

    midpoint = (
        bid + ask
    ) / 2.0

    if midpoint <= 0.0:
        return 0.0

    return (
        (ask - bid)
        / midpoint
        * 10_000.0
    )


def _cached_mark(
    lane: Any,
    symbol: str,
    now: float,
) -> dict[str, Any]:
    with lane._lock:
        cache = getattr(
            lane,
            "_v1618_exit_mark_cache",
            {},
        )

        row = copy.deepcopy(
            cache.get(symbol)
            or {}
        )

    observed = _n(
        row.get("observed_at")
    )

    if (
        _n(row.get("bid")) > 0.0
        and observed > 0.0
        and now - observed
        <= EXIT_MARK_CACHE_SECONDS
    ):
        row["cached"] = True
        return row

    return {}


def _fresh_mark(
    lane: Any,
    symbol: str,
    now: float,
) -> dict[str, Any]:
    cached = _cached_mark(
        lane,
        symbol,
        now,
    )

    if cached:
        return cached

    try:
        bid, ask = _fresh_bid(
            lane.testnet,
            symbol,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": (
                "fresh_testnet_bid_error"
            ),
            "error_type": (
                type(exc).__name__
            ),
            "observed_at": now,
            "cached": False,
        }

    bid = max(
        0.0,
        _n(bid),
    )

    ask = max(
        0.0,
        _n(ask),
    )

    if bid <= 0.0:
        return {
            "available": False,
            "reason": (
                "fresh_testnet_bid_unavailable"
            ),
            "bid": bid,
            "ask": ask,
            "observed_at": now,
            "cached": False,
        }

    row = {
        "available": True,
        "reason": (
            "fresh_testnet_executable_bid"
        ),
        "bid": bid,
        "ask": ask,
        "spread_bps": (
            _spread_bps(
                bid,
                ask,
            )
        ),
        "observed_at": now,
        "cached": False,
    }

    with lane._lock:
        cache = getattr(
            lane,
            "_v1618_exit_mark_cache",
            None,
        )

        if not isinstance(
            cache,
            dict,
        ):
            cache = {}
            lane._v1618_exit_mark_cache = (
                cache
            )

        cache[symbol] = (
            copy.deepcopy(row)
        )

    return row


class _ExitMarkServiceProxy:
    def __init__(
        self,
        service: Any,
        *,
        symbol: str,
        bid: float,
        ask: float,
        spread_bps: float,
    ) -> None:
        self._service = service
        self._symbol = symbol
        self._bid = bid
        self._ask = ask
        self._spread_bps = (
            spread_bps
        )

        self.internal_midpoint = 0.0
        self.internal_spread_bps = 0.0

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self._service,
            name,
        )

    def collective_signal(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        signal = copy.deepcopy(
            self._service.collective_signal(
                symbol
            )
        )

        normalized = str(
            symbol or ""
        ).upper()

        if normalized != self._symbol:
            return signal

        micro = signal.get(
            "microstructure"
        )

        if not isinstance(
            micro,
            dict,
        ):
            micro = {}

        features = micro.get(
            "features"
        )

        if not isinstance(
            features,
            dict,
        ):
            features = {}

        self.internal_midpoint = max(
            0.0,
            _n(
                features.get(
                    "midpoint"
                )
            ),
        )

        self.internal_spread_bps = max(
            0.0,
            _n(
                features.get(
                    "spread_bps"
                )
            ),
        )

        features[
            "testnet_internal_midpoint"
        ] = self.internal_midpoint

        features[
            "testnet_internal_spread_bps"
        ] = self.internal_spread_bps

        # v1.60.18:
        # Exit PnL/risk decisions must use
        # a price that can actually be sold
        # on the Testnet execution venue.
        features["midpoint"] = (
            self._bid
        )

        if self._spread_bps > 0.0:
            features["spread_bps"] = (
                self._spread_bps
            )

        features[
            "testnet_authoritative_exit_bid"
        ] = self._bid

        features[
            "testnet_authoritative_exit_ask"
        ] = self._ask

        micro["features"] = (
            features
        )

        signal["microstructure"] = (
            micro
        )

        signal[
            "testnet_authoritative_exit_mark"
        ] = {
            "bid": self._bid,
            "ask": self._ask,
            "spread_bps": (
                self._spread_bps
            ),
            "live_authority": False,
        }

        return signal


def _record_mark(
    lane: Any,
    *,
    symbol: str,
    now: float,
    mark: dict[str, Any],
    proxy: _ExitMarkServiceProxy | None = None,
    outcome: dict[str, Any] | None = None,
) -> None:
    with lane._lock:
        lane.state[
            "v1618_authoritative_exit_mark_checks"
        ] = (
            int(
                lane.state.get(
                    "v1618_authoritative_exit_mark_checks"
                )
                or 0
            )
            + 1
        )

        if (
            mark.get("available")
            is not True
        ):
            lane.state[
                "v1618_exit_mark_unavailable_waits"
            ] = (
                int(
                    lane.state.get(
                        "v1618_exit_mark_unavailable_waits"
                    )
                    or 0
                )
                + 1
            )

        live = (
            lane.state.get(
                "active"
            )
            or {}
        ).get(symbol)

        sentinel = {}

        if isinstance(
            live,
            dict,
        ):
            sentinel = (
                live.get(
                    "last_sentinel"
                )
                or {}
            )

        lane.state[
            "v1618_last_authoritative_exit_mark"
        ] = {
            "symbol": symbol,
            "available": (
                mark.get("available")
                is True
            ),
            "fresh_bid": (
                mark.get("bid")
            ),
            "fresh_ask": (
                mark.get("ask")
            ),
            "fresh_spread_bps": (
                mark.get(
                    "spread_bps"
                )
            ),
            "cached": bool(
                mark.get("cached")
            ),
            "mark_reason": (
                mark.get("reason")
            ),
            "error_type": (
                mark.get("error_type")
            ),
            "internal_midpoint": (
                proxy.internal_midpoint
                if proxy is not None
                else None
            ),
            "internal_spread_bps": (
                proxy.internal_spread_bps
                if proxy is not None
                else None
            ),
            "sentinel_price": (
                sentinel.get("price")
            ),
            "sentinel_gross_bps": (
                sentinel.get(
                    "gross_bps"
                )
            ),
            "sentinel_reason": (
                sentinel.get("reason")
            ),
            "outcome_reason": (
                (
                    outcome
                    or {}
                ).get("reason")
            ),
            "position_remains_active": (
                isinstance(
                    live,
                    dict,
                )
            ),
            "live_authority": False,
            "observed_at": now,
        }

        lane._save_locked()


def install_testnet_authoritative_exit_mark_v1618() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1618_authoritative_exit_mark_installed",
        False,
    ):
        return

    original_manage = (
        HyperSpeedCollectiveTestnetLane._manage_active
    )

    original_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def manage_active(
        self: Any,
        service: Any,
        snapshot: dict[str, Any],
        symbol: str,
        record: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        normalized = str(
            symbol
        ).upper()

        exchange = getattr(
            self.testnet,
            "exchange",
            None,
        )

        if (
            exchange is None
            or not callable(
                getattr(
                    exchange,
                    "fetch_ticker",
                    None,
                )
            )
        ):
            # Preserve legacy/non-exchange
            # Testnet adapters exactly.
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        current_total = max(
            0.0,
            _n(
                (
                    snapshot.get(
                        "positions"
                    )
                    or {}
                ).get(normalized)
            ),
        )

        if current_total <= 0.0:
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        with self._lock:
            queue = copy.deepcopy(
                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).get(normalized)
            )

        if isinstance(
            queue,
            dict,
        ):
            # v1.60.17 handles stale profit
            # queues; v1.60.15 handles active
            # protective price-limit watches.
            # Do not add another HTTP probe.
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        mark = _fresh_mark(
            self,
            normalized,
            now,
        )

        if (
            mark.get("available")
            is not True
        ):
            _record_mark(
                self,
                symbol=normalized,
                now=now,
                mark=mark,
            )

            return self._decision(
                "waiting_for_authoritative_testnet_exit_mark",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "reason": (
                        mark.get(
                            "reason"
                        )
                    ),
                    "error_type": (
                        mark.get(
                            "error_type"
                        )
                    ),
                    "order_submitted": False,
                    "position_remains_active": True,
                    "internal_midpoint_exit_allowed": False,
                    "live_authority": False,
                },
            )

        proxy = _ExitMarkServiceProxy(
            service,
            symbol=normalized,
            bid=max(
                0.0,
                _n(mark.get("bid")),
            ),
            ask=max(
                0.0,
                _n(mark.get("ask")),
            ),
            spread_bps=max(
                0.0,
                _n(
                    mark.get(
                        "spread_bps"
                    )
                ),
            ),
        )

        result = original_manage(
            self,
            proxy,
            snapshot,
            symbol,
            record,
            now=now,
        )

        _record_mark(
            self,
            symbol=normalized,
            now=now,
            mark=mark,
            proxy=proxy,
            outcome=result,
        )

        return result

    def health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_health(
                self
            )
        )

        with self._lock:
            last = copy.deepcopy(
                self.state.get(
                    "v1618_last_authoritative_exit_mark"
                )
                or {}
            )

            checks = int(
                self.state.get(
                    "v1618_authoritative_exit_mark_checks"
                )
                or 0
            )

            waits = int(
                self.state.get(
                    "v1618_exit_mark_unavailable_waits"
                )
                or 0
            )

        payload["version"] = (
            "1.60.18"
        )

        payload[
            "testnet_authoritative_exit_mark"
        ] = {
            "version": "1.60.18",
            "enabled": True,
            "authoritative_source": (
                "fresh_bybit_testnet_executable_bid"
            ),
            "cache_seconds": (
                EXIT_MARK_CACHE_SECONDS
            ),
            "active_exit_internal_midpoint_authority": False,
            "entry_signal_pricing_changed": False,
            "mtf_ranking_changed": False,
            "price_limit_protection_lowered": False,
            "modeled_cost_floor_lowered": False,
            "protective_exit_path_preserved": True,
            "checks": checks,
            "unavailable_waits": waits,
            "last_mark": last,
            "live_authority": False,
        }

        payload["live_authority"] = (
            False
        )

        return payload

    HyperSpeedCollectiveTestnetLane._manage_active = (
        manage_active
    )

    HyperSpeedCollectiveTestnetLane.health = (
        health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.18"
    )

    VelocitySniperTestnetLane.VERSION = (
        "1.60.18"
    )

    HyperSpeedCollectiveTestnetLane._v1618_authoritative_exit_mark_installed = (
        True
    )
