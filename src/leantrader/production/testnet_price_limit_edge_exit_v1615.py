from __future__ import annotations

import copy
from typing import Any

from .testnet_exit_price_guard_v1611 import (
    _fresh_bid,
    _price_limit,
)

PRICE_LIMIT_PROBE_SECONDS = 2.0
POST_REJECTION_COOLDOWN_SECONDS = 15.0
PREPARATION_RETRY_SECONDS = 15.0


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _base_exit_reason(value: Any) -> str:
    reason = str(
        value
        or "fast_collective_testnet_exit"
    )

    marker = ":corrected_recycle"

    if marker in reason:
        reason = reason.split(
            marker,
            1,
        )[0]

    return reason


def _watch_snapshot(
    self: Any,
    symbol: str,
) -> dict[str, Any]:
    with self._lock:
        return copy.deepcopy(
            (
                self.state.get(
                    "v1615_price_limit_watch"
                )
                or {}
            ).get(symbol)
            or {}
        )


def _record_watch(
    self: Any,
    *,
    symbol: str,
    now: float,
    reason: str,
    fresh_bid: float = 0.0,
    fresh_ask: float = 0.0,
    sell_limit: float = 0.0,
    next_probe_at: float | None = None,
    executable_boundary: bool = False,
    preparation: dict[str, Any] | None = None,
) -> None:
    with self._lock:
        watches = self.state.setdefault(
            "v1615_price_limit_watch",
            {},
        )

        previous = watches.get(symbol) or {}

        watches[symbol] = {
            "symbol": symbol,
            "reason": reason,
            "observed_at": now,
            "fresh_bid": max(
                0.0,
                fresh_bid,
            ),
            "fresh_ask": max(
                0.0,
                fresh_ask,
            ),
            "sell_limit": max(
                0.0,
                sell_limit,
            ),
            "executable_boundary": bool(
                executable_boundary
            ),
            "checks": int(
                previous.get("checks")
                or 0
            )
            + 1,
            "next_probe_at": (
                now
                + PRICE_LIMIT_PROBE_SECONDS
                if next_probe_at is None
                else float(next_probe_at)
            ),
            "preparation": copy.deepcopy(
                preparation or {}
            ),
            "order_submitted_by_watch": False,
            "live_authority": False,
        }

        self.state[
            "v1615_price_limit_checks"
        ] = (
            int(
                self.state.get(
                    "v1615_price_limit_checks"
                )
                or 0
            )
            + 1
        )

        self._save_locked()


def _recent_rejection_wait(
    testnet: Any,
    symbol: str,
    now: float,
) -> float:
    row = (
        testnet.state.get(
            "v1611_last_price_limit_rejection"
        )
        or {}
    )

    if (
        str(
            row.get("symbol")
            or ""
        ).upper()
        != symbol
    ):
        return 0.0

    observed = _n(
        row.get("observed_at")
    )

    if observed <= 0.0:
        return 0.0

    age = max(
        0.0,
        now - observed,
    )

    return max(
        0.0,
        POST_REJECTION_COOLDOWN_SECONDS
        - age,
    )


def _clear_verified_price_limit_cooldown(
    testnet: Any,
    symbol: str,
) -> None:
    with testnet._io_lock:
        (
            testnet.state.get(
                "v1611_price_limit_blocked_until"
            )
            or {}
        ).pop(
            symbol,
            None,
        )

        testnet._save_state()


def install_testnet_price_limit_edge_exit_v1615() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .testnet_execution import (
        BybitTestnetExecutionEngine,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1615_price_limit_edge_exit_installed",
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

        with self._lock:
            queue = copy.deepcopy(
                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).get(normalized)
            )

        if not isinstance(
            queue,
            dict,
        ):
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        watch = _watch_snapshot(
            self,
            normalized,
        )

        next_probe = _n(
            watch.get(
                "next_probe_at"
            )
        )

        if next_probe > now:
            return self._decision(
                "price_limit_exit_watch",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "reason": watch.get(
                        "reason"
                    ),
                    "fresh_bid": watch.get(
                        "fresh_bid"
                    ),
                    "sell_limit": watch.get(
                        "sell_limit"
                    ),
                    "retry_in_seconds": max(
                        0.0,
                        next_probe - now,
                    ),
                    "order_submitted": False,
                    "position_remains_active": True,
                    "live_authority": False,
                },
            )

        exchange = getattr(
            self.testnet,
            "exchange",
            None,
        )

        if exchange is None:
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        try:
            bid, ask = _fresh_bid(
                self.testnet,
                normalized,
            )

            limit = _price_limit(
                self.testnet,
                normalized,
            )

        except Exception:
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        if (
            limit.get("supported")
            is not True
            or limit.get("ok")
            is not True
        ):
            # Endpoint uncertainty remains fail-closed and uses
            # the existing bounded recovery path.
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        sell_limit = max(
            0.0,
            _n(
                limit.get(
                    "sell_limit"
                )
            ),
        )

        if (
            bid <= 0.0
            or sell_limit <= 0.0
            or bid + 1e-12
            < sell_limit
        ):
            _record_watch(
                self,
                symbol=normalized,
                now=now,
                reason=(
                    "bybit_sell_boundary_not_executable"
                ),
                fresh_bid=bid,
                fresh_ask=ask,
                sell_limit=sell_limit,
                executable_boundary=False,
            )

            return self._decision(
                "price_limit_exit_watch",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "fresh_bid": bid,
                    "fresh_ask": ask,
                    "sell_limit": sell_limit,
                    "order_submitted": False,
                    "position_remains_active": True,
                    "deferral_incremented": False,
                    "live_authority": False,
                },
            )

        rejection_wait = (
            _recent_rejection_wait(
                self.testnet,
                normalized,
                now,
            )
        )

        if rejection_wait > 0.0:
            _record_watch(
                self,
                symbol=normalized,
                now=now,
                reason=(
                    "recent_price_limit_rejection_cooldown"
                ),
                fresh_bid=bid,
                fresh_ask=ask,
                sell_limit=sell_limit,
                executable_boundary=True,
                next_probe_at=(
                    now
                    + min(
                        PRICE_LIMIT_PROBE_SECONDS,
                        rejection_wait,
                    )
                ),
            )

            return self._decision(
                "price_limit_exit_watch",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "fresh_bid": bid,
                    "sell_limit": sell_limit,
                    "retry_in_seconds": (
                        rejection_wait
                    ),
                    "order_submitted": False,
                    "position_remains_active": True,
                    "live_authority": False,
                },
            )

        # The public boundary is now executable. Only now may the
        # stale v1.60.11 internal cooldown be cleared.
        _clear_verified_price_limit_cooldown(
            self.testnet,
            normalized,
        )

        current_quantity = max(
            0.0,
            _n(
                (
                    snapshot.get(
                        "positions"
                    )
                    or {}
                ).get(
                    normalized
                ),
                _n(
                    record.get(
                        "quantity"
                    )
                ),
            ),
        )

        preparation = (
            self.testnet.prepare_sell(
                normalized,
                current_quantity,
                bid,
            )
        )

        prep_status = str(
            preparation.get(
                "status"
            )
            or ""
        )

        if prep_status == "dust":
            with self._lock:
                (
                    self.state.get(
                        "active"
                    )
                    or {}
                ).pop(
                    normalized,
                    None,
                )

                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).pop(
                    normalized,
                    None,
                )

                self.state.setdefault(
                    "v1615_dust_reclassifications",
                    [],
                ).append(
                    {
                        "symbol": normalized,
                        "recorded_at": now,
                        "preparation": copy.deepcopy(
                            preparation
                        ),
                        "counted_as_executed_close": False,
                        "live_authority": False,
                    }
                )

                self.state[
                    "v1615_dust_reclassifications"
                ] = self.state[
                    "v1615_dust_reclassifications"
                ][-100:]

                self._save_locked()

            return self._decision(
                "active_exit_reclassified_dust",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "preparation": preparation,
                    "counted_as_executed_close": False,
                    "live_authority": False,
                },
            )

        if prep_status != "executable":
            _record_watch(
                self,
                symbol=normalized,
                now=now,
                reason=(
                    "price_limit_clear_but_sell_preparation_blocked"
                ),
                fresh_bid=bid,
                fresh_ask=ask,
                sell_limit=sell_limit,
                executable_boundary=True,
                next_probe_at=(
                    now
                    + PREPARATION_RETRY_SECONDS
                ),
                preparation=preparation,
            )

            return self._decision(
                "price_limit_clear_exit_preparation_waiting",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "preparation": preparation,
                    "order_submitted": False,
                    "position_remains_active": True,
                    "live_authority": False,
                },
            )

        with self._lock:
            live_queue = (
                self.state.get(
                    "deferred_exit_recoveries"
                )
                or {}
            ).get(normalized)

            if isinstance(
                live_queue,
                dict,
            ):
                live_queue[
                    "next_retry_at"
                ] = now

                source_event = (
                    live_queue.get(
                        "source_event"
                    )
                    or {}
                )

                if isinstance(
                    source_event,
                    dict,
                ):
                    source_event[
                        "reason"
                    ] = _base_exit_reason(
                        source_event.get(
                            "reason"
                        )
                    )

                    source_event[
                        "price"
                    ] = bid

                    source_event[
                        "quantity"
                    ] = max(
                        0.0,
                        _n(
                            preparation.get(
                                "executable_quantity"
                            ),
                            current_quantity,
                        ),
                    )

                self._save_locked()

        _record_watch(
            self,
            symbol=normalized,
            now=now,
            reason=(
                "sell_boundary_executable_retry_released"
            ),
            fresh_bid=bid,
            fresh_ask=ask,
            sell_limit=sell_limit,
            executable_boundary=True,
            preparation=preparation,
        )

        # Existing v1.60.8/v1.60.9 logic performs the actual
        # reconciled/idempotent submission and handles any race.
        return original_manage(
            self,
            service,
            self.testnet.safe_snapshot(),
            symbol,
            record,
            now=now,
        )

    def health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_health(self)
        )

        with self._lock:
            watches = copy.deepcopy(
                self.state.get(
                    "v1615_price_limit_watch"
                )
                or {}
            )

        payload["version"] = "1.60.15"

        payload[
            "price_limit_edge_exit"
        ] = {
            "version": "1.60.15",
            "enabled": True,
            "probe_seconds": (
                PRICE_LIMIT_PROBE_SECONDS
            ),
            "post_rejection_cooldown_seconds": (
                POST_REJECTION_COOLDOWN_SECONDS
            ),
            "public_boundary_probe_only_while_blocked": True,
            "order_submission_while_bid_below_sell_limit": False,
            "clear_internal_cooldown_only_after_verified_boundary": True,
            "watch_count": len(
                watches
            ),
            "checks": int(
                self.state.get(
                    "v1615_price_limit_checks"
                )
                or 0
            ),
            "watches": watches,
            "reason_growth_bounded": True,
            "ambiguous_resubmission_allowed": False,
            "live_authority": False,
        }

        payload["live_authority"] = False

        return payload

    HyperSpeedCollectiveTestnetLane._manage_active = (
        manage_active
    )

    HyperSpeedCollectiveTestnetLane.health = (
        health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.15"
    )

    VelocitySniperTestnetLane.VERSION = (
        "1.60.15"
    )

    BybitTestnetExecutionEngine.VERSION = (
        "3.1"
    )

    HyperSpeedCollectiveTestnetLane._v1615_price_limit_edge_exit_installed = (
        True
    )
