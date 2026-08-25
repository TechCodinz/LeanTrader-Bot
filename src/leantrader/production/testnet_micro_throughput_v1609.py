from __future__ import annotations

import copy
import time
from typing import Any


EXIT_RETRY_BASE_SECONDS = 15.0
EXIT_RETRY_MAX_SECONDS = 300.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _defer_exit(
    self: Any,
    pending: dict[str, Any],
    now: float,
    *,
    reason: str,
) -> dict[str, Any]:
    event = copy.deepcopy(
        pending.get("source_event")
        or pending.get("event")
        or {}
    )

    symbol = str(event.get("symbol") or "").upper()

    if not symbol:
        self._clear_pending_if_event(
            event.get("event_id")
        )
        return self._decision(
            "exit_recovery_missing_symbol",
            details={"live_authority": False},
        )

    with self._lock:
        queue = self.state.setdefault(
            "deferred_exit_recoveries",
            {},
        )

        previous = queue.get(symbol) or {}

        previous_deferrals = max(
            int(previous.get("deferrals") or 0),
            int(pending.get("deferred_deferrals") or 0),
        )

        deferrals = previous_deferrals + 1

        delay = min(
            EXIT_RETRY_MAX_SECONDS,
            EXIT_RETRY_BASE_SECONDS
            * (2 ** min(5, max(0, deferrals - 1))),
        )

        recovery_attempt = max(
            1,
            int(pending.get("recovery_attempt") or 0),
        )

        queue[symbol] = {
            "symbol": symbol,
            "kind": "exit_recovery",
            "source_event": event,
            "assessment": copy.deepcopy(
                pending.get("assessment") or {}
            ),
            "recovery_attempt": recovery_attempt,
            "deferrals": deferrals,
            "deferred_at": now,
            "next_retry_at": now + delay,
            "reason": reason,
            "live_authority": False,
        }

        # Critical v1.60.9 behavior:
        # a confirmed terminal/blocked exit no longer owns the
        # single global fast-lane submission slot.
        self.state["pending_event"] = None
        self._save_locked()

    return self._decision(
        "exit_recovery_deferred_nonblocking",
        details={
            "kind": "exit",
            "symbol": symbol,
            "recovery_attempt": recovery_attempt,
            "deferrals": deferrals,
            "retry_in_seconds": delay,
            "next_retry_at": now + delay,
            "position_remains_active": True,
            "risk_capacity_released": False,
            "global_entry_slot_released": True,
            "ambiguous_order_resubmission_allowed": False,
            "live_authority": False,
        },
    )


def install_testnet_micro_throughput_v1609() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1609_micro_throughput_installed",
        False,
    ):
        return

    original_submit = (
        HyperSpeedCollectiveTestnetLane._submit_pending
    )
    original_manage = (
        HyperSpeedCollectiveTestnetLane._manage_active
    )
    original_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def submit_pending(
        self: Any,
        pending: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        kind = str(pending.get("kind") or "")
        attempt = int(
            pending.get("recovery_attempt") or 0
        )

        deferred_retry_due = bool(
            pending.get("deferred_retry_due")
        )

        # Existing v1.60.8 recovery state, including a persisted
        # recovery_attempt=111, is immediately moved away from the
        # global pending slot without submitting another order.
        if (
            not deferred_retry_due
            and (
                kind == "exit_recovery"
                or (
                    kind == "exit"
                    and attempt > 0
                )
            )
        ):
            return _defer_exit(
                self,
                pending,
                now,
                reason="existing_exit_recovery_isolated",
            )

        result = original_submit(
            self,
            pending,
            now,
        )

        result_reason = str(
            result.get("reason") or ""
        )

        # Once an order is authoritatively terminal with zero fill,
        # or fresh balance cannot currently make the exit executable,
        # isolate only this symbol. Do not stall the entire 0.5s lane.
        if result_reason in {
            "zero_fill_exit_reconciled_for_corrected_retry",
            "exit_recovery_waiting_for_executable_balance",
            "exit_waiting_for_executable_balance",
            "exit_recycle_cooldown",
        }:
            current = self._pending()

            if isinstance(current, dict):
                return _defer_exit(
                    self,
                    current,
                    now,
                    reason=result_reason,
                )

        return result

    def manage_active(
        self: Any,
        service: Any,
        snapshot: dict[str, Any],
        symbol: str,
        record: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        normalized = str(symbol).upper()

        with self._lock:
            queue = copy.deepcopy(
                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).get(normalized)
            )

        if not isinstance(queue, dict):
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        current_quantity = _number(
            (
                snapshot.get("positions")
                or {}
            ).get(normalized)
        )

        # If reconciliation proves the position is already absent,
        # let canonical active-position cleanup run normally.
        if current_quantity <= 0.0:
            with self._lock:
                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).pop(normalized, None)
                self._save_locked()

            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        next_retry = _number(
            queue.get("next_retry_at")
        )

        if now < next_retry:
            return self._decision(
                "exit_recovery_deferred_nonblocking",
                details={
                    "kind": "exit",
                    "symbol": normalized,
                    "current_quantity": current_quantity,
                    "retry_in_seconds": max(
                        0.0,
                        next_retry - now,
                    ),
                    "next_retry_at": next_retry,
                    "global_entry_slot_released": True,
                    "position_remains_active": True,
                    "live_authority": False,
                },
            )

        # Retry one isolated exit attempt using the same safety path:
        # fresh reconciliation, fresh free balance, exchange precision,
        # exchange minimums and no ambiguous resubmission.
        with self._lock:
            (
                self.state.get(
                    "deferred_exit_recoveries"
                )
                or {}
            ).pop(normalized, None)
            self._save_locked()

        source_event = copy.deepcopy(
            queue.get("source_event") or {}
        )

        try:
            signal = service.collective_signal(
                normalized
            )
            features = (
                (
                    signal.get("microstructure")
                    or {}
                ).get("features")
                or {}
            )

            fresh_price = _number(
                features.get("midpoint")
            )

            if (
                signal.get("fresh") is True
                and fresh_price > 0.0
            ):
                source_event["price"] = (
                    fresh_price
                )

        except Exception:
            pass

        recovery = {
            "kind": "exit_recovery",
            "source_event": source_event,
            "assessment": copy.deepcopy(
                queue.get("assessment") or {}
            ),
            "created_at": now,
            "retry_not_before": 0.0,
            "recovery_attempt": int(
                queue.get("recovery_attempt")
                or 1
            ),
            "deferred_deferrals": int(
                queue.get("deferrals") or 1
            ),
            "deferred_retry_due": True,
        }

        self._set_pending(recovery)

        return self._submit_pending(
            recovery,
            now=now,
        )

    def health(self: Any) -> dict[str, Any]:
        payload = original_health(self)

        with self._lock:
            queue = copy.deepcopy(
                self.state.get(
                    "deferred_exit_recoveries"
                )
                or {}
            )

        deferred = []

        for symbol, row in queue.items():
            if not isinstance(row, dict):
                continue

            deferred.append(
                {
                    "symbol": str(symbol),
                    "recovery_attempt": int(
                        row.get(
                            "recovery_attempt"
                        )
                        or 0
                    ),
                    "deferrals": int(
                        row.get("deferrals")
                        or 0
                    ),
                    "next_retry_at": (
                        row.get(
                            "next_retry_at"
                        )
                    ),
                    "reason": row.get("reason"),
                    "position_remains_active": True,
                    "live_authority": False,
                }
            )

        snapshot = {}

        try:
            snapshot = (
                self.testnet.safe_snapshot()
            )
        except Exception:
            pass

        payload.update(
            {
                "version": "1.60.9",
                "exit_recovery_isolation": {
                    "enabled": True,
                    "per_symbol": True,
                    "terminal_zero_fill_nonblocking": True,
                    "global_pending_slot_released_after_terminal_failure": True,
                    "position_remains_in_risk_accounting": True,
                    "minimum_retry_seconds": EXIT_RETRY_BASE_SECONDS,
                    "maximum_retry_seconds": EXIT_RETRY_MAX_SECONDS,
                    "exponential_backoff": True,
                    "ambiguous_order_resubmission_allowed": False,
                    "live_authority": False,
                },
                "deferred_exit_recoveries": deferred,
                "deferred_exit_recovery_count": len(
                    deferred
                ),
                "executor_entry_budget": {
                    "daily_entry_order_count": int(
                        snapshot.get(
                            "daily_entry_order_count"
                        )
                        or 0
                    ),
                    "daily_entry_submitted_usd": _number(
                        snapshot.get(
                            "daily_entry_submitted_usd"
                        )
                    ),
                    "daily_total_order_count": int(
                        snapshot.get(
                            "daily_total_order_count",
                            snapshot.get(
                                "daily_order_count"
                            ),
                        )
                        or 0
                    ),
                    "daily_total_submitted_usd": _number(
                        snapshot.get(
                            "daily_total_submitted_usd",
                            snapshot.get(
                                "daily_submitted_usd"
                            ),
                        )
                    ),
                },
                "live_authority": False,
            }
        )

        # Do not display stale paper-growth compounding as actual
        # Testnet compounding.
        actual_eligible = bool(
            payload.get(
                "actual_testnet_profit_compounding_eligible"
            )
        )

        last_sizing = (
            payload.get("last_sizing")
            or {}
        )

        payload[
            "principal_protected_compounding"
        ] = bool(
            actual_eligible
            and last_sizing.get(
                "compounding"
            )
        )

        return payload

    HyperSpeedCollectiveTestnetLane._submit_pending = (
        submit_pending
    )
    HyperSpeedCollectiveTestnetLane._manage_active = (
        manage_active
    )
    HyperSpeedCollectiveTestnetLane.health = health
    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.9"
    )
    VelocitySniperTestnetLane.VERSION = "1.60.9"

    HyperSpeedCollectiveTestnetLane._v1609_micro_throughput_installed = True
