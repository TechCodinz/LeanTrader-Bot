"""v1.60.32 pending-order liveness for Bybit Testnet.

A persisted fast-lane pending order in ``submitting`` or ``open`` state must
remain fail-closed and must never be resubmitted.  It must also not become a
permanent latch merely because no reconciliation error existed before the
pending path inspected it.

This patch periodically asks the existing authoritative Testnet executor to
reconcile that exact persisted order.  All existing v1.60.29/v1.60.31 terminal
handling remains authoritative after reconciliation.

It also retries the v1.60.31 startup orphan recovery only when that startup
recovery did not previously complete successfully.

No exchange order is created here.  Live authority remains false.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from .testnet_terminal_pending_recovery_v1629 import (
    NON_TERMINAL_ORDER_STATES,
    _authoritative_order,
    _deterministic_client_order_id,
)

VERSION = "1.60.32"

PENDING_RECONCILE_INTERVAL_SECONDS = 2.0
STARTUP_RECOVERY_RETRY_SECONDS = 10.0


def reconcile_pending_order_v1632(
    lane: Any,
    *,
    now: float,
) -> dict[str, Any]:
    """Reconcile one already-persisted unresolved pending order.

    This function never submits or resubmits an exchange order.
    """

    pending = lane._pending()

    if not isinstance(pending, dict):
        return {
            "attempted": False,
            "reason": "no_pending_event",
            "live_authority": False,
        }

    event = copy.deepcopy(
        pending.get("event")
        or pending.get("source_event")
        or {}
    )

    symbol = str(event.get("symbol") or "").upper()

    if not symbol:
        return {
            "attempted": False,
            "reason": "pending_symbol_missing",
            "live_authority": False,
        }

    try:
        client_order_id = _deterministic_client_order_id(event)
    except Exception:
        return {
            "attempted": False,
            "reason": "pending_client_id_unavailable",
            "symbol": symbol,
            "live_authority": False,
        }

    order = _authoritative_order(
        lane.testnet,
        client_order_id,
    )

    if not isinstance(order, dict):
        return {
            "attempted": False,
            "reason": "authoritative_order_not_persisted",
            "symbol": symbol,
            "client_order_id": client_order_id,
            "live_authority": False,
        }

    before_status = str(
        order.get("status") or ""
    ).lower()

    if before_status not in NON_TERMINAL_ORDER_STATES:
        return {
            "attempted": False,
            "reason": "pending_order_already_terminal",
            "symbol": symbol,
            "client_order_id": client_order_id,
            "before_status": before_status,
            "live_authority": False,
        }

    last = float(
        getattr(
            lane,
            "_v1632_last_pending_reconcile_at",
            0.0,
        )
        or 0.0
    )

    if (
        last > 0.0
        and now - last
        < PENDING_RECONCILE_INTERVAL_SECONDS
    ):
        return {
            "attempted": False,
            "reason": "pending_reconciliation_throttled",
            "symbol": symbol,
            "client_order_id": client_order_id,
            "before_status": before_status,
            "retry_in_seconds": max(
                0.0,
                PENDING_RECONCILE_INTERVAL_SECONDS
                - (now - last),
            ),
            "live_authority": False,
        }

    lane._v1632_last_pending_reconcile_at = now

    ok = False
    error_type = None
    reconciliation = {}

    try:
        reconciliation = (
            lane.testnet.reconcile_required()
            or {}
        )
        ok = True
    except Exception as exc:
        error_type = type(exc).__name__

    refreshed = _authoritative_order(
        lane.testnet,
        client_order_id,
    )

    after_status = (
        str(refreshed.get("status") or "").lower()
        if isinstance(refreshed, dict)
        else before_status
    )

    return {
        "attempted": True,
        "ok": ok,
        "reason": (
            "authoritative_pending_reconciliation_complete"
            if ok
            else "authoritative_pending_reconciliation_ambiguous"
        ),
        "symbol": symbol,
        "client_order_id": client_order_id,
        "before_status": before_status,
        "after_status": after_status,
        "error_type": error_type,
        "reconciled": reconciliation.get("reconciled"),
        "checked": reconciliation.get("checked"),
        "order_submitted": False,
        "resubmission_allowed": False,
        "live_authority": False,
    }


def retry_startup_orphan_recovery_v1632(
    lane: Any,
    *,
    now: float,
) -> dict[str, Any]:
    """Retry v1.60.31 only if its startup recovery never succeeded."""

    with lane._lock:
        startup = copy.deepcopy(
            lane.state.get("v1631_startup_recovery")
        )

    if (
        isinstance(startup, dict)
        and startup.get("ok") is True
    ):
        return {
            "attempted": False,
            "reason": "v1631_startup_recovery_already_ok",
            "live_authority": False,
        }

    method = getattr(
        lane,
        "recover_orphaned_terminal_buys_v1631",
        None,
    )

    if not callable(method):
        return {
            "attempted": False,
            "reason": "v1631_recovery_method_unavailable",
            "live_authority": False,
        }

    last = float(
        getattr(
            lane,
            "_v1632_last_startup_recovery_retry_at",
            0.0,
        )
        or 0.0
    )

    if (
        last > 0.0
        and now - last
        < STARTUP_RECOVERY_RETRY_SECONDS
    ):
        return {
            "attempted": False,
            "reason": "startup_recovery_retry_throttled",
            "live_authority": False,
        }

    lane._v1632_last_startup_recovery_retry_at = now

    try:
        outcome = method(now=now)
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "reason": "v1631_startup_recovery_retry_exception",
            "error_type": type(exc).__name__,
            "order_submitted": False,
            "live_authority": False,
        }

    return {
        "attempted": True,
        "ok": outcome.get("ok") is True,
        "reason": "v1631_startup_recovery_retried",
        "outcome": copy.deepcopy(outcome),
        "order_submitted": False,
        "live_authority": False,
    }


def install_testnet_pending_liveness_v1632() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1632_pending_liveness_installed",
        False,
    ):
        return

    original_step = HyperSpeedCollectiveTestnetLane.step
    original_health = HyperSpeedCollectiveTestnetLane.health

    def step(
        self: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        observed_now = (
            time.time()
            if now is None
            else float(now)
        )

        pending_outcome = (
            reconcile_pending_order_v1632(
                self,
                now=observed_now,
            )
        )

        startup_outcome = (
            retry_startup_orphan_recovery_v1632(
                self,
                now=observed_now,
            )
        )

        if (
            pending_outcome.get("attempted") is True
            or startup_outcome.get("attempted") is True
        ):
            with self._lock:
                self.state["v1632_pending_liveness"] = {
                    "version": VERSION,
                    "pending_reconciliation": (
                        copy.deepcopy(
                            pending_outcome
                        )
                    ),
                    "startup_orphan_retry": (
                        copy.deepcopy(
                            startup_outcome
                        )
                    ),
                    "order_submitted": False,
                    "resubmission_allowed": False,
                    "testnet_only": True,
                    "live_authority": False,
                }
                self._save_locked()

        # Existing v1.60.31/v1.60.29 logic remains authoritative.
        return original_step(
            self,
            now=observed_now,
        )

    def health(self: Any) -> dict[str, Any]:
        payload = original_health(self)

        with self._lock:
            recent = copy.deepcopy(
                self.state.get(
                    "v1632_pending_liveness"
                )
                or {}
            )

        payload["pending_order_liveness"] = {
            "version": VERSION,
            "enabled": True,
            "unresolved_pending_is_periodically_reconciled": True,
            "pending_reconciliation_interval_seconds": (
                PENDING_RECONCILE_INTERVAL_SECONDS
            ),
            "v1631_failed_startup_recovery_is_retried": True,
            "startup_recovery_retry_seconds": (
                STARTUP_RECOVERY_RETRY_SECONDS
            ),
            "order_submitted": False,
            "resubmission_allowed": False,
            "fail_closed": True,
            "recent": recent,
            "testnet_only": True,
            "live_authority": False,
        }

        payload["live_authority"] = False
        return payload

    HyperSpeedCollectiveTestnetLane.step = step
    HyperSpeedCollectiveTestnetLane.health = health

    HyperSpeedCollectiveTestnetLane.VERSION = VERSION
    VelocitySniperTestnetLane.VERSION = VERSION

    HyperSpeedCollectiveTestnetLane._v1632_pending_liveness_installed = True
