"""v1.60.31 side-aware terminal pending-entry recovery.

Bybit Testnet only. This patch closes the v1.60.29 orphan-position hole where a
terminal *buy* pending event could be reconciled by the generic terminal-pending
path, have its latch cleared, yet never recreate the fast-lane ``active`` row.
The executor then correctly held real Testnet inventory while the fast lane had
no sentinel ownership for it.

The repair is deliberately narrow:

* only terminal ``closed`` BUY orders with a real fill are eligible;
* authoritative executor reconciliation must succeed first;
* the executor must still hold a positive position for the same symbol;
* no open/submitting order for the symbol may exist;
* startup recovery additionally requires the v1.60.29 terminal reconciliation
  record and that the matching BUY is still the latest real filled order for
  the symbol;
* no exchange order is submitted, no executor position/PnL is fabricated or
  mutated, and all payloads retain ``live_authority=False``.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from .testnet_residual_dust_cycle_v1627 import _n, _timestamp
from .testnet_terminal_pending_recovery_v1629 import (
    NON_TERMINAL_ORDER_STATES,
    _authoritative_order,
    _deterministic_client_order_id,
)

VERSION = "1.60.31"
MAX_STARTUP_RECONCILIATIONS = 50
MAX_RECOVERED_CLIENT_IDS = 200
MAX_RECOVERY_ROWS = 100


def _symbol_unresolved_order_exists(
    state: dict[str, Any],
    symbol: str,
    *,
    exclude_client_id: str = "",
) -> bool:
    normalized = str(symbol or "").upper()
    excluded = str(exclude_client_id or "")

    for client_id, row in (state.get("orders") or {}).items():
        if not isinstance(row, dict):
            continue
        if excluded and str(client_id) == excluded:
            continue
        if str(row.get("symbol") or "").upper() != normalized:
            continue
        if str(row.get("status") or "").lower() in NON_TERMINAL_ORDER_STATES:
            return True

    return False


def _latest_filled_order(
    state: dict[str, Any],
    symbol: str,
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    normalized = str(symbol or "").upper()
    rows: list[tuple[float, str, dict[str, Any]]] = []

    for client_id, row in (state.get("orders") or {}).items():
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").upper() != normalized:
            continue
        if str(row.get("status") or "").lower() != "closed":
            continue
        if _n(row.get("filled")) <= 0.0:
            continue
        rows.append((_timestamp(row), str(client_id), copy.deepcopy(row)))

    if not rows:
        return None, None

    _ts, client_id, row = max(rows, key=lambda item: (item[0], item[1]))
    return client_id, row


def _entry_mode_from_reason(reason: Any) -> str:
    text = str(reason or "")
    prefix = "fast_collective_testnet_entry:"
    if text.startswith(prefix):
        value = text[len(prefix) :].strip()
        if value:
            return value
    return "recovered_terminal_buy"


def _build_active_record(
    *,
    pending: dict[str, Any],
    order: dict[str, Any],
    symbol: str,
    remaining: float,
    now: float,
    client_order_id: str,
    source: str,
) -> dict[str, Any] | None:
    event = copy.deepcopy((pending or {}).get("event") or {})
    assessment = copy.deepcopy((pending or {}).get("assessment") or {})

    filled = max(0.0, _n(order.get("filled")))
    filled_cost = max(0.0, _n(order.get("filled_cost")))

    entry_price = max(0.0, _n(order.get("average")))
    if entry_price <= 0.0 and filled > 0.0 and filled_cost > 0.0:
        entry_price = filled_cost / filled
    if entry_price <= 0.0:
        entry_price = max(0.0, _n(event.get("price")))
    if entry_price <= 0.0:
        return None

    entered_at = _timestamp(order)
    if entered_at <= 0.0:
        entered_at = max(0.0, _n(event.get("timestamp")))
    if entered_at <= 0.0:
        entered_at = now

    reason = order.get("reason") or event.get("reason")
    entry_mode = str(assessment.get("entry_mode") or "") or _entry_mode_from_reason(
        reason
    )

    if not assessment:
        assessment = {
            "allowed": True,
            "entry_mode": entry_mode,
            "reason": "v1631_recovered_terminal_buy",
            "recovered_from_terminal_buy": True,
            "recovery_source": source,
            "testnet_exploration_authority": True,
            "live_authority": False,
        }
    else:
        assessment["recovered_from_terminal_buy"] = True
        assessment["recovery_source"] = source
        assessment["live_authority"] = False

    initial_quantity = max(remaining, filled)
    entry_notional = max(filled_cost, initial_quantity * entry_price)
    target_hold = max(5.0, _n(assessment.get("target_hold_seconds"), 30.0))

    return {
        "symbol": symbol,
        "quantity": remaining,
        "initial_quantity": initial_quantity,
        "entry_price": entry_price,
        "entry_notional_usd": entry_notional,
        "peak_price": entry_price,
        "entered_at": entered_at,
        "target_hold_seconds": target_hold,
        "entry_event_id": event.get("event_id"),
        "entry_mode": entry_mode,
        "intelligence": assessment,
        "recovered_by": VERSION,
        "recovered_client_order_id": client_order_id,
        "recovery_source": source,
        "testnet_only": True,
        "live_authority": False,
    }


def install_testnet_terminal_buy_recovery_v1631() -> None:
    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane
    from .velocity_sniper_testnet import VelocitySniperTestnetLane

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1631_terminal_buy_recovery_installed",
        False,
    ):
        return

    original_submit_pending = HyperSpeedCollectiveTestnetLane._submit_pending
    original_start = HyperSpeedCollectiveTestnetLane.start
    original_health = HyperSpeedCollectiveTestnetLane.health

    def _record_recovery(
        self: Any,
        *,
        symbol: str,
        client_order_id: str,
        quantity: float,
        source: str,
        now: float,
        pending_latch_cleared: bool,
    ) -> None:
        rows = list(self.state.get("v1631_terminal_buy_recoveries") or [])
        rows.append(
            {
                "symbol": symbol,
                "client_order_id": client_order_id,
                "quantity": quantity,
                "source": source,
                "pending_latch_cleared": pending_latch_cleared,
                "order_submitted": False,
                "executor_state_mutated": False,
                "global_realized_pnl_mutated": False,
                "timestamp": now,
                "testnet_only": True,
                "live_authority": False,
            }
        )
        self.state["v1631_terminal_buy_recoveries"] = rows[-MAX_RECOVERY_ROWS:]

        recovered_ids = list(self.state.get("v1631_recovered_client_ids") or [])
        if client_order_id not in recovered_ids:
            recovered_ids.append(client_order_id)
        self.state["v1631_recovered_client_ids"] = recovered_ids[
            -MAX_RECOVERED_CLIENT_IDS:
        ]

    def _activate_terminal_buy(
        self: Any,
        *,
        pending: dict[str, Any],
        order: dict[str, Any],
        client_order_id: str,
        snapshot: dict[str, Any],
        now: float,
        source: str,
        clear_pending: bool,
    ) -> dict[str, Any]:
        event = copy.deepcopy((pending or {}).get("event") or {})
        symbol = str(event.get("symbol") or order.get("symbol") or "").upper()
        remaining = max(0.0, _n((snapshot.get("positions") or {}).get(symbol)))

        if not symbol or remaining <= 0.0:
            return {
                "activated": False,
                "reason": "authoritative_buy_position_absent",
                "symbol": symbol,
                "live_authority": False,
            }

        record = _build_active_record(
            pending=pending,
            order=order,
            symbol=symbol,
            remaining=remaining,
            now=now,
            client_order_id=client_order_id,
            source=source,
        )
        if record is None:
            return {
                "activated": False,
                "reason": "terminal_buy_entry_price_unprovable",
                "symbol": symbol,
                "live_authority": False,
            }

        with self._lock:
            active = self.state.setdefault("active", {})
            existing = active.get(symbol)
            recovered_ids = set(self.state.get("v1631_recovered_client_ids") or [])

            if existing is None:
                active[symbol] = record

                if client_order_id not in recovered_ids:
                    self.state["entries_today"] = int(
                        self.state.get("entries_today") or 0
                    ) + 1

            pending_latch_cleared = False
            if clear_pending:
                latch = self.state.get("pending_event")
                if isinstance(latch, dict):
                    latch_event = latch.get("event") or latch.get("source_event") or {}
                    try:
                        latch_client_id = _deterministic_client_order_id(latch_event)
                    except Exception:
                        latch_client_id = ""
                    if latch_client_id == client_order_id:
                        self.state["pending_event"] = None
                        pending_latch_cleared = True
                elif latch is None:
                    pending_latch_cleared = True

            _record_recovery(
                self,
                symbol=symbol,
                client_order_id=client_order_id,
                quantity=remaining,
                source=source,
                now=now,
                pending_latch_cleared=pending_latch_cleared,
            )

            self.state["last_error"] = None
            self.state["last_action"] = {
                "action": "terminal_pending_buy_reconciled_active",
                "symbol": symbol,
                "client_order_id": client_order_id,
                "status": "closed",
                "quantity": remaining,
                "price": record.get("entry_price"),
                "source": source,
                "pending_latch_cleared": pending_latch_cleared,
                "order_submitted": False,
                "timestamp": now,
                "testnet_only": True,
                "live_authority": False,
            }
            self._save_locked()

        return {
            "activated": True,
            "reason": "terminal_buy_active_restored",
            "symbol": symbol,
            "quantity": remaining,
            "entry_price": record.get("entry_price"),
            "pending_latch_cleared": pending_latch_cleared,
            "order_submitted": False,
            "executor_state_mutated": False,
            "global_realized_pnl_mutated": False,
            "testnet_only": True,
            "live_authority": False,
        }

    def _submit_pending(
        self: Any,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        event = copy.deepcopy((pending or {}).get("event") or {})
        kind = str((pending or {}).get("kind") or "").lower()
        side = str(event.get("side") or "").lower()
        symbol = str(event.get("symbol") or "").upper()

        # v1.60.29's generic terminal reconciliation is correct for exits. Only
        # intercept terminal ENTRY buys so a real executor position can never be
        # left without fast-lane sentinel ownership.
        if kind != "entry" or side != "buy" or not symbol:
            return original_submit_pending(self, pending, now=now)

        testnet_state = getattr(self.testnet, "state", None)
        testnet_lock = getattr(self.testnet, "_io_lock", None)
        if not isinstance(testnet_state, dict) or testnet_lock is None:
            return original_submit_pending(self, pending, now=now)

        try:
            client_order_id = _deterministic_client_order_id(event)
        except Exception:
            return original_submit_pending(self, pending, now=now)

        order = _authoritative_order(self.testnet, client_order_id)
        if not isinstance(order, dict):
            return original_submit_pending(self, pending, now=now)

        status = str(order.get("status") or "").lower()
        filled = max(0.0, _n(order.get("filled")))

        if status in NON_TERMINAL_ORDER_STATES:
            return original_submit_pending(self, pending, now=now)

        if (
            status != "closed"
            or filled <= 0.0
            or str(order.get("side") or side).lower() != "buy"
        ):
            return original_submit_pending(self, pending, now=now)

        if _symbol_unresolved_order_exists(
            testnet_state,
            symbol,
            exclude_client_id=client_order_id,
        ):
            return self._decision(
                "terminal_buy_reconciliation_unresolved_order_fail_closed",
                details={
                    "kind": kind,
                    "symbol": symbol,
                    "client_order_id": client_order_id,
                    "order_submitted": False,
                    "pending_latch_cleared": False,
                    "position_remains_unmodified": True,
                    "live_authority": False,
                },
            )

        try:
            self.testnet.reconcile_required()
            snapshot = self.testnet.safe_snapshot()
        except Exception:
            return self._decision(
                "terminal_buy_reconciliation_ambiguous",
                details={
                    "kind": kind,
                    "symbol": symbol,
                    "client_order_id": client_order_id,
                    "order_submitted": False,
                    "pending_latch_cleared": False,
                    "position_remains_unmodified": True,
                    "live_authority": False,
                },
            )

        recovery = _activate_terminal_buy(
            self,
            pending=pending,
            order=order,
            client_order_id=client_order_id,
            snapshot=snapshot,
            now=now,
            source="pending_entry_terminal_closed",
            clear_pending=True,
        )

        if recovery.get("activated") is not True:
            # When the authoritative position is absent, preserve v1.60.29's
            # existing terminal semantics rather than fabricating an active row.
            return original_submit_pending(self, pending, now=now)

        return self._decision(
            "terminal_pending_buy_reconciled",
            details={
                "kind": kind,
                "symbol": symbol,
                "client_order_id": client_order_id,
                "status": status,
                "filled": filled,
                **recovery,
            },
        )

    def recover_orphaned_terminal_buys_v1631(
        self: Any,
        *,
        now: float | None = None,
        limit: int = MAX_STARTUP_RECONCILIATIONS,
    ) -> dict[str, Any]:
        """Boundedly restore only proven v1.60.29 orphaned terminal buys."""

        observed_now = time.time() if now is None else float(now)
        testnet_state = getattr(self.testnet, "state", None)
        testnet_lock = getattr(self.testnet, "_io_lock", None)

        if not isinstance(testnet_state, dict) or testnet_lock is None:
            return {
                "ok": False,
                "reason": "executor_state_unavailable",
                "recovered": 0,
                "live_authority": False,
            }

        try:
            self.testnet.reconcile_required()
            snapshot = self.testnet.safe_snapshot()
        except Exception as exc:
            return {
                "ok": False,
                "reason": "executor_reconciliation_ambiguous",
                "error": type(exc).__name__,
                "recovered": 0,
                "live_authority": False,
            }

        with self._lock:
            reconciliations = copy.deepcopy(
                self.state.get("v1629_terminal_pending_reconciliations") or []
            )
            already_active = set((self.state.get("active") or {}).keys())
            recovered_ids = set(self.state.get("v1631_recovered_client_ids") or [])

        inspected = 0
        recovered: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        bounded = list(reversed(reconciliations))[: max(0, int(limit))]
        for reconciliation in bounded:
            if not isinstance(reconciliation, dict):
                continue

            inspected += 1
            if reconciliation.get("position_retired") is True:
                continue

            symbol = str(reconciliation.get("symbol") or "").upper()
            client_order_id = str(reconciliation.get("client_order_id") or "")
            if not symbol or not client_order_id:
                continue
            if client_order_id in recovered_ids or symbol in already_active:
                continue

            order = _authoritative_order(self.testnet, client_order_id)
            if not isinstance(order, dict):
                continue
            if str(order.get("side") or "").lower() != "buy":
                continue
            if str(order.get("status") or "").lower() != "closed":
                continue
            if _n(order.get("filled")) <= 0.0:
                continue

            latest_client_id, latest_order = _latest_filled_order(
                testnet_state,
                symbol,
            )
            if latest_client_id != client_order_id or not isinstance(
                latest_order, dict
            ):
                skipped.append(
                    {
                        "symbol": symbol,
                        "client_order_id": client_order_id,
                        "reason": "matching_buy_not_latest_filled_order",
                    }
                )
                continue

            if _symbol_unresolved_order_exists(
                testnet_state,
                symbol,
                exclude_client_id=client_order_id,
            ):
                skipped.append(
                    {
                        "symbol": symbol,
                        "client_order_id": client_order_id,
                        "reason": "unresolved_symbol_order_fail_closed",
                    }
                )
                continue

            remaining = max(
                0.0,
                _n((snapshot.get("positions") or {}).get(symbol)),
            )
            if remaining <= 0.0:
                continue

            event = {
                "event_id": reconciliation.get("event_id"),
                "symbol": symbol,
                "side": "buy",
                "price": order.get("average"),
                "reason": order.get("reason"),
            }
            pending = {
                "kind": "entry",
                "event": event,
                "assessment": {
                    "allowed": True,
                    "entry_mode": _entry_mode_from_reason(order.get("reason")),
                    "reason": "v1631_startup_orphan_recovery",
                    "recovered_from_terminal_buy": True,
                    "testnet_exploration_authority": True,
                    "live_authority": False,
                },
            }

            outcome = _activate_terminal_buy(
                self,
                pending=pending,
                order=order,
                client_order_id=client_order_id,
                snapshot=snapshot,
                now=observed_now,
                source="startup_v1629_orphan_reconciliation",
                clear_pending=False,
            )
            if outcome.get("activated") is True:
                recovered.append(outcome)
                recovered_ids.add(client_order_id)
                already_active.add(symbol)

        summary = {
            "ok": True,
            "version": VERSION,
            "inspected": inspected,
            "recovered": len(recovered),
            "skipped": len(skipped),
            "recovered_symbols": [row.get("symbol") for row in recovered],
            "recent_skips": skipped[-20:],
            "order_submitted": False,
            "executor_state_mutated": False,
            "global_realized_pnl_mutated": False,
            "testnet_only": True,
            "live_authority": False,
        }

        with self._lock:
            self.state["v1631_startup_recovery"] = copy.deepcopy(summary)
            self._save_locked()

        return summary

    def start(self: Any) -> None:
        try:
            recover_orphaned_terminal_buys_v1631(self)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self.state["v1631_startup_recovery"] = {
                    "ok": False,
                    "version": VERSION,
                    "reason": "startup_recovery_exception",
                    "error": type(exc).__name__,
                    "order_submitted": False,
                    "live_authority": False,
                }
                self._save_locked()
        original_start(self)

    def health(self: Any) -> dict[str, Any]:
        payload = original_health(self)
        with self._lock:
            recent = copy.deepcopy(
                self.state.get("v1631_terminal_buy_recoveries") or []
            )
            startup = copy.deepcopy(self.state.get("v1631_startup_recovery") or {})

        payload["terminal_buy_recovery"] = {
            "version": VERSION,
            "enabled": True,
            "side_aware_terminal_pending": True,
            "restores_active_only_from_real_closed_filled_buy": True,
            "requires_authoritative_reconciliation": True,
            "requires_positive_executor_position": True,
            "startup_requires_v1629_reconciliation_evidence": True,
            "startup_requires_matching_buy_latest_filled_order": True,
            "unresolved_symbol_orders_fail_closed": True,
            "order_submitted": False,
            "executor_state_mutated": False,
            "global_realized_pnl_mutated": False,
            "startup": startup,
            "recoveries": len(recent),
            "recent": recent[-20:],
            "testnet_only": True,
            "live_authority": False,
        }
        payload["live_authority"] = False
        return payload

    HyperSpeedCollectiveTestnetLane._submit_pending = _submit_pending
    HyperSpeedCollectiveTestnetLane.recover_orphaned_terminal_buys_v1631 = (
        recover_orphaned_terminal_buys_v1631
    )
    HyperSpeedCollectiveTestnetLane.start = start
    HyperSpeedCollectiveTestnetLane.health = health

    HyperSpeedCollectiveTestnetLane.VERSION = VERSION
    VelocitySniperTestnetLane.VERSION = VERSION
    HyperSpeedCollectiveTestnetLane._v1631_terminal_buy_recovery_installed = True
