from __future__ import annotations

import copy
from typing import Any


def install_testnet_exit_recycle_compat_v1608() -> None:
    """Preserve legacy observability and repair v1.60.8 wrapper signatures.

    The real Bybit executor remains authoritative for reconciliation, free
    balances, execution methods, Testnet-only authority and order safety.
    Compatibility here is read-only except for adapting the v1.60.8 capacity
    wrapper to the current Hyper lane method signature and clearing the
    canonical pending_event key after an explicit dust recycle.
    """

    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_testnet_exit_recycle_compat_v1608_installed",
        False,
    ):
        return

    original_init = HyperSpeedCollectiveTestnetLane.__init__

    def init_with_snapshot_compat(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        snapshot = getattr(self.testnet, "safe_snapshot", None)
        health = getattr(self.testnet, "health", None)
        if not callable(snapshot) and callable(health):
            self.testnet.safe_snapshot = health

    HyperSpeedCollectiveTestnetLane.__init__ = init_with_snapshot_compat

    broken_capacity = HyperSpeedCollectiveTestnetLane._adaptive_position_capacity
    base_capacity = None
    for cell in getattr(broken_capacity, "__closure__", None) or ():
        candidate = cell.cell_contents
        if callable(candidate) and getattr(candidate, "__name__", "") == "_adaptive_position_capacity":
            base_capacity = candidate
            break

    if base_capacity is not None:
        def adaptive_position_capacity(
            self: Any,
            supervisor: dict[str, Any],
            snapshot: dict[str, Any],
            *,
            candidate_count: int,
            entries_today: int,
        ) -> dict[str, Any]:
            adjusted = copy.deepcopy(snapshot)
            adjusted["daily_order_count"] = int(
                snapshot.get("daily_entry_order_count") or 0
            )
            adjusted["daily_submitted_usd"] = float(
                snapshot.get("daily_entry_submitted_usd") or 0.0
            )
            return base_capacity(
                self,
                supervisor,
                adjusted,
                candidate_count=candidate_count,
                entries_today=entries_today,
            )

        HyperSpeedCollectiveTestnetLane._adaptive_position_capacity = adaptive_position_capacity

    original_pending = HyperSpeedCollectiveTestnetLane._pending

    def pending_with_dust_cleanup(self: Any) -> dict[str, Any] | None:
        pending = original_pending(self)
        if not isinstance(pending, dict):
            return pending
        event = pending.get("event") or pending.get("source_event") or {}
        symbol = str(event.get("symbol") or "").upper()
        if not symbol:
            return pending
        with self._lock:
            active = self.state.get("active") or {}
            dust_recycles = self.state.get("dust_recycles") or []
            recycled = any(
                isinstance(row, dict)
                and str(row.get("symbol") or "").upper() == symbol
                for row in dust_recycles[-10:]
            )
            if symbol not in active and recycled:
                self.state["pending_event"] = None
                self.state.setdefault("last_exit_by_symbol", {})[symbol] = float(
                    dust_recycles[-1].get("recorded_at") or 0.0
                ) if isinstance(dust_recycles[-1], dict) else 0.0
                self._save_locked()
                return None
        return pending

    HyperSpeedCollectiveTestnetLane._pending = pending_with_dust_cleanup
    HyperSpeedCollectiveTestnetLane._testnet_exit_recycle_compat_v1608_installed = True
