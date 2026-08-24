
from __future__ import annotations

import copy
import time
from typing import Any

from .fast_collective_testnet import FastCollectiveTestnetLane


class HyperSpeedCollectiveTestnetLane(FastCollectiveTestnetLane):
    """Multi-position fast Testnet router with one sentinel per position."""

    VERSION = "1.57.1"

    def __init__(
        self,
        *args: Any,
        maximum_concurrent_positions: int = 6,
        maximum_entries_per_cycle: int = 3,
        reentry_cooldown_seconds: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.maximum_concurrent_positions = max(
            2,
            min(10, int(maximum_concurrent_positions)),
        )
        self.maximum_entries_per_cycle = max(
            1,
            min(
                self.maximum_concurrent_positions,
                int(maximum_entries_per_cycle),
            ),
        )
        self.reentry_cooldown_seconds = max(
            5.0,
            float(reentry_cooldown_seconds),
        )
        with self._lock:
            self.state.setdefault(
                "last_exit_by_symbol",
                {},
            )
            self._save_locked()

    def step(
        self,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        service = self.service_provider()
        if service is None:
            return self._decision("waiting_for_fast_swarm")

        self._refresh_day(now)

        pending = self._pending()
        if pending is not None:
            self._submit_pending(pending, now=now)
            if self._pending() is not None:
                return self._decision("testnet_event_in_flight")

        active = self._active_snapshot()
        exit_actions: list[dict[str, Any]] = []

        for symbol, record in list(active.items()):
            snapshot = self.testnet.safe_snapshot()
            result = self._manage_active(
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )
            details = result.get("details") or {}
            if details.get("kind") == "exit":
                exit_actions.append(
                    {
                        "symbol": symbol,
                        "status": details.get("status"),
                    }
                )
            if self._pending() is not None:
                return self._decision(
                    "testnet_event_in_flight",
                    details={"exit_actions": exit_actions},
                )

        snapshot = self.testnet.safe_snapshot()
        supervisor = self.supervisory_provider() or {}
        gate = self._supervisor_allows_entries(
            supervisor,
            now=now,
        )
        if not gate["allowed"]:
            return self._decision(
                gate["reason"],
                details={**gate, "exit_actions": exit_actions},
            )
        if snapshot.get("kill_switch_active") is True:
            return self._decision(
                "testnet_kill_switch",
                details={"exit_actions": exit_actions},
            )
        if snapshot.get("last_reconciliation_errors") or []:
            return self._decision(
                "testnet_reconciliation_not_clear",
                details={"exit_actions": exit_actions},
            )

        positions = {
            str(symbol).upper(): self._number(quantity)
            for symbol, quantity in (snapshot.get("positions") or {}).items()
            if self._number(quantity) > 0.0
        }
        slots = max(
            0,
            self.maximum_concurrent_positions - len(positions),
        )
        with self._lock:
            entries_today = int(self.state.get("entries_today") or 0)
            last_exit = dict(
                self.state.get("last_exit_by_symbol") or {}
            )
        daily_slots = max(
            0,
            self.maximum_entries_per_day - entries_today,
        )
        entry_limit = min(
            slots,
            daily_slots,
            self.maximum_entries_per_cycle,
        )
        if entry_limit <= 0:
            return self._decision(
                "holding_or_capacity_full",
                details={
                    "positions": sorted(positions),
                    "slots": slots,
                    "daily_slots": daily_slots,
                    "exit_actions": exit_actions,
                },
            )

        canonical_open = {
            str(symbol).upper()
            for symbol in (
                supervisor.get("canonical_open_positions")
                or []
            )
        }
        candidates = service.collective_candidates(limit=18)
        relaxed = bool(
            self.started_at > 0.0
            and now - self.started_at >= self.bootstrap_after_seconds
        )
        supervisor_symbols = supervisor.get("symbols") or {}
        assessed: list[tuple[str, dict[str, Any]]] = []

        for symbol in candidates:
            normalized = str(symbol or "").upper()
            if (
                not normalized
                or normalized in positions
                or normalized in canonical_open
                or (
                    now
                    - self._number(last_exit.get(normalized))
                    < self.reentry_cooldown_seconds
                )
            ):
                continue
            try:
                signal = service.collective_signal(normalized)
                row = self.assess_candidate(
                    signal,
                    supervisor_symbols.get(normalized, {}),
                    relaxed=relaxed,
                )
            except Exception as exc:
                row = {
                    "allowed": False,
                    "reason": f"candidate_error:{type(exc).__name__}",
                }
            assessed.append((normalized, row))

        allowed = [
            (symbol, row)
            for symbol, row in assessed
            if row.get("allowed") is True
        ]
        if not allowed:
            top = sorted(
                assessed,
                key=lambda item: (
                    self._number(item[1].get("decision_score")),
                    self._number(item[1].get("mtf_confidence")),
                    self._number(item[1].get("micro_confidence")),
                ),
                reverse=True,
            )[:5]
            return self._decision(
                "no_aligned_long_candidate",
                details={
                    "relaxed": relaxed,
                    "positions": sorted(positions),
                    "exit_actions": exit_actions,
                    "top": [
                        {
                            "symbol": symbol,
                            "reason": row.get("reason"),
                            "score": row.get("decision_score"),
                            "mtf_confidence": row.get("mtf_confidence"),
                            "micro_confidence": row.get("micro_confidence"),
                        }
                        for symbol, row in top
                    ],
                },
            )

        selected = sorted(
            allowed,
            key=lambda item: (
                bool(item[1].get("cost_qualified")),
                self._number(item[1].get("decision_score")),
                self._number(item[1].get("quality")),
            ),
            reverse=True,
        )[:entry_limit]

        opened: list[str] = []
        for symbol, assessment in selected:
            price = self._number(assessment.get("price"))
            assessment = {
                **assessment,
                "target_hold_seconds": self._target_hold_seconds(assessment),
            }
            event = self._new_event(
                symbol=symbol,
                side="buy",
                quantity=self.order_usd / max(price, 1e-12),
                price=price,
                reason=(
                    "fast_collective_testnet_entry:"
                    + str(
                        assessment.get("entry_mode")
                        or "exploration_probe"
                    )
                ),
                now=now,
            )
            pending = {
                "kind": "entry",
                "event": event,
                "assessment": assessment,
                "created_at": now,
            }
            self._set_pending(pending)
            self._submit_pending(pending, now=now)
            if symbol in self._active_snapshot():
                opened.append(symbol)
            if self._pending() is not None:
                break

        after = self.testnet.safe_snapshot()
        return self._decision(
            "fast_multi_route_cycle",
            details={
                "opened": opened,
                "exit_actions": exit_actions,
                "entry_limit": entry_limit,
                "positions_after": sorted(
                    symbol
                    for symbol, quantity in (
                        after.get("positions") or {}
                    ).items()
                    if self._number(quantity) > 0.0
                ),
            },
        )

    def _manage_active(
        self,
        service: Any,
        snapshot: dict[str, Any],
        symbol: str,
        record: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        current_total = self._number(
            (snapshot.get("positions") or {}).get(symbol)
        )
        fast_quantity = min(
            self._number(record.get("quantity")),
            current_total,
        )
        if fast_quantity <= 0.0:
            with self._lock:
                (self.state.get("active") or {}).pop(symbol, None)
                self.state["last_action"] = {
                    "action": "fast_position_absent",
                    "symbol": symbol,
                    "timestamp": now,
                }
                self._save_locked()
            return self._decision(
                "active_position_already_absent",
                details={"symbol": symbol},
            )

        signal = service.collective_signal(symbol)
        micro = signal.get("microstructure") or {}
        features = micro.get("features") or {}
        price = self._number(features.get("midpoint"))
        if signal.get("fresh") is not True or price <= 0.0:
            return self._decision(
                "waiting_for_fresh_exit_mark",
                details={"symbol": symbol},
            )

        entry_price = self._number(record.get("entry_price"))
        entered_at = self._number(record.get("entered_at"))
        previous_peak = max(
            entry_price,
            self._number(record.get("peak_price"), entry_price),
        )
        peak_price = max(previous_peak, price)
        age_seconds = max(0.0, now - entered_at)
        gross_bps = (
            (price / entry_price - 1.0) * 10_000.0
            if entry_price > 0.0
            else 0.0
        )
        peak_gain_bps = (
            (peak_price / entry_price - 1.0) * 10_000.0
            if entry_price > 0.0
            else 0.0
        )
        retrace_bps = (
            (price / peak_price - 1.0) * 10_000.0
            if peak_price > 0.0
            else 0.0
        )
        spread_bps = self._number(
            features.get("spread_bps"),
            1_000_000.0,
        )
        target_hold = max(
            45.0,
            min(
                self.maximum_hold_seconds,
                self._number(record.get("target_hold_seconds"), 90.0),
            ),
        )

        reason = None
        if gross_bps >= self.take_profit_bps:
            reason = "take_profit"
        elif gross_bps <= -self.stop_loss_bps:
            reason = "stop_loss"
        elif peak_gain_bps >= 45.0 and retrace_bps <= -20.0:
            reason = "sentinel_trailing_profit"
        elif self.strong_short_reversal(signal):
            reason = "micro_mtf_reversal"
        elif spread_bps > 30.0:
            reason = "liquidity_spread_deterioration"
        elif age_seconds >= target_hold:
            reason = "dynamic_time_exit"

        sentinel = {
            "symbol": symbol,
            "price": price,
            "age_seconds": age_seconds,
            "target_hold_seconds": target_hold,
            "gross_bps": gross_bps,
            "peak_price": peak_price,
            "peak_gain_bps": peak_gain_bps,
            "retrace_from_peak_bps": retrace_bps,
            "spread_bps": spread_bps,
            "reason": reason or "sentinel_hold",
        }
        with self._lock:
            live = (self.state.get("active") or {}).get(symbol)
            if live is not None:
                live["peak_price"] = peak_price
                live["last_sentinel"] = dict(sentinel)
                if current_total < self._number(live.get("quantity")):
                    live["quantity"] = current_total
                self._save_locked()

        if reason is None:
            return self._decision(
                "holding_fast_testnet_position",
                details=sentinel,
            )

        event = self._new_event(
            symbol=symbol,
            side="sell",
            quantity=fast_quantity,
            price=price,
            reason="fast_collective_testnet_exit:" + reason,
            now=now,
            remaining_quantity=max(0.0, current_total - fast_quantity),
        )
        pending = {
            "kind": "exit",
            "event": event,
            "assessment": {
                **sentinel,
                "exit_reason": reason,
            },
            "created_at": now,
        }
        self._set_pending(pending)
        return self._submit_pending(pending, now=now)

    def _target_hold_seconds(
        self,
        assessment: dict[str, Any],
    ) -> float:
        horizons = [
            self._number(row.get("horizon_seconds"))
            for row in (
                assessment.get("micro_support")
                or []
            )
            if isinstance(row, dict)
            and self._number(row.get("horizon_seconds")) > 0.0
        ]
        base = max(horizons) if horizons else 30.0
        multiplier = (
            4.0
            if assessment.get("cost_qualified") is True
            else 3.0
        )
        lower = (
            90.0
            if assessment.get("cost_qualified") is True
            else 60.0
        )
        upper = (
            self.maximum_hold_seconds
            if assessment.get("cost_qualified") is True
            else min(180.0, self.maximum_hold_seconds)
        )
        return max(lower, min(upper, base * multiplier))

    def _submit_pending(
        self,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        event = copy.deepcopy(pending["event"])
        results = self.testnet.mirror_events([event])
        result = dict(results[0]) if results else {}
        snapshot = self.testnet.safe_snapshot()

        symbol = str(event["symbol"]).upper()
        status = str(result.get("status") or "").lower()
        filled = max(0.0, self._number(result.get("filled")))
        current_total = self._number(
            (snapshot.get("positions") or {}).get(symbol)
        )
        terminal = status in self.TERMINAL_ORDER_STATES

        with self._lock:
            kind = str(pending.get("kind") or "")

            if kind == "entry":
                active = self.state.setdefault("active", {})
                if (
                    filled <= 0.0
                    and current_total > 0.0
                    and status not in {"skipped", "rejected", "canceled"}
                ):
                    filled = current_total
                if filled > 0.0 and symbol not in active:
                    assessment = pending.get("assessment") or {}
                    entry_price = self._number(
                        result.get("average"),
                        self._number(event.get("price")),
                    )
                    active[symbol] = {
                        "symbol": symbol,
                        "quantity": filled,
                        "initial_quantity": filled,
                        "entry_price": entry_price,
                        "peak_price": entry_price,
                        "entered_at": now,
                        "entry_event_id": event.get("event_id"),
                        "entry_mode": assessment.get("entry_mode"),
                        "target_hold_seconds": assessment.get(
                            "target_hold_seconds"
                        ),
                        "intelligence": copy.deepcopy(assessment),
                        "last_sentinel": None,
                    }
                    self.state["entries_today"] = (
                        int(self.state.get("entries_today") or 0) + 1
                    )
                    self.state["last_action"] = {
                        "action": "buy",
                        "symbol": symbol,
                        "status": status,
                        "quantity": filled,
                        "price": entry_price,
                        "timestamp": now,
                        "entry_mode": assessment.get("entry_mode"),
                        "support_groups": assessment.get("support_groups"),
                    }
                if terminal:
                    self.state["pending_event"] = None

            elif kind == "exit":
                active = self.state.setdefault("active", {})
                record = active.get(symbol)
                if record is None:
                    self.state["pending_event"] = None
                else:
                    before = self._number(record.get("quantity"))
                    sold = min(before, filled)
                    if sold > 0.0:
                        remaining_fast = max(0.0, before - sold)
                    elif current_total < before:
                        remaining_fast = max(0.0, current_total)
                    else:
                        remaining_fast = before

                    if remaining_fast <= max(1e-12, before * 0.001):
                        record = active.pop(symbol)
                        exit_price = self._number(
                            result.get("average"),
                            self._number(event.get("price")),
                        )
                        entry_price = self._number(record.get("entry_price"))
                        gross_bps = (
                            (exit_price / entry_price - 1.0) * 10_000.0
                            if entry_price > 0.0
                            else 0.0
                        )
                        net_bps = gross_bps - self.round_trip_cost_bps
                        closed = {
                            "symbol": symbol,
                            "quantity": record.get("initial_quantity"),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_bps": gross_bps,
                            "net_bps_after_model": net_bps,
                            "modeled_round_trip_cost_bps": (
                                self.round_trip_cost_bps
                            ),
                            "entered_at": record.get("entered_at"),
                            "exited_at": now,
                            "hold_seconds": max(
                                0.0,
                                now - self._number(record.get("entered_at")),
                            ),
                            "exit_reason": (
                                pending.get("assessment") or {}
                            ).get("exit_reason"),
                            "entry_mode": record.get("entry_mode"),
                            "intelligence": record.get("intelligence"),
                            "testnet_only": True,
                            "live_authority": False,
                        }
                        history = list(self.state.get("closed") or [])
                        history.append(closed)
                        self.state["closed"] = history[-250:]
                        self.state["exits_today"] = (
                            int(self.state.get("exits_today") or 0) + 1
                        )
                        self.state.setdefault(
                            "last_exit_by_symbol",
                            {},
                        )[symbol] = now
                        self.state["last_action"] = {
                            "action": "sell",
                            "symbol": symbol,
                            "status": status,
                            "price": exit_price,
                            "timestamp": now,
                            "gross_bps": gross_bps,
                            "net_bps_after_model": net_bps,
                            "exit_reason": closed.get("exit_reason"),
                        }
                        self.state["pending_event"] = None
                    else:
                        record["quantity"] = remaining_fast
                        if terminal:
                            self.state["pending_event"] = None

            self.state["last_error"] = None
            self._save_locked()

        return self._decision(
            "testnet_event_processed",
            details={
                "kind": pending.get("kind"),
                "symbol": symbol,
                "status": status,
                "filled": filled,
                "current_total_quantity": current_total,
            },
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload.update(
            {
                "version": self.VERSION,
                "maximum_concurrent_positions": (
                    self.maximum_concurrent_positions
                ),
                "maximum_entries_per_cycle": (
                    self.maximum_entries_per_cycle
                ),
                "reentry_cooldown_seconds": (
                    self.reentry_cooldown_seconds
                ),
                "multi_position_router": True,
                "per_position_hyper_speed_sentinel": True,
                "independent_position_exit": True,
                "immediate_slot_reuse": True,
                "single_position_restriction": False,
                "live_authority": False,
            }
        )
        return payload
