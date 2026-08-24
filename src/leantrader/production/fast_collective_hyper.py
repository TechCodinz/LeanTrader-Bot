
from __future__ import annotations

import copy
import time
from typing import Any

from .fast_collective_testnet import FastCollectiveTestnetLane


class HyperSpeedCollectiveTestnetLane(FastCollectiveTestnetLane):
    """Multi-position fast Testnet router with one sentinel per position."""

    VERSION = "1.59.1"

    def __init__(
        self,
        *args: Any,
        maximum_concurrent_positions: int = 6,
        maximum_entries_per_cycle: int = 3,
        reentry_cooldown_seconds: float = 20.0,
        starting_equity: float = 50.0,
        maximum_order_usd: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.starting_equity = max(
            0.01,
            float(starting_equity),
        )

        # This is only an additional local ceiling. The Bybit Testnet
        # executor still enforces its own stricter order/position/daily caps.
        self.maximum_order_usd = max(
            self.order_usd,
            float(maximum_order_usd),
        )
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
            1.0,
            float(reentry_cooldown_seconds),
        )
        with self._lock:
            self.state.setdefault(
                "last_exit_by_symbol",
                {},
            )
            self.state.setdefault(
                "last_sizing",
                {},
            )
            self._save_locked()

    def _fast_open_notional(
        self,
    ) -> float:
        with self._lock:
            active = copy.deepcopy(
                self.state.get("active")
                or {}
            )

        total = 0.0

        for record in active.values():
            if not isinstance(record, dict):
                continue

            total += (
                self._number(
                    record.get("quantity")
                )
                * self._number(
                    record.get("entry_price")
                )
            )

        return max(0.0, total)

    def _compound_order_notional(
        self,
        supervisor: dict[str, Any],
        *,
        slots: int,
    ) -> dict[str, Any]:
        """Allocate only from canonical principal-protected deployable capital."""

        growth = (
            supervisor.get("capital_growth")
            or {}
        )

        # Before the first capital snapshot exists, remain bounded at the
        # original fixed exploration amount instead of inventing capital.
        if not isinstance(growth, dict) or not growth:
            return {
                "allowed": True,
                "reason": "fixed_fallback_before_capital_snapshot",
                "order_notional_usd": min(
                    self.maximum_order_usd,
                    self.order_usd,
                ),
                "compounding": False,
                "live_authority": False,
            }

        if (
            growth.get(
                "new_entries_allowed"
            )
            is False
        ):
            return {
                "allowed": False,
                "reason": "capital_growth_new_entries_blocked",
                "capital_state": growth.get("state"),
                "live_authority": False,
            }

        risk_multiplier = max(
            0.0,
            min(
                1.0,
                self._number(
                    growth.get(
                        "risk_multiplier"
                    ),
                    1.0,
                ),
            ),
        )

        remaining = max(
            0.0,
            self._number(
                growth.get(
                    "remaining_deployable_notional"
                )
            ),
        )

        fast_open = (
            self._fast_open_notional()
        )

        available_pool = max(
            0.0,
            remaining - fast_open,
        )

        slot_count = max(
            1,
            int(slots),
        )

        slot_budget = (
            available_pool
            / slot_count
        )

        order_notional = min(
            self.maximum_order_usd,
            slot_budget
            * risk_multiplier,
        )

        if (
            risk_multiplier <= 0.0
            or order_notional < 0.50
        ):
            return {
                "allowed": False,
                "reason": "capital_growth_insufficient_deployable_budget",
                "capital_state": growth.get("state"),
                "remaining_deployable_notional": remaining,
                "fast_open_notional": fast_open,
                "available_pool": available_pool,
                "slot_budget": slot_budget,
                "risk_multiplier": risk_multiplier,
                "live_authority": False,
            }

        return {
            "allowed": True,
            "reason": "principal_protected_compound_budget",
            "order_notional_usd": order_notional,
            "compounding": True,
            "capital_state": growth.get("state"),
            "equity": self._number(
                growth.get("equity")
            ),
            "peak_equity": self._number(
                growth.get("peak_equity")
            ),
            "protected_principal": self._number(
                growth.get(
                    "protected_principal"
                )
            ),
            "locked_profit": self._number(
                growth.get("locked_profit")
            ),
            "reinvestable_realized_profit": self._number(
                growth.get(
                    "reinvestable_realized_profit"
                )
            ),
            "remaining_deployable_notional": remaining,
            "fast_open_notional": fast_open,
            "available_pool": available_pool,
            "slot_budget": slot_budget,
            "risk_multiplier": risk_multiplier,
            "martingale": False,
            "live_authority": False,
        }

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

        sizing = self._compound_order_notional(
            supervisor,
            slots=slots,
        )

        with self._lock:
            self.state["last_sizing"] = (
                copy.deepcopy(sizing)
            )
            self._save_locked()

        if sizing.get("allowed") is not True:
            return self._decision(
                str(
                    sizing.get("reason")
                    or "capital_growth_blocked"
                ),
                details={
                    "sizing": sizing,
                    "exit_actions": exit_actions,
                },
            )

        order_notional_usd = self._number(
            sizing.get(
                "order_notional_usd"
            )
        )

        canonical_open = {
            str(symbol).upper()
            for symbol in (
                supervisor.get("canonical_open_positions")
                or []
            )
        }
        candidates = service.collective_candidates(limit=24)
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
                bool(
                    item[1].get(
                        "velocity_sniper"
                    )
                ),
                self._number(
                    item[1].get(
                        "decision_score"
                    )
                ),
                bool(
                    item[1].get(
                        "cost_qualified"
                    )
                ),
                self._number(
                    item[1].get("quality")
                ),
            ),
            reverse=True,
        )[:entry_limit]

        opened: list[str] = []
        for symbol, assessment in selected:
            price = self._number(assessment.get("price"))
            assessment = {
                **assessment,
                "target_hold_seconds": self._target_hold_seconds(assessment),
                "order_notional_usd": order_notional_usd,
                "capital_growth_sizing": copy.deepcopy(
                    sizing
                ),
            }
            event = self._new_event(
                symbol=symbol,
                side="buy",
                quantity=order_notional_usd / max(price, 1e-12),
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

        velocity_bps_s = self._number(
            features.get(
                "midpoint_velocity_bps_per_second"
            )
        )
        acceleration_bps_s2 = self._number(
            features.get(
                "midpoint_acceleration_bps_per_second2"
            )
        )
        trend_5s_bps = self._number(
            features.get(
                "recent_midpoint_trend_bps_5s"
            )
        )
        range_5s_bps = max(
            0.0,
            self._number(
                features.get(
                    "recent_midpoint_range_bps_5s"
                )
            ),
        )
        depth_imbalance = self._number(
            features.get("depth_imbalance")
        )
        microprice_shift_bps = self._number(
            features.get(
                "microprice_shift_bps"
            )
        )

        target_hold = max(
            5.0,
            min(
                self.maximum_hold_seconds,
                self._number(
                    record.get(
                        "target_hold_seconds"
                    ),
                    12.0,
                ),
            ),
        )

        dynamic_take_profit_bps = max(
            self.round_trip_cost_bps + 10.0,
            min(
                self.take_profit_bps,
                self.round_trip_cost_bps
                + max(
                    10.0,
                    min(
                        30.0,
                        range_5s_bps * 0.75,
                    ),
                ),
            ),
        )

        dynamic_stop_loss_bps = max(
            20.0,
            min(
                self.stop_loss_bps,
                max(
                    20.0,
                    range_5s_bps * 0.50,
                ),
            ),
        )

        reason = None

        if gross_bps >= dynamic_take_profit_bps:
            reason = "velocity_take_profit"

        elif gross_bps <= -dynamic_stop_loss_bps:
            reason = "velocity_stop_loss"

        elif (
            peak_gain_bps
            >= self.round_trip_cost_bps + 10.0
            and retrace_bps <= -10.0
        ):
            reason = "velocity_trailing_profit"

        elif self.strong_short_reversal(signal):
            reason = "micro_mtf_reversal"

        elif (
            age_seconds >= 1.5
            and velocity_bps_s <= -0.75
            and trend_5s_bps <= -2.0
        ):
            reason = "velocity_reversal"

        elif spread_bps > 25.0:
            reason = "liquidity_spread_deterioration"

        elif (
            age_seconds >= 3.0
            and gross_bps
            >= self.round_trip_cost_bps + 5.0
            and velocity_bps_s <= 0.20
            and trend_5s_bps <= 2.0
        ):
            reason = "velocity_profit_decay"

        elif (
            age_seconds >= 6.0
            and velocity_bps_s <= 0.15
            and acceleration_bps_s2 <= 0.0
            and trend_5s_bps < 2.0
        ):
            reason = "velocity_decay"

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
            "midpoint_velocity_bps_per_second": (
                velocity_bps_s
            ),
            "midpoint_acceleration_bps_per_second2": (
                acceleration_bps_s2
            ),
            "recent_midpoint_trend_bps_5s": (
                trend_5s_bps
            ),
            "recent_midpoint_range_bps_5s": (
                range_5s_bps
            ),
            "depth_imbalance": depth_imbalance,
            "microprice_shift_bps": (
                microprice_shift_bps
            ),
            "dynamic_take_profit_bps": (
                dynamic_take_profit_bps
            ),
            "dynamic_stop_loss_bps": (
                dynamic_stop_loss_bps
            ),
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
            self._number(
                row.get("horizon_seconds")
            )
            for row in (
                assessment.get("micro_support")
                or []
            )
            if (
                isinstance(row, dict)
                and self._number(
                    row.get("horizon_seconds")
                ) > 0.0
            )
        ]

        velocity = bool(
            assessment.get("velocity_sniper")
        )

        if velocity:
            base = (
                min(horizons)
                if horizons
                else 5.0
            )

            if (
                assessment.get(
                    "cost_qualified"
                )
                is True
            ):
                return max(
                    8.0,
                    min(
                        self.maximum_hold_seconds,
                        30.0,
                        base * 2.0,
                    ),
                )

            return max(
                5.0,
                min(
                    self.maximum_hold_seconds,
                    15.0,
                    base * 1.25,
                ),
            )

        base = (
            min(horizons)
            if horizons
            else 15.0
        )

        lower = (
            20.0
            if assessment.get(
                "cost_qualified"
            )
            is True
            else 12.0
        )

        return max(
            lower,
            min(
                self.maximum_hold_seconds,
                base * 2.0,
            ),
        )

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
                        "entry_notional_usd": self._number(
                            assessment.get(
                                "order_notional_usd"
                            ),
                            filled * entry_price,
                        ),
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

                        entry_notional_usd = self._number(
                            record.get(
                                "entry_notional_usd"
                            )
                        )

                        if entry_notional_usd <= 0.0:
                            entry_notional_usd = (
                                self._number(
                                    record.get(
                                        "initial_quantity"
                                    )
                                )
                                * entry_price
                            )

                        modeled_net_pnl_usd = (
                            entry_notional_usd
                            * net_bps
                            / 10_000.0
                        )

                        closed = {
                            "symbol": symbol,
                            "quantity": record.get("initial_quantity"),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_bps": gross_bps,
                            "net_bps_after_model": net_bps,
                            "entry_notional_usd": entry_notional_usd,
                            "modeled_net_pnl_usd": modeled_net_pnl_usd,
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
                            "modeled_net_pnl_usd": modeled_net_pnl_usd,
                            "entry_notional_usd": entry_notional_usd,
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

    def _closed_entry_notional_usd(
        self,
        row: dict[str, Any],
    ) -> float:
        explicit = self._number(
            row.get("entry_notional_usd")
        )

        if explicit > 0.0:
            return explicit

        return max(
            0.0,
            self._number(row.get("quantity"))
            * self._number(row.get("entry_price")),
        )

    def _closed_modeled_net_pnl_usd(
        self,
        row: dict[str, Any],
    ) -> float:
        # v1.59+ rows persist this directly. Older rows are reconstructed
        # from their actual Testnet quantity/entry and the same >=30 bps
        # modeled net result; no historical profitability is fabricated.
        if row.get("modeled_net_pnl_usd") is not None:
            return self._number(
                row.get("modeled_net_pnl_usd")
            )

        return (
            self._closed_entry_notional_usd(row)
            * self._number(
                row.get("net_bps_after_model")
            )
            / 10_000.0
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()

        with self._lock:
            closed = copy.deepcopy(
                self.state.get("closed")
                or []
            )
            last_sizing = copy.deepcopy(
                self.state.get(
                    "last_sizing"
                )
                or {}
            )

        net_rows = [
            self._number(
                row.get(
                    "net_bps_after_model"
                )
            )
            for row in closed
            if isinstance(row, dict)
        ]

        positive_bps = sum(
            value
            for value in net_rows
            if value > 0.0
        )

        negative_bps = abs(
            sum(
                value
                for value in net_rows
                if value < 0.0
            )
        )

        wins = sum(
            1
            for value in net_rows
            if value > 0.0
        )

        now = time.time()

        completed_last_hour = sum(
            1
            for row in closed
            if isinstance(row, dict)
            and self._number(
                row.get("exited_at")
            )
            >= now - 3_600.0
        )

        modeled_net_pnl_usd = sum(
            self._closed_modeled_net_pnl_usd(
                row
            )
            for row in closed
            if isinstance(row, dict)
        )

        closed_last_hour = [
            row
            for row in closed
            if (
                isinstance(row, dict)
                and self._number(
                    row.get("exited_at")
                )
                >= now - 3_600.0
            )
        ]

        modeled_net_pnl_last_hour = sum(
            self._closed_modeled_net_pnl_usd(
                row
            )
            for row in closed_last_hour
        )

        capital_turnover_last_hour = sum(
            self._closed_entry_notional_usd(
                row
            )
            for row in closed_last_hour
        )

        hourly_return_fraction = (
            modeled_net_pnl_last_hour
            / self.starting_equity
            if self.starting_equity > 0.0
            else 0.0
        )

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
                "principal_protected_compounding": True,
                "martingale": False,
                "maximum_compound_order_usd": (
                    self.maximum_order_usd
                ),
                "last_sizing": last_sizing,
                "closed_count": len(net_rows),
                "win_rate": (
                    wins / len(net_rows)
                    if net_rows
                    else None
                ),
                "average_net_bps": (
                    sum(net_rows)
                    / len(net_rows)
                    if net_rows
                    else None
                ),
                "profit_factor": (
                    positive_bps
                    / negative_bps
                    if negative_bps > 0.0
                    else None
                ),
                "modeled_net_pnl_usd": (
                    modeled_net_pnl_usd
                ),
                "modeled_net_pnl_last_hour_usd": (
                    modeled_net_pnl_last_hour
                ),
                "capital_turnover_last_hour_usd": (
                    capital_turnover_last_hour
                ),
                "modeled_return_per_hour_fraction": (
                    hourly_return_fraction
                ),
                "completed_trades_last_hour": (
                    completed_last_hour
                ),
                "live_authority": False,
            }
        )
        return payload
