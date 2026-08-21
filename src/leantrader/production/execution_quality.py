from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class ExecutionQualityIntelligence:
    """Persistent paper-fill and cost-drag intelligence with no order authority."""

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    MINIMUM_COST_FLOOR_BPS = 30.0
    EVENT_LIMIT = 5_000
    OUTCOME_LIMIT = 5_000
    SEEN_LIMIT = 20_000
    COST_SCENARIOS_BPS = (30.0, 50.0, 75.0, 100.0, 150.0, 250.0)

    def __init__(
        self,
        state_path: Path,
        *,
        modeled_round_trip_cost_bps: float = 30.0,
    ) -> None:
        if float(modeled_round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("execution intelligence cannot lower the 30-bps cost floor")
        self.state_path = state_path
        self.modeled_round_trip_cost_bps = float(modeled_round_trip_cost_bps)
        self.last_error: str | None = None
        self.state = self._load()

    def start(self) -> None:
        self.state = self._load()

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _authority_denied() -> dict[str, bool]:
        return {
            "research_only": True,
            "advisory_only": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_modify_orders": False,
            "can_modify_sizing": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, Any]:
        rows = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        samples = len(rows)
        if not samples:
            return {
                "samples": 0,
                "average": 0.0,
                "median": None,
                "p95": None,
                "best": None,
                "worst": None,
            }
        ordered = sorted(rows)
        p95_index = min(samples - 1, max(0, math.ceil(0.95 * samples) - 1))
        return {
            "samples": samples,
            "average": sum(rows) / samples,
            "median": statistics.median(rows),
            "p95": ordered[p95_index],
            "best": min(rows),
            "worst": max(rows),
        }

    @staticmethod
    def _return_statistics(values: list[float]) -> dict[str, Any]:
        rows = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        samples = len(rows)
        if not samples:
            return {
                "samples": 0,
                "wins": 0,
                "win_rate": 0.0,
                "average_net_return": 0.0,
                "cumulative_net_return": 0.0,
                "profit_factor": None,
                "max_losing_streak": 0,
            }
        wins = sum(1 for value in rows if value > 0.0)
        gross_profit = sum(value for value in rows if value > 0.0)
        gross_loss = abs(sum(value for value in rows if value < 0.0))
        losing_streak = 0
        maximum_losing_streak = 0
        for value in rows:
            losing_streak = losing_streak + 1 if value <= 0.0 else 0
            maximum_losing_streak = max(maximum_losing_streak, losing_streak)
        return {
            "samples": samples,
            "wins": wins,
            "win_rate": wins / samples,
            "average_net_return": sum(rows) / samples,
            "cumulative_net_return": sum(rows),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
            "max_losing_streak": maximum_losing_streak,
        }

    def _cost_survival(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        returns = [
            _finite(row.get("realized_return"), math.nan)
            for row in outcomes
            if isinstance(row, dict)
        ]
        returns = [value for value in returns if math.isfinite(value)]
        for scenario in self.COST_SCENARIOS_BPS:
            incremental_cost = max(
                0.0, scenario - self.modeled_round_trip_cost_bps
            ) / 10_000.0
            stressed = [value - incremental_cost for value in returns]
            stats = self._return_statistics(stressed)
            result[f"{scenario:.0f}bps"] = {
                "assumed_round_trip_cost_bps": scenario,
                "incremental_cost_applied": incremental_cost,
                "positive_after_cost_rate": (
                    sum(1 for value in stressed if value > 0.0) / len(stressed)
                    if stressed
                    else 0.0
                ),
                **stats,
            }
        return result

    def observe(
        self,
        *,
        events: list[dict[str, Any]],
        reference_prices: dict[str, float],
    ) -> dict[str, Any]:
        stored_events = self.state.setdefault("events", [])
        outcomes = self.state.setdefault("outcomes", [])
        seen = self.state.setdefault("seen_event_ids", {})
        recorded: list[str] = []
        skipped = 0

        for event in events:
            if not isinstance(event, dict):
                skipped += 1
                continue
            event_id = str(event.get("event_id") or "").strip()
            if not event_id or event_id in seen:
                skipped += 1
                continue
            symbol = str(event.get("symbol") or "").strip().upper()
            side = str(event.get("side") or "").strip().lower()
            fill_price = _finite(event.get("price"))
            quantity = max(0.0, _finite(event.get("quantity")))
            fee = max(0.0, _finite(event.get("fee")))
            reference = _finite(reference_prices.get(symbol))
            if (
                not symbol
                or side not in {"buy", "sell"}
                or fill_price <= 0.0
                or quantity <= 0.0
                or reference <= 0.0
            ):
                skipped += 1
                continue

            shortfall_bps = (
                (fill_price / reference - 1.0) * 10_000.0
                if side == "buy"
                else (reference / fill_price - 1.0) * 10_000.0
            )
            traded_value = quantity * fill_price
            fee_bps = fee / max(traded_value, 1e-12) * 10_000.0
            row = {
                "event_id": event_id,
                "observed_at": time.time(),
                "event_timestamp": str(event.get("timestamp") or ""),
                "symbol": symbol,
                "side": side,
                "reason": str(event.get("reason") or "")[:120],
                "reference_price": reference,
                "fill_price": fill_price,
                "quantity": quantity,
                "traded_value": traded_value,
                "adverse_shortfall_bps": shortfall_bps,
                "fee_bps": fee_bps,
                "single_leg_drag_bps": shortfall_bps + fee_bps,
                "paper_fill": True,
                **self._authority_denied(),
            }
            stored_events.append(row)
            seen[event_id] = time.time()
            recorded.append(event_id)

            if side == "sell":
                realized_return = _finite(event.get("realized_return"), math.nan)
                if math.isfinite(realized_return):
                    outcomes.append(
                        {
                            "event_id": event_id,
                            "observed_at": time.time(),
                            "symbol": symbol,
                            "reason": str(event.get("reason") or "")[:120],
                            "realized_return": realized_return,
                            "realized_pnl": _finite(event.get("realized_pnl")),
                            "remaining_quantity": max(
                                0.0, _finite(event.get("remaining_quantity"))
                            ),
                            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
                            **self._authority_denied(),
                        }
                    )

        self.state["events"] = stored_events[-self.EVENT_LIMIT :]
        self.state["outcomes"] = outcomes[-self.OUTCOME_LIMIT :]
        if len(seen) > self.SEEN_LIMIT:
            ordered = sorted(seen.items(), key=lambda item: float(item[1]))
            for key, _ in ordered[: len(seen) - self.SEEN_LIMIT]:
                seen.pop(key, None)

        self.state["cycles"] = int(self.state.get("cycles") or 0) + 1
        self.state["last"] = {
            "observed_at": time.time(),
            "recorded_event_ids": recorded,
            "skipped_events": skipped,
        }
        self._save()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        events = list(self.state.get("events") or [])
        outcomes = list(self.state.get("outcomes") or [])
        shortfalls = [
            _finite(row.get("adverse_shortfall_bps"), math.nan)
            for row in events
            if isinstance(row, dict)
        ]
        fees = [
            _finite(row.get("fee_bps"), math.nan)
            for row in events
            if isinstance(row, dict)
        ]
        drags = [
            _finite(row.get("single_leg_drag_bps"), math.nan)
            for row in events
            if isinstance(row, dict)
        ]
        by_symbol: dict[str, dict[str, Any]] = {}
        for symbol in sorted(
            {
                str(row.get("symbol") or "")
                for row in events
                if isinstance(row, dict)
            }
        ):
            symbol_rows = [
                row
                for row in events
                if isinstance(row, dict) and row.get("symbol") == symbol
            ]
            by_symbol[symbol] = {
                "events": len(symbol_rows),
                "shortfall_bps": self._distribution(
                    [
                        _finite(row.get("adverse_shortfall_bps"))
                        for row in symbol_rows
                    ]
                ),
                "single_leg_drag_bps": self._distribution(
                    [_finite(row.get("single_leg_drag_bps")) for row in symbol_rows]
                ),
            }
        realized_returns = [
            _finite(row.get("realized_return"), math.nan)
            for row in outcomes
            if isinstance(row, dict)
        ]
        result = {
            "cycles": int(self.state.get("cycles") or 0),
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "paper_fill_events_retained": len(events),
            "realized_exit_outcomes_retained": len(outcomes),
            "last": dict(self.state.get("last") or {}),
            "adverse_shortfall_bps": self._distribution(shortfalls),
            "fee_bps": self._distribution(fees),
            "single_leg_drag_bps": self._distribution(drags),
            "realized_return_statistics": self._return_statistics(
                realized_returns
            ),
            "cost_survival": self._cost_survival(outcomes),
            "by_symbol": by_symbol,
            "actual_market_impact_unobservable_in_paper": True,
            "paper_fill_model_not_live_execution": True,
            **self._authority_denied(),
        }
        return result

    def health(self) -> dict[str, Any]:
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "state_path": str(self.state_path),
            "cycles": int(self.state.get("cycles") or 0),
            "paper_fill_events": len(self.state.get("events") or []),
            "realized_exit_outcomes": len(self.state.get("outcomes") or []),
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "cost_scenarios_bps": list(self.COST_SCENARIOS_BPS),
            **self._authority_denied(),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "cycles": 0,
            "events": [],
            "outcomes": [],
            "seen_event_ids": {},
            "last": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("cycles", 0)
                payload.setdefault("events", [])
                payload.setdefault("outcomes", [])
                payload.setdefault("seen_event_ids", {})
                payload.setdefault("last", {})
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)
