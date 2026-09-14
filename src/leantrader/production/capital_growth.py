from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class CapitalGrowthGovernor:
    """Principal-protecting compounding governor; never a signal generator.

    It locks a configured portion of starting principal plus a non-reinvested
    share of realized profits, reduces risk under drawdown, and never uses
    martingale sizing or increases an upstream risk budget.
    """

    VERSION = "2.0"

    def __init__(
        self,
        state_path: Path,
        *,
        starting_equity: float,
        principal_floor_fraction: float = 0.70,
        profit_reinvest_fraction: float = 0.50,
    ) -> None:
        if starting_equity <= 0:
            raise ValueError("starting equity must be positive")
        if not 0.0 <= principal_floor_fraction <= 1.0:
            raise ValueError("principal floor fraction must be in [0, 1]")
        if not 0.0 <= profit_reinvest_fraction <= 1.0:
            raise ValueError("profit reinvest fraction must be in [0, 1]")
        self.state_path = state_path
        self.starting_equity = float(starting_equity)
        self.principal_floor_fraction = float(principal_floor_fraction)
        self.profit_reinvest_fraction = float(profit_reinvest_fraction)
        self.peak_equity = float(starting_equity)
        self.locked_profit = 0.0
        self.observations = 0
        self.last: dict[str, Any] = {}
        self.last_error: str | None = None
        self._load()

    def start(self) -> None:
        self._load()

    def stop(self) -> None:
        self._save()

    def evaluate(
        self,
        *,
        equity: float,
        realized_pnl: float = 0.0,
        open_notional: float = 0.0,
    ) -> dict[str, Any]:
        equity = max(0.0, float(equity))
        open_notional = max(0.0, float(open_notional))
        self.peak_equity = max(self.peak_equity, equity)
        realized_gain = max(0.0, float(realized_pnl))
        self.locked_profit = max(
            self.locked_profit,
            realized_gain * (1.0 - self.profit_reinvest_fraction),
        )
        base_principal_floor = self.starting_equity * self.principal_floor_fraction
        protected_principal = min(self.peak_equity, base_principal_floor + self.locked_profit)
        drawdown = (
            0.0
            if self.peak_equity <= 0
            else max(0.0, (self.peak_equity - equity) / self.peak_equity)
        )
        reinvestable_profit = realized_gain * self.profit_reinvest_fraction
        deployable_equity = max(0.0, equity - protected_principal)
        remaining_deployable_notional = max(0.0, deployable_equity - open_notional)

        if equity <= protected_principal or remaining_deployable_notional <= 0:
            risk_multiplier = 0.0
            new_entries_allowed = False
            state = "principal_floor"
        elif drawdown >= 0.20:
            risk_multiplier = 0.25
            new_entries_allowed = True
            state = "deep_drawdown"
        elif drawdown >= 0.10:
            risk_multiplier = 0.50
            new_entries_allowed = True
            state = "drawdown"
        elif drawdown >= 0.05:
            risk_multiplier = 0.75
            new_entries_allowed = True
            state = "defensive"
        else:
            risk_multiplier = 1.0
            new_entries_allowed = True
            state = "normal"

        result = {
            "state": state,
            "equity": equity,
            "peak_equity": self.peak_equity,
            "base_principal_floor": base_principal_floor,
            "locked_profit": self.locked_profit,
            "protected_principal": protected_principal,
            "drawdown": drawdown,
            "reinvestable_realized_profit": reinvestable_profit,
            "deployable_equity": deployable_equity,
            "open_notional": open_notional,
            "remaining_deployable_notional": remaining_deployable_notional,
            "risk_multiplier": _clip(risk_multiplier),
            "new_entries_allowed": new_entries_allowed,
            "martingale": False,
            "execution_authority": False,
            "can_increase_upstream_risk": False,
            "evaluated_at": time.time(),
        }
        self.last = result
        self.observations += 1
        self.last_error = None
        self._save()
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            "healthy": self.last_error is None,
            "observations": self.observations,
            "principal_floor_fraction": self.principal_floor_fraction,
            "profit_reinvest_fraction": self.profit_reinvest_fraction,
            "locked_profit": self.locked_profit,
            "risk_multiplier": float(self.last.get("risk_multiplier", 1.0)),
            "protected_principal": float(
                self.last.get("protected_principal", self.starting_equity * self.principal_floor_fraction)
            ),
            "deployable_equity": float(self.last.get("deployable_equity", self.starting_equity)),
            "open_notional": float(self.last.get("open_notional", 0.0)),
            "remaining_deployable_notional": float(
                self.last.get("remaining_deployable_notional", self.starting_equity)
            ),
            "new_entries_allowed": bool(self.last.get("new_entries_allowed", True)),
            "martingale": False,
            "execution_authority": False,
            "can_increase_upstream_risk": False,
            "error": self.last_error,
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.peak_equity = max(
                self.starting_equity,
                float(payload.get("peak_equity", self.starting_equity)),
            )
            self.locked_profit = max(0.0, float(payload.get("locked_profit", 0.0)))
            self.observations = int(payload.get("observations", 0))
            self.last = dict(payload.get("last") or {})
            self.last_error = None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.peak_equity = self.starting_equity
            self.locked_profit = 0.0
            self.observations = 0
            self.last = {}
            self.last_error = f"{type(exc).__name__}: {exc}"

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "peak_equity": self.peak_equity,
            "locked_profit": self.locked_profit,
            "observations": self.observations,
            "last": self.last,
            "updated_at": time.time(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
