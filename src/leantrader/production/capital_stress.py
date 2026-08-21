from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class CapitalStressSimulator:
    """Deterministic small-account survival research with no sizing authority."""

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    MINIMUM_COST_FLOOR_BPS = 30.0
    HISTORY_LIMIT = 1_000

    def __init__(
        self,
        state_path: Path,
        *,
        starting_equity: float,
        principal_floor_fraction: float,
        risk_per_trade_fraction: float,
        max_daily_loss_fraction: float,
        max_drawdown_fraction: float,
        modeled_round_trip_cost_bps: float = 30.0,
    ) -> None:
        if float(starting_equity) <= 0.0:
            raise ValueError("capital stress starting equity must be positive")
        for name, value in (
            ("principal floor", principal_floor_fraction),
            ("risk per trade", risk_per_trade_fraction),
            ("daily loss", max_daily_loss_fraction),
            ("drawdown", max_drawdown_fraction),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"capital stress {name} must be between zero and one")
        if float(modeled_round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("capital stress cannot lower the 30-bps cost floor")
        self.state_path = state_path
        self.starting_equity = float(starting_equity)
        self.principal_floor_fraction = float(principal_floor_fraction)
        self.risk_per_trade_fraction = float(risk_per_trade_fraction)
        self.max_daily_loss_fraction = float(max_daily_loss_fraction)
        self.max_drawdown_fraction = float(max_drawdown_fraction)
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
            "not_a_forecast": True,
            "cannot_override_capital_governor": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_modify_orders": False,
            "can_modify_sizing": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }

    def _scenario(
        self,
        *,
        name: str,
        description: str,
        loss: float,
        equity: float,
        peak_equity: float,
        protected_principal: float,
        horizon: str,
    ) -> dict[str, Any]:
        bounded_loss = max(0.0, min(equity, float(loss)))
        projected_equity = max(0.0, equity - bounded_loss)
        loss_fraction = bounded_loss / max(equity, 1e-12)
        projected_drawdown = (
            max(0.0, peak_equity - projected_equity)
            / max(peak_equity, 1e-12)
        )
        return {
            "scenario": name,
            "description": description,
            "horizon": horizon,
            "projected_loss": bounded_loss,
            "projected_loss_fraction": loss_fraction,
            "projected_equity": projected_equity,
            "projected_drawdown_fraction": projected_drawdown,
            "principal_floor_breach": projected_equity < protected_principal,
            "daily_loss_limit_breach": (
                horizon in {"immediate", "single_cycle"}
                and loss_fraction >= self.max_daily_loss_fraction
            ),
            "drawdown_limit_breach": (
                projected_drawdown >= self.max_drawdown_fraction
            ),
            **self._authority_denied(),
        }

    def evaluate(
        self,
        *,
        equity: float,
        cash: float,
        peak_equity: float,
        positions: dict[str, dict[str, Any]],
        execution_quality: dict[str, Any],
    ) -> dict[str, Any]:
        equity = max(0.0, _finite(equity))
        cash = max(0.0, min(equity, _finite(cash)))
        peak_equity = max(equity, _finite(peak_equity, equity))
        normalized: dict[str, dict[str, float]] = {}
        for symbol, row in sorted(positions.items()):
            if not isinstance(row, dict):
                continue
            notional = max(0.0, _finite(row.get("notional")))
            price = _finite(row.get("price"))
            atr = max(0.0, _finite(row.get("atr")))
            if notional <= 0.0 or price <= 0.0:
                continue
            normalized[str(symbol).upper()] = {
                "notional": notional,
                "price": price,
                "atr": atr,
                "atr_fraction": atr / price,
            }

        gross_exposure = sum(row["notional"] for row in normalized.values())
        largest_position = max(
            (row["notional"] for row in normalized.values()),
            default=0.0,
        )
        exposure_fraction = gross_exposure / max(equity, 1e-12)
        largest_position_fraction = largest_position / max(equity, 1e-12)
        concentration = (
            sum((row["notional"] / gross_exposure) ** 2 for row in normalized.values())
            if gross_exposure > 0.0
            else 0.0
        )
        protected_principal = self.starting_equity * self.principal_floor_fraction
        configured_cost = self.modeled_round_trip_cost_bps / 10_000.0

        atr_shock_loss = sum(
            row["notional"]
            * min(0.25, max(0.03, 3.0 * row["atr_fraction"]))
            for row in normalized.values()
        )
        realized_stats = (
            execution_quality.get("realized_return_statistics")
            if isinstance(execution_quality, dict)
            else {}
        )
        if not isinstance(realized_stats, dict):
            realized_stats = {}
        observed_losing_streak = max(
            0, int(realized_stats.get("max_losing_streak") or 0)
        )
        conservative_losing_streak = max(5, observed_losing_streak)
        losing_streak_loss = equity * conservative_losing_streak * (
            self.risk_per_trade_fraction + configured_cost
        )

        scenario_inputs = [
            (
                "baseline_liquidation",
                "Exit all open paper positions at the configured round-trip cost.",
                gross_exposure * configured_cost,
                "immediate",
            ),
            (
                "slippage_spike_100bps",
                "Exit exposure under a 100-bps all-in cost shock.",
                gross_exposure * 0.01,
                "immediate",
            ),
            (
                "correlated_market_shock",
                "All open positions fall eight percent together and exits cost 100 bps.",
                gross_exposure * 0.09,
                "single_cycle",
            ),
            (
                "liquidity_gap",
                "A twelve-percent adverse gap combines with 150-bps liquidation cost.",
                gross_exposure * 0.135,
                "single_cycle",
            ),
            (
                "exchange_outage",
                "Exposure remains trapped through a fifteen-percent adverse move.",
                gross_exposure * 0.15,
                "single_cycle",
            ),
            (
                "atr_volatility_shock",
                "Each position suffers three ATR with a three-percent minimum shock.",
                atr_shock_loss + gross_exposure * 0.01,
                "single_cycle",
            ),
            (
                "losing_streak",
                "Configured per-trade risk and costs repeat across a conservative losing streak.",
                losing_streak_loss,
                "multi_trade",
            ),
        ]
        scenarios = [
            self._scenario(
                name=name,
                description=description,
                loss=loss,
                equity=equity,
                peak_equity=peak_equity,
                protected_principal=protected_principal,
                horizon=horizon,
            )
            for name, description, loss, horizon in scenario_inputs
        ]
        worst = max(
            scenarios,
            key=lambda row: float(row["projected_loss"]),
            default={
                "scenario": "none",
                "projected_loss": 0.0,
                "projected_equity": equity,
                "projected_loss_fraction": 0.0,
                "projected_drawdown_fraction": 0.0,
                "principal_floor_breach": False,
                "daily_loss_limit_breach": False,
                "drawdown_limit_breach": False,
            },
        )

        any_floor_breach = any(
            row["principal_floor_breach"] for row in scenarios
        )
        any_drawdown_breach = any(
            row["drawdown_limit_breach"] for row in scenarios
        )
        any_daily_breach = any(
            row["daily_loss_limit_breach"] for row in scenarios
        )
        if equity <= 0.0:
            stress_state = "capital_exhausted"
        elif any_floor_breach:
            stress_state = "principal_floor_at_risk"
        elif any_drawdown_breach:
            stress_state = "drawdown_limit_at_risk"
        elif any_daily_breach:
            stress_state = "daily_loss_limit_at_risk"
        else:
            stress_state = "within_configured_stress_buffer"

        result = {
            "evaluated_at": time.time(),
            "stress_state": stress_state,
            "equity": equity,
            "cash": cash,
            "peak_equity": peak_equity,
            "starting_equity": self.starting_equity,
            "protected_principal": protected_principal,
            "capital_buffer_to_floor": equity - protected_principal,
            "gross_open_exposure": gross_exposure,
            "exposure_fraction": exposure_fraction,
            "largest_position_fraction": largest_position_fraction,
            "position_concentration_hhi": concentration,
            "positions_stressed": len(normalized),
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "configured_limits": {
                "risk_per_trade_fraction": self.risk_per_trade_fraction,
                "max_daily_loss_fraction": self.max_daily_loss_fraction,
                "max_drawdown_fraction": self.max_drawdown_fraction,
                "principal_floor_fraction": self.principal_floor_fraction,
            },
            "execution_outcome_samples": int(
                realized_stats.get("samples") or 0
            ),
            "observed_max_losing_streak": observed_losing_streak,
            "stress_losing_streak": conservative_losing_streak,
            "scenarios": scenarios,
            "worst_scenario": dict(worst),
            "worst_projected_equity": float(worst["projected_equity"]),
            "worst_projected_loss_fraction": float(
                worst["projected_loss_fraction"]
            ),
            "survives_worst_scenario": float(worst["projected_equity"]) > 0.0,
            "principal_floor_survives_all": not any_floor_breach,
            "daily_limit_survives_immediate_scenarios": not any_daily_breach,
            "drawdown_limit_survives_all": not any_drawdown_breach,
            "observed_execution_evidence_mature": int(
                realized_stats.get("samples") or 0
            ) >= 100,
            **self._authority_denied(),
        }
        history = self.state.setdefault("history", [])
        history.append(
            {
                "evaluated_at": result["evaluated_at"],
                "stress_state": stress_state,
                "equity": equity,
                "gross_open_exposure": gross_exposure,
                "worst_scenario": worst["scenario"],
                "worst_projected_equity": worst["projected_equity"],
            }
        )
        self.state["history"] = history[-self.HISTORY_LIMIT :]
        self.state["evaluations"] = int(self.state.get("evaluations") or 0) + 1
        self.state["last"] = result
        self._save()
        return result

    def health(self) -> dict[str, Any]:
        last = self.state.get("last")
        if not isinstance(last, dict):
            last = {}
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "state_path": str(self.state_path),
            "evaluations": int(self.state.get("evaluations") or 0),
            "stress_state": str(last.get("stress_state") or "waiting_for_first_evaluation"),
            "starting_equity": self.starting_equity,
            "principal_floor_fraction": self.principal_floor_fraction,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "scenario_count": len(last.get("scenarios") or []),
            **self._authority_denied(),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "evaluations": 0,
            "history": [],
            "last": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("evaluations", 0)
                payload.setdefault("history", [])
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
