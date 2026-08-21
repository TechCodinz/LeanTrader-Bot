from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


class StrategyObservatory:
    """Ungated shadow evidence with transaction costs applied per closed episode.

    v1 charged a full round-trip cost to every consecutive observation. That was a
    deliberately conservative stress score, but it is not a valid estimate of the
    P&L of a signal that remains in the same direction across multiple polls.

    v2 separates two evidence types:
      * directional observations: gross next-observation skill, diagnostic only;
      * closed shadow episodes: cost-adjusted evidence, authoritative for Brain use.

    A shadow episode opens when a non-zero signal appears and closes only when its
    direction flips. The round-trip cost is charged once at episode close.
    """

    VERSION = "3.0"
    SCHEMA_VERSION = 3
    EVIDENCE_AUTHORITY = "costed_shadow_episode_v2"
    PROFITABILITY_AUTHORITY = "prospective_paper_net_of_costs_v1"
    MAX_CYCLE_RETURNS = 20_000
    MAX_FUNNEL_CYCLES = 2_000

    def __init__(self, state_path: Path, *, round_trip_cost_bps: float = 30.0) -> None:
        if round_trip_cost_bps < 0:
            raise ValueError("observatory costs cannot be negative")
        self.state_path = state_path
        self.round_trip_cost_bps = float(round_trip_cost_bps)
        self.state = self._load()
        self.calls = 0

    @staticmethod
    def _record_metric(
        record: dict[str, Any],
        *,
        value: float,
        symbol: str,
        value_key: str,
        ewma_key: str,
        alpha: float = 0.10,
    ) -> None:
        record["samples"] = int(record.get("samples", 0)) + 1
        record["wins"] = int(record.get("wins", 0)) + int(value > 0)
        record[value_key] = float(record.get(value_key, 0.0)) + value
        previous_ewma = float(record.get(ewma_key, 0.0))
        record[ewma_key] = value if record["samples"] == 1 else (1.0 - alpha) * previous_ewma + alpha * value
        record["negative_streak"] = 0 if value > 0 else int(record.get("negative_streak", 0)) + 1
        record["last_value"] = value

        symbols = record.setdefault("symbols", {})
        symbol_record = symbols.setdefault(
            symbol,
            {
                "samples": 0,
                "wins": 0,
                value_key: 0.0,
                ewma_key: 0.0,
                "negative_streak": 0,
                "last_value": None,
            },
        )
        symbol_record["samples"] = int(symbol_record.get("samples", 0)) + 1
        symbol_record["wins"] = int(symbol_record.get("wins", 0)) + int(value > 0)
        symbol_record[value_key] = float(symbol_record.get(value_key, 0.0)) + value
        symbol_previous_ewma = float(symbol_record.get(ewma_key, 0.0))
        symbol_record[ewma_key] = (
            value
            if symbol_record["samples"] == 1
            else (1.0 - alpha) * symbol_previous_ewma + alpha * value
        )
        symbol_record["negative_streak"] = (
            0 if value > 0 else int(symbol_record.get("negative_streak", 0)) + 1
        )
        symbol_record["last_value"] = value

    @staticmethod
    def _normalize_signals(
        signals: list[dict[str, Any]], timeframe_signals: dict[str, float]
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for signal in signals:
            name = str(signal.get("engine", "")).strip()
            score = float(signal.get("score") or 0.0)
            if name and math.isfinite(score):
                normalized[f"engine:{name}"] = max(-1.0, min(1.0, score))
        for timeframe, score_value in timeframe_signals.items():
            score = float(score_value)
            if math.isfinite(score):
                normalized[f"timeframe:{timeframe}"] = max(-1.0, min(1.0, score))
        return normalized

    def observe(
        self,
        symbol: str,
        price: float,
        signals: list[dict[str, Any]],
        timeframe_signals: dict[str, float],
    ) -> dict[str, Any]:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("observatory requires a positive finite price")

        normalized = self._normalize_signals(signals, timeframe_signals)
        now = time.time()
        pending = self.state.setdefault("pending", {}).setdefault(symbol, {})
        episodes = self.state.setdefault("episodes", {}).setdefault(symbol, {})
        directional_outcomes: list[dict[str, Any]] = []
        closed_episodes: list[dict[str, Any]] = []

        # Diagnostic gross directional skill. No transaction cost is charged here
        # because this is explicitly not trade-P&L evidence.
        for name, previous in list(pending.items()):
            previous_price = float(previous.get("price") or 0.0)
            previous_score = float(previous.get("score") or 0.0)
            if previous_price <= 0 or abs(previous_score) < 1e-12:
                continue
            market_return = price / previous_price - 1.0
            direction = 1.0 if previous_score > 0 else -1.0
            gross_return = direction * market_return
            record = self.state.setdefault("directional_strategies", {}).setdefault(
                name,
                {
                    "samples": 0,
                    "wins": 0,
                    "cumulative_gross_return": 0.0,
                    "ewma_gross_return": 0.0,
                    "negative_streak": 0,
                    "last_value": None,
                    "symbols": {},
                },
            )
            self._record_metric(
                record,
                value=gross_return,
                symbol=symbol,
                value_key="cumulative_gross_return",
                ewma_key="ewma_gross_return",
            )
            directional_outcomes.append({"strategy": name, "gross_return": gross_return})

        # Costed shadow episodes. A persistent signal is one held episode, not a
        # brand-new round trip every poll. Cost is charged once when direction flips.
        for name, score in normalized.items():
            if abs(score) < 1e-12:
                continue
            direction = 1 if score > 0 else -1
            active = episodes.get(name)
            if active is None:
                episodes[name] = {
                    "direction": direction,
                    "entry_price": price,
                    "entry_score": score,
                    "opened_at": now,
                    "observations": 1,
                }
                continue

            active_direction = int(active.get("direction") or 0)
            if active_direction == direction:
                active["observations"] = int(active.get("observations", 0)) + 1
                active["last_score"] = score
                active["last_seen_at"] = now
                continue

            entry_price = float(active.get("entry_price") or 0.0)
            if entry_price > 0 and active_direction in (-1, 1):
                gross_return = active_direction * (price / entry_price - 1.0)
                net_return = gross_return - self.round_trip_cost_bps / 10_000.0
                record = self.state.setdefault("strategies", {}).setdefault(
                    name,
                    {
                        "samples": 0,
                        "wins": 0,
                        "cumulative_net_return": 0.0,
                        "ewma_net_return": 0.0,
                        "negative_streak": 0,
                        "last_value": None,
                        "symbols": {},
                    },
                )
                self._record_metric(
                    record,
                    value=net_return,
                    symbol=symbol,
                    value_key="cumulative_net_return",
                    ewma_key="ewma_net_return",
                )
                closed_episodes.append(
                    {
                        "strategy": name,
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "observations": int(active.get("observations", 1)),
                    }
                )

            episodes[name] = {
                "direction": direction,
                "entry_price": price,
                "entry_score": score,
                "opened_at": now,
                "observations": 1,
            }

        self.state["pending"][symbol] = {
            name: {"price": price, "score": score, "timestamp": now} for name, score in normalized.items()
        }
        self.state["last_observation_epoch"] = now
        self.calls += 1
        self._save()
        return {
            "symbol": symbol,
            "signals_observed": len(normalized),
            "directional_outcomes_recorded": len(directional_outcomes),
            "episodes_closed": len(closed_episodes),
            # Backward-compatible result key; now means authoritative costed episodes.
            "outcomes_recorded": len(closed_episodes),
            "outcomes": closed_episodes,
            "evidence_authority": self.EVIDENCE_AUTHORITY,
        }

    def evidence(self, strategy: str, symbol: str | None = None) -> dict[str, Any]:
        """Return costed closed-episode evidence only.

        Diagnostic next-observation directional skill is intentionally excluded so
        the Brain cannot quarantine or downsize from per-poll pseudo-P&L.
        """
        name = str(strategy).strip()
        empty = {
            "authority": self.EVIDENCE_AUTHORITY,
            "samples": 0,
            "wins": 0,
            "average_net_return": 0.0,
            "ewma_net_return": 0.0,
        }
        if not name:
            return dict(empty)
        record = self.state.get("strategies", {}).get(name, {})
        if not isinstance(record, dict):
            return dict(empty)
        if symbol is not None:
            row = (record.get("symbols") or {}).get(symbol, {})
            if not isinstance(row, dict):
                row = {}
            samples = int(row.get("samples", 0))
            cumulative = float(row.get("cumulative_net_return", 0.0))
            wins = int(row.get("wins", 0))
            average = cumulative / samples if samples else 0.0
            ewma = float(row.get("ewma_net_return", average))
            return {
                "authority": self.EVIDENCE_AUTHORITY,
                "scope": "symbol",
                "strategy": name,
                "symbol": symbol,
                "samples": samples,
                "wins": wins,
                "win_rate": wins / samples if samples else 0.0,
                "cumulative_net_return": cumulative,
                "average_net_return": average,
                "ewma_net_return": ewma,
            }
        samples = int(record.get("samples", 0))
        cumulative = float(record.get("cumulative_net_return", 0.0))
        wins = int(record.get("wins", 0))
        average = cumulative / samples if samples else 0.0
        return {
            "authority": self.EVIDENCE_AUTHORITY,
            "scope": "global",
            "strategy": name,
            "samples": samples,
            "wins": wins,
            "win_rate": wins / samples if samples else 0.0,
            "cumulative_net_return": cumulative,
            "average_net_return": average,
            "ewma_net_return": float(record.get("ewma_net_return", average)),
            "negative_streak": int(record.get("negative_streak", 0)),
            "last_net_return": record.get("last_value"),
        }

    @staticmethod
    def _summary(record: dict[str, Any], *, value_key: str, ewma_key: str) -> dict[str, Any]:
        samples = int(record.get("samples", 0))
        cumulative = float(record.get(value_key, 0.0))
        return {
            "samples": samples,
            "wins": int(record.get("wins", 0)),
            "win_rate": int(record.get("wins", 0)) / samples if samples else 0.0,
            value_key: cumulative,
            "average_return": cumulative / samples if samples else 0.0,
            ewma_key: float(record.get(ewma_key, 0.0)),
            "negative_streak": int(record.get("negative_streak", 0)),
            "last_value": record.get("last_value"),
            "symbols": len(record.get("symbols", {})),
        }


    @staticmethod
    def _finite_metric(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def record_cycle(
        self,
        *,
        equity: float,
        cash: float,
        realized_pnl: float,
        starting_equity: float,
        open_positions: int,
        paper_trade_events: int,
        execution_funnel: dict[str, Any],
        decisions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist net portfolio evidence and final no-trade attribution.

        This method is observational only. It cannot promote a strategy, change a
        threshold, submit an order, or grant testnet/live authority.
        """
        equity = self._finite_metric(equity, "equity")
        cash = self._finite_metric(cash, "cash")
        realized_pnl = self._finite_metric(realized_pnl, "realized_pnl")
        starting_equity = self._finite_metric(starting_equity, "starting_equity")
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if open_positions < 0 or paper_trade_events < 0:
            raise ValueError("portfolio counts cannot be negative")
        if not isinstance(execution_funnel, dict) or not isinstance(decisions, dict):
            raise ValueError("cycle evidence must use mapping inputs")

        now = time.time()
        portfolio = self.state.setdefault("portfolio", {})
        authoritative_start = float(portfolio.get("starting_equity") or starting_equity)
        if authoritative_start <= 0:
            authoritative_start = starting_equity
        previous_equity = float(portfolio.get("equity") or authoritative_start)
        cycle_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        cycle_returns = [
            float(value)
            for value in portfolio.get("cycle_returns", [])
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        cycle_returns.append(cycle_return)
        cycle_returns = cycle_returns[-self.MAX_CYCLE_RETURNS :]

        peak_equity = max(
            authoritative_start,
            float(portfolio.get("peak_equity") or authoritative_start),
            equity,
        )
        current_drawdown = max(0.0, (peak_equity - equity) / max(peak_equity, 1e-12))
        max_drawdown = max(
            current_drawdown,
            float(portfolio.get("max_drawdown_pct") or 0.0),
        )
        net_pnl = equity - authoritative_start
        unrealized_pnl = net_pnl - realized_pnl
        portfolio.update(
            {
                "authority": self.PROFITABILITY_AUTHORITY,
                "cycles": int(portfolio.get("cycles", 0)) + 1,
                "starting_equity": authoritative_start,
                "equity": equity,
                "cash": cash,
                "net_pnl": net_pnl,
                "net_return": net_pnl / authoritative_start,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "peak_equity": peak_equity,
                "current_drawdown_pct": current_drawdown,
                "max_drawdown_pct": max_drawdown,
                "open_positions": int(open_positions),
                "paper_trade_events": int(paper_trade_events),
                "cycle_returns": cycle_returns,
                "first_observed_at": float(portfolio.get("first_observed_at") or now),
                "last_observed_at": now,
            }
        )

        funnel = self.state.setdefault(
            "decision_funnel",
            {
                "cycles": 0,
                "totals": {},
                "entry_block_reasons": {},
                "final_route_reasons": {},
                "regimes": {},
                "recent_cycles": [],
            },
        )
        funnel["cycles"] = int(funnel.get("cycles", 0)) + 1
        totals = funnel.setdefault("totals", {})
        aggregate_fields = (
            "symbols_evaluated",
            "base_enter_candidates",
            "router_approved_pre_brain",
            "brain_approved",
            "cognitive_governance_reviewed",
            "cognitive_governance_vetoes",
            "cognitive_governance_reductions",
            "final_route_allowed",
            "entry_attempts",
            "entry_failures",
            "buy_events",
            "sell_events",
            "entry_blocks",
        )
        for key in aggregate_fields:
            value = int(execution_funnel.get(key, 0) or 0)
            if value < 0:
                raise ValueError(f"execution_funnel.{key} cannot be negative")
            totals[key] = int(totals.get(key, 0)) + value

        block_reasons = funnel.setdefault("entry_block_reasons", {})
        for reason, count in (execution_funnel.get("entry_block_reasons") or {}).items():
            normalized = str(reason or "unspecified").strip() or "unspecified"
            block_reasons[normalized] = int(block_reasons.get(normalized, 0)) + max(
                0, int(count or 0)
            )

        route_reasons = funnel.setdefault("final_route_reasons", {})
        regimes = funnel.setdefault("regimes", {})
        for symbol, decision in decisions.items():
            if not isinstance(decision, dict):
                continue
            allowed = decision.get("allowed") is True
            reason = "allowed" if allowed else str(
                decision.get("reason") or "unspecified_block"
            ).strip()
            route_reasons[reason] = int(route_reasons.get(reason, 0)) + 1
            regime = str(decision.get("regime") or "unknown").strip() or "unknown"
            regime_row = regimes.setdefault(regime, {"evaluated": 0, "allowed": 0})
            regime_row["evaluated"] = int(regime_row.get("evaluated", 0)) + 1
            regime_row["allowed"] = int(regime_row.get("allowed", 0)) + int(allowed)

        recent = list(funnel.get("recent_cycles", []))
        recent.append(
            {
                "timestamp": now,
                "equity": equity,
                "net_return": net_pnl / authoritative_start,
                "symbols_evaluated": int(execution_funnel.get("symbols_evaluated", 0) or 0),
                "final_route_allowed": int(
                    execution_funnel.get("final_route_allowed", 0) or 0
                ),
                "entry_attempts": int(execution_funnel.get("entry_attempts", 0) or 0),
                "buy_events": int(execution_funnel.get("buy_events", 0) or 0),
                "entry_blocks": int(execution_funnel.get("entry_blocks", 0) or 0),
                "halted": bool(execution_funnel.get("halted")),
            }
        )
        funnel["recent_cycles"] = recent[-self.MAX_FUNNEL_CYCLES :]
        funnel["last_observed_at"] = now
        self.state["last_profitability_cycle_epoch"] = now
        self._save()
        return self.profitability_snapshot()

    def profitability_snapshot(self) -> dict[str, Any]:
        portfolio = self.state.get("portfolio", {})
        returns = [
            float(value)
            for value in portfolio.get("cycle_returns", [])
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        samples = len(returns)
        average = sum(returns) / samples if samples else 0.0
        volatility = (
            math.sqrt(sum((value - average) ** 2 for value in returns) / samples)
            if samples
            else 0.0
        )
        downside = [value for value in returns if value < 0]
        downside_deviation = (
            math.sqrt(sum(value * value for value in downside) / len(downside))
            if downside
            else 0.0
        )
        gross_profit = sum(value for value in returns if value > 0)
        gross_loss = abs(sum(value for value in returns if value < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        positive_cycles = sum(1 for value in returns if value > 0)
        return {
            "authority": self.PROFITABILITY_AUTHORITY,
            "state": (
                "collecting_prospective_evidence"
                if int(portfolio.get("cycles", 0)) > 0
                else "waiting_for_first_cycle"
            ),
            "cycles": int(portfolio.get("cycles", 0)),
            "return_samples": samples,
            "starting_equity": float(portfolio.get("starting_equity", 0.0)),
            "equity": float(portfolio.get("equity", 0.0)),
            "cash": float(portfolio.get("cash", 0.0)),
            "net_pnl": float(portfolio.get("net_pnl", 0.0)),
            "net_return": float(portfolio.get("net_return", 0.0)),
            "realized_pnl": float(portfolio.get("realized_pnl", 0.0)),
            "unrealized_pnl": float(portfolio.get("unrealized_pnl", 0.0)),
            "peak_equity": float(portfolio.get("peak_equity", 0.0)),
            "current_drawdown_pct": float(
                portfolio.get("current_drawdown_pct", 0.0)
            ),
            "max_drawdown_pct": float(portfolio.get("max_drawdown_pct", 0.0)),
            "average_cycle_return": average,
            "cycle_return_volatility": volatility,
            "downside_deviation": downside_deviation,
            "positive_cycle_rate": positive_cycles / samples if samples else 0.0,
            "profit_factor": profit_factor,
            "open_positions": int(portfolio.get("open_positions", 0)),
            "paper_trade_events": int(portfolio.get("paper_trade_events", 0)),
            "promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def decision_funnel_snapshot(self) -> dict[str, Any]:
        funnel = self.state.get("decision_funnel", {})
        totals = dict(funnel.get("totals", {}))
        evaluated = int(totals.get("symbols_evaluated", 0))
        allowed = int(totals.get("final_route_allowed", 0))
        buys = int(totals.get("buy_events", 0))
        return {
            "cycles": int(funnel.get("cycles", 0)),
            "totals": totals,
            "final_route_allow_rate": allowed / evaluated if evaluated else 0.0,
            "paper_entry_conversion_rate": buys / evaluated if evaluated else 0.0,
            "entry_block_reasons": dict(funnel.get("entry_block_reasons", {})),
            "final_route_reasons": dict(funnel.get("final_route_reasons", {})),
            "regimes": dict(funnel.get("regimes", {})),
            "recent_cycles_retained": len(funnel.get("recent_cycles", [])),
            "execution_authority": False,
        }

    def health(self) -> dict[str, Any]:
        strategies = {
            name: self._summary(record, value_key="cumulative_net_return", ewma_key="ewma_net_return")
            for name, record in self.state.get("strategies", {}).items()
        }
        directional = {
            name: self._summary(
                record,
                value_key="cumulative_gross_return",
                ewma_key="ewma_gross_return",
            )
            for name, record in self.state.get("directional_strategies", {}).items()
        }
        pending = self.state.get("pending", {})
        episodes = self.state.get("episodes", {})
        return {
            "persistent": True,
            "state_path": str(self.state_path),
            "schema_version": self.SCHEMA_VERSION,
            "evidence_model": "episode_costed_v2",
            "evidence_authority": self.EVIDENCE_AUTHORITY,
            "calls": self.calls,
            "strategies_measured": len(strategies),
            "directional_strategies_measured": len(directional),
            "pending_symbols": len(pending),
            "pending_signals": sum(len(rows) for rows in pending.values()),
            "open_shadow_episodes": sum(len(rows) for rows in episodes.values()),
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "cost_application": "once_per_closed_shadow_episode",
            "router_gates_applied": False,
            "paper_observation_authority": True,
            "testnet_authority": False,
            "live_authority": False,
            "legacy_v1_preserved": isinstance(self.state.get("legacy_v1"), dict),
            "strategies": strategies,
            "directional_strategies": directional,
            "profitability_intelligence": self.profitability_snapshot(),
            "decision_funnel": self.decision_funnel_snapshot(),
            "promotion_authority": False,
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": self.SCHEMA_VERSION,
                "pending": {},
                "episodes": {},
                "strategies": {},
                "directional_strategies": {},
                "portfolio": {},
                "decision_funnel": {},
            }
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == self.SCHEMA_VERSION:
                payload.setdefault("pending", {})
                payload.setdefault("episodes", {})
                payload.setdefault("strategies", {})
                payload.setdefault("directional_strategies", {})
                payload.setdefault("portfolio", {})
                payload.setdefault("decision_funnel", {})
                return payload
            if payload.get("schema_version") == 2:
                payload["schema_version"] = self.SCHEMA_VERSION
                payload.setdefault("portfolio", {})
                payload.setdefault("decision_funnel", {})
                payload["migrated_from_v2_at"] = time.time()
                return payload
            if payload.get("schema_version") == 1:
                # Preserve contaminated v1 evidence for forensics/audit, but never
                # feed it into new Brain authority.
                return {
                    "schema_version": self.SCHEMA_VERSION,
                    "pending": {},
                    "episodes": {},
                    "strategies": {},
                    "directional_strategies": {},
                    "portfolio": {},
                    "decision_funnel": {},
                    "legacy_v1": payload,
                    "migrated_at": time.time(),
                }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {
            "schema_version": self.SCHEMA_VERSION,
            "pending": {},
            "episodes": {},
            "strategies": {},
            "directional_strategies": {},
            "portfolio": {},
            "decision_funnel": {},
        }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
