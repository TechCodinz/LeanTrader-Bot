from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


class StrategyObservatory:
    """Ungated paper evidence for every engine and timeframe signal."""

    VERSION = "1.0"

    def __init__(self, state_path: Path, *, round_trip_cost_bps: float = 30.0) -> None:
        if round_trip_cost_bps < 0:
            raise ValueError("observatory costs cannot be negative")
        self.state_path = state_path
        self.round_trip_cost_bps = round_trip_cost_bps
        self.state = self._load()
        self.calls = 0

    def observe(
        self,
        symbol: str,
        price: float,
        signals: list[dict[str, Any]],
        timeframe_signals: dict[str, float],
    ) -> dict[str, Any]:
        if not math.isfinite(price) or price <= 0:
            raise ValueError("observatory requires a positive finite price")
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

        pending = self.state.setdefault("pending", {}).setdefault(symbol, {})
        outcomes: list[dict[str, Any]] = []
        for name, previous in list(pending.items()):
            previous_price = float(previous.get("price") or 0.0)
            previous_score = float(previous.get("score") or 0.0)
            if previous_price <= 0 or abs(previous_score) < 1e-12:
                continue
            market_return = price / previous_price - 1.0
            direction = 1.0 if previous_score > 0 else -1.0
            net_return = direction * market_return - self.round_trip_cost_bps / 10_000.0
            record = self.state.setdefault("strategies", {}).setdefault(
                name,
                {
                    "samples": 0,
                    "wins": 0,
                    "cumulative_net_return": 0.0,
                    "last_net_return": None,
                    "symbols": {},
                },
            )
            record["samples"] = int(record["samples"]) + 1
            record["wins"] = int(record["wins"]) + int(net_return > 0)
            record["cumulative_net_return"] = float(record["cumulative_net_return"]) + net_return
            record["last_net_return"] = net_return
            symbol_record = record["symbols"].setdefault(symbol, {"samples": 0, "net_return": 0.0})
            symbol_record["samples"] = int(symbol_record["samples"]) + 1
            symbol_record["net_return"] = float(symbol_record["net_return"]) + net_return
            outcomes.append({"strategy": name, "net_return": net_return})

        now = time.time()
        self.state["pending"][symbol] = {
            name: {"price": price, "score": score, "timestamp": now} for name, score in normalized.items()
        }
        self.state["last_observation_epoch"] = now
        self.calls += 1
        self._save()
        return {
            "symbol": symbol,
            "signals_observed": len(normalized),
            "outcomes_recorded": len(outcomes),
            "outcomes": outcomes,
        }

    def health(self) -> dict[str, Any]:
        strategies: dict[str, dict[str, Any]] = {}
        for name, record in self.state.get("strategies", {}).items():
            samples = int(record.get("samples", 0))
            strategies[name] = {
                "samples": samples,
                "wins": int(record.get("wins", 0)),
                "win_rate": int(record.get("wins", 0)) / samples if samples else 0.0,
                "cumulative_net_return": float(record.get("cumulative_net_return", 0.0)),
                "last_net_return": record.get("last_net_return"),
                "symbols": len(record.get("symbols", {})),
            }
        pending = self.state.get("pending", {})
        return {
            "persistent": True,
            "state_path": str(self.state_path),
            "calls": self.calls,
            "strategies_measured": len(strategies),
            "pending_symbols": len(pending),
            "pending_signals": sum(len(rows) for rows in pending.values()),
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "router_gates_applied": False,
            "paper_observation_authority": True,
            "testnet_authority": False,
            "live_authority": False,
            "strategies": strategies,
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "pending": {}, "strategies": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == 1:
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {"schema_version": 1, "pending": {}, "strategies": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
