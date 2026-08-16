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

    VERSION = "2.0"
    SCHEMA_VERSION = 2
    EVIDENCE_AUTHORITY = "costed_shadow_episode_v2"

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
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": self.SCHEMA_VERSION,
                "pending": {},
                "episodes": {},
                "strategies": {},
                "directional_strategies": {},
            }
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == self.SCHEMA_VERSION:
                payload.setdefault("pending", {})
                payload.setdefault("episodes", {})
                payload.setdefault("strategies", {})
                payload.setdefault("directional_strategies", {})
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
        }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
