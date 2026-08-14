from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


class MarketEvidenceGate:
    """Classify symbol performance from closed, net-of-cost paper outcomes."""

    VERSION = "1.0"

    def __init__(self, state_path: Path, *, minimum_samples: int = 8, rolling_window: int = 50) -> None:
        if minimum_samples < 3:
            raise ValueError("market evidence minimum_samples must be at least 3")
        if rolling_window < minimum_samples:
            raise ValueError("market evidence rolling_window cannot be smaller than minimum_samples")
        self.state_path = state_path
        self.minimum_samples = minimum_samples
        self.rolling_window = rolling_window
        self.state = self._load()

    def assess(self, symbol: str) -> dict[str, Any]:
        record = self.state.setdefault("symbols", {}).setdefault(
            symbol,
            {"outcomes": [], "assessments": 0, "last_trade": None},
        )
        record["assessments"] = int(record.get("assessments", 0)) + 1
        outcomes = [float(value) for value in record.get("outcomes", [])][-self.rolling_window :]
        samples = len(outcomes)
        wins = [value for value in outcomes if value > 0]
        losses = [value for value in outcomes if value < 0]
        expectancy = float(np.mean(outcomes)) if outcomes else 0.0
        win_rate = len(wins) / samples if samples else 0.0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else (math.inf if wins else 0.0)

        if samples < self.minimum_samples:
            classification, allowed, multiplier = "exploration", True, 0.35
        elif expectancy > 0 and profit_factor >= 1.10 and win_rate >= 0.35:
            classification, allowed, multiplier = "qualified", True, 1.0
        elif expectancy >= -0.001 and profit_factor >= 0.90:
            classification, allowed, multiplier = "probation", True, 0.25
        else:
            # A quarantined market receives one small evidence probe every 100
            # assessments; otherwise it could never demonstrate recovery.
            probe = int(record["assessments"]) % 100 == 0
            classification = "quarantined_probe" if probe else "quarantined"
            allowed, multiplier = probe, 0.10 if probe else 0.0
        record["classification"] = classification
        return {
            "classification": classification,
            "allowed": allowed,
            "size_multiplier": multiplier,
            "samples": samples,
            "expectancy": expectancy,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "rolling_window": self.rolling_window,
        }

    def record(
        self,
        symbol: str,
        realized_return: float,
        predicted_probability: float | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(realized_return):
            raise ValueError("market evidence requires a finite realized return")
        if predicted_probability is not None and (
            not math.isfinite(predicted_probability) or not 0.0 <= predicted_probability <= 1.0
        ):
            raise ValueError("predicted probability must be finite and between zero and one")
        record = self.state.setdefault("symbols", {}).setdefault(
            symbol,
            {"outcomes": [], "assessments": 0, "last_trade": None},
        )
        outcomes = [float(value) for value in record.get("outcomes", [])]
        outcomes.append(float(realized_return))
        record["outcomes"] = outcomes[-self.rolling_window :]
        if predicted_probability is not None:
            forecasts = list(record.get("forecasts", []))
            forecasts.append(
                {
                    "probability": float(predicted_probability),
                    "outcome": int(realized_return > 0),
                }
            )
            record["forecasts"] = forecasts[-self.rolling_window :]
        record["last_trade"] = dt.datetime.now(dt.UTC).isoformat()
        self._save()
        return self.assess(symbol)

    def health(self) -> dict[str, Any]:
        classifications: dict[str, int] = {}
        samples = 0
        forecasts: list[dict[str, Any]] = []
        for record in self.state.get("symbols", {}).values():
            name = str(record.get("classification", "unassessed"))
            classifications[name] = classifications.get(name, 0) + 1
            samples += len(record.get("outcomes", []))
            forecasts.extend(record.get("forecasts", []))
        calibration: dict[str, Any] = {"samples": len(forecasts), "state": "waiting_for_closed_trades"}
        if forecasts:
            probabilities = np.asarray([row["probability"] for row in forecasts], dtype=float)
            outcomes = np.asarray([row["outcome"] for row in forecasts], dtype=float)
            brier = float(np.mean((probabilities - outcomes) ** 2))
            ece = 0.0
            for lower in np.linspace(0.0, 1.0, 10, endpoint=False):
                upper = lower + 0.1
                mask = (probabilities >= lower) & (
                    probabilities < upper if upper < 1.0 else probabilities <= upper
                )
                if mask.any():
                    ece += float(mask.mean()) * abs(
                        float(probabilities[mask].mean() - outcomes[mask].mean())
                    )
            calibration = {
                "samples": len(forecasts),
                "state": "measured",
                "brier_score": brier,
                "expected_calibration_error": ece,
            }
        return {
            "persistent": True,
            "state_path": str(self.state_path),
            "minimum_samples": self.minimum_samples,
            "rolling_window": self.rolling_window,
            "symbols_observed": len(self.state.get("symbols", {})),
            "closed_trade_samples": samples,
            "classifications": classifications,
            "cold_start": "bounded_exploration",
            "quarantine_retest": "one_in_100_assessments",
            "online_calibration": calibration,
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "symbols": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == 1 and isinstance(payload.get("symbols"), dict):
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {"schema_version": 1, "symbols": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)


class BoundedDecisionRouter:
    """Route adaptive and ultra-engine evidence into paper/Testnet authority."""

    VERSION = "1.0"

    def __init__(
        self,
        evidence: MarketEvidenceGate,
        *,
        minimum_advanced_confidence: float = 0.20,
        minimum_combined_score: float = 0.20,
        negative_consensus_veto: float = -0.25,
    ) -> None:
        self.evidence = evidence
        self.minimum_advanced_confidence = minimum_advanced_confidence
        self.minimum_combined_score = minimum_combined_score
        self.negative_consensus_veto = negative_consensus_veto
        self.routes = 0
        self.approved = 0
        self.blocked: dict[str, int] = {}
        self.last_routes: dict[str, dict[str, Any]] = {}

    def route(
        self,
        *,
        symbol: str,
        base_enter: bool,
        base_score: float,
        base_confidence: float,
        advanced: dict[str, Any],
    ) -> dict[str, Any]:
        swarm = advanced.get("swarm") or {}
        advanced_score = float(swarm.get("score") or 0.0)
        advanced_confidence = float(swarm.get("confidence") or 0.0)
        signals = list(advanced.get("signals") or [])
        contributing = [
            str(signal.get("engine"))
            for signal in signals
            if float(signal.get("confidence") or 0.0) > 0
        ]
        combined_score = float(np.clip(0.70 * base_score + 0.30 * advanced_score, -1.0, 1.0))
        evidence = self.evidence.assess(symbol)
        reason = "approved"
        allowed = True
        if not base_enter:
            allowed, reason = False, "adaptive_signal_not_ready"
        elif advanced.get("news_blackout") is True:
            allowed, reason = False, "high_impact_news_blackout"
        elif len(contributing) < 3:
            allowed, reason = False, "insufficient_ultra_engine_evidence"
        elif advanced_confidence < self.minimum_advanced_confidence:
            allowed, reason = False, "low_ultra_consensus_confidence"
        elif advanced_score <= self.negative_consensus_veto:
            allowed, reason = False, "negative_ultra_consensus"
        elif combined_score < self.minimum_combined_score:
            allowed, reason = False, "combined_score_below_threshold"
        elif not evidence["allowed"]:
            allowed, reason = False, "market_evidence_quarantine"

        size_multiplier = float(evidence["size_multiplier"]) if allowed else 0.0
        result = {
            "allowed": allowed,
            "reason": reason,
            "combined_score": combined_score,
            "predicted_probability": float(np.clip(0.5 + combined_score / 2.0, 0.0, 1.0)),
            "base_score": base_score,
            "base_confidence": base_confidence,
            "advanced_score": advanced_score,
            "advanced_confidence": advanced_confidence,
            "contributing_engines": contributing,
            "size_multiplier": size_multiplier,
            "market_evidence": evidence,
            "authority": "paper_and_testnet_only",
            "live_authority": False,
        }
        self.routes += 1
        if allowed:
            self.approved += 1
        else:
            self.blocked[reason] = self.blocked.get(reason, 0) + 1
        self.last_routes[symbol] = result
        return result

    def record_outcome(
        self,
        symbol: str,
        realized_return: float,
        predicted_probability: float | None = None,
    ) -> dict[str, Any]:
        return self.evidence.record(symbol, realized_return, predicted_probability)

    def health(self) -> dict[str, Any]:
        return {
            "routes": self.routes,
            "approved": self.approved,
            "blocked": dict(self.blocked),
            "symbols_routed": len(self.last_routes),
            "minimum_advanced_confidence": self.minimum_advanced_confidence,
            "minimum_combined_score": self.minimum_combined_score,
            "negative_consensus_veto": self.negative_consensus_veto,
            "adaptive_weight": 0.70,
            "ultra_weight": 0.30,
            "paper_authority": True,
            "testnet_authority": True,
            "live_authority": False,
            "market_evidence": self.evidence.health(),
        }
