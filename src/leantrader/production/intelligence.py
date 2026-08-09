from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENTS = ("trend", "momentum", "mean_reversion")
DEFAULT_WEIGHTS = {
    "trend": {"trend": 0.55, "momentum": 0.35, "mean_reversion": 0.10},
    "range": {"trend": 0.15, "momentum": 0.20, "mean_reversion": 0.65},
    "high_volatility": {"trend": 0.45, "momentum": 0.35, "mean_reversion": 0.20},
}


@dataclass(frozen=True)
class QualityReport:
    valid: bool
    score: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class IntelligenceDecision:
    close: float
    atr: float
    enter_long: bool
    trend_up: bool
    regime: str
    confidence: float
    quality_score: float
    component_scores: dict[str, float]
    weights: dict[str, float]
    rationale: tuple[str, ...]


class AdaptiveIntelligence:
    """Deterministic ensemble with bounded, evidence-gated paper-trade adaptation."""

    VERSION = "2.0"

    def __init__(
        self,
        state_path: Path,
        *,
        learning_rate: float = 0.08,
        min_samples: int = 5,
        minimum_weight: float = 0.10,
        maximum_weight: float = 0.70,
    ) -> None:
        if not 0 < learning_rate <= 0.25:
            raise ValueError("learning_rate must be in (0, 0.25]")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        self.state_path = state_path
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.minimum_weight = minimum_weight
        self.maximum_weight = maximum_weight
        self.weights = {regime: values.copy() for regime, values in DEFAULT_WEIGHTS.items()}
        self.pending_samples = {regime: 0 for regime in DEFAULT_WEIGHTS}
        self.evidence = {regime: {name: 0.0 for name in COMPONENTS} for regime in DEFAULT_WEIGHTS}
        self.promotions = 0
        self.last_promotion: str | None = None
        self._load()

    def evaluate(self, frame: pd.DataFrame) -> IntelligenceDecision:
        quality = self._quality(frame)
        if not quality.valid:
            raise ValueError("market data rejected: " + "; ".join(quality.issues))

        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(
            axis=1
        )
        atr = float(true_range.rolling(14).mean().iloc[-1])
        atr_floor = max(atr, float(close.iloc[-1]) * 1e-6)

        ema_fast = close.ewm(span=50, adjust=False).mean()
        ema_slow = close.ewm(span=200, adjust=False).mean()
        trend_strength = float(abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / atr_floor)
        volatility = float(atr / max(float(close.iloc[-1]), 1e-9))
        historical_volatility = (true_range / close.replace(0, np.nan)).rolling(100).median().iloc[-1]

        if math.isfinite(float(historical_volatility)) and volatility > float(historical_volatility) * 2.0:
            regime = "high_volatility"
        elif trend_strength >= 1.5:
            regime = "trend"
        else:
            regime = "range"

        mean_20 = close.rolling(20).mean().iloc[-1]
        std_20 = max(float(close.rolling(20).std(ddof=0).iloc[-1]), atr_floor * 0.25)
        component_scores = {
            "trend": float(np.tanh((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (2.0 * atr_floor))),
            "momentum": float(np.tanh((close.iloc[-1] - close.iloc[-20]) / (3.0 * atr_floor))),
            "mean_reversion": float(-np.tanh((close.iloc[-1] - mean_20) / (2.0 * std_20))),
        }
        weights = self.weights[regime].copy()
        score = sum(component_scores[name] * weights[name] for name in COMPONENTS)
        confidence = min(1.0, abs(score)) * quality.score
        threshold = 0.30 if regime != "high_volatility" else 0.55
        enter_long = score >= threshold and quality.score >= 0.90
        trend_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
        rationale = (
            f"regime={regime}",
            f"ensemble={score:.3f}",
            f"threshold={threshold:.2f}",
            f"data_quality={quality.score:.2f}",
        )
        return IntelligenceDecision(
            close=float(close.iloc[-1]),
            atr=atr,
            enter_long=enter_long,
            trend_up=trend_up,
            regime=regime,
            confidence=confidence,
            quality_score=quality.score,
            component_scores=component_scores,
            weights=weights,
            rationale=rationale,
        )

    def learn(self, metadata: dict[str, Any], realized_return: float) -> bool:
        """Stage evidence and promote a bounded weight update after enough closed trades."""
        regime = str(metadata.get("regime", ""))
        scores = metadata.get("component_scores", {})
        if regime not in self.weights or not isinstance(scores, dict):
            return False
        if not all(name in scores and math.isfinite(float(scores[name])) for name in COMPONENTS):
            return False

        outcome = float(np.clip(realized_return / 0.02, -1.0, 1.0))
        self.pending_samples[regime] += 1
        for name in COMPONENTS:
            self.evidence[regime][name] += float(np.clip(float(scores[name]), -1.0, 1.0)) * outcome

        promoted = False
        if self.pending_samples[regime] >= self.min_samples:
            samples = self.pending_samples[regime]
            proposed = {
                name: self.weights[regime][name] * math.exp(self.learning_rate * self.evidence[regime][name] / samples)
                for name in COMPONENTS
            }
            self.weights[regime] = self._bounded_normalize(proposed)
            self.pending_samples[regime] = 0
            self.evidence[regime] = {name: 0.0 for name in COMPONENTS}
            self.promotions += 1
            self.last_promotion = dt.datetime.now(dt.UTC).isoformat()
            promoted = True
        self._save()
        return promoted

    def health(self) -> dict[str, Any]:
        return {
            "adaptive": True,
            "bounded": True,
            "promotions": self.promotions,
            "last_promotion": self.last_promotion,
            "pending_samples": self.pending_samples.copy(),
            "weights": {regime: values.copy() for regime, values in self.weights.items()},
        }

    @staticmethod
    def _quality(frame: pd.DataFrame) -> QualityReport:
        required = {"open", "high", "low", "close", "volume"}
        issues: list[str] = []
        if not required.issubset(frame.columns):
            missing = sorted(required - set(frame.columns))
            return QualityReport(False, 0.0, (f"missing columns: {','.join(missing)}",))
        if len(frame) < 220:
            return QualityReport(False, 0.0, ("fewer than 220 candles",))

        numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            issues.append("non-finite OHLCV values")
        if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
            issues.append("non-positive prices")
        if (numeric["volume"] < 0).any():
            issues.append("negative volume")
        if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any():
            issues.append("high below candle values")
        if (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any():
            issues.append("low above candle values")
        if (numeric["high"] < numeric["low"]).any():
            issues.append("high below low")

        if "timestamp" in frame.columns:
            timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
            if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
                issues.append("timestamps invalid or not monotonic")
            elif timestamps.duplicated().any():
                issues.append("duplicate timestamps")
            elif len(timestamps) > 2:
                deltas = timestamps.diff().dropna()
                median = float(deltas.median())
                if median <= 0 or (deltas > median * 3.0).any():
                    issues.append("material candle gaps")

        score = max(0.0, 1.0 - 0.25 * len(issues))
        return QualityReport(not issues, score, tuple(issues))

    def _bounded_normalize(self, weights: dict[str, float]) -> dict[str, float]:
        if len(weights) * self.minimum_weight > 1 or len(weights) * self.maximum_weight < 1:
            raise ValueError("weight bounds cannot sum to one")
        remaining = 1.0
        free = set(weights)
        result: dict[str, float] = {}
        while free:
            positive_total = sum(max(0.0, weights[name]) for name in free)
            if positive_total <= 0:
                proposed = {name: remaining / len(free) for name in free}
            else:
                proposed = {name: remaining * max(0.0, weights[name]) / positive_total for name in free}
            below = {name for name, value in proposed.items() if value < self.minimum_weight}
            above = {name for name, value in proposed.items() if value > self.maximum_weight}
            if not below and not above:
                result.update(proposed)
                break
            for name in below:
                result[name] = self.minimum_weight
                remaining -= self.minimum_weight
                free.remove(name)
            for name in above - below:
                result[name] = self.maximum_weight
                remaining -= self.maximum_weight
                free.remove(name)
        return result

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0:
            return {name: 1.0 / len(weights) for name in weights}
        return {name: max(0.0, value) / total for name, value in weights.items()}

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1:
                return
            loaded_weights = data.get("weights", {})
            for regime in DEFAULT_WEIGHTS:
                values = loaded_weights.get(regime, {})
                if set(values) == set(COMPONENTS):
                    self.weights[regime] = self._bounded_normalize({name: float(values[name]) for name in COMPONENTS})
            self.pending_samples.update(
                {k: int(v) for k, v in data.get("pending_samples", {}).items() if k in self.pending_samples}
            )
            for regime, values in data.get("evidence", {}).items():
                if regime in self.evidence and isinstance(values, dict):
                    self.evidence[regime].update(
                        {name: float(value) for name, value in values.items() if name in COMPONENTS}
                    )
            self.promotions = int(data.get("promotions", 0))
            self.last_promotion = data.get("last_promotion")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # A corrupt optional learning state must never prevent safe paper startup.
            return

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "weights": self.weights,
            "pending_samples": self.pending_samples,
            "evidence": self.evidence,
            "promotions": self.promotions,
            "last_promotion": self.last_promotion,
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
