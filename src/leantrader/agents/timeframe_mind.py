from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class TimeframeAssessment:
    symbol: str
    timeframe: str
    direction: str
    independently_qualified: bool
    confidence: float
    expected_edge_bps: float
    modeled_round_trip_cost_bps: float
    recent_momentum_bps: float
    directional_consistency: float
    path_efficiency: float
    q75_abs_move_bps: float
    samples: int
    reason: str
    research_hypothesis: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiTimeframeMind:
    """Independent descriptive qualification for each agent timeframe.

    This is deliberately a paper/research hypothesis layer. It measures whether
    recent movement on one timeframe is directional and economically large
    enough to clear modeled costs; it does not inherit authority from another
    timeframe and it does not claim predictive profitability.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        minimum_samples: int = 30,
        lookback_bars: int = 12,
        minimum_confidence: float = 0.55,
        capture_efficiency: float = 0.35,
        minimum_modeled_round_trip_cost_bps: float = 30.0,
    ) -> None:
        if minimum_samples < 20:
            raise ValueError("minimum_samples must be at least 20")
        if lookback_bars < 3:
            raise ValueError("lookback_bars must be at least 3")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0.0 < capture_efficiency <= 1.0:
            raise ValueError("capture_efficiency must be in (0, 1]")
        if minimum_modeled_round_trip_cost_bps < 30.0:
            raise ValueError("modeled round-trip cost floor cannot be below 30 bps")
        self.minimum_samples = int(minimum_samples)
        self.lookback_bars = int(lookback_bars)
        self.minimum_confidence = float(minimum_confidence)
        self.capture_efficiency = float(capture_efficiency)
        self.minimum_modeled_round_trip_cost_bps = float(minimum_modeled_round_trip_cost_bps)
        self.assessments = 0
        self.qualified = 0

    @staticmethod
    def _clean(frame: pd.DataFrame, minimum_samples: int) -> pd.Series:
        if not isinstance(frame, pd.DataFrame) or "close" not in frame.columns:
            raise ValueError("timeframe assessment requires close candles")
        closes = pd.to_numeric(frame["close"], errors="coerce")
        closes = closes[closes.notna() & (closes > 0)].astype(float)
        if len(closes) < minimum_samples:
            raise ValueError("insufficient timeframe samples")
        return closes

    def assess(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: pd.DataFrame,
        modeled_round_trip_cost_bps: float | None = None,
    ) -> TimeframeAssessment:
        closes = self._clean(candles, self.minimum_samples)
        returns = closes.pct_change().dropna() * 10_000.0
        returns = returns[returns.map(math.isfinite)]
        if len(returns) < self.minimum_samples - 1:
            raise ValueError("insufficient finite timeframe returns")
        window = returns.iloc[-min(self.lookback_bars, len(returns)) :]
        signed_sum = float(window.sum())
        if signed_sum > 0:
            direction = "long"
            same_direction = window > 0
        elif signed_sum < 0:
            direction = "short"
            same_direction = window < 0
        else:
            direction = "flat"
            same_direction = window == 0

        consistency = float(same_direction.mean()) if len(window) else 0.0
        start_index = max(0, len(closes) - len(window) - 1)
        path = closes.iloc[start_index:]
        path_changes = path.diff().dropna().abs()
        path_distance = float(path_changes.sum())
        displacement = abs(float(path.iloc[-1]) - float(path.iloc[0])) if len(path) >= 2 else 0.0
        efficiency = 0.0 if path_distance <= 0 else max(0.0, min(1.0, displacement / path_distance))
        q75 = float(window.abs().quantile(0.75)) if len(window) else 0.0
        momentum = abs(signed_sum)
        # Bound the economic hypothesis by both path momentum and typical bar
        # movement so a single spike cannot manufacture a huge expected edge.
        typical_path_budget = q75 * max(1.0, len(window) / 4.0)
        gross_capture = min(momentum, typical_path_budget) * self.capture_efficiency
        expected_edge = gross_capture * consistency * max(0.25, efficiency)
        confidence = max(0.0, min(1.0, consistency * (0.5 + 0.5 * efficiency)))
        requested_cost = (
            self.minimum_modeled_round_trip_cost_bps
            if modeled_round_trip_cost_bps is None
            else float(modeled_round_trip_cost_bps)
        )
        if not math.isfinite(requested_cost):
            raise ValueError("modeled_round_trip_cost_bps must be finite")
        modeled_cost = max(self.minimum_modeled_round_trip_cost_bps, requested_cost)

        reason = "qualified"
        independently_qualified = True
        if direction == "flat":
            independently_qualified, reason = False, "flat_timeframe"
        elif confidence < self.minimum_confidence:
            independently_qualified, reason = False, "timeframe_confidence_below_threshold"
        elif expected_edge <= modeled_cost:
            independently_qualified, reason = False, "timeframe_edge_does_not_clear_modeled_cost"

        result = TimeframeAssessment(
            symbol=str(symbol).upper(),
            timeframe=str(timeframe),
            direction=direction,
            independently_qualified=independently_qualified,
            confidence=confidence,
            expected_edge_bps=expected_edge,
            modeled_round_trip_cost_bps=modeled_cost,
            recent_momentum_bps=momentum,
            directional_consistency=consistency,
            path_efficiency=efficiency,
            q75_abs_move_bps=q75,
            samples=len(returns),
            reason=reason,
        )
        self.assessments += 1
        if independently_qualified:
            self.qualified += 1
        return result

    def assess_many(
        self,
        *,
        symbol: str,
        frames: Mapping[str, pd.DataFrame],
        modeled_round_trip_cost_bps: float | None = None,
    ) -> dict[str, TimeframeAssessment]:
        result: dict[str, TimeframeAssessment] = {}
        for timeframe, frame in frames.items():
            try:
                result[str(timeframe)] = self.assess(
                    symbol=symbol,
                    timeframe=str(timeframe),
                    candles=frame,
                    modeled_round_trip_cost_bps=modeled_round_trip_cost_bps,
                )
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def agrees_with_position(assessment: TimeframeAssessment, *, side: str) -> bool:
        return assessment.independently_qualified and assessment.direction == str(side).lower()

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "assessments": self.assessments,
            "qualified": self.qualified,
            "minimum_samples": self.minimum_samples,
            "minimum_confidence": self.minimum_confidence,
            "capture_efficiency": self.capture_efficiency,
            "minimum_modeled_round_trip_cost_bps": self.minimum_modeled_round_trip_cost_bps,
            "independent_timeframe_qualification": True,
            "research_hypothesis": True,
            "predictive_profit_claim": False,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
