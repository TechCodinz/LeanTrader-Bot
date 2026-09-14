from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import pandas as pd

from .opportunity_radar import OpportunitySnapshot


@dataclass(frozen=True)
class MovementProfile:
    symbol: str
    samples: int
    nominal_price: float
    median_abs_move_bps: float
    q75_abs_move_bps: float
    q90_abs_move_bps: float
    realized_volatility_bps: float
    movement_frequency_per_minute: float
    directional_persistence: float
    reversal_tendency: float
    liquidity_score: float
    fill_probability: float
    spread_bps: float
    expected_capture_bps: float
    capture_efficiency: float
    source: str = "measured_1m_candles_and_discovery"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketMovementProfiler:
    """Turn recent market movement into conservative opportunity economics.

    The profiler measures movement; it does not predict profit. Expected capture
    is a bounded fraction of observed q75 absolute movement and remains subject
    to the radar's >=30 bps round-trip cost floor before an opportunity can be
    considered qualified.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        minimum_samples: int = 30,
        capture_efficiency: float = 0.50,
        movement_threshold_bps: float = 20.0,
        reference_quote_volume_usd: float = 1_000_000.0,
        maximum_acceptable_spread_bps: float = 25.0,
    ) -> None:
        if minimum_samples < 10:
            raise ValueError("minimum_samples must be at least 10")
        if not 0.0 < capture_efficiency <= 1.0:
            raise ValueError("capture_efficiency must be in (0, 1]")
        if movement_threshold_bps <= 0:
            raise ValueError("movement_threshold_bps must be positive")
        if reference_quote_volume_usd <= 0:
            raise ValueError("reference_quote_volume_usd must be positive")
        if maximum_acceptable_spread_bps <= 0:
            raise ValueError("maximum_acceptable_spread_bps must be positive")
        self.minimum_samples = int(minimum_samples)
        self.capture_efficiency = float(capture_efficiency)
        self.movement_threshold_bps = float(movement_threshold_bps)
        self.reference_quote_volume_usd = float(reference_quote_volume_usd)
        self.maximum_acceptable_spread_bps = float(maximum_acceptable_spread_bps)
        self.profiles = 0
        self.rejected = 0

    @staticmethod
    def _finite_nonnegative(value: Any, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return number

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, minimum_samples: int) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("candles must be a pandas DataFrame")
        if "close" not in frame.columns:
            raise ValueError("candles require a close column")
        cleaned = frame.copy()
        cleaned["close"] = pd.to_numeric(cleaned["close"], errors="coerce")
        cleaned = cleaned[cleaned["close"].notna() & (cleaned["close"] > 0)]
        if len(cleaned) < minimum_samples:
            raise ValueError("insufficient movement samples")
        return cleaned

    def profile(
        self,
        *,
        symbol: str,
        candles: pd.DataFrame,
        quote_volume_usd: float,
        spread_bps: float,
        nominal_price: float | None = None,
        timeframe_seconds: float = 60.0,
    ) -> MovementProfile:
        frame = self._validate_frame(candles, self.minimum_samples)
        quote_volume = self._finite_nonnegative(quote_volume_usd, "quote_volume_usd")
        spread = self._finite_nonnegative(spread_bps, "spread_bps")
        timeframe_seconds = float(timeframe_seconds)
        if not math.isfinite(timeframe_seconds) or timeframe_seconds <= 0:
            raise ValueError("timeframe_seconds must be positive and finite")

        closes = frame["close"].astype(float)
        returns_bps = closes.pct_change().dropna() * 10_000.0
        returns_bps = returns_bps[returns_bps.map(math.isfinite)]
        if len(returns_bps) < self.minimum_samples - 1:
            raise ValueError("insufficient finite movement samples")
        absolute = returns_bps.abs()
        price = float(nominal_price if nominal_price is not None else closes.iloc[-1])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("nominal_price must be positive and finite")

        q50 = float(absolute.quantile(0.50))
        q75 = float(absolute.quantile(0.75))
        q90 = float(absolute.quantile(0.90))
        realized_volatility = float(returns_bps.std(ddof=0))
        minutes_per_observation = timeframe_seconds / 60.0
        duration_minutes = max(minutes_per_observation, len(returns_bps) * minutes_per_observation)
        qualifying_moves = int((absolute >= self.movement_threshold_bps).sum())
        movement_frequency = qualifying_moves / duration_minutes

        signs = returns_bps.map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        nonzero_pairs = [
            (int(previous), int(current))
            for previous, current in zip(signs.iloc[:-1], signs.iloc[1:])
            if previous != 0 and current != 0
        ]
        persistence = (
            sum(1 for previous, current in nonzero_pairs if previous == current) / len(nonzero_pairs)
            if nonzero_pairs
            else 0.0
        )
        reversal = 1.0 - persistence if nonzero_pairs else 0.0

        # Log-volume scaling prevents one enormous venue from dominating while
        # remaining monotonic in actual executable quote liquidity.
        liquidity = 0.0
        if quote_volume > 0:
            liquidity = min(
                1.0,
                max(0.0, math.log10(1.0 + quote_volume) / math.log10(1.0 + self.reference_quote_volume_usd)),
            )
        spread_quality = max(0.0, 1.0 - spread / self.maximum_acceptable_spread_bps)
        fill_probability = max(0.0, min(1.0, 0.65 * liquidity + 0.35 * spread_quality))
        expected_capture = q75 * self.capture_efficiency

        result = MovementProfile(
            symbol=str(symbol).upper(),
            samples=len(returns_bps),
            nominal_price=price,
            median_abs_move_bps=q50,
            q75_abs_move_bps=q75,
            q90_abs_move_bps=q90,
            realized_volatility_bps=realized_volatility,
            movement_frequency_per_minute=movement_frequency,
            directional_persistence=persistence,
            reversal_tendency=reversal,
            liquidity_score=liquidity,
            fill_probability=fill_probability,
            spread_bps=spread,
            expected_capture_bps=expected_capture,
            capture_efficiency=self.capture_efficiency,
        )
        self.profiles += 1
        return result

    def to_opportunity_snapshot(
        self,
        profile: MovementProfile,
        *,
        fee_bps: float,
        slippage_bps: float,
        adverse_selection_bps: float = 0.0,
    ) -> OpportunitySnapshot:
        persistence_quality = max(profile.directional_persistence, profile.reversal_tendency)
        return OpportunitySnapshot(
            symbol=profile.symbol,
            nominal_price=profile.nominal_price,
            movement_frequency_per_minute=profile.movement_frequency_per_minute,
            expected_capture_bps=profile.expected_capture_bps,
            liquidity_score=profile.liquidity_score,
            fill_probability=profile.fill_probability,
            persistence_score=persistence_quality,
            spread_bps=profile.spread_bps,
            fee_bps=self._finite_nonnegative(fee_bps, "fee_bps"),
            slippage_bps=self._finite_nonnegative(slippage_bps, "slippage_bps"),
            adverse_selection_bps=self._finite_nonnegative(adverse_selection_bps, "adverse_selection_bps"),
            source=profile.source,
        )

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "profiles": self.profiles,
            "rejected": self.rejected,
            "minimum_samples": self.minimum_samples,
            "capture_efficiency": self.capture_efficiency,
            "movement_threshold_bps": self.movement_threshold_bps,
            "nominal_price_is_selection_factor": False,
            "predictive_claim": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
