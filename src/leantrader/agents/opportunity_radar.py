from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


def _unit(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("opportunity inputs must be finite")
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class OpportunitySnapshot:
    symbol: str
    nominal_price: float
    movement_frequency_per_minute: float
    expected_capture_bps: float
    liquidity_score: float
    fill_probability: float
    persistence_score: float
    spread_bps: float
    fee_bps: float
    slippage_bps: float
    adverse_selection_bps: float = 0.0
    source: str = "market_sensor_fabric"


@dataclass(frozen=True)
class OpportunityScore:
    symbol: str
    score: float
    net_capture_bps: float
    modeled_round_trip_cost_bps: float
    movement_frequency_per_minute: float
    quality_multiplier: float
    nominal_price: float
    qualified: bool
    reason: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MicroOpportunityRadar:
    """Cost-aware ranking for continuously moving market opportunities.

    Nominal price is deliberately excluded from the score. A 0.00000x market is
    attractive only when its executable percentage economics are superior.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        minimum_modeled_round_trip_cost_bps: float = 30.0,
        minimum_net_capture_bps: float = 0.0,
        minimum_liquidity_score: float = 0.20,
        minimum_fill_probability: float = 0.50,
    ) -> None:
        floor = float(minimum_modeled_round_trip_cost_bps)
        if not math.isfinite(floor) or floor < 30.0:
            raise ValueError("modeled round-trip cost floor cannot be below 30 bps")
        self.minimum_modeled_round_trip_cost_bps = floor
        self.minimum_net_capture_bps = float(minimum_net_capture_bps)
        self.minimum_liquidity_score = _unit(minimum_liquidity_score)
        self.minimum_fill_probability = _unit(minimum_fill_probability)
        self.scored = 0
        self.qualified = 0

    @staticmethod
    def _nonnegative_finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
        return value

    def score(self, snapshot: OpportunitySnapshot) -> OpportunityScore:
        price = float(snapshot.nominal_price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("nominal_price must be positive and finite")
        frequency = self._nonnegative_finite(
            snapshot.movement_frequency_per_minute, "movement_frequency_per_minute"
        )
        expected_capture = self._nonnegative_finite(snapshot.expected_capture_bps, "expected_capture_bps")
        spread = self._nonnegative_finite(snapshot.spread_bps, "spread_bps")
        fees = self._nonnegative_finite(snapshot.fee_bps, "fee_bps")
        slippage = self._nonnegative_finite(snapshot.slippage_bps, "slippage_bps")
        adverse = self._nonnegative_finite(snapshot.adverse_selection_bps, "adverse_selection_bps")
        liquidity = _unit(snapshot.liquidity_score)
        fill = _unit(snapshot.fill_probability)
        persistence = _unit(snapshot.persistence_score)

        observed_cost = spread + fees + slippage + adverse
        modeled_cost = max(self.minimum_modeled_round_trip_cost_bps, observed_cost)
        net_capture = expected_capture - modeled_cost
        quality = liquidity * fill * persistence
        # Profit velocity proxy: executable net edge multiplied by how often the
        # market presents the movement and by execution-quality probability.
        velocity = max(0.0, net_capture) * frequency * quality

        reason = "qualified"
        qualified = True
        if liquidity < self.minimum_liquidity_score:
            qualified, reason = False, "insufficient_liquidity"
        elif fill < self.minimum_fill_probability:
            qualified, reason = False, "insufficient_fill_probability"
        elif net_capture <= self.minimum_net_capture_bps:
            qualified, reason = False, "non_positive_net_capture_after_costs"
        elif velocity <= 0:
            qualified, reason = False, "no_executable_profit_velocity"

        self.scored += 1
        if qualified:
            self.qualified += 1
        return OpportunityScore(
            symbol=str(snapshot.symbol).upper(),
            score=velocity if qualified else 0.0,
            net_capture_bps=net_capture,
            modeled_round_trip_cost_bps=modeled_cost,
            movement_frequency_per_minute=frequency,
            quality_multiplier=quality,
            nominal_price=price,
            qualified=qualified,
            reason=reason,
            source=str(snapshot.source),
        )

    def rank(self, snapshots: Iterable[OpportunitySnapshot]) -> list[OpportunityScore]:
        scores = [self.score(row) for row in snapshots]
        return sorted(
            scores,
            key=lambda row: (
                row.qualified,
                row.score,
                row.net_capture_bps,
                row.movement_frequency_per_minute,
            ),
            reverse=True,
        )

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "scored": self.scored,
            "qualified": self.qualified,
            "minimum_modeled_round_trip_cost_bps": self.minimum_modeled_round_trip_cost_bps,
            "nominal_price_is_selection_factor": False,
            "objective": "net_profit_velocity_after_costs",
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
