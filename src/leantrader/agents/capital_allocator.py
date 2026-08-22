from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .opportunity_radar import OpportunityScore
from .shared_position_graph import PositionCoordinator


@dataclass(frozen=True)
class CapitalStage:
    name: str
    minimum_equity: float
    max_tranche_fraction: float
    max_concurrent_agents: int


DEFAULT_STAGES = (
    CapitalStage("micro", 0.0, 0.04, 8),
    CapitalStage("growth", 200.0, 0.03, 12),
    CapitalStage("expansion", 1_000.0, 0.02, 20),
    CapitalStage("scale", 10_000.0, 0.01, 32),
)


class SwarmCapitalAllocator:
    """Stage-aware capital allocation bounded by upstream deployable capital.

    This allocator never creates risk budget. It can only divide an upstream
    paper risk/deployable budget among independently qualified swarm agents.
    """

    VERSION = "1.0"

    def __init__(
        self,
        coordinator: PositionCoordinator,
        *,
        stages: tuple[CapitalStage, ...] = DEFAULT_STAGES,
        profit_reinvest_fraction: float = 0.50,
    ) -> None:
        if not stages:
            raise ValueError("at least one capital stage is required")
        if not 0.0 <= profit_reinvest_fraction <= 1.0:
            raise ValueError("profit_reinvest_fraction must be in [0, 1]")
        ordered = tuple(sorted(stages, key=lambda row: row.minimum_equity))
        if ordered[0].minimum_equity > 0:
            raise ValueError("capital stages must cover starting equity")
        self.coordinator = coordinator
        self.stages = ordered
        self.profit_reinvest_fraction = float(profit_reinvest_fraction)
        self.realized_pnl = 0.0
        self.recycled_profit = 0.0
        self.allocations = 0
        self.blocked = 0

    def stage_for(self, equity: float) -> CapitalStage:
        equity = float(equity)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("equity must be positive and finite")
        selected = self.stages[0]
        for stage in self.stages:
            if equity >= stage.minimum_equity:
                selected = stage
            else:
                break
        return selected

    def allocate(
        self,
        *,
        symbol: str,
        equity: float,
        requested_notional: float,
        upstream_remaining_deployable_notional: float,
        opportunity: OpportunityScore,
        confidence: float,
        evidence_multiplier: float = 1.0,
        active_agents: int = 0,
    ) -> dict[str, Any]:
        equity = float(equity)
        requested = max(0.0, float(requested_notional))
        upstream = max(0.0, float(upstream_remaining_deployable_notional))
        confidence = max(0.0, min(1.0, float(confidence)))
        evidence_multiplier = max(0.0, min(1.0, float(evidence_multiplier)))
        stage = self.stage_for(equity)
        reason = "allocated"

        if not opportunity.qualified:
            allocation, reason = 0.0, f"opportunity:{opportunity.reason}"
        elif active_agents >= stage.max_concurrent_agents:
            allocation, reason = 0.0, "stage_concurrency_cap"
        elif requested <= 0 or upstream <= 0:
            allocation, reason = 0.0, "no_deployable_capital"
        elif confidence <= 0 or evidence_multiplier <= 0:
            allocation, reason = 0.0, "no_qualified_confidence"
        else:
            capacity = self.coordinator.remaining_capacity(symbol=symbol, equity=equity)["available"]
            per_tranche_cap = equity * stage.max_tranche_fraction
            # The quality multiplier only reduces the requested budget; it can
            # never enlarge the upstream request or risk ceiling.
            quality_multiplier = confidence * evidence_multiplier
            allocation = min(requested, upstream, capacity, per_tranche_cap) * quality_multiplier
            if allocation <= 0:
                allocation, reason = 0.0, "coordinated_exposure_cap"

        if allocation > 0:
            self.allocations += 1
        else:
            self.blocked += 1
        return {
            "allocated_notional": allocation,
            "requested_notional": requested,
            "stage": stage.name,
            "stage_max_tranche_fraction": stage.max_tranche_fraction,
            "stage_max_concurrent_agents": stage.max_concurrent_agents,
            "opportunity_score": opportunity.score,
            "net_capture_bps": opportunity.net_capture_bps,
            "confidence": confidence,
            "evidence_multiplier": evidence_multiplier,
            "reason": reason,
            "uses_upstream_deployable_budget_only": True,
            "can_increase_upstream_risk": False,
            "martingale": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def record_realized_pnl(self, realized_pnl: float) -> dict[str, float]:
        value = float(realized_pnl)
        if not math.isfinite(value):
            raise ValueError("realized_pnl must be finite")
        self.realized_pnl += value
        if value > 0:
            self.recycled_profit += value * self.profit_reinvest_fraction
        elif value < 0:
            self.recycled_profit = max(0.0, self.recycled_profit + value)
        return {
            "realized_pnl": self.realized_pnl,
            "recycled_profit": self.recycled_profit,
        }

    def health(self, *, equity: float) -> dict[str, Any]:
        stage = self.stage_for(equity)
        return {
            "version": self.VERSION,
            "stage": asdict(stage),
            "allocations": self.allocations,
            "blocked": self.blocked,
            "realized_pnl": self.realized_pnl,
            "recycled_profit": self.recycled_profit,
            "profit_reinvest_fraction": self.profit_reinvest_fraction,
            "martingale": False,
            "can_increase_upstream_risk": False,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
