from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Iterable

from .capital_allocator import SwarmCapitalAllocator
from .opportunity_radar import MicroOpportunityRadar, OpportunityScore, OpportunitySnapshot
from .shared_position_graph import AgentRole, PositionCoordinator, TrancheState


@dataclass
class SwarmAgent:
    agent_id: str
    role: AgentRole
    timeframe: str
    symbol: str
    spawned_at: float = field(default_factory=time.time)
    active: bool = True
    decisions: int = 0
    accepted: int = 0


class MarketSwarmOrchestrator:
    """Paper-only multi-agent coordination across markets and timeframes.

    Fast agents may open the first tranche, while independently-qualified
    longer-timeframe agents can join the same shared symbol position. Each agent
    owns and can close only its own tranche. The global coordinator retains the
    aggregate exposure view.
    """

    VERSION = "1.0"

    DEFAULT_SPECIALISTS: tuple[tuple[AgentRole, str], ...] = (
        (AgentRole.SCALP, "micro"),
        (AgentRole.SCALP, "1m"),
        (AgentRole.MOMENTUM, "5m"),
        (AgentRole.MOMENTUM, "15m"),
        (AgentRole.TREND, "1h"),
        (AgentRole.TREND, "4h"),
        (AgentRole.REVERSAL, "5m"),
        (AgentRole.ARBITRAGE, "cross-venue"),
    )

    def __init__(
        self,
        *,
        radar: MicroOpportunityRadar | None = None,
        coordinator: PositionCoordinator | None = None,
        allocator: SwarmCapitalAllocator | None = None,
        minimum_agent_confidence: float = 0.50,
    ) -> None:
        self.radar = radar or MicroOpportunityRadar()
        self.coordinator = coordinator or PositionCoordinator()
        self.allocator = allocator or SwarmCapitalAllocator(self.coordinator)
        if self.allocator.coordinator is not self.coordinator:
            raise ValueError("allocator and swarm must share the same PositionCoordinator")
        if not 0.0 <= minimum_agent_confidence <= 1.0:
            raise ValueError("minimum_agent_confidence must be in [0, 1]")
        self.minimum_agent_confidence = float(minimum_agent_confidence)
        self.agents: dict[str, SwarmAgent] = {}
        self._agent_sequence = 0
        self.decisions = 0
        self.accepted = 0
        self.blocked: dict[str, int] = {}

    def spawn_agent(self, *, role: AgentRole, timeframe: str, symbol: str) -> SwarmAgent:
        self._agent_sequence += 1
        symbol = str(symbol).upper()
        role = AgentRole(role)
        agent_id = f"{role.value}:{timeframe}:{symbol}:{self._agent_sequence}"
        agent = SwarmAgent(agent_id=agent_id, role=role, timeframe=str(timeframe), symbol=symbol)
        self.agents[agent_id] = agent
        return agent

    def activate_specialists(
        self,
        opportunity: OpportunityScore,
        *,
        specialists: Iterable[tuple[AgentRole, str]] | None = None,
    ) -> list[SwarmAgent]:
        if not opportunity.qualified:
            return []
        return [
            self.spawn_agent(role=role, timeframe=timeframe, symbol=opportunity.symbol)
            for role, timeframe in (specialists or self.DEFAULT_SPECIALISTS)
        ]

    def scan(self, snapshots: Iterable[OpportunitySnapshot]) -> list[OpportunityScore]:
        return self.radar.rank(snapshots)

    @property
    def active_agents(self) -> int:
        return sum(1 for agent in self.agents.values() if agent.active)

    @property
    def capital_active_agents(self) -> int:
        return sum(1 for agent in self.agents.values() if agent.active and agent.accepted > 0)

    def consider_join(
        self,
        *,
        agent_id: str,
        opportunity: OpportunityScore,
        side: str,
        entry_price: float,
        requested_notional: float,
        equity: float,
        upstream_remaining_deployable_notional: float,
        confidence: float,
        expected_edge_bps: float,
        independently_qualified: bool,
        evidence_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        agent = self.agents.get(agent_id)
        if agent is None or not agent.active:
            raise KeyError(f"unknown or inactive agent: {agent_id}")
        if agent.symbol != opportunity.symbol:
            raise ValueError("agent cannot join an opportunity for another symbol")
        confidence = float(confidence)
        expected_edge_bps = float(expected_edge_bps)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not math.isfinite(expected_edge_bps):
            raise ValueError("expected_edge_bps must be finite")

        self.decisions += 1
        agent.decisions += 1
        reason = "approved"
        if not independently_qualified:
            reason = "agent_timeframe_not_independently_qualified"
        elif confidence < self.minimum_agent_confidence:
            reason = "agent_confidence_below_threshold"
        elif expected_edge_bps <= opportunity.modeled_round_trip_cost_bps:
            reason = "agent_edge_does_not_clear_modeled_cost"
        elif not opportunity.qualified:
            reason = f"opportunity:{opportunity.reason}"

        if reason != "approved":
            self.blocked[reason] = self.blocked.get(reason, 0) + 1
            return self._decision_payload(agent, opportunity, False, reason, 0.0, None)

        allocation = self.allocator.allocate(
            symbol=agent.symbol,
            equity=equity,
            requested_notional=requested_notional,
            upstream_remaining_deployable_notional=upstream_remaining_deployable_notional,
            opportunity=opportunity,
            confidence=confidence,
            evidence_multiplier=evidence_multiplier,
            active_agents=self.capital_active_agents,
        )
        allocated = float(allocation["allocated_notional"])
        if allocated <= 0:
            reason = str(allocation["reason"])
            self.blocked[reason] = self.blocked.get(reason, 0) + 1
            return self._decision_payload(agent, opportunity, False, reason, 0.0, None)

        tranche = self.coordinator.attach_tranche(
            agent_id=agent.agent_id,
            role=agent.role,
            timeframe=agent.timeframe,
            symbol=agent.symbol,
            side=side,
            entry_price=entry_price,
            capital=allocated,
            confidence=confidence,
            expected_edge_bps=expected_edge_bps,
            equity=equity,
        )
        agent.accepted += 1
        self.accepted += 1
        return self._decision_payload(agent, opportunity, True, "approved", allocated, tranche.tranche_id)

    def close_agent_tranche(self, *, agent_id: str, tranche_id: str, exit_price: float) -> dict[str, Any]:
        agent = self.agents.get(agent_id)
        if agent is None:
            raise KeyError(f"unknown agent: {agent_id}")
        owned = None
        for position in self.coordinator.positions.values():
            candidate = position.tranches.get(tranche_id)
            if candidate is not None:
                owned = candidate
                break
        if owned is None:
            raise KeyError(f"unknown tranche_id: {tranche_id}")
        if owned.agent_id != agent_id:
            raise PermissionError("an agent cannot close another agent's tranche")
        if owned.state != TrancheState.OPEN:
            raise ValueError("tranche is already closed")
        tranche = self.coordinator.close_tranche(tranche_id, exit_price=exit_price)
        recycle = self.allocator.record_realized_pnl(tranche.realized_pnl)
        return {
            "tranche_id": tranche.tranche_id,
            "agent_id": agent_id,
            "symbol": tranche.symbol,
            "role": tranche.role.value,
            "timeframe": tranche.timeframe,
            "realized_pnl": tranche.realized_pnl,
            "recycled_profit": recycle["recycled_profit"],
            "other_tranches_remain_independent": True,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def retire_agent(self, agent_id: str) -> None:
        agent = self.agents.get(agent_id)
        if agent is None:
            raise KeyError(f"unknown agent: {agent_id}")
        agent.active = False

    def _decision_payload(
        self,
        agent: SwarmAgent,
        opportunity: OpportunityScore,
        allowed: bool,
        reason: str,
        allocated_notional: float,
        tranche_id: str | None,
    ) -> dict[str, Any]:
        return {
            "allowed": bool(allowed),
            "reason": reason,
            "agent_id": agent.agent_id,
            "role": agent.role.value,
            "timeframe": agent.timeframe,
            "symbol": agent.symbol,
            "opportunity_score": opportunity.score,
            "net_capture_bps": opportunity.net_capture_bps,
            "allocated_notional": allocated_notional,
            "tranche_id": tranche_id,
            "independent_timeframe_authority": True,
            "shared_position_coordination": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def health(self, *, equity: float) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "active_agents": self.active_agents,
            "capital_active_agents": self.capital_active_agents,
            "agents": [{**asdict(agent), "role": agent.role.value} for agent in self.agents.values()],
            "decisions": self.decisions,
            "accepted": self.accepted,
            "blocked": dict(self.blocked),
            "radar": self.radar.health(),
            "capital_allocator": self.allocator.health(equity=equity),
            "positions": self.coordinator.snapshot(),
            "multi_timeframe_shared_positions": True,
            "profit_recycling": True,
            "whole_universe_compatible": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
