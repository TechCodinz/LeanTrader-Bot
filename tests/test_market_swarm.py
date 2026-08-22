from __future__ import annotations

import pytest

from leantrader.agents.capital_allocator import SwarmCapitalAllocator
from leantrader.agents.opportunity_radar import MicroOpportunityRadar, OpportunitySnapshot
from leantrader.agents.shared_position_graph import AgentRole, PositionCoordinator, TrancheState
from leantrader.agents.swarm_orchestrator import MarketSwarmOrchestrator


def _snapshot(
    symbol: str,
    *,
    price: float,
    frequency: float = 4.0,
    capture_bps: float = 75.0,
    liquidity: float = 0.90,
    fill: float = 0.95,
    persistence: float = 0.80,
    spread_bps: float = 4.0,
    fee_bps: float = 8.0,
    slippage_bps: float = 5.0,
    adverse_bps: float = 2.0,
) -> OpportunitySnapshot:
    return OpportunitySnapshot(
        symbol=symbol,
        nominal_price=price,
        movement_frequency_per_minute=frequency,
        expected_capture_bps=capture_bps,
        liquidity_score=liquidity,
        fill_probability=fill,
        persistence_score=persistence,
        spread_bps=spread_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        adverse_selection_bps=adverse_bps,
    )


def test_nominal_price_is_not_an_opportunity_advantage() -> None:
    radar = MicroOpportunityRadar()
    tiny = radar.score(_snapshot("TINY/USDT", price=0.0000042))
    expensive = radar.score(_snapshot("HIGH/USDT", price=42_000.0))
    assert tiny.score == pytest.approx(expensive.score)
    assert tiny.net_capture_bps == pytest.approx(expensive.net_capture_bps)
    assert radar.health()["nominal_price_is_selection_factor"] is False


def test_fast_liquid_market_wins_by_percentage_economics_not_price() -> None:
    radar = MicroOpportunityRadar()
    ranked = radar.rank(
        [
            _snapshot("FAST/USDT", price=0.000008, frequency=8.0, capture_bps=90.0),
            _snapshot("SLOW/USDT", price=70_000.0, frequency=0.4, capture_bps=48.0),
        ]
    )
    assert ranked[0].symbol == "FAST/USDT"
    assert ranked[0].qualified is True
    assert ranked[0].score > ranked[1].score


def test_cost_model_cannot_be_loosened_below_thirty_bps() -> None:
    with pytest.raises(ValueError, match="30 bps"):
        MicroOpportunityRadar(minimum_modeled_round_trip_cost_bps=29.99)


def test_scalp_can_close_while_higher_timeframe_tranche_remains_open() -> None:
    swarm = MarketSwarmOrchestrator()
    opportunity = swarm.scan([_snapshot("MOVE/USDT", price=0.001)])[0]
    scalp = swarm.spawn_agent(role=AgentRole.SCALP, timeframe="micro", symbol=opportunity.symbol)
    trend = swarm.spawn_agent(role=AgentRole.TREND, timeframe="1h", symbol=opportunity.symbol)

    scalp_join = swarm.consider_join(
        agent_id=scalp.agent_id,
        opportunity=opportunity,
        side="long",
        entry_price=0.001,
        requested_notional=5.0,
        equity=50.0,
        upstream_remaining_deployable_notional=15.0,
        confidence=0.90,
        expected_edge_bps=80.0,
        independently_qualified=True,
    )
    trend_join = swarm.consider_join(
        agent_id=trend.agent_id,
        opportunity=opportunity,
        side="long",
        entry_price=0.001,
        requested_notional=5.0,
        equity=50.0,
        upstream_remaining_deployable_notional=15.0,
        confidence=0.80,
        expected_edge_bps=70.0,
        independently_qualified=True,
    )
    assert scalp_join["allowed"] is True
    assert trend_join["allowed"] is True

    closed = swarm.close_agent_tranche(
        agent_id=scalp.agent_id,
        tranche_id=scalp_join["tranche_id"],
        exit_price=0.0011,
    )
    assert closed["realized_pnl"] > 0
    position = swarm.coordinator.positions[("MOVE/USDT", "long")]
    assert position.tranches[scalp_join["tranche_id"]].state == TrancheState.CLOSED
    assert position.tranches[trend_join["tranche_id"]].state == TrancheState.OPEN
    assert position.open_notional > 0
    assert swarm.allocator.recycled_profit > 0


def test_timeframe_agent_must_independently_qualify_before_joining() -> None:
    swarm = MarketSwarmOrchestrator()
    opportunity = swarm.scan([_snapshot("MOVE/USDT", price=2.0)])[0]
    trend = swarm.spawn_agent(role=AgentRole.TREND, timeframe="4h", symbol=opportunity.symbol)
    decision = swarm.consider_join(
        agent_id=trend.agent_id,
        opportunity=opportunity,
        side="long",
        entry_price=2.0,
        requested_notional=5.0,
        equity=50.0,
        upstream_remaining_deployable_notional=15.0,
        confidence=0.95,
        expected_edge_bps=90.0,
        independently_qualified=False,
    )
    assert decision["allowed"] is False
    assert decision["reason"] == "agent_timeframe_not_independently_qualified"
    assert swarm.coordinator.total_open_notional == 0


def test_agent_cannot_close_another_agents_tranche() -> None:
    swarm = MarketSwarmOrchestrator()
    opportunity = swarm.scan([_snapshot("MOVE/USDT", price=1.0)])[0]
    scalp = swarm.spawn_agent(role=AgentRole.SCALP, timeframe="1m", symbol=opportunity.symbol)
    trend = swarm.spawn_agent(role=AgentRole.TREND, timeframe="1h", symbol=opportunity.symbol)
    joined = swarm.consider_join(
        agent_id=trend.agent_id,
        opportunity=opportunity,
        side="long",
        entry_price=1.0,
        requested_notional=2.0,
        equity=50.0,
        upstream_remaining_deployable_notional=15.0,
        confidence=1.0,
        expected_edge_bps=80.0,
        independently_qualified=True,
    )
    with pytest.raises(PermissionError):
        swarm.close_agent_tranche(
            agent_id=scalp.agent_id,
            tranche_id=joined["tranche_id"],
            exit_price=1.1,
        )
    position = swarm.coordinator.positions[("MOVE/USDT", "long")]
    assert position.tranches[joined["tranche_id"]].state == TrancheState.OPEN


def test_coordinated_symbol_exposure_caps_collective_agent_risk() -> None:
    coordinator = PositionCoordinator(
        max_symbol_exposure_fraction=0.05,
        max_portfolio_exposure_fraction=0.50,
    )
    allocator = SwarmCapitalAllocator(coordinator)
    swarm = MarketSwarmOrchestrator(coordinator=coordinator, allocator=allocator)
    opportunity = swarm.scan([_snapshot("CAP/USDT", price=1.0)])[0]
    agents = [
        swarm.spawn_agent(role=AgentRole.SCALP, timeframe=f"{index}m", symbol=opportunity.symbol)
        for index in range(1, 4)
    ]
    decisions = [
        swarm.consider_join(
            agent_id=agent.agent_id,
            opportunity=opportunity,
            side="long",
            entry_price=1.0,
            requested_notional=10.0,
            equity=50.0,
            upstream_remaining_deployable_notional=20.0,
            confidence=1.0,
            expected_edge_bps=90.0,
            independently_qualified=True,
        )
        for agent in agents
    ]
    assert decisions[0]["allocated_notional"] == pytest.approx(2.0)
    assert decisions[1]["allocated_notional"] == pytest.approx(0.5)
    assert decisions[2]["allowed"] is False
    assert coordinator.symbol_open_notional("CAP/USDT") == pytest.approx(2.5)


def test_swarm_never_grants_testnet_live_or_automatic_promotion() -> None:
    swarm = MarketSwarmOrchestrator()
    health = swarm.health(equity=50.0)
    assert health["automatic_promotion"] is False
    assert health["execution_authority"] is False
    assert health["testnet_authority"] is False
    assert health["live_authority"] is False
    assert health["capital_allocator"]["can_increase_upstream_risk"] is False
    assert health["radar"]["minimum_modeled_round_trip_cost_bps"] >= 30.0
