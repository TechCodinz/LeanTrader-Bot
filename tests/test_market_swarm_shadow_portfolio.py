from __future__ import annotations

from pathlib import Path

from leantrader.agents.shared_position_graph import AgentRole, PositionCoordinator
from leantrader.agents.swarm_evidence import SwarmOutcomeJournal, build_v142_swarm_manifests
from leantrader.agents.swarm_shadow_portfolio import SwarmShadowPortfolio
from leantrader.production.prospective_validation import ProspectiveValidationLab


def test_shadow_portfolio_costs_profit_and_preserves_principal(tmp_path: Path):
    portfolio = SwarmShadowPortfolio(
        tmp_path / "portfolio.json",
        starting_equity=50.0,
        fee_bps=10.0,
        slippage_bps=5.0,
        principal_floor_fraction=0.70,
        profit_reinvest_fraction=0.50,
    )
    assert portfolio.remaining_deployable_notional({}) == 15.0
    row = portfolio.open_tranche(
        tranche_id="t1",
        agent_id="a1",
        role="scalp",
        timeframe="1m",
        symbol="AAA/USDT",
        side="long",
        reference_price=1.0,
        notional=2.0,
        confidence=0.8,
        expected_edge_bps=80.0,
        modeled_round_trip_cost_bps=30.0,
    )
    assert row["notional"] == 2.0
    preview = portfolio.preview_net_return("t1", reference_price=1.02)
    assert preview["net_pnl"] > 0.0
    episode = portfolio.close_tranche("t1", reference_price=1.02, reason="test")
    assert episode["net_return"] > 0.0
    assert episode["fee_bps_per_side"] == 10.0
    assert episode["slippage_bps_per_side"] == 5.0
    health = portfolio.health()
    assert health["realized_pnl"] > 0.0
    assert health["locked_profit"] > 0.0
    assert health["canonical_paper_ledger_mutation"] is False
    assert health["live_authority"] is False


def test_shadow_portfolio_persists_open_tranche(tmp_path: Path):
    path = tmp_path / "portfolio.json"
    first = SwarmShadowPortfolio(path, starting_equity=50.0, fee_bps=10.0, slippage_bps=5.0)
    first.open_tranche(
        tranche_id="persisted",
        agent_id="agent",
        role="momentum",
        timeframe="5m",
        symbol="BBB/USDT",
        side="long",
        reference_price=2.0,
        notional=1.0,
        confidence=0.8,
        expected_edge_bps=90.0,
        modeled_round_trip_cost_bps=30.0,
    )
    second = SwarmShadowPortfolio(path, starting_equity=50.0, fee_bps=10.0, slippage_bps=5.0)
    assert second.open_symbols() == {"BBB/USDT"}
    assert second.has_open_agent("agent") is True
    assert second.health()["open_tranches"] == 1


def test_position_coordinator_persists_agent_ownership(tmp_path: Path):
    path = tmp_path / "positions.json"
    first = PositionCoordinator(state_path=path, max_symbol_exposure_fraction=0.1, max_portfolio_exposure_fraction=0.2)
    tranche = first.attach_tranche(
        agent_id="agent-1",
        role=AgentRole.SCALP,
        timeframe="1m",
        symbol="AAA/USDT",
        side="long",
        entry_price=1.0,
        capital=2.0,
        confidence=0.8,
        expected_edge_bps=80.0,
        equity=50.0,
    )
    second = PositionCoordinator(state_path=path, max_symbol_exposure_fraction=0.1, max_portfolio_exposure_fraction=0.2)
    restored = second.open_tranches()
    assert len(restored) == 1
    assert restored[0].tranche_id == tranche.tranche_id
    assert restored[0].agent_id == "agent-1"


def test_swarm_manifests_register_before_future_outcomes(tmp_path: Path):
    lab = ProspectiveValidationLab(tmp_path / "prospective.json", minimum_samples=100, round_trip_cost_bps=30.0)
    manifests = build_v142_swarm_manifests(minimum_samples=100, round_trip_cost_bps=30.0)
    result = lab.observe_cycle(
        observatory_authority=lab.EVIDENCE_AUTHORITY,
        observed_round_trip_cost_bps=30.0,
        strategy_episodes=[],
        foundry_manifests=manifests,
        market_rows={},
    )
    assert len(result["experiments_registered"]) == len(manifests)
    assert all(row["automatic_promotion"] is False for row in manifests)
    assert all(row["research_protocol"]["untouched_holdout_required"] is True for row in manifests)
    assert all(row["research_protocol"]["partition_plan"]["untouched_holdout_samples"] >= 100 for row in manifests)


def test_closed_outcome_journal_is_acknowledge_once(tmp_path: Path):
    journal = SwarmOutcomeJournal(tmp_path / "outcomes.json")
    episode_id = journal.append(
        {
            "strategy": "swarm_scalp_1m",
            "symbol": "AAA/USDT",
            "regime": "long_1m",
            "opened_at": 100.0,
            "closed_at": 200.0,
            "net_return": 0.01,
        }
    )
    pending = journal.pending()
    assert len(pending) == 1
    assert pending[0]["episode_id"] == episode_id
    assert pending[0]["evidence_authority"] == "costed_shadow_episode_v2"
    assert journal.acknowledge([episode_id]) == 1
    assert journal.pending() == []
    assert journal.acknowledge([episode_id]) == 0
