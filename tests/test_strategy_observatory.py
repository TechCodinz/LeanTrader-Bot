from __future__ import annotations

import json

import pytest

from leantrader.production.strategy_observatory import StrategyObservatory


def test_directional_skill_is_gross_and_costs_apply_only_when_episode_closes(tmp_path):
    path = tmp_path / "observatory.json"
    observer = StrategyObservatory(path, round_trip_cost_bps=10)

    first = observer.observe(
        "BTC/USDT",
        100.0,
        [{"engine": "smart_scalping", "score": 0.8}],
        {},
    )
    assert first["outcomes_recorded"] == 0

    # Same direction: diagnostic gross skill updates, but no new round-trip cost.
    second = observer.observe(
        "BTC/USDT",
        102.0,
        [{"engine": "smart_scalping", "score": 0.2}],
        {},
    )
    assert second["directional_outcomes_recorded"] == 1
    assert second["episodes_closed"] == 0
    assert observer.evidence("engine:smart_scalping", "BTC/USDT")["samples"] == 0

    # Direction flips: one held shadow episode closes and pays cost once.
    third = observer.observe(
        "BTC/USDT",
        103.0,
        [{"engine": "smart_scalping", "score": -0.4}],
        {},
    )
    assert third["episodes_closed"] == 1
    evidence = observer.evidence("engine:smart_scalping", "BTC/USDT")
    assert evidence["authority"] == "costed_shadow_episode_v2"
    assert evidence["samples"] == 1
    assert evidence["average_net_return"] == pytest.approx(0.029)

    health = observer.health()
    assert health["evidence_model"] == "episode_costed_v2"
    assert health["cost_application"] == "once_per_closed_shadow_episode"
    assert health["directional_strategies_measured"] == 1


def test_symbol_scoped_episode_evidence_prevents_cross_market_quarantine(tmp_path):
    path = tmp_path / "observatory.json"
    observer = StrategyObservatory(path, round_trip_cost_bps=10)

    observer.observe("BTC/USDT", 100.0, [{"engine": "bounded_decision_router", "score": 0.8}], {})
    observer.observe("BTC/USDT", 102.0, [{"engine": "bounded_decision_router", "score": -0.8}], {})
    observer.observe("ETH/USDT", 100.0, [{"engine": "bounded_decision_router", "score": -0.8}], {})
    observer.observe("ETH/USDT", 102.0, [{"engine": "bounded_decision_router", "score": 0.8}], {})

    btc = observer.evidence("engine:bounded_decision_router", "BTC/USDT")
    eth = observer.evidence("engine:bounded_decision_router", "ETH/USDT")
    global_evidence = observer.evidence("engine:bounded_decision_router")

    assert btc["scope"] == "symbol"
    assert btc["samples"] == 1
    assert btc["average_net_return"] == pytest.approx(0.019)
    assert eth["samples"] == 1
    assert eth["average_net_return"] == pytest.approx(-0.021)
    assert global_evidence["samples"] == 2
    assert global_evidence["average_net_return"] == pytest.approx(-0.001)


def test_v1_state_is_preserved_but_never_authoritative(tmp_path):
    path = tmp_path / "observatory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pending": {"BTC/USDT": {}},
                "strategies": {
                    "engine:bounded_decision_router": {
                        "samples": 100,
                        "cumulative_net_return": -0.50,
                    }
                },
            }
        )
    )
    observer = StrategyObservatory(path)
    assert observer.health()["legacy_v1_preserved"] is True
    evidence = observer.evidence("engine:bounded_decision_router", "BTC/USDT")
    assert evidence["samples"] == 0
    assert evidence["authority"] == "costed_shadow_episode_v2"


def _execution_funnel(**overrides):
    payload = {
        "symbols_evaluated": 1,
        "base_enter_candidates": 1,
        "router_approved_pre_brain": 1,
        "brain_approved": 1,
        "cognitive_governance_reviewed": 1,
        "cognitive_governance_vetoes": 0,
        "cognitive_governance_reductions": 0,
        "final_route_allowed": 1,
        "entry_attempts": 0,
        "entry_failures": 0,
        "buy_events": 0,
        "sell_events": 0,
        "entry_blocks": 0,
        "entry_block_reasons": {},
        "halted": False,
    }
    payload.update(overrides)
    return payload


def test_profitability_intelligence_tracks_net_equity_drawdown_and_funnel(tmp_path):
    path = tmp_path / "observatory.json"
    observer = StrategyObservatory(path, round_trip_cost_bps=30)

    observer.record_cycle(
        equity=100.0,
        cash=100.0,
        realized_pnl=0.0,
        starting_equity=100.0,
        open_positions=0,
        paper_trade_events=0,
        execution_funnel=_execution_funnel(),
        decisions={
            "BTC/USDT": {
                "allowed": True,
                "reason": "allowed",
                "regime": "trend",
            }
        },
    )
    observer.record_cycle(
        equity=110.0,
        cash=90.0,
        realized_pnl=0.0,
        starting_equity=100.0,
        open_positions=1,
        paper_trade_events=1,
        execution_funnel=_execution_funnel(
            entry_attempts=1,
            buy_events=1,
        ),
        decisions={
            "BTC/USDT": {
                "allowed": True,
                "reason": "allowed",
                "regime": "trend",
            }
        },
    )
    snapshot = observer.record_cycle(
        equity=99.0,
        cash=99.0,
        realized_pnl=-2.0,
        starting_equity=100.0,
        open_positions=0,
        paper_trade_events=2,
        execution_funnel=_execution_funnel(
            brain_approved=0,
            final_route_allowed=0,
            entry_blocks=1,
            entry_block_reasons={"brain:negative_expectancy": 1},
        ),
        decisions={
            "BTC/USDT": {
                "allowed": False,
                "reason": "brain:negative_expectancy",
                "regime": "range",
            }
        },
    )

    assert snapshot["authority"] == "prospective_paper_net_of_costs_v1"
    assert snapshot["state"] == "collecting_prospective_evidence"
    assert snapshot["cycles"] == 3
    assert snapshot["equity"] == pytest.approx(99.0)
    assert snapshot["net_pnl"] == pytest.approx(-1.0)
    assert snapshot["net_return"] == pytest.approx(-0.01)
    assert snapshot["realized_pnl"] == pytest.approx(-2.0)
    assert snapshot["unrealized_pnl"] == pytest.approx(1.0)
    assert snapshot["peak_equity"] == pytest.approx(110.0)
    assert snapshot["max_drawdown_pct"] == pytest.approx(0.10)
    assert snapshot["profit_factor"] == pytest.approx(1.0)
    assert snapshot["promotion_authority"] is False
    assert snapshot["testnet_authority"] is False
    assert snapshot["live_authority"] is False

    funnel = observer.decision_funnel_snapshot()
    assert funnel["cycles"] == 3
    assert funnel["totals"]["symbols_evaluated"] == 3
    assert funnel["totals"]["final_route_allowed"] == 2
    assert funnel["totals"]["buy_events"] == 1
    assert funnel["entry_block_reasons"]["brain:negative_expectancy"] == 1
    assert funnel["final_route_reasons"]["allowed"] == 2
    assert funnel["final_route_reasons"]["brain:negative_expectancy"] == 1
    assert funnel["regimes"]["trend"] == {"evaluated": 2, "allowed": 2}
    assert funnel["regimes"]["range"] == {"evaluated": 1, "allowed": 0}
    assert funnel["execution_authority"] is False

    reloaded = StrategyObservatory(path)
    assert reloaded.profitability_snapshot()["cycles"] == 3
    assert reloaded.health()["schema_version"] == 3


def test_v2_strategy_evidence_migrates_without_gaining_promotion_authority(tmp_path):
    path = tmp_path / "observatory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "pending": {},
                "episodes": {},
                "directional_strategies": {},
                "strategies": {
                    "engine:bounded_decision_router": {
                        "samples": 2,
                        "wins": 1,
                        "cumulative_net_return": 0.01,
                        "ewma_net_return": 0.002,
                        "symbols": {},
                    }
                },
            }
        )
    )

    observer = StrategyObservatory(path)
    evidence = observer.evidence("engine:bounded_decision_router")
    assert evidence["samples"] == 2
    assert evidence["cumulative_net_return"] == pytest.approx(0.01)
    assert observer.health()["schema_version"] == 3
    assert observer.profitability_snapshot()["promotion_authority"] is False


def test_profitability_cycle_rejects_non_finite_or_invalid_portfolio_inputs(tmp_path):
    observer = StrategyObservatory(tmp_path / "observatory.json")
    with pytest.raises(ValueError, match="equity must be finite"):
        observer.record_cycle(
            equity=float("nan"),
            cash=100.0,
            realized_pnl=0.0,
            starting_equity=100.0,
            open_positions=0,
            paper_trade_events=0,
            execution_funnel={},
            decisions={},
        )
    with pytest.raises(ValueError, match="starting_equity must be positive"):
        observer.record_cycle(
            equity=100.0,
            cash=100.0,
            realized_pnl=0.0,
            starting_equity=0.0,
            open_positions=0,
            paper_trade_events=0,
            execution_funnel={},
            decisions={},
        )
