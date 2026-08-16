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
