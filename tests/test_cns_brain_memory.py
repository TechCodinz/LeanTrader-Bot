from __future__ import annotations

from pathlib import Path

from leantrader.production.brain import TradingBrain
from leantrader.production.capital_growth import CapitalGrowthGovernor
from leantrader.production.cns import CentralNervousSystem
from leantrader.production.memory_retention import MarketFingerprint, MemoryRetentionEngine


def test_memory_promotes_only_closed_outcomes(tmp_path: Path):
    memory = MemoryRetentionEngine(tmp_path / "memory.json", max_episodes=100)
    fp = MarketFingerprint(regime="trend", trend=0.6, momentum=0.4, ultra_score=0.5, ultra_confidence=0.8)
    memory.remember_decision("d1", symbol="BTC/USDT", strategy="smart_scalping", fingerprint=fp, confidence=0.8)
    assert memory.health()["retained_episodes"] == 0
    assert memory.health()["pending_working_memory"] == 1
    memory.close_decision("d1", net_return=-0.01)
    assert memory.health()["retained_episodes"] == 1
    assert memory.health()["closed_outcomes"] == 1
    evidence = memory.semantic_evidence(symbol="BTC/USDT", regime="trend", strategy="smart_scalping")
    assert evidence["samples"] == 1
    assert evidence["wins"] == 0


def test_memory_recalls_similar_positive_and_negative_experience(tmp_path: Path):
    memory = MemoryRetentionEngine(tmp_path / "memory.json", max_episodes=100)
    fp = MarketFingerprint(regime="trend", trend=0.7, momentum=0.5, spread_bps=3.0, ultra_score=0.6, ultra_confidence=0.8)
    memory.record_closed_observation(observation_id="a", symbol="BTC/USDT", strategy="smart_scalping", fingerprint=fp, confidence=0.8, net_return=0.01)
    memory.record_closed_observation(observation_id="b", symbol="BTC/USDT", strategy="smart_scalping", fingerprint=fp, confidence=0.7, net_return=-0.02)
    summary = memory.summarize(symbol="BTC/USDT", fingerprint=fp)
    assert summary["samples"] == 2
    assert summary["weighted_net_return"] < 0.01
    assert 0.0 <= summary["win_rate"] <= 1.0


def test_cns_fuses_without_execution_authority(tmp_path: Path):
    cns = CentralNervousSystem(tmp_path / "cns.json")
    packet = cns.integrate(
        symbol="BTC/USDT",
        adaptive={"score": 0.4, "confidence": 0.7},
        advanced={"swarm": {"score": 0.5, "confidence": 0.8}, "liquidity": {"spread_bps": 4.0, "imbalance": 0.2}},
        routed={"allowed": True, "score": 0.45},
        memory_summary={"support": 0.6, "weighted_net_return": 0.004},
    )
    assert packet["execution_authority"] is False
    assert 0.0 <= packet["signal_coherence"] <= 1.0
    assert 0.0 <= packet["risk_pressure"] <= 1.0


def test_brain_can_only_reduce_upstream_risk(tmp_path: Path):
    brain = TradingBrain(tmp_path / "brain.json", min_strategy_samples=50, negative_expectancy_floor=-0.001)
    result = brain.evaluate(
        symbol="BTC/USDT",
        cns={"signal_coherence": 0.8, "risk_pressure": 0.1, "action_bias": 0.6, "safety_blocks": []},
        memory={"support": 0.8, "weighted_net_return": 0.01},
        strategy_evidence={"authority": "costed_shadow_episode_v2", "samples": 100, "cumulative_net_return": 0.05},
        upstream_allowed=True,
    )
    assert result["allow_entry"] is True
    assert 0.0 <= result["risk_multiplier"] <= 1.0
    assert 0.0 <= result["confidence_multiplier"] <= 1.0
    assert result["execution_authority"] is False
    assert result["can_increase_upstream_risk"] is False


def test_brain_downsizes_sufficient_negative_evidence(tmp_path: Path):
    brain = TradingBrain(tmp_path / "brain.json", min_strategy_samples=50, negative_expectancy_floor=-0.001)
    result = brain.evaluate(
        symbol="BTC/USDT",
        cns={"signal_coherence": 0.7, "risk_pressure": 0.05, "action_bias": 0.3, "safety_blocks": []},
        memory={"support": 0.7, "weighted_net_return": -0.01},
        strategy_evidence={"authority": "costed_shadow_episode_v2", "samples": 128, "cumulative_net_return": -0.20},
        upstream_allowed=True,
    )
    assert result["risk_multiplier"] <= 0.20
    assert "negative_strategy_evidence" in result["reasons"]
    assert "negative_similar_memory" in result["reasons"]


def test_capital_growth_never_martingales_or_exceeds_upstream_budget(tmp_path: Path):
    governor = CapitalGrowthGovernor(tmp_path / "capital.json", starting_equity=50.0)
    healthy = governor.evaluate(equity=55.0, realized_pnl=5.0)
    drawdown = governor.evaluate(equity=45.0, realized_pnl=-5.0)
    assert healthy["risk_multiplier"] <= 1.0
    assert drawdown["risk_multiplier"] < healthy["risk_multiplier"]
    assert drawdown["martingale"] is False
    assert drawdown["can_increase_upstream_risk"] is False


def test_memory_close_is_idempotent_for_restart_replay(tmp_path: Path):
    memory = MemoryRetentionEngine(tmp_path / "memory.json", max_episodes=100)
    fp = MarketFingerprint(regime="range", trend=-0.1, momentum=0.2)
    memory.remember_decision(
        "replay-1",
        symbol="ETH/USDT",
        strategy="bounded_decision_router",
        fingerprint=fp,
        confidence=0.7,
    )
    first = memory.close_decision("replay-1", net_return=0.003)
    second = memory.close_decision("replay-1", net_return=0.003)
    assert first["decision_id"] == second["decision_id"]
    assert memory.health()["closed_outcomes"] == 1
    assert memory.health()["retained_episodes"] == 1


def test_brain_uses_per_sample_expectancy_not_cumulative_sum(tmp_path: Path):
    brain = TradingBrain(tmp_path / "brain.json")
    result = brain.evaluate(
        symbol="BTC/USDT",
        cns={"signal_coherence": 0.9, "risk_pressure": 0.0, "action_bias": 0.5, "safety_blocks": []},
        memory={"support": 0.0, "weighted_net_return": 0.0},
        strategy_evidence={"authority": "costed_shadow_episode_v2", "samples": 100, "cumulative_net_return": -0.05},
        upstream_allowed=True,
    )
    assert result["strategy_expectancy"] == -0.0005
    assert "negative_strategy_evidence" not in result["reasons"]


def test_brain_quarantines_persistent_negative_expectancy(tmp_path: Path):
    brain = TradingBrain(
        tmp_path / "brain.json",
        min_strategy_samples=50,
        negative_expectancy_floor=-0.001,
        quarantine_min_samples=100,
        quarantine_expectancy_floor=-0.004,
    )
    result = brain.evaluate(
        symbol="BTC/USDT",
        cns={"signal_coherence": 0.9, "risk_pressure": 0.0, "action_bias": 0.5, "safety_blocks": []},
        memory={"support": 0.0, "weighted_net_return": 0.0},
        strategy_evidence={"authority": "costed_shadow_episode_v2", "samples": 120, "average_net_return": -0.005},
        upstream_allowed=True,
    )
    assert result["allow_entry"] is False
    assert result["risk_multiplier"] == 0.0
    assert result["strategy_quarantined"] is True
    assert "strategy_quarantined" in result["reasons"]


def test_capital_growth_blocks_new_entries_at_protected_floor(tmp_path: Path):
    governor = CapitalGrowthGovernor(
        tmp_path / "capital.json",
        starting_equity=50.0,
        principal_floor_fraction=0.70,
        profit_reinvest_fraction=0.50,
    )
    governor.evaluate(equity=55.0, realized_pnl=5.0)
    floor_state = governor.evaluate(equity=37.5, realized_pnl=5.0)
    assert floor_state["protected_principal"] == 37.5
    assert floor_state["new_entries_allowed"] is False
    assert floor_state["risk_multiplier"] == 0.0
    assert floor_state["martingale"] is False


def test_capital_growth_caps_total_open_notional_to_deployable_budget(tmp_path: Path):
    governor = CapitalGrowthGovernor(
        tmp_path / "capital.json",
        starting_equity=50.0,
        principal_floor_fraction=0.70,
        profit_reinvest_fraction=0.50,
    )
    first = governor.evaluate(equity=50.0, realized_pnl=0.0, open_notional=10.0)
    assert first["deployable_equity"] == 15.0
    assert first["remaining_deployable_notional"] == 5.0
    assert first["new_entries_allowed"] is True

    exhausted = governor.evaluate(equity=50.0, realized_pnl=0.0, open_notional=15.0)
    assert exhausted["remaining_deployable_notional"] == 0.0
    assert exhausted["new_entries_allowed"] is False
    assert exhausted["risk_multiplier"] == 0.0


def test_brain_ignores_untrusted_legacy_shadow_evidence(tmp_path: Path):
    brain = TradingBrain(tmp_path / "brain.json", min_strategy_samples=10, quarantine_min_samples=20)
    result = brain.evaluate(
        symbol="BTC/USDT",
        cns={"signal_coherence": 0.9, "risk_pressure": 0.0, "action_bias": 0.5, "safety_blocks": []},
        memory={"support": 0.0, "weighted_net_return": 0.0},
        strategy_evidence={"samples": 500, "average_net_return": -0.50},
        upstream_allowed=True,
    )
    assert result["strategy_samples"] == 0
    assert result["strategy_quarantined"] is False
    assert "negative_strategy_evidence" not in result["reasons"]
    assert result["strategy_evidence_authority"] == "untrusted_or_legacy"


def test_brain_releases_v21_quarantines_on_migration(tmp_path: Path):
    path = tmp_path / "brain.json"
    path.write_text(
        __import__("json").dumps(
            {
                "version": "2.1",
                "evaluations": 10,
                "vetoes": 4,
                "downsizes": 2,
                "last": {},
                "quarantined_strategies": {
                    "bounded_decision_router:BTC/USDT": {
                        "strategy": "bounded_decision_router:BTC/USDT",
                        "samples": 100,
                        "expectancy": -0.005,
                    }
                },
            }
        )
    )
    brain = TradingBrain(path)
    assert brain.quarantined_strategies == {}
    assert "bounded_decision_router:BTC/USDT" in brain.legacy_quarantined_strategies
