from __future__ import annotations

import numpy as np
import pandas as pd

from leantrader.production.active_research import ActiveResearchPlanner
from leantrader.production.adversarial_critic import AdversarialCritic
from leantrader.production.hypothesis_lab import HypothesisLab
from leantrader.production.intelligence_council import IntelligenceCouncil
from leantrader.production.market_world_model import MarketWorldModel
from leantrader.production.meta_cognition import MetaCognitiveSelfModel
from leantrader.production.tail_risk_sentinel import TailRiskSentinel


def frame(n: int = 320, *, last_volume: float = 1.0) -> pd.DataFrame:
    close = np.linspace(100.0, 108.0, n)
    volume = np.ones(n)
    volume[-1] = last_volume
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": volume,
        }
    )


def advanced(*, imbalance: float = -0.8, spread_bps: float = 4.0) -> dict:
    signals = [
        {"engine": "smart_scalping", "score": 0.5, "confidence": 0.7},
        {"engine": "technical_structure", "score": 0.6, "confidence": 0.8},
        {"engine": "spectral_harmonics", "score": -0.2, "confidence": 0.5},
        {"engine": "pattern_memory", "score": 0.1, "confidence": 0.3},
    ]
    return {
        "signals": signals,
        "swarm": {"score": 0.45, "confidence": 0.65},
        "liquidity": {"imbalance": imbalance, "spread_bps": spread_bps, "available": True},
        "news_blackout": False,
    }


def test_world_model_builds_persistent_senses_and_rare_scope(tmp_path):
    model = MarketWorldModel(tmp_path / "world.json")
    result = model.observe_symbol(
        "BTC/USDT",
        frame(last_volume=25.0),
        adaptive={"score": 0.7, "confidence": 0.8, "regime": "trend"},
        advanced=advanced(imbalance=-0.9),
        public_context={"available": True, "score": -0.6, "confidence": 0.5},
        timeframe_signals={"1m": 0.8, "15m": 0.7, "1h": -0.8, "1d": -0.7},
        timeframe_coverage=1.0,
    )
    assert result["execution_authority"] is False
    assert 0.0 <= result["senses"]["novelty"] <= 1.0
    assert result["senses"]["volume_shock"] > 0
    assert "liquidity_price_divergence" in result["latent_patterns"]
    assert result["unknowns"] == [] or isinstance(result["unknowns"], list)
    market = model.observe_market({"BTC/USDT": frame(), "ETH/USDT": frame()})
    assert market["symbols_modeled"] == 1
    assert model.health()["execution_authority"] is False


def test_self_model_learns_specialist_reliability_from_closed_outcomes(tmp_path):
    self_model = MetaCognitiveSelfModel(tmp_path / "self.json")
    metadata = {
        "symbol": "BTC/USDT",
        "component_scores": {"trend": 0.8, "momentum": 0.5},
        "weights": {"trend": 0.7, "momentum": 0.3},
        "advanced_shadow": {
            "signals": [{"engine": "technical_structure", "score": 0.7}],
            "swarm": {"score": 0.6},
        },
        "decision_route": {"combined_score": 0.6},
        "intelligence_council": {"consensus_score": 0.55},
        "adversarial_critic": {"adjusted_score": 0.4},
    }
    for _ in range(6):
        self_model.record_outcome(metadata, 0.01)
    trust = self_model.specialist_trust("advanced:technical_structure")
    assert trust["samples"] == 6
    assert trust["reliability"] > 0.5
    result = self_model.assess_symbol(
        symbol="BTC/USDT",
        world={
            "state_confidence": 0.7,
            "knowledge_state": "measured",
            "unknowns": [],
            "senses": {"novelty": 0.1, "model_disagreement": 0.1},
            "adaptive": {"score": 0.5, "confidence": 0.7},
            "swarm": {"score": 0.4, "confidence": 0.7},
        },
        cns={"signal_coherence": 0.7, "risk_pressure": 0.1},
        memory={"support": 0.7, "contextual_samples": 8, "weighted_net_return": 0.002},
        route={"router_allowed_pre_brain": True, "allowed": True},
        brain={"allow_entry": True, "reasons": []},
        strategy_evidence={"samples": 10, "ewma_net_return": 0.001},
        engine_health={"market_data": {"healthy": True}},
    )
    assert result["execution_authority"] is False
    assert result["can_modify_code"] is False
    assert "specialist_trust" in result


def test_council_and_critic_are_bounded_and_uncertainty_aware(tmp_path):
    council = IntelligenceCouncil(tmp_path / "council.json")
    self_state = {"meta_confidence": 0.7, "uncertainty": 0.2, "specialist_trust": {}}
    world = {
        "data_quality": 0.9,
        "knowledge_state": "out_of_distribution",
        "senses": {"rare_scope_score": 0.8, "novelty": 0.8},
    }
    decision = council.deliberate(
        symbol="BTC/USDT",
        adaptive={"score": 0.7, "confidence": 0.8},
        advanced=advanced(),
        world=world,
        self_model=self_state,
        memory={"support": 0.0, "weighted_net_return": 0.0},
        public_context={"available": True, "score": -0.5, "confidence": 0.5},
    )
    assert decision["execution_authority"] is False
    assert decision["rare_scope_research_candidate"] is True

    critic = AdversarialCritic(tmp_path / "critic.json")
    review = critic.review(
        symbol="BTC/USDT",
        council=decision,
        world={
            **world,
            "state_confidence": 0.2,
            "senses": {
                "novelty": 0.8,
                "liquidity_stress": 0.8,
                "volatility_shock": 0.8,
                "timeframe_fracture": 0.7,
            },
        },
        self_model={"contradictions": ["x"]},
        memory={"support": 0.1, "contextual_samples": 1},
        route={"allowed": True},
        brain={"reasons": []},
        public_context={"available": False},
    )
    assert review["adjusted_confidence"] <= decision["confidence"]
    assert review["risk_guidance_multiplier"] <= 1.0
    assert review["execution_authority"] is False


def test_hypothesis_lab_requires_falsifiable_forward_resolution(tmp_path):
    lab = HypothesisLab(tmp_path / "hyp.json", horizon_observations=2)
    world = {
        "regime": "compression",
        "price": 100.0,
        "latent_patterns": ["compression_with_participation_anomaly"],
        "senses": {"rare_scope_score": 0.8, "novelty": 0.7},
        "features": {"vol_short": 0.001},
    }
    first = lab.observe(
        symbol="BTC/USDT",
        world=world,
        council={"confidence": 0.7},
        critic={"confidence_haircut": 0.2, "falsification_questions": ["q"]},
    )
    assert first["generated"]
    lab.observe(
        symbol="BTC/USDT",
        world={**world, "price": 100.05, "regime_changed": False},
        council={"confidence": 0.7},
        critic={"confidence_haircut": 0.2},
    )
    third = lab.observe(
        symbol="BTC/USDT",
        world={**world, "price": 101.0, "regime_changed": True},
        council={"confidence": 0.7},
        critic={"confidence_haircut": 0.2},
    )
    assert third["resolved"]
    assert lab.health()["research_only"] is True


def test_active_research_knows_missing_data_adapters(tmp_path):
    planner = ActiveResearchPlanner(tmp_path / "research.json")
    result = planner.plan_symbol(
        symbol="BTC/USDT",
        world={
            "knowledge_state": "out_of_distribution",
            "timeframe_coverage": 1.0,
            "latent_patterns": ["volatility_liquidity_coupling"],
            "senses": {"rare_scope_score": 0.9},
            "unknowns": [],
        },
        self_model={"unknowns": [], "uncertainty": 0.8},
        council={"disagreement": 0.6},
        critic={"falsification_questions": []},
        hypotheses={"active_for_symbol": []},
        engine_health={
            "market_data": {"healthy": True},
            "memory_retention": {"healthy": True},
            "strategy_observatory": {"healthy": True},
        },
        public_context_health={"market_data_fresh": True, "news_fresh": True},
        arbitrage={"available": True},
    )
    assert "liquidations" in result["missing_adapters"]
    assert result["execution_authority"] is False
    backlog = planner.adapter_backlog()
    assert any(row["source"] == "liquidations" for row in backlog)


def test_tail_risk_sentinel_uses_compound_evidence_not_single_magic_signal(tmp_path):
    sentinel = TailRiskSentinel(tmp_path / "tail.json")
    result = sentinel.assess(
        symbol="BTC/USDT",
        world={
            "senses": {
                "price_shock": 0.9,
                "volatility_shock": 0.9,
                "liquidity_stress": 0.9,
                "novelty": 0.8,
                "model_disagreement": 0.5,
            }
        },
        market_world={"correlation_fracture": 0.4, "cross_sectional_dispersion": 0.06},
        advanced={"news_blackout": True},
        runtime_errors={},
    )
    assert result["state"] in {"severe", "extreme"}
    assert result["risk_guidance_multiplier"] <= 0.2
    assert result["can_halt_execution"] is False
    assert sentinel.health()["legacy_black_swan_code_loaded"] is False


def test_world_model_discovers_research_only_cross_market_lead_lag(tmp_path):
    rng = np.random.default_rng(7)
    leader_returns = rng.normal(0.0, 0.002, 100)
    follower_returns = np.concatenate([np.zeros(2), leader_returns[:-2]]) + rng.normal(0.0, 0.00015, 100)

    def from_returns(values: np.ndarray) -> pd.DataFrame:
        close = 100.0 * np.exp(np.cumsum(values))
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(close), 10.0),
            }
        )

    model = MarketWorldModel(tmp_path / "leadlag.json")
    market = model.observe_market(
        {
            "BTC/USDT": from_returns(leader_returns),
            "ETH/USDT": from_returns(follower_returns),
        }
    )
    candidates = market["lead_lag_research_candidates"]
    assert candidates
    strongest = candidates[0]
    assert strongest["leader"] == "BTC/USDT"
    assert strongest["follower"] == "ETH/USDT"
    assert strongest["lag_steps"] == 2
    assert market["relationship_discovery_is_not_trade_authority"] is True


def test_active_research_requests_falsification_for_lead_lag_candidate(tmp_path):
    planner = ActiveResearchPlanner(tmp_path / "leadlag_research.json")
    result = planner.plan_symbol(
        symbol="BTC/USDT",
        world={
            "knowledge_state": "measured",
            "timeframe_coverage": 1.0,
            "latent_patterns": [],
            "senses": {"rare_scope_score": 0.2},
            "unknowns": [],
        },
        self_model={"unknowns": [], "uncertainty": 0.2},
        council={"disagreement": 0.2},
        critic={"falsification_questions": []},
        hypotheses={"active_for_symbol": []},
        engine_health={
            "market_data": {"healthy": True},
            "memory_retention": {"healthy": True},
            "strategy_observatory": {"healthy": True},
        },
        public_context_health={"market_data_fresh": True, "news_fresh": True},
        arbitrage={"available": True},
        market_world={
            "lead_lag_research_candidates": [
                {
                    "leader": "BTC/USDT",
                    "follower": "ETH/USDT",
                    "lag_steps": 2,
                    "correlation": 0.8,
                    "contemporaneous_correlation": 0.1,
                    "incremental_strength": 0.7,
                }
            ]
        },
    )
    assert any("lead/lag" in task["question"] for task in result["tasks"])
    assert "rates_fx_cross_asset" in result["missing_adapters"]
    assert result["execution_authority"] is False


def test_self_model_deduplicates_low_value_direction_conflict_history(tmp_path):
    self_model = MetaCognitiveSelfModel(tmp_path / "self.json")
    common = dict(
        symbol="BTC/USDT",
        cns={"signal_coherence": 0.5, "risk_pressure": 0.1},
        memory={"support": 0.2, "contextual_samples": 2},
        route={"router_allowed_pre_brain": False, "allowed": False},
        brain={"allow_entry": False, "reasons": []},
        strategy_evidence={"authority": "costed_shadow_episode_v2", "samples": 0},
        engine_health={"market_data": {"healthy": True}},
    )
    # Tiny opposite signs are ensemble noise and must not become contradictions.
    low = self_model.assess_symbol(
        world={
            "state_confidence": 0.6,
            "knowledge_state": "measured",
            "unknowns": [],
            "senses": {"novelty": 0.1, "model_disagreement": 0.2},
            "adaptive": {"score": 0.05, "confidence": 0.8},
            "swarm": {"score": -0.06, "confidence": 0.8},
        },
        **common,
    )
    assert "adaptive_swarm_direction_conflict" not in low["contradictions"]

    world = {
        "state_confidence": 0.6,
        "knowledge_state": "measured",
        "unknowns": [],
        "senses": {"novelty": 0.1, "model_disagreement": 0.7},
        "adaptive": {"score": 0.65, "confidence": 0.8},
        "swarm": {"score": -0.55, "confidence": 0.75},
    }
    for _ in range(5):
        result = self_model.assess_symbol(world=world, **common)
        assert "adaptive_swarm_direction_conflict" in result["contradictions"]
    # Same ongoing conflict is summarized, not appended once per poll.
    assert len(self_model.state["contradiction_history"]) == 1
    assert self_model.state["contradiction_summary"]["adaptive_swarm_direction_conflict"]["occurrences"] == 5


def test_self_model_archives_legacy_contradiction_history_on_load(tmp_path):
    path = tmp_path / "self.json"
    path.write_text(
        __import__("json").dumps({
            "schema_version": 1,
            "assessments": 10,
            "closed_outcomes": 1,
            "system_observations": 1,
            "latest": {},
            "system": {},
            "specialist_trust": {},
            "outcome_history": [],
            "contradiction_history": [{"symbol": "BTC/USDT", "contradictions": ["x"]}],
        })
    )
    model = MetaCognitiveSelfModel(path)
    assert model.state["contradiction_history"] == []
    assert len(model.state["legacy_contradiction_history_v1"]) == 1
    assert model.state["contradiction_event_model"] == "episode_dedup_v2"
