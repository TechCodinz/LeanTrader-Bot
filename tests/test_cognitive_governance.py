from __future__ import annotations

from leantrader.production.brain import TradingBrain
from leantrader.production.cognitive_governance import CognitiveGovernanceBridge


def calm_inputs() -> dict:
    return {
        "world": {
            "state_confidence": 0.8,
            "data_quality": 0.9,
            "knowledge_state": "measured",
            "unknowns": [],
            "senses": {"novelty": 0.1},
        },
        "self_model": {
            "uncertainty": 0.15,
            "meta_confidence": 0.8,
            "contradictions": [],
        },
        "council": {
            "consensus_score": 0.45,
            "confidence": 0.7,
            "uncertainty": 0.2,
            "disagreement": 0.15,
        },
        "critic": {
            "adjusted_score": 0.35,
            "adjusted_confidence": 0.65,
            "confidence_haircut": 0.1,
            "risk_guidance_multiplier": 1.0,
            "objections": [],
        },
        "hypothesis": {"active_for_symbol": []},
        "tail_risk": {
            "state": "normal",
            "severity": 0.1,
            "risk_guidance_multiplier": 1.0,
        },
        "research_plan": {"missing_adapters": [], "degraded_sources": []},
        "memory": {"support": 0.7, "contextual_samples": 8, "weighted_net_return": 0.002},
        "public_context": {"available": True, "fresh": True, "confidence": 0.7},
        "sensor_context": {
            "derivatives": {"status": "available"},
            "flow_intelligence": {"status": "available"},
            "liquidations": {"status": "available"},
        },
    }


def test_cognitive_governance_preserves_calm_upstream_approval(tmp_path):
    bridge = CognitiveGovernanceBridge(tmp_path / "cognitive.json")
    result = bridge.evaluate(symbol="BTC/USDT", upstream_allowed=True, **calm_inputs())
    assert result["allow_entry"] is True
    assert result["risk_multiplier"] == 1.0
    assert result["confidence_multiplier"] == 1.0
    assert result["action"] == "allow"
    assert result["execution_authority"] is False
    assert result["can_increase_upstream_risk"] is False


def test_cognitive_governance_tail_risk_can_only_veto_new_risk(tmp_path):
    bridge = CognitiveGovernanceBridge(tmp_path / "cognitive.json")
    inputs = calm_inputs()
    inputs["tail_risk"] = {
        "state": "severe",
        "severity": 0.88,
        "risk_guidance_multiplier": 0.2,
    }
    result = bridge.evaluate(symbol="BTC/USDT", upstream_allowed=True, **inputs)
    assert result["allow_entry"] is False
    assert result["risk_multiplier"] == 0.0
    assert result["action"] == "veto"
    assert "tail_risk_severe" in result["reasons"]


def test_cognitive_governance_reduces_ood_uncertain_state_without_claiming_edge(tmp_path):
    bridge = CognitiveGovernanceBridge(tmp_path / "cognitive.json")
    inputs = calm_inputs()
    inputs["world"].update({"state_confidence": 0.2, "knowledge_state": "out_of_distribution"})
    inputs["world"]["senses"]["novelty"] = 0.92
    inputs["self_model"]["uncertainty"] = 0.8
    inputs["critic"].update({"confidence_haircut": 0.8, "risk_guidance_multiplier": 0.3})
    result = bridge.evaluate(symbol="BTC/USDT", upstream_allowed=True, **inputs)
    assert result["allow_entry"] is True
    assert 0.0 < result["risk_multiplier"] <= 0.3
    assert result["confidence_multiplier"] <= 0.45
    assert result["action"] == "reduce"
    assert result["evidence_authority"] == "bounded_cognitive_safety_v1"


def test_cognitive_governance_never_resurrects_rejected_upstream_entry(tmp_path):
    bridge = CognitiveGovernanceBridge(tmp_path / "cognitive.json")
    result = bridge.evaluate(symbol="BTC/USDT", upstream_allowed=False, **calm_inputs())
    assert result["allow_entry"] is False
    assert result["risk_multiplier"] == 0.0
    assert "upstream_not_allowed" in result["reasons"]


def test_brain_second_stage_review_never_increases_base_risk(tmp_path):
    brain = TradingBrain(tmp_path / "brain.json")
    base = {
        "symbol": "BTC/USDT",
        "allow_entry": True,
        "risk_multiplier": 0.4,
        "confidence_multiplier": 0.8,
        "reasons": [],
        "execution_authority": False,
    }
    reviewed = brain.apply_cognitive_governance(
        symbol="BTC/USDT",
        base_decision=base,
        governance={
            "allow_entry": True,
            "risk_multiplier": 0.9,
            "confidence_multiplier": 0.95,
            "reasons": [],
        },
    )
    assert reviewed["allow_entry"] is True
    assert reviewed["risk_multiplier"] == 0.4
    assert reviewed["confidence_multiplier"] == 0.8
    assert reviewed["can_increase_upstream_risk"] is False


def test_brain_second_stage_review_applies_cognitive_veto(tmp_path):
    brain = TradingBrain(tmp_path / "brain.json")
    base = {
        "symbol": "BTC/USDT",
        "allow_entry": True,
        "risk_multiplier": 0.7,
        "confidence_multiplier": 0.9,
        "reasons": [],
    }
    reviewed = brain.apply_cognitive_governance(
        symbol="BTC/USDT",
        base_decision=base,
        governance={
            "allow_entry": False,
            "risk_multiplier": 0.0,
            "confidence_multiplier": 0.4,
            "reasons": ["tail_risk_extreme"],
        },
    )
    assert reviewed["allow_entry"] is False
    assert reviewed["risk_multiplier"] == 0.0
    assert "cognitive:tail_risk_extreme" in reviewed["reasons"]
    assert brain.health()["cognitive_vetoes"] == 1
