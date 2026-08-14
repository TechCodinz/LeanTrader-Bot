from __future__ import annotations

import math

import pytest

from leantrader.production.decision_router import BoundedDecisionRouter, MarketEvidenceGate


def advanced(score: float = 0.55, confidence: float = 0.70, blackout: bool = False):
    return {
        "swarm": {"score": score, "confidence": confidence},
        "news_blackout": blackout,
        "signals": [
            {"engine": "smart_scalping", "score": 0.5, "confidence": 0.7},
            {"engine": "technical_structure", "score": 0.6, "confidence": 0.8},
            {"engine": "spectral_harmonics", "score": 0.4, "confidence": 0.6},
            {"engine": "fluid_liquidity", "score": 0.2, "confidence": 0.9},
        ],
    }


def test_market_evidence_uses_bounded_exploration_then_qualifies(tmp_path):
    gate = MarketEvidenceGate(tmp_path / "router.json", minimum_samples=3, rolling_window=10)
    assert gate.assess("BTC/USDT")["classification"] == "exploration"
    for outcome in (0.01, 0.02, 0.005):
        result = gate.record("BTC/USDT", outcome)
    assert result["classification"] == "qualified"
    assert result["allowed"] is True
    assert math.isinf(result["profit_factor"])


def test_closed_outcomes_measure_online_calibration(tmp_path):
    gate = MarketEvidenceGate(tmp_path / "router.json", minimum_samples=3, rolling_window=10)
    gate.record("BTC/USDT", 0.01, 0.8)
    gate.record("BTC/USDT", -0.01, 0.6)
    calibration = gate.health()["online_calibration"]
    assert calibration["state"] == "measured"
    assert calibration["samples"] == 2
    assert 0 <= calibration["brier_score"] <= 1


def test_losing_market_is_quarantined_but_not_forgotten(tmp_path):
    path = tmp_path / "router.json"
    gate = MarketEvidenceGate(path, minimum_samples=3, rolling_window=10)
    for outcome in (-0.01, -0.02, -0.005):
        result = gate.record("DOGE/USDT", outcome)
    assert result["classification"] == "quarantined"
    assert result["allowed"] is False
    assert MarketEvidenceGate(path, minimum_samples=3, rolling_window=10).health()["closed_trade_samples"] == 3


def test_router_combines_adaptive_ultra_and_market_evidence(tmp_path):
    router = BoundedDecisionRouter(
        MarketEvidenceGate(tmp_path / "router.json", minimum_samples=3, rolling_window=10)
    )
    approved = router.route(
        symbol="BTC/USDT",
        base_enter=True,
        base_score=0.6,
        base_confidence=0.8,
        advanced=advanced(),
    )
    assert approved["allowed"] is True
    assert approved["authority"] == "paper_and_testnet_only"
    assert approved["live_authority"] is False
    assert approved["size_multiplier"] == pytest.approx(0.35)
    assert 0 <= approved["predicted_probability"] <= 1
    assert len(approved["contributing_engines"]) == 4

    negative = router.route(
        symbol="ETH/USDT",
        base_enter=True,
        base_score=0.6,
        base_confidence=0.8,
        advanced=advanced(score=-0.8),
    )
    assert negative["allowed"] is False
    assert negative["reason"] == "negative_ultra_consensus"

    blackout = router.route(
        symbol="SOL/USDT",
        base_enter=True,
        base_score=0.6,
        base_confidence=0.8,
        advanced=advanced(blackout=True),
    )
    assert blackout["allowed"] is False
    assert blackout["reason"] == "high_impact_news_blackout"


def test_router_rejects_insufficient_or_low_confidence_ultra_evidence(tmp_path):
    router = BoundedDecisionRouter(
        MarketEvidenceGate(tmp_path / "router.json", minimum_samples=3, rolling_window=10)
    )
    sparse = advanced()
    sparse["signals"] = sparse["signals"][:2]
    result = router.route(
        symbol="BTC/USDT", base_enter=True, base_score=0.7, base_confidence=0.8, advanced=sparse
    )
    assert result["reason"] == "insufficient_ultra_engine_evidence"

    result = router.route(
        symbol="ETH/USDT",
        base_enter=True,
        base_score=0.7,
        base_confidence=0.8,
        advanced=advanced(confidence=0.1),
    )
    assert result["reason"] == "low_ultra_consensus_confidence"
