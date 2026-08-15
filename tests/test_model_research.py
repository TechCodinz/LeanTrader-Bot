from __future__ import annotations

import json

import pytest

from leantrader.production.model_research import (
    ModelResearchEngine,
    ModelResearchError,
    StructuredResearchProvider,
)

PROPOSAL = {
    "hypothesis": "Reduce fast-horizon weight when measured disagreement rises.",
    "confidence": 0.62,
    "evidence_refs": ["strategy_observatory:timeframe:1m"],
    "risk_flags": ["regime drift"],
    "candidate": {
        "component_weight_deltas": {"trend": 0.02, "mean_reversion": -0.02},
        "timeframe_group_deltas": {"fast": -0.05, "strategic": 0.05},
        "router_threshold_delta": 0.01,
        "risk_size_multiplier": 0.75,
    },
}


def test_openai_provider_and_engine_accept_bounded_research_only(tmp_path):
    key_path = tmp_path / "key"
    key_path.write_text("test-key-not-real", encoding="utf-8")
    observed = {}

    def post(url, headers, payload):
        observed.update({"url": url, "headers": headers, "payload": payload})
        return {"output_text": json.dumps(PROPOSAL)}

    provider = StructuredResearchProvider(
        provider="openai",
        model="research-model",
        api_key_path=key_path,
        http_post=post,
    )
    engine = ModelResearchEngine(
        tmp_path / "research.json",
        enabled=True,
        interval_cycles=10,
        provider=provider,
    )
    result = engine.observe({"exchange": "bybit", "closed_samples": 20})
    assert result["accepted"] is True
    assert result["proposal"]["status"] == "pending_causal_replay"
    assert result["proposal"]["testnet_authority"] is False
    assert result["proposal"]["live_authority"] is False
    assert observed["url"] == "https://api.openai.com/v1/responses"
    assert "test-key-not-real" not in json.dumps(observed["payload"])
    fingerprint = result["proposal"]["fingerprint"]
    assert engine.record_validation(
        fingerprint,
        windows=5,
        net_return=0.02,
        max_drawdown=0.04,
        brier_score=0.20,
    ) is True
    health = engine.health()
    assert health["eligible_paper_challengers"] == 1
    assert health["automatic_live_promotion"] is False
    assert health["execution_authority"] is False


def test_model_research_rejects_unbounded_or_execution_controls(tmp_path):
    class FakeProvider:
        provider = "openai"
        model = "fake"

        @staticmethod
        def propose(_evidence):
            return {**PROPOSAL, "candidate": {**PROPOSAL["candidate"], "leverage": 100}}

        @staticmethod
        def health():
            return {"configured": True}

    engine = ModelResearchEngine(
        tmp_path / "research.json",
        enabled=True,
        interval_cycles=10,
        provider=FakeProvider(),
    )
    with pytest.raises(ModelResearchError, match="unsupported candidate controls"):
        engine.observe({"exchange": "bybit"})
    assert engine.health()["failures"] == 1


def test_disabled_model_research_is_explicit_and_network_free(tmp_path):
    engine = ModelResearchEngine(tmp_path / "research.json", enabled=False, interval_cycles=10)
    result = engine.observe({"exchange": "bybit"})
    assert result["reason"] == "disabled"
    assert result["calls"] == 0
    assert result["execution_authority"] is False
