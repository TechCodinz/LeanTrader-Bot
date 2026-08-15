from __future__ import annotations

import json
from pathlib import Path

from leantrader.production.cns import CentralNervousSystem
from leantrader.production.error_attribution import ErrorAttributionTracker
from leantrader.production.memory_retention import MarketFingerprint, MemoryRetentionEngine


def test_legacy_fill_history_is_low_support_prior(tmp_path: Path):
    legacy = tmp_path / "memory.jsonl"
    legacy.write_text(
        "\n".join(
            [
                json.dumps({"type": "fill", "symbol": "BTC/USDT", "pnl_pct": 0.01}),
                json.dumps({"type": "fill", "symbol": "BTC/USDT", "pnl_pct": -0.005}),
                json.dumps({"type": "open", "symbol": "BTC/USDT"}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    memory = MemoryRetentionEngine(tmp_path / "retention.json", legacy_memory_path=legacy)
    summary = memory.summarize(symbol="BTC/USDT", fingerprint=MarketFingerprint(regime="trend"))
    assert summary["contextual_samples"] == 0
    assert summary["legacy_samples"] == 2
    assert summary["source"] == "legacy_closed_fill_prior"
    assert 0 < summary["support"] <= 0.20
    assert memory.health()["legacy_fill_outcomes"] == 2


def test_contextual_closed_outcomes_dominate_legacy_prior(tmp_path: Path):
    legacy = tmp_path / "memory.jsonl"
    legacy.write_text(json.dumps({"type": "fill", "symbol": "BTC/USDT", "pnl_pct": -0.5}) + "\n")
    memory = MemoryRetentionEngine(tmp_path / "retention.json", legacy_memory_path=legacy)
    fp = MarketFingerprint(regime="trend", trend=0.7, momentum=0.5)
    for i in range(4):
        memory.record_closed_observation(
            observation_id=f"x{i}",
            symbol="BTC/USDT",
            strategy="router",
            fingerprint=fp,
            confidence=0.8,
            net_return=0.01,
        )
    summary = memory.summarize(symbol="BTC/USDT", fingerprint=fp)
    assert summary["contextual_samples"] == 4
    assert summary["support"] > 0.5
    assert summary["weighted_net_return"] > 0
    assert summary["source"] == "contextual_closed_outcomes_with_legacy_prior"


def test_cns_scales_memory_bias_by_support(tmp_path: Path):
    cns = CentralNervousSystem(tmp_path / "cns.json")
    packet = cns.integrate(
        symbol="BTC/USDT",
        adaptive={"score": 0.0, "confidence": 0.5},
        advanced={"swarm": {"score": 0.0, "confidence": 0.5}},
        routed={"allowed": True, "combined_score": 0.0},
        memory_summary={"support": 0.1, "weighted_net_return": 0.10},
    )
    assert abs(packet["memory_effect"]) <= abs(packet["memory_bias"]) * 0.100001


def test_optional_error_cooldown_and_recovery(tmp_path: Path):
    tracker = ErrorAttributionTracker(tmp_path / "errors.json", cooldown_after=2, cooldown_seconds=60)
    tracker.failure("BTC:order_book", "boom", optional=True, component="order_book", symbol="BTC", now=100)
    assert tracker.should_attempt("BTC:order_book", now=101)
    tracker.failure("BTC:order_book", "boom", optional=True, component="order_book", symbol="BTC", now=102)
    assert not tracker.should_attempt("BTC:order_book", now=103)
    assert tracker.should_attempt("BTC:order_book", now=163)
    tracker.success("BTC:order_book", now=164)
    assert tracker.should_attempt("BTC:order_book", now=165)
