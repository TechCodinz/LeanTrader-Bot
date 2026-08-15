from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from leantrader.production.intelligence import AdaptiveIntelligence


def market_frame(rows: int = 320) -> pd.DataFrame:
    close = np.linspace(90.0, 110.0, rows)
    return pd.DataFrame(
        {
            "timestamp": np.arange(rows) * 900_000,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.ones(rows),
        }
    )


def test_decision_is_deterministic_explainable_and_bounded(tmp_path):
    intelligence = AdaptiveIntelligence(tmp_path / "intelligence.json")
    first = intelligence.evaluate(market_frame())
    second = intelligence.evaluate(market_frame())

    assert first == second
    assert first.regime in {"trend", "range", "high_volatility"}
    assert set(first.component_scores) == {
        "trend",
        "momentum",
        "mean_reversion",
        "bollinger_breakout",
    }
    assert sum(first.weights.values()) == pytest.approx(1.0)
    assert all(0.10 <= value <= 0.70 for value in first.weights.values())
    assert first.quality_score == 1.0
    assert len(first.rationale) == 8
    assert first.multi_timeframe_confirmed is True
    assert first.multi_timeframe_coverage == 1.0
    assert first.session_allowed is True


def test_bad_market_data_is_rejected_before_signals(tmp_path):
    frame = market_frame()
    frame.loc[10, "close"] = float("nan")
    intelligence = AdaptiveIntelligence(tmp_path / "intelligence.json")
    with pytest.raises(ValueError, match="market data rejected"):
        intelligence.evaluate(frame)


def test_multi_timeframe_and_fx_session_are_fail_closed(tmp_path):
    falling = market_frame()
    falling["close"] = falling["close"].iloc[::-1].to_numpy()
    intelligence = AdaptiveIntelligence(tmp_path / "intelligence.json")
    decision = intelligence.evaluate(
        market_frame(),
        context_frames={"1h": falling},
        symbol="EUR/USD",
    )
    assert decision.multi_timeframe_confirmed is False
    assert decision.multi_timeframe_score < 0
    assert decision.enter_long is False


def test_learning_promotes_only_after_evidence_gate_and_persists(tmp_path):
    path = tmp_path / "intelligence.json"
    intelligence = AdaptiveIntelligence(path, min_samples=3, learning_rate=0.20)
    metadata = {
        "regime": "trend",
        "component_scores": {
            "trend": 1.0,
            "momentum": 0.5,
            "mean_reversion": -1.0,
            "bollinger_breakout": 1.0,
        },
    }
    original = intelligence.weights["trend"].copy()

    assert intelligence.learn(metadata, 0.02) is False
    assert intelligence.learn(metadata, 0.02) is False
    assert intelligence.weights["trend"] == original
    assert intelligence.learn(metadata, 0.02) is True
    assert intelligence.weights["trend"]["trend"] > original["trend"]
    assert all(0.10 <= value <= 0.70 for value in intelligence.weights["trend"].values())
    assert sum(intelligence.weights["trend"].values()) == pytest.approx(1.0)

    restored = AdaptiveIntelligence(path)
    assert restored.weights["trend"] == pytest.approx(intelligence.weights["trend"])
    assert json.loads(path.read_text())["schema_version"] == 1
