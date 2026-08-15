from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from leantrader.production.advanced_engines import (
    ArbitrageEngine,
    EngineSignal,
    LiquidityFluidEngine,
    MoonScoutEngine,
    NewsAwarenessEngine,
    PatternMemoryEngine,
    PortfolioRiskEngine,
    SmartScalpingEngine,
    SpectralHarmonicsEngine,
    SwarmConsensusEngine,
    TechnicalStructureEngine,
    UltraEngineSuite,
)


def frame(rows: int = 320, slope: float = 0.05, volume_multiplier: float = 1.0) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    close = 100.0 + slope * x + np.sin(x / 8.0)
    volume = np.full(rows, 100.0)
    volume[-1] *= volume_multiplier
    return pd.DataFrame(
        {
            "timestamp": x * 900_000,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": volume,
        }
    )


def test_smart_scalping_and_spectral_signals_are_deterministic():
    market = frame()
    scalp = SmartScalpingEngine()
    spectral = SpectralHarmonicsEngine()
    assert scalp.evaluate(market) == scalp.evaluate(market)
    assert spectral.evaluate(market) == spectral.evaluate(market)
    assert -1 <= scalp.evaluate(market).score <= 1
    assert "dominant_period" in spectral.evaluate(market).rationale


def test_technical_structure_rehabilitates_main_indicators_without_lookahead():
    engine = TechnicalStructureEngine()
    result = engine.evaluate(frame())
    assert result == engine.evaluate(frame())
    assert -1 <= result.score <= 1
    assert "adx=" in result.rationale
    assert engine.health()["indicators"] == ["macd", "adx", "stochastic", "obv", "liquidity_sweeps"]

    flat = frame(slope=0.0)
    flat["close"] = 100.0
    flat["open"] = 100.0
    flat["high"] = 100.0
    flat["low"] = 100.0
    assert np.isfinite(engine.evaluate(flat).score)


def test_liquidity_engine_measures_spread_imbalance_and_impact():
    result = LiquidityFluidEngine().evaluate(
        {
            "bids": [[99.9, 5.0], [99.8, 5.0]],
            "asks": [[100.1, 2.0], [100.2, 8.0]],
        },
        desired_qty=3.0,
    )
    assert result["spread_bps"] == pytest.approx(20.0)
    assert result["buy_impact_bps"] > 0
    assert result["safe_buy_qty_30bps"] == 10.0


def test_news_engine_uses_real_items_and_blackout_window(tmp_path):
    now = dt.datetime(2026, 8, 10, 12, 0, tzinfo=dt.UTC)
    path = tmp_path / "news.json"
    path.write_text(
        json.dumps(
            [
                {
                    "timestamp": now.isoformat(),
                    "title": "BTC bullish adoption surge",
                    "symbols": ["BTCUSDT"],
                    "impact": "high",
                }
            ]
        )
    )
    result = NewsAwarenessEngine(path).evaluate("BTC/USDT", now)
    assert result["sentiment"] > 0
    assert result["blackout"] is True
    assert result["matched_items"] == 1


def test_news_ingestion_validates_and_deduplicates(tmp_path):
    path = tmp_path / "news.json"
    engine = NewsAwarenessEngine(path)
    item = {
        "timestamp": "2026-08-10T12:00:00+00:00",
        "title": "EUR rate hike",
        "source": "calendar",
        "symbols": ["EUR/USD"],
        "impact": "high",
    }
    assert engine.ingest([item, item, {"timestamp": "invalid", "title": "bad"}]) == 1
    assert engine.ingest([item]) == 0
    assert engine.health()["items"] == 1


def test_news_matching_supports_long_base_symbols_and_rejects_future_clock_skew(tmp_path):
    now = dt.datetime.now(dt.UTC)
    path = tmp_path / "news.json"
    engine = NewsAwarenessEngine(path, max_future_skew_seconds=300)
    assert engine.ingest(
        [
            {
                "timestamp": now.isoformat(),
                "title": "DOGE bullish adoption surge",
                "source": "feed",
                "symbols": ["DOGE"],
                "impact": "high",
            },
            {
                "timestamp": (now + dt.timedelta(hours=1)).isoformat(),
                "title": "DOGE impossible future item",
                "source": "feed",
                "symbols": ["DOGE"],
                "impact": "high",
            },
        ]
    ) == 1
    result = engine.evaluate("DOGE/USDT", now)
    assert result["matched_items"] == 1
    assert result["blackout"] is True
    assert engine.health()["future_items_rejected"] == 1


def test_pattern_memory_requires_evidence_and_persists(tmp_path):
    path = tmp_path / "memory.json"
    memory = PatternMemoryEngine(path)
    for index in range(5):
        memory.remember({"trend": 0.8 + index * 0.01, "momentum": 0.5}, 0.7, "BTC")
    recalled = memory.recall({"trend": 0.82, "momentum": 0.5})
    assert recalled.score > 0
    assert recalled.confidence > 0
    assert len(PatternMemoryEngine(path).records) == 5


def test_swarm_penalizes_disagreement():
    swarm = SwarmConsensusEngine()
    aligned = swarm.combine([EngineSignal("a", 0.8, 0.9, ""), EngineSignal("b", 0.7, 0.9, "")])
    divided = swarm.combine([EngineSignal("a", 0.8, 0.9, ""), EngineSignal("b", -0.8, 0.9, "")])
    assert aligned.confidence > divided.confidence


def test_moon_scout_and_portfolio_risk_use_observed_frames():
    frames = {
        "BTC/USDT": frame(slope=0.08, volume_multiplier=4.0),
        "ETH/USDT": frame(slope=0.02),
        "SOL/USDT": frame(slope=-0.01),
    }
    ranking = MoonScoutEngine().rank(frames)
    assert ranking[0]["symbol"] == "BTC/USDT"
    risk = PortfolioRiskEngine().analyze(frames, {"BTC/USDT": 5.0, "ETH/USDT": 3.0})
    assert risk["var_95_usd"] >= 0
    assert risk["concentration"] == pytest.approx(0.625)


def test_market_snapshot_actively_runs_moon_scout_and_arbitrage(tmp_path):
    suite = UltraEngineSuite(tmp_path / "memory.json", tmp_path / "news.json")
    quotes = [
        {
            "venue": "bybit",
            "symbol": "BTC/USDT",
            "bid": 100.0,
            "ask": 100.1,
            "bid_quantity": 2.0,
            "ask_quantity": 2.0,
            "fee_bps": 1.0,
            "slippage_bps": 1.0,
        },
        {
            "venue": "okx",
            "symbol": "BTC/USDT",
            "bid": 100.5,
            "ask": 100.6,
            "bid_quantity": 2.0,
            "ask_quantity": 2.0,
            "fee_bps": 1.0,
            "slippage_bps": 1.0,
        },
    ]
    snapshot = suite.market_snapshot({"BTC/USDT": frame()}, {}, quotes)
    assert snapshot["moon_scout_ranking"][0]["symbol"] == "BTC/USDT"
    assert snapshot["arbitrage_opportunities"][0]["buy_venue"] == "bybit"
    health = suite.health()
    assert health["activity"]["moon_scout_dynamic_scanner"]["state"] == "active"
    assert health["activity"]["arbitrage"]["state"] == "active"


def test_arbitrage_subtracts_all_costs_and_never_executes():
    opportunities = ArbitrageEngine().scan(
        [
            {
                "venue": "a",
                "symbol": "BTC/USDT",
                "bid": 99.9,
                "ask": 100.0,
                "fee_bps": 2,
                "slippage_bps": 1,
                "ask_quantity": 2,
                "bid_quantity": 2,
            },
            {
                "venue": "b",
                "symbol": "BTC/USDT",
                "bid": 100.2,
                "ask": 100.3,
                "fee_bps": 2,
                "slippage_bps": 1,
                "ask_quantity": 1,
                "bid_quantity": 1,
            },
        ],
        minimum_net_bps=5,
    )
    assert opportunities[0]["net_bps"] == pytest.approx(14.0)
    assert opportunities[0]["max_quantity"] == 1.0
    assert opportunities[0]["liquidity_verified"] is True
    assert opportunities[0]["execution_authority"] is False


def test_ultra_suite_exposes_real_capability_map(tmp_path):
    suite = UltraEngineSuite(tmp_path / "memory.json", tmp_path / "news.json")
    result = suite.evaluate_symbol(
        "BTC/USDT",
        frame(),
        {"bids": [[99.9, 5.0]], "asks": [[100.1, 5.0]]},
        desired_qty=0.01,
    )
    assert result["swarm"]["engine"] == "swarm_hivemind"
    assert any(signal["engine"] == "technical_structure" for signal in result["signals"])
    assert any(signal["engine"] == "fluid_liquidity" for signal in result["signals"])
    assert result["liquidity"]["available"] is True
    health = suite.health()
    assert health["legacy_random_engines_loaded"] is False
    assert "frequency_harmonics_ultrasonic" in health["capabilities"]
    assert health["activity"]["fluid_liquidity"]["state"] == "active"


def test_ultra_suite_uses_fast_scalping_frame_and_complete_timeframe_matrix(tmp_path):
    suite = UltraEngineSuite(tmp_path / "memory.json", tmp_path / "news.json")
    contexts = {
        timeframe: frame(slope=0.02)
        for timeframe in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M")
    }
    result = suite.evaluate_symbol("BTC/USDT", frame(), context_frames=contexts)
    assert result["timeframe_matrix"]["valid"] == 13
    assert set(result["timeframe_matrix"]["groups"]) == {"fast", "tactical", "strategic"}
    assert any(signal["engine"] == "multi_timeframe_matrix" for signal in result["signals"])
    assert suite.health()["activity"]["multi_timeframe_matrix"]["successes"] == 1


def test_ultra_suite_isolates_and_reports_individual_engine_failure(tmp_path):
    suite = UltraEngineSuite(tmp_path / "memory.json", tmp_path / "news.json")
    suite.spectral.evaluate = lambda _frame: (_ for _ in ()).throw(RuntimeError("spectral failure"))
    result = suite.evaluate_symbol("BTC/USDT", frame())
    spectral = next(signal for signal in result["signals"] if signal["engine"] == "spectral_harmonics")
    assert spectral["confidence"] == 0.0
    assert suite.health()["activity"]["spectral_harmonics"]["state"] == "failed"
    assert "spectral failure" in suite.health()["activity"]["spectral_harmonics"]["last_error"]
