from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

import leantrader.production.runner as runner_module
from leantrader.production.runner import PaperRunner, atr_sized_notional
from leantrader.production.settings import Settings


class FakeFeed:
    def candles(self, _symbol: str, _timeframe: str, limit: int) -> pd.DataFrame:
        close = np.linspace(90.0, 110.0, limit)
        return pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "volume": np.ones(limit),
            }
        )


def test_one_cycle_writes_healthy_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env()
    result = PaperRunner(settings, FakeFeed()).cycle()
    assert result["mode"] == "paper"
    assert result["healthy"] is True
    assert result["errors"] == {}
    assert result["runtime"] == "verified-multi-engine-v12.6-world-model-self-awareness"
    assert set(result["engines"]) == {
        "market_data",
        "exchange_intelligence",
        "exchange_protection",
        "market_temporal_guard",
        "cross_venue_arbitrage",
        "market_universe",
        "model_research",
        "paper_ledger",
        "adaptive_intelligence",
        "advanced_shadow_suite",
        "public_market_context",
        "research_governor",
        "decision_router",
        "error_attribution",
        "strategy_observatory",
        "memory_retention",
        "central_nervous_system",
        "trading_brain",
        "market_world_model",
        "meta_cognitive_self_model",
        "intelligence_council",
        "adversarial_critic",
        "hypothesis_lab",
        "active_research_planner",
        "tail_risk_sentinel",
        "capital_growth",
        "operations_safety",
    }
    assert all(engine["healthy"] for engine in result["engines"].values())
    assert result["decisions"]["BTC/USDT"]["quality_score"] == 1.0
    assert result["advanced_shadow"]["execution_authority"] is False
    assert result["market_world_model"]["execution_authority"] is False
    assert result["meta_cognitive_self_model"]["consciousness_claim"] is False
    assert result["intelligence_council"]["execution_authority"] is False
    assert result["adversarial_critic"]["shadow_only"] is True
    assert result["hypothesis_lab"]["research_only"] is True
    assert result["active_research"]["execution_authority"] is False
    assert result["tail_risk_sentinel"]["shadow_only"] is True
    assert result["research_governor"]["capital_preservation"]["state"] == "normal"
    assert result["capital_growth"]["martingale"] is False
    assert result["capital_growth"]["can_increase_upstream_risk"] is False
    assert result["engines"]["central_nervous_system"]["execution_authority"] is False
    assert result["engines"]["trading_brain"]["can_increase_upstream_risk"] is False
    assert result["engines"]["memory_retention"]["causal_closed_outcomes_only"] is True
    assert result["operation_metrics"]["written"] is True
    assert settings.metrics_path.exists()
    assert settings.heartbeat_path.exists()
    observatory = json.loads(settings.strategy_observatory_state_path.read_text())
    pending = observatory["pending"]["BTC/USDT"]
    assert {
        "engine:adaptive_component:trend",
        "engine:adaptive_component:momentum",
        "engine:adaptive_component:mean_reversion",
        "engine:adaptive_component:bollinger_breakout",
        "engine:adaptive_ensemble",
        "engine:swarm_hivemind",
        "engine:bounded_decision_router",
        "timeframe:1m",
        "timeframe:1M",
    } <= set(pending)


def test_atr_sizing_respects_risk_position_and_order_caps():
    notional = atr_sized_notional(
        equity=1000.0,
        price=100.0,
        atr=2.0,
        stop_multiple=2.0,
        risk_fraction=0.01,
        position_cap_fraction=0.10,
        order_cap=500.0,
    )
    assert notional == 100.0

    with_existing = atr_sized_notional(
        equity=1000.0,
        price=100.0,
        atr=2.0,
        stop_multiple=2.0,
        risk_fraction=0.01,
        position_cap_fraction=0.10,
        order_cap=500.0,
        existing_notional=80.0,
    )
    assert with_existing == 20.0


def test_high_impact_news_blocks_new_paper_entry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "BTC/USDT")
    settings = Settings.from_env()
    settings.news_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.news_state_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                    "title": "BTC rate decision",
                    "symbols": ["BTCUSDT"],
                    "impact": "high",
                }
            ]
        ),
        encoding="utf-8",
    )
    result = PaperRunner(settings, FakeFeed()).cycle()
    assert result["entry_blocks"] == {"BTC/USDT": "high_impact_news_blackout"}
    assert result["open_positions"] == []
    assert settings.provenance_path.exists()


def test_shadow_context_failure_blocks_entry_and_alert_failure_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "BTC/USDT")
    settings = Settings.from_env()
    runner = PaperRunner(settings, FakeFeed())

    original_call = runner.engines.call

    def fail_optional_engines(name, method, *args, **kwargs):
        if name == "advanced_shadow_suite" and method == "evaluate_symbol":
            raise RuntimeError("shadow context failed")
        if name == "operations_safety" and method == "alert_events":
            raise RuntimeError("alert service failed")
        return original_call(name, method, *args, **kwargs)

    monkeypatch.setattr(runner.engines, "call", fail_optional_engines)
    result = runner.cycle()

    assert result["entry_blocks"] == {"BTC/USDT": "decision_route_unavailable"}
    assert result["open_positions"] == []
    assert result["operation_alerts"] == [{"sent": False, "reason": "alert_engine_unavailable"}]
    assert "operations_safety:alerts" in result["errors"]


def test_testnet_engine_is_required_and_visible_when_enabled(monkeypatch, tmp_path):
    class FakeTestnetEngine:
        VERSION = "fake-testnet"

        def __init__(self, **_kwargs):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def mirror_events(self, events):
            return [{"mirrored": event["side"]} for event in events]

        def eligible_symbols(self, quote="USDT"):
            assert quote == "USDT"
            return {"BTC/USDT"}

        def health(self):
            return {
                "environment": "testnet",
                "sandbox_endpoint_verified": self.started,
                "live_authority": False,
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "BTC/USDT")
    monkeypatch.setenv("BYBIT_TESTNET_ENABLED", "true")
    monkeypatch.setenv("BYBIT_TESTNET_CONFIRM", "I_UNDERSTAND_TESTNET_ONLY")
    monkeypatch.setattr(runner_module, "BybitTestnetExecutionEngine", FakeTestnetEngine)

    result = PaperRunner(Settings.from_env(), FakeFeed()).cycle()
    testnet = result["engines"]["bybit_testnet_execution"]
    assert testnet["required"] is True
    assert testnet["healthy"] is True
    assert testnet["environment"] == "testnet"
    assert testnet["sandbox_endpoint_verified"] is True
    assert testnet["live_authority"] is False


def test_dynamic_universe_scans_exchange_candidates_in_rotating_batches(monkeypatch, tmp_path):
    class DynamicFeed(FakeFeed):
        def discover_markets(self, **_kwargs):
            return {
                "candidates": [
                    {"symbol": "BTC/USDT"},
                    {"symbol": "ETH/USDT"},
                    {"symbol": "SOL/USDT"},
                ],
                "rejection_counts": {"insufficient_volume": 4},
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "AUTO")
    monkeypatch.setenv("MARKET_SCAN_BATCH_SIZE", "2")
    settings = Settings.from_env()
    runner = PaperRunner(settings, DynamicFeed())
    first = runner.cycle()
    second = runner.cycle()

    assert first["cycle_symbols"] == ["BTC/USDT", "ETH/USDT"]
    # Positions opened in the first batch stay at the front for exit safety;
    # SOL still enters the next rotating batch, proving forward coverage.
    assert second["cycle_symbols"] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    universe = second["engines"]["market_universe"]
    assert universe["eligible_symbols"] == 3
    assert universe["full_sweeps"] == 1
    assert universe["all_eligible_markets_rotate"] is True


def test_immature_symbol_history_is_availability_and_skips_context_matrix(monkeypatch, tmp_path):
    class MixedHistoryFeed(FakeFeed):
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
            self.calls.append((symbol, timeframe))
            frame = super().candles(symbol, timeframe, limit)
            if symbol == "KII/USDT" and timeframe == "15m":
                return frame.tail(109).reset_index(drop=True)
            return frame

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "BTC/USDT,KII/USDT")
    feed = MixedHistoryFeed()
    runner = PaperRunner(Settings.from_env(), feed)

    result = runner.cycle()

    assert result["healthy"] is True
    assert result["errors"] == {}
    assert "BTC/USDT" in result["decisions"]
    assert "KII/USDT" not in result["decisions"]
    row = runner.error_attribution.records["KII/USDT"]
    assert row["availability_state"] == "unavailable"
    assert row["component"] == "symbol_history"
    assert row["failures"] == 0
    assert row["unavailable_count"] == 1
    assert all(
        timeframe == "15m"
        for symbol, timeframe in feed.calls
        if symbol == "KII/USDT"
    )
