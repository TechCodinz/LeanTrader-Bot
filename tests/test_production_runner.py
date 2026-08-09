from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

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
    assert result["runtime"] == "verified-multi-engine-v3"
    assert set(result["engines"]) == {
        "market_data",
        "paper_ledger",
        "adaptive_intelligence",
        "advanced_shadow_suite",
        "research_governor",
        "operations_safety",
    }
    assert all(engine["healthy"] for engine in result["engines"].values())
    assert result["decisions"]["BTC/USDT"]["quality_score"] == 1.0
    assert result["advanced_shadow"]["execution_authority"] is False
    assert result["research_governor"]["capital_preservation"]["state"] == "normal"
    assert settings.heartbeat_path.exists()


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

    assert result["entry_blocks"] == {"BTC/USDT": "advanced_context_unavailable"}
    assert result["open_positions"] == []
    assert result["operation_alerts"] == [{"sent": False, "reason": "alert_engine_unavailable"}]
    assert "operations_safety:alerts" in result["errors"]
