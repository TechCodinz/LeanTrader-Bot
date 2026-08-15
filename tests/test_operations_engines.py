from __future__ import annotations

import datetime as dt
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from leantrader.production.operations_engines import (
    DataProvenanceEngine,
    ExecutionRealityEngine,
    ForexEngine,
    MarketManipulationEngine,
    PrometheusMetricsEngine,
    ReconciliationEngine,
    StrategyCapacityEngine,
    TelegramAlertEngine,
)


def book() -> dict:
    return {
        "bids": [[99.9, 2.0], [99.8, 4.0]],
        "asks": [[100.1, 1.0], [100.2, 2.0]],
    }


def candles(volume_multiplier: float = 1.0) -> pd.DataFrame:
    close = np.linspace(99.99, 100.0, 40)
    volume = np.full(40, 100.0)
    volume[-1] *= volume_multiplier
    return pd.DataFrame({"close": close, "volume": volume})


def test_execution_reality_models_depth_cost_and_partial_fill():
    fill = ExecutionRealityEngine().market_fill(
        book(),
        side="buy",
        quantity=4.0,
        fee_bps=10.0,
        latency_bps=2.0,
    )
    assert fill.filled_quantity == 3.0
    assert fill.partial is True
    assert fill.average_price > 100.1
    assert fill.fee > 0


def test_forex_engine_supports_xauusd_and_provider_formats():
    engine = ForexEngine()
    assert engine.normalize("XAU/USD", "oanda") == "XAU_USD"
    assert engine.normalize("EUR/USD", "mt5") == "EURUSD"
    assert engine.pip_size("XAUUSD") == 0.01
    assert engine.pip_size("EURUSD") == 0.0001
    assert engine.risk_units("EURUSD", equity=1000, risk_fraction=0.01, stop_distance=0.002) == 5000
    assert engine.session_allowed(dt.datetime(2026, 8, 10, 12, tzinfo=dt.UTC)) is True


def test_reconciliation_reports_exact_mismatch():
    result = ReconciliationEngine().compare({"BTC/USDT": 0.1}, {"BTC/USDT": 0.08})
    assert result["reconciled"] is False
    assert result["mismatches"][0]["delta"] == pytest.approx(-0.02)


def test_manipulation_engine_flags_cancellation_and_wash_like_volume():
    previous = {"bids": [[99.9, 100.0]], "asks": [[100.1, 100.0]]}
    current = {"bids": [[99.9, 5.0]], "asks": [[100.1, 100.0]]}
    result = MarketManipulationEngine().evaluate(previous, current, candles(8.0))
    assert result["spoof_like"] is True
    assert result["wash_volume_like"] is True


def test_strategy_capacity_stops_at_impact_budget():
    result = StrategyCapacityEngine().estimate(book(), "buy", impact_cap_bps=15.0)
    assert result["max_quantity"] == 1.0
    assert result["max_notional"] == 100.1


def test_provenance_is_stable_and_append_only(tmp_path):
    path = tmp_path / "provenance.jsonl"
    engine = DataProvenanceEngine(path)
    first = engine.record("BTC/USDT", {"score": 0.5, "model": "ensemble"})
    second = engine.record("BTC/USDT", {"model": "ensemble", "score": 0.5})
    assert first == second
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["fingerprint"] == first


def test_telegram_is_safely_disabled_without_credentials():
    result = TelegramAlertEngine(token="", chat_id="").send("paper event")
    assert result == {"sent": False, "reason": "telegram not configured"}


def test_telegram_token_can_be_loaded_from_secret_file(monkeypatch, tmp_path):
    token_file = tmp_path / "telegram_bot_token"
    token_file.write_text("test-token-value", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_FILE", str(token_file))
    engine = TelegramAlertEngine(chat_id="test-chat")
    assert engine.token == "test-token-value"
    assert engine.health()["configured"] is True


def test_telegram_publishes_gated_free_paid_moon_arbitrage_and_testnet_link(monkeypatch):
    monkeypatch.setenv("TELEGRAM_FREE_CHAT_ID", "free-chat")
    monkeypatch.setenv("TELEGRAM_PAID_CHAT_ID", "paid-chat")
    monkeypatch.setenv("TELEGRAM_TESTNET_TRADE_URL", "https://testnet.bybit.com/")
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        requests.append((urllib.parse.parse_qs(request.data.decode()), timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    engine = TelegramAlertEngine(token="test-token", chat_id="admin-chat", monitor_interval_cycles=1)
    status = {
        "healthy": True,
        "runtime": "verified-test",
        "equity": 50.0,
        "open_positions": [],
        "errors": {},
        "engines": {
            "exchange_protection": {
                "authorization_checks": 3,
                "block_reasons": {"api_key_ip_bound": 1},
            }
        },
        "testnet_execution": {"enabled": True},
        "decisions": {
            "BTC/USDT": {
                "confidence": 0.90,
                "enter_long": True,
                "regime": "trend",
                "multi_timeframe_score": 0.5,
                "route": {"allowed": True, "reason": "approved"},
            }
        },
        "advanced_shadow": {
            "market": {
                "moon_scout_ranking": [
                    {"symbol": "SOL/USDT", "score": 1.5, "momentum": 0.08, "volume_spike": 3.0}
                ],
                "arbitrage_opportunities": [
                    {
                        "symbol": "BTC/USDT",
                        "buy_venue": "bybit",
                        "sell_venue": "okx",
                        "buy_price": 100.0,
                        "sell_price": 100.5,
                        "net_bps": 20.0,
                        "liquidity_verified": True,
                    }
                ],
            }
        },
    }
    results = engine.publish_cycle(status)
    assert any(result.get("sent") for result in results)
    chats = [payload["chat_id"][0] for payload, _timeout in requests]
    assert {"admin-chat", "free-chat", "paid-chat"} <= set(chats)
    paid_payloads = [payload for payload, _timeout in requests if payload["chat_id"] == ["paid-chat"]]
    assert any("reply_markup" in payload for payload in paid_payloads)
    assert any(
        "Exchange protection blocked authority" in payload["text"][0]
        for payload, _timeout in requests
    )
    health = engine.health()
    assert health["sent"] == len(requests)
    assert health["outbound_only"] is True
    assert health["execution_authority"] is False


def test_prometheus_metrics_are_atomic_and_use_canonical_status(tmp_path):
    path = tmp_path / "leantrader.prom"
    result = PrometheusMetricsEngine(path).write(
        {
            "healthy": True,
            "equity": 51.5,
            "cash": 48.0,
            "realized_pnl": 1.5,
            "open_positions": ["BTC/USDT"],
            "errors": {},
            "halt_reason": None,
            "engines": {"paper_ledger": {"healthy": True}},
        }
    )
    text = path.read_text()
    assert result["written"] is True
    assert "leantrader_equity_usd 51.5" in text
    assert 'leantrader_engine_healthy{engine="paper_ledger"} 1' in text
