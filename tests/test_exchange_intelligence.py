from __future__ import annotations

from typing import Any, ClassVar

import pytest

from leantrader.production.exchange_intelligence import (
    ExchangeIntelligence,
    sort_timeframes,
    timeframe_seconds,
)


class FakeExchange:
    id = "okx"
    apiKey = ""
    secret = ""
    rateLimit = 100
    urls: ClassVar[dict[str, Any]] = {"api": "https://www.okx.com", "test": "https://www.okx.com/demo"}
    timeframes: ClassVar[dict[str, str]] = {"1m": "1m", "5m": "5m", "1h": "1H", "1d": "1D"}
    has: ClassVar[dict[str, bool]] = {
        "fetchMarkets": True,
        "fetchTickers": True,
        "fetchTicker": True,
        "fetchOHLCV": True,
        "fetchOrderBook": True,
        "fetchTrades": True,
        "fetchTime": True,
        "fetchFundingRate": True,
        "fetchOpenInterest": True,
        "fetchBalance": True,
        "createOrder": True,
        "cancelOrder": True,
        "fetchOrder": True,
        "fetchOpenOrders": True,
        "fetchClosedOrders": True,
        "fetchMyTrades": True,
    }
    markets: ClassVar[dict[str, dict[str, Any]]] = {
        "BTC/USDT": {
            "id": "BTC-USDT",
            "active": True,
            "type": "spot",
            "spot": True,
            "margin": True,
            "swap": False,
            "future": False,
            "option": False,
            "quote": "USDT",
            "precision": {"amount": 1e-6, "price": 0.1},
            "limits": {"amount": {"min": 1e-6}, "cost": {"min": 1.0}},
            "maker": 0.0008,
            "taker": 0.001,
        },
        "BTC/USDT:USDT": {
            "id": "BTC-USDT-SWAP",
            "active": True,
            "type": "swap",
            "spot": False,
            "margin": False,
            "swap": True,
            "future": False,
            "option": False,
            "quote": "USDT",
            "settle": "USDT",
            "linear": True,
            "inverse": False,
            "contractSize": 0.01,
            "precision": {"amount": 0.01, "price": 0.1},
            "limits": {"amount": {"min": 0.01}},
            "maker": 0.0002,
            "taker": 0.0005,
        },
    }


class FakeFeed:
    def __init__(self) -> None:
        self.exchange = FakeExchange()
        self.loaded = False

    def _load_markets(self) -> None:
        self.loaded = True


def test_exchange_intelligence_attests_capabilities_rules_and_auto_timeframes(tmp_path):
    feed = FakeFeed()
    engine = ExchangeIntelligence(
        state_path=tmp_path / "exchange.json",
        exchange_id="okx",
        feed=feed,
        base_timeframe="5m",
        requested_timeframes=(),
    )
    engine.start()
    assert feed.loaded is True
    assert engine.resolve_timeframes() == ("1m", "5m", "1h", "1d")
    health = engine.health()
    assert health["provider_rules_dynamic"] is True
    assert health["market_types"] == {"spot": 1, "margin": 1, "swap": 1, "future": 0, "option": 0}
    assert health["credentials_loaded"] is False
    assert health["execution_authority"] is False
    assert health["capabilities"]["fetchTime"] is True
    rules = engine.market_rules("BTC/USDT:USDT")
    assert rules["swap"] is True
    assert rules["contract_size"] == 0.01
    assert rules["execution_authority"] is False
    assert (tmp_path / "exchange.json").exists()


def test_exchange_intelligence_rejects_adapter_identity_mismatch(tmp_path):
    engine = ExchangeIntelligence(
        state_path=tmp_path / "exchange.json",
        exchange_id="binance",
        feed=FakeFeed(),
        base_timeframe="5m",
        requested_timeframes=(),
    )
    with pytest.raises(RuntimeError, match="resolved to okx"):
        engine.start()


def test_timeframe_parser_preserves_minutes_and_months():
    assert timeframe_seconds("1m") == 60
    assert timeframe_seconds("1M") == 2_592_000
    assert sort_timeframes({"1d", "5m", "1h", "1m"}) == ("1m", "5m", "1h", "1d")
    with pytest.raises(ValueError):
        timeframe_seconds("tick")
