from __future__ import annotations

from typing import Any, ClassVar

from leantrader.production.arbitrage_monitor import CrossVenueQuoteCollector


class Exchange:
    apiKey = ""
    secret = ""

    def __init__(self, venue: str, bid: float, ask: float) -> None:
        self.id = venue
        self.bid = bid
        self.ask = ask
        self.markets: ClassVar[dict[str, dict[str, Any]]] = {}

    def load_markets(self):
        self.markets = {"BTC/USDT": {"active": True, "taker": 0.001}}
        return self.markets

    def fetch_tickers(self, symbols):
        assert symbols == ["BTC/USDT"]
        return {
            "BTC/USDT": {
                "bid": self.bid,
                "ask": self.ask,
                "bidVolume": 2.0,
                "askVolume": 1.5,
            }
        }


class Feed:
    def __init__(self) -> None:
        self.exchange = Exchange("bybit", 100.0, 100.1)


def test_cross_venue_collector_uses_real_two_sided_public_quotes():
    feed = Feed()
    exchanges = {"okx": Exchange("okx", 100.5, 100.6)}
    collector = CrossVenueQuoteCollector(
        primary_feed=feed,
        venues=("bybit", "okx"),
        exchange_factory=lambda venue: exchanges[venue],
    )
    collector.start()
    result = collector.collect(("BTC/USDT",))
    assert result["available"] is True
    assert result["successful_venues"] == ["bybit", "okx"]
    assert len(result["quotes"]) == 2
    assert all(row["read_only"] for row in result["quotes"])
    assert collector.health()["execution_authority"] is False


def test_cross_venue_collector_is_explicitly_unavailable_with_one_venue():
    collector = CrossVenueQuoteCollector(primary_feed=Feed(), venues=("bybit",))
    result = collector.collect(("BTC/USDT",))
    assert result["available"] is False
    assert result["reason"] == "configure_at_least_two_public_venues"
