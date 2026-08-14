from __future__ import annotations

from leantrader.production.market_universe import MarketUniverse
from leantrader.production.runner import MarketFeed


class DiscoveryExchange:
    def __init__(self):
        self.has = {"fetchTickers": True}
        self.markets = {
            "BTC/USDT": {"spot": True, "active": True, "quote": "USDT", "base": "BTC"},
            "ETH/USDT": {"spot": True, "active": True, "quote": "USDT", "base": "ETH"},
            "TINY/USDT": {"spot": True, "active": True, "quote": "USDT", "base": "TINY"},
            "BAD3L/USDT": {"spot": True, "active": True, "quote": "USDT", "base": "BAD3L"},
            "BTC/USDC": {"spot": True, "active": True, "quote": "USDC", "base": "BTC"},
            "OLD/USDT": {"spot": True, "active": False, "quote": "USDT", "base": "OLD"},
            "BTC/USDT:USDT": {"spot": False, "active": True, "quote": "USDT", "base": "BTC"},
        }

    def load_markets(self):
        return self.markets

    def fetch_tickers(self):
        return {
            "BTC/USDT": {"last": 100.0, "quoteVolume": 5_000_000.0, "bid": 99.9, "ask": 100.1},
            "ETH/USDT": {"last": 50.0, "quoteVolume": 2_000_000.0, "bid": 49.9, "ask": 50.1},
            "TINY/USDT": {"last": 1.0, "quoteVolume": 10_000.0, "bid": 0.99, "ask": 1.01},
            "BAD3L/USDT": {"last": 1.0, "quoteVolume": 4_000_000.0, "bid": 0.999, "ask": 1.001},
        }


def test_market_discovery_filters_only_ineligible_markets_and_ranks_liquidity():
    feed = MarketFeed.__new__(MarketFeed)
    feed.exchange = DiscoveryExchange()
    feed._markets_loaded = False
    feed._last_discovery = {}

    result = feed.discover_markets(
        quote="USDT",
        min_quote_volume_usd=250_000.0,
        max_spread_bps=75.0,
    )

    assert [item["symbol"] for item in result["candidates"]] == ["BTC/USDT", "ETH/USDT"]
    assert result["rejection_counts"]["leveraged_token"] == 1
    assert result["rejection_counts"]["insufficient_volume"] == 1
    assert result["rejection_counts"]["not_active_spot"] == 2
    assert result["rejection_counts"]["quote_mismatch"] == 1


def test_dynamic_universe_rotates_through_every_eligible_market_and_persists(tmp_path):
    state_path = tmp_path / "universe.json"
    universe = MarketUniverse(
        state_path=state_path,
        mode="dynamic",
        configured_symbols=(),
        quote="USDT",
        batch_size=2,
        refresh_seconds=3600,
    )
    candidates = [
        {"symbol": "BTC/USDT"},
        {"symbol": "ETH/USDT"},
        {"symbol": "SOL/USDT"},
        {"symbol": "DOGE/USDT"},
        {"symbol": "XRP/USDT"},
    ]
    universe.refresh(candidates, allowed_symbols={item["symbol"] for item in candidates})

    scans = [universe.next_batch() for _ in range(3)]
    assert set().union(*map(set, scans)) == {item["symbol"] for item in candidates}
    assert universe.health()["full_sweeps"] == 1

    restarted = MarketUniverse(
        state_path=state_path,
        mode="dynamic",
        configured_symbols=(),
        quote="USDT",
        batch_size=2,
        refresh_seconds=3600,
    )
    assert restarted.health()["eligible_symbols"] == 5
    assert restarted.health()["full_sweeps"] == 1


def test_testnet_intersection_and_open_position_priority(tmp_path):
    universe = MarketUniverse(
        state_path=tmp_path / "universe.json",
        mode="dynamic",
        configured_symbols=(),
        quote="USDT",
        batch_size=1,
        refresh_seconds=3600,
    )
    universe.refresh(
        [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}, {"symbol": "SOL/USDT"}],
        allowed_symbols={"BTC/USDT", "SOL/USDT"},
    )
    selected = universe.next_batch(mandatory_symbols={"LEGACY/USDT"})
    assert selected[0] == "LEGACY/USDT"
    assert set(universe.symbols) == {"BTC/USDT", "SOL/USDT"}
