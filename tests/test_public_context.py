from __future__ import annotations

import json
import time

from leantrader.production.public_context import PublicMarketContextEngine

RSS = """<?xml version="1.0"?><rss><channel><item>
<title>BTC ETF approval drives adoption surge</title>
<pubDate>Fri, 14 Aug 2026 12:00:00 GMT</pubDate>
</item></channel></rss>"""


def test_public_context_collects_market_cap_global_trending_and_news(tmp_path):
    def fetch_json(url: str):
        if "/coins/markets?" in url:
            return [
                {
                    "id": "bitcoin",
                    "symbol": "btc",
                    "name": "Bitcoin",
                    "market_cap": 2_000_000_000_000,
                    "market_cap_rank": 1,
                    "fully_diluted_valuation": 2_100_000_000_000,
                    "total_volume": 30_000_000_000,
                    "circulating_supply": 20_000_000,
                    "price_change_percentage_1h_in_currency": 0.5,
                    "price_change_percentage_24h": 2.0,
                    "price_change_percentage_7d_in_currency": 4.0,
                    "market_cap_change_percentage_24h": 1.5,
                    "last_updated": "2026-08-14T12:00:00Z",
                }
            ]
        if url.endswith("/global"):
            return {
                "data": {
                    "total_market_cap": {"usd": 3_000_000_000_000},
                    "total_volume": {"usd": 100_000_000_000},
                    "market_cap_percentage": {"btc": 55, "eth": 12},
                    "market_cap_change_percentage_24h_usd": 1.2,
                }
            }
        if url.endswith("/search/trending"):
            return {"coins": [{"item": {"symbol": "BTC"}}]}
        raise AssertionError(url)

    fixed_now = 1786791600.0  # 2026-08-15T11:00:00Z; source item remains within 24h.
    engine = PublicMarketContextEngine(
        tmp_path / "context.json",
        json_fetcher=fetch_json,
        text_fetcher=lambda _url: RSS,
        now_fn=lambda: fixed_now,
    )
    result = engine.refresh(("BTC/USDT", "ETH/USDT"))
    assert result["updated"] is True
    assert len(result["news_items"]) == 2
    assert result["news_items"][0]["symbols"] == ["BTC"]
    assert result["news_items"][0]["impact"] == "high"
    context = engine.evaluate("BTC/USDT")
    assert context["available"] is True
    assert context["market_cap_usd"] == 2_000_000_000_000
    assert context["market_cap_rank"] == 1
    assert context["trending"] is True
    assert context["score"] > 0
    assert result["market_data_fresh"] is True
    assert result["news_fresh"] is True
    saved = json.loads((tmp_path / "context.json").read_text())
    assert saved["successful_sources"] == [
        "coingecko_markets",
        "coingecko_global",
        "coingecko_trending",
        "coindesk",
        "cointelegraph",
    ]


def test_public_context_reports_provider_failures_without_fabricating_data(tmp_path):
    def fail(_url: str):
        raise RuntimeError("offline")

    engine = PublicMarketContextEngine(
        tmp_path / "context.json",
        json_fetcher=fail,
        text_fetcher=fail,
    )
    result = engine.refresh(("BTC/USDT",))
    assert result["updated"] is False
    assert result["failures"] == 1
    assert set(result["last_errors"]) == {
        "coingecko_markets",
        "coingecko_global",
        "coingecko_trending",
        "coindesk",
        "cointelegraph",
    }
    assert engine.evaluate("BTC/USDT")["available"] is False


def test_fresh_news_can_supply_non_directional_context_without_faking_symbol_market_data(tmp_path):
    now = time.time()
    state = tmp_path / "context.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "markets": {"BTC": {"market_cap_usd": 1_000_000}},
                "global": {},
                "trending": [],
                "last_success_epoch": now,
                "successful_sources": ["coindesk"],
                "last_news_refresh_epoch": now,
                "latest_news_item_epoch": now,
                "news_sources_successful": ["coindesk"],
            }
        )
    )
    engine = PublicMarketContextEngine(state)
    assert engine.health()["news_fresh"] is True
    assert engine.health()["market_data_fresh"] is False
    context = engine.evaluate("BTC/USDT")
    assert context["available"] is True
    assert context["symbol_market_available"] is False
    assert context["score"] == 0.0
    assert context["confidence"] <= 0.12


def test_public_context_batches_large_symbol_universe_without_losing_all_market_context(tmp_path):
    seen_sizes = []

    def fetch_json(url: str):
        if "/coins/markets?" in url:
            from urllib.parse import parse_qs, urlparse
            symbols = parse_qs(urlparse(url).query).get("symbols", [""])[0].split(",")
            symbols = [value for value in symbols if value]
            seen_sizes.append(len(symbols))
            if len(symbols) > 50:
                raise RuntimeError("400 Bad Request")
            return [
                {
                    "id": value, "symbol": value, "name": value.upper(),
                    "market_cap": 1_000_000_000, "market_cap_rank": 10,
                    "total_volume": 10_000_000, "circulating_supply": 1_000_000,
                    "price_change_percentage_1h_in_currency": 0.1,
                    "price_change_percentage_24h": 0.2,
                    "price_change_percentage_7d_in_currency": 0.3,
                    "market_cap_change_percentage_24h": 0.1,
                    "last_updated": "2026-08-17T00:00:00Z",
                } for value in symbols
            ]
        if url.endswith("/global"):
            return {"data": {"total_market_cap": {"usd": 1}, "total_volume": {"usd": 1}, "market_cap_percentage": {"btc": 50, "eth": 10}, "market_cap_change_percentage_24h_usd": 0.1}}
        if url.endswith("/search/trending"):
            return {"coins": []}
        raise AssertionError(url)

    symbols = tuple(f"T{i}/USDT" for i in range(120))
    engine = PublicMarketContextEngine(
        tmp_path / "context.json", json_fetcher=fetch_json, text_fetcher=lambda _url: RSS,
        now_fn=lambda: 1786946000.0,
    )
    result = engine.refresh(symbols)
    assert result["updated"] is True
    assert max(seen_sizes) <= 50
    assert result["markets"] == 120
    assert engine.evaluate("T119/USDT")["symbol_market_available"] is True
