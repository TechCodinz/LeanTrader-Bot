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


def test_news_success_cannot_make_stale_market_context_look_fresh(tmp_path):
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
    assert engine.evaluate("BTC/USDT")["available"] is False
