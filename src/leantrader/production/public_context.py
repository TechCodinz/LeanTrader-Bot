from __future__ import annotations

import datetime as dt
import email.utils
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


class PublicMarketContextEngine:
    """Cached, read-only fundamentals, global context, trending data and news."""

    VERSION = "1.0"
    COINGECKO = "https://api.coingecko.com/api/v3"
    NEWS_SOURCES: ClassVar[dict[str, str]] = {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
    }
    HIGH_IMPACT: ClassVar[set[str]] = {
        "approval",
        "ban",
        "bankruptcy",
        "breach",
        "cpi",
        "crash",
        "default",
        "delist",
        "delisting",
        "etf",
        "emergency",
        "exploit",
        "fed",
        "fomc",
        "hack",
        "inflation",
        "lawsuit",
        "liquidation",
        "maintenance",
        "outage",
        "rate",
        "sanctions",
        "sec",
    }

    def __init__(
        self,
        state_path: Path,
        *,
        enabled: bool = True,
        refresh_seconds: int = 900,
        json_fetcher: Callable[[str], Any] | None = None,
        text_fetcher: Callable[[str], str] | None = None,
    ) -> None:
        if refresh_seconds < 300:
            raise ValueError("public context refresh must be at least 300 seconds")
        self.state_path = state_path
        self.enabled = enabled
        self.refresh_seconds = refresh_seconds
        self.json_fetcher = json_fetcher or self._fetch_json
        self.text_fetcher = text_fetcher or self._fetch_text
        self.state = self._load()
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.last_errors: dict[str, str] = {}

    def refresh(self, symbols: list[str] | tuple[str, ...]) -> dict[str, Any]:
        if not self.enabled:
            return {"updated": False, "news_items": [], **self.health()}
        self.attempts += 1
        now = time.time()
        last_success = float(self.state.get("last_success_epoch") or 0.0)
        if last_success and now - last_success < self.refresh_seconds:
            return {"updated": False, "news_items": [], **self.health()}

        bases = sorted({str(symbol).split("/", 1)[0].upper() for symbol in symbols if "/" in str(symbol)})
        errors: dict[str, str] = {}
        successful_sources: list[str] = []
        news_items: list[dict[str, Any]] = []
        market_rows: dict[str, dict[str, Any]] = {}

        for offset in range(0, len(bases), 100):
            batch = bases[offset : offset + 100]
            query = urllib.parse.urlencode(
                {
                    "vs_currency": "usd",
                    "symbols": ",".join(value.lower() for value in batch),
                    "include_tokens": "top",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": 1,
                    "sparkline": "false",
                    "price_change_percentage": "1h,24h,7d",
                }
            )
            try:
                payload = self.json_fetcher(f"{self.COINGECKO}/coins/markets?{query}")
                if not isinstance(payload, list):
                    raise TypeError("coins/markets response must be a list")
                for row in payload:
                    symbol = str(row.get("symbol", "")).upper()
                    if symbol not in batch or symbol in market_rows:
                        continue
                    market_rows[symbol] = {
                        "provider_id": str(row.get("id", "")),
                        "name": str(row.get("name", symbol)),
                        "market_cap_usd": self._number(row.get("market_cap")),
                        "market_cap_rank": self._integer(row.get("market_cap_rank")),
                        "fully_diluted_valuation_usd": self._number(row.get("fully_diluted_valuation")),
                        "total_volume_usd": self._number(row.get("total_volume")),
                        "circulating_supply": self._number(row.get("circulating_supply")),
                        "price_change_percentage_1h": self._number(
                            row.get("price_change_percentage_1h_in_currency")
                        ),
                        "price_change_percentage_24h": self._number(
                            row.get("price_change_percentage_24h_in_currency")
                            if row.get("price_change_percentage_24h_in_currency") is not None
                            else row.get("price_change_percentage_24h")
                        ),
                        "price_change_percentage_7d": self._number(
                            row.get("price_change_percentage_7d_in_currency")
                        ),
                        "market_cap_change_percentage_24h": self._number(
                            row.get("market_cap_change_percentage_24h")
                        ),
                        "updated_at": str(row.get("last_updated", "")),
                    }
                successful_sources.append("coingecko_markets")
            except Exception as exc:  # noqa: BLE001 - provider failures remain explicit and non-fabricated
                errors["coingecko_markets"] = f"{type(exc).__name__}: {exc}"
                break

        global_context: dict[str, Any] | None = None
        try:
            payload = self.json_fetcher(f"{self.COINGECKO}/global")
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            global_context = {
                "total_market_cap_usd": self._number((data.get("total_market_cap") or {}).get("usd")),
                "total_volume_usd": self._number((data.get("total_volume") or {}).get("usd")),
                "btc_dominance_pct": self._number((data.get("market_cap_percentage") or {}).get("btc")),
                "eth_dominance_pct": self._number((data.get("market_cap_percentage") or {}).get("eth")),
                "market_cap_change_percentage_24h": self._number(
                    data.get("market_cap_change_percentage_24h_usd")
                ),
            }
            successful_sources.append("coingecko_global")
        except Exception as exc:  # noqa: BLE001
            errors["coingecko_global"] = f"{type(exc).__name__}: {exc}"

        trending: list[str] | None = None
        try:
            payload = self.json_fetcher(f"{self.COINGECKO}/search/trending")
            trending = [
                str((row.get("item") or {}).get("symbol", "")).upper()
                for row in (payload.get("coins", []) if isinstance(payload, dict) else [])
                if str((row.get("item") or {}).get("symbol", "")).strip()
            ]
            successful_sources.append("coingecko_trending")
        except Exception as exc:  # noqa: BLE001
            errors["coingecko_trending"] = f"{type(exc).__name__}: {exc}"

        for source, url in self.NEWS_SOURCES.items():
            try:
                news_items.extend(self._parse_rss(self.text_fetcher(url), source, bases))
                successful_sources.append(source)
            except Exception as exc:  # noqa: BLE001
                errors[source] = f"{type(exc).__name__}: {exc}"

        news_sources = [source for source in self.NEWS_SOURCES if source in successful_sources]
        news_timestamps: list[dt.datetime] = []
        for item in news_items:
            try:
                timestamp = dt.datetime.fromisoformat(str(item["timestamp"]))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=dt.UTC)
                news_timestamps.append(timestamp.astimezone(dt.UTC))
            except (KeyError, TypeError, ValueError):
                continue

        if market_rows:
            self.state["markets"] = market_rows
            self.state["last_market_refresh_epoch"] = now
        if global_context is not None:
            self.state["global"] = global_context
        if trending is not None:
            self.state["trending"] = trending
        self.state["last_attempt_epoch"] = now
        self.state["successful_sources"] = list(dict.fromkeys(successful_sources))
        self.state["last_errors"] = errors
        self.state["news_sources_successful"] = news_sources
        self.state["news_items_collected"] = len(news_items)
        if news_sources:
            self.state["last_news_refresh_epoch"] = now
        if news_timestamps:
            self.state["latest_news_item_epoch"] = max(value.timestamp() for value in news_timestamps)
        if successful_sources:
            self.state["last_success_epoch"] = now
            self.successes += 1
        else:
            self.failures += 1
        self.last_errors = errors
        self._save()
        return {"updated": bool(successful_sources), "news_items": news_items, **self.health()}

    def evaluate(self, symbol: str) -> dict[str, Any]:
        base = symbol.split("/", 1)[0].upper()
        row = dict((self.state.get("markets") or {}).get(base, {}))
        age = time.time() - float(self.state.get("last_market_refresh_epoch") or 0.0)
        fresh = bool(row) and age <= self.refresh_seconds * 2
        if not fresh:
            return {
                "available": False,
                "fresh": False,
                "score": 0.0,
                "confidence": 0.0,
                "reason": "market_context_missing_or_stale",
            }
        changes = [
            float(row.get("price_change_percentage_1h") or 0.0),
            float(row.get("price_change_percentage_24h") or 0.0),
            float(row.get("market_cap_change_percentage_24h") or 0.0),
        ]
        score = float(np.tanh((0.20 * changes[0] + 0.50 * changes[1] + 0.30 * changes[2]) / 8.0))
        trending = base in set(self.state.get("trending") or [])
        if trending:
            score = min(1.0, score + 0.10)
        market_cap = float(row.get("market_cap_usd") or 0.0)
        confidence = min(0.65, 0.20 + max(0.0, math.log10(max(market_cap, 1.0)) - 6.0) * 0.07)
        return {
            "available": True,
            "fresh": True,
            "score": score,
            "confidence": confidence,
            "trending": trending,
            **row,
        }

    def health(self) -> dict[str, Any]:
        last_success = float(self.state.get("last_success_epoch") or 0.0)
        age = time.time() - last_success if last_success else None
        last_market_refresh = float(self.state.get("last_market_refresh_epoch") or 0.0)
        market_refresh_age = time.time() - last_market_refresh if last_market_refresh else None
        last_news_refresh = float(self.state.get("last_news_refresh_epoch") or 0.0)
        news_refresh_age = time.time() - last_news_refresh if last_news_refresh else None
        latest_news_item = float(self.state.get("latest_news_item_epoch") or 0.0)
        news_item_delta = time.time() - latest_news_item if latest_news_item else None
        news_item_age = max(0.0, news_item_delta) if news_item_delta is not None else None
        news_item_future_skew = max(0.0, -news_item_delta) if news_item_delta is not None else None
        return {
            "provider": "coingecko_plus_rss",
            "enabled": self.enabled,
            "read_only": True,
            "execution_authority": False,
            "refresh_seconds": self.refresh_seconds,
            "markets": len(self.state.get("markets") or {}),
            "trending_symbols": len(self.state.get("trending") or []),
            "global_context_available": bool(self.state.get("global")),
            "last_market_refresh_epoch": last_market_refresh or None,
            "market_refresh_age_seconds": market_refresh_age,
            "market_data_fresh": (
                bool(self.state.get("markets"))
                and market_refresh_age is not None
                and market_refresh_age <= self.refresh_seconds * 2
            ),
            "successful_sources": list(self.state.get("successful_sources") or []),
            "news_sources_successful": list(self.state.get("news_sources_successful") or []),
            "news_items_collected": int(self.state.get("news_items_collected") or 0),
            "last_news_refresh_epoch": last_news_refresh or None,
            "news_refresh_age_seconds": news_refresh_age,
            "latest_news_item_epoch": latest_news_item or None,
            "latest_news_item_age_seconds": news_item_age,
            "latest_news_item_future_skew_seconds": news_item_future_skew,
            "news_fresh": (
                bool(self.state.get("news_sources_successful"))
                and news_refresh_age is not None
                and news_refresh_age <= self.refresh_seconds * 2
                and news_item_age is not None
                and news_item_age <= 86_400
                and news_item_future_skew is not None
                and news_item_future_skew <= 300
            ),
            "last_errors": dict(self.state.get("last_errors") or self.last_errors),
            "last_success_epoch": last_success or None,
            "age_seconds": age,
            "fresh": age is not None and age <= self.refresh_seconds * 2,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "state_path": str(self.state_path),
        }

    @classmethod
    def _parse_rss(cls, payload: str, source: str, bases: list[str]) -> list[dict[str, Any]]:
        root = ET.fromstring(payload)
        items: list[dict[str, Any]] = []
        for node in root.findall(".//item")[:100]:
            title = str(node.findtext("title") or "").strip()
            published = str(node.findtext("pubDate") or node.findtext("date") or "").strip()
            if not title or not published:
                continue
            timestamp = email.utils.parsedate_to_datetime(published)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.UTC)
            tokens = set(re.findall(r"[A-Z0-9]+", title.upper()))
            matched = [base for base in bases if base in tokens]
            lower_tokens = {token.lower() for token in tokens}
            items.append(
                {
                    "timestamp": timestamp.astimezone(dt.UTC).isoformat(),
                    "title": title,
                    "source": source,
                    "symbols": matched,
                    "impact": "high" if lower_tokens & cls.HIGH_IMPACT else "medium",
                }
            )
        return items

    @staticmethod
    def _fetch_json(url: str) -> Any:
        return json.loads(PublicMarketContextEngine._fetch_text(url))

    @staticmethod
    def _fetch_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "LeanTrader/1.0 market-research"})
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "markets": {}, "global": {}, "trending": []}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == 1:
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {"schema_version": 1, "markets": {}, "global": {}, "trending": []}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
