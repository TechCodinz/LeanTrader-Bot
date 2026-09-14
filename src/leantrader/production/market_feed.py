"""The historical public market-data feed, lifted out of runner_v141.

This is the read-only CCXT feed the mature fast service used: candles, order
books and market discovery with volume, spread and spot filtering. It loads no
credentials and places no orders, which is why market intelligence can run at
full speed regardless of which execution environment is selected.

It lived inside runner_v141's 3,390-line module alongside the paper runner.
Lifting it out is what lets the fast service be constructed without importing
the runner stack -- the class body below is unchanged.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd


class MarketFeed:
    """Public, read-only CCXT market-data feed. No credentials are loaded."""

    def __init__(self, exchange_id: str) -> None:
        import ccxt  # type: ignore

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"unknown CCXT exchange: {exchange_id}")
        self.exchange = exchange_class({"enableRateLimit": True, "timeout": 20_000})
        self._markets_loaded = False
        self._last_discovery: dict[str, Any] = {}
        self._candle_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}
        self._candle_cache_hits = 0
        self._candle_cache_misses = 0

    def _load_markets(self) -> None:
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        self._load_markets()
        key = (symbol, timeframe, limit)
        cached = self._candle_cache.get(key)
        ttl = self._timeframe_cache_seconds(timeframe)
        if cached is not None and time.monotonic() - cached[0] < ttl:
            self._candle_cache_hits += 1
            return cached[1].copy()
        self._candle_cache_misses += 1
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            raise RuntimeError(f"no candles returned for {symbol}")
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._candle_cache[key] = (time.monotonic(), frame)
        return frame.copy()

    @staticmethod
    def _timeframe_cache_seconds(timeframe: str) -> float:
        try:
            seconds = timeframe_seconds(timeframe)
        except ValueError:
            seconds = 60
        return max(15.0, min(float(seconds) * 0.25, 3_600.0))

    def order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        self._load_markets()
        if not self.exchange.has.get("fetchOrderBook", False):
            raise RuntimeError("exchange does not support order-book inspection")
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def discover_markets(
        self,
        *,
        quote: str,
        min_quote_volume_usd: float,
        max_spread_bps: float,
    ) -> dict[str, Any]:
        """Return every active, liquid spot market ranked by quote volume."""
        self._load_markets()
        if not self.exchange.has.get("fetchTickers", False):
            raise RuntimeError("exchange does not support bulk ticker discovery")
        if str(getattr(self.exchange, "id", "")).lower() == "bybit":
            tickers = self.exchange.fetch_tickers(
                params={"category": "spot"}
            )
        else:
            tickers = self.exchange.fetch_tickers()
        accepted: list[dict[str, Any]] = []
        rejected: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1

        for symbol, market in self.exchange.markets.items():
            if not market.get("spot") or market.get("active") is False:
                reject("not_active_spot")
                continue
            if str(market.get("quote", "")).upper() != quote.upper():
                reject("quote_mismatch")
                continue
            if market.get("leveraged") is True or self._looks_leveraged(str(market.get("base", ""))):
                reject("leveraged_token")
                continue
            ticker = tickers.get(symbol) or {}
            last = float(ticker.get("last") or ticker.get("close") or 0.0)
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            if quote_volume <= 0:
                quote_volume = float(ticker.get("baseVolume") or 0.0) * last
            if last <= 0 or quote_volume < min_quote_volume_usd:
                reject("insufficient_volume")
                continue

            try:
                percentage_24h = float(
                    ticker.get("percentage")
                )
            except (TypeError, ValueError):
                percentage_24h = 0.0

            if not math.isfinite(percentage_24h):
                percentage_24h = 0.0

            if percentage_24h == 0.0:
                try:
                    open_price = float(
                        ticker.get("open") or 0.0
                    )
                except (TypeError, ValueError):
                    open_price = 0.0

                if open_price > 0.0:
                    percentage_24h = (
                        last / open_price - 1.0
                    ) * 100.0

            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            if bid <= 0 or ask <= 0 or ask < bid:
                reject("invalid_spread")
                continue
            spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
            if spread_bps > max_spread_bps:
                reject("spread_too_wide")
                continue
            accepted.append(
                {
                    "symbol": symbol.upper(),
                    "quote_volume_usd": quote_volume,
                    "spread_bps": spread_bps,
                    "last": last,
                    "percentage_24h": (
                        percentage_24h
                    ),
                    "absolute_percentage_24h": abs(
                        percentage_24h
                    ),
                }
            )
        accepted.sort(key=lambda item: (-float(item["quote_volume_usd"]), str(item["symbol"])))
        self._last_discovery = {
            "quote": quote.upper(),
            "eligible": len(accepted),
            "markets_seen": len(self.exchange.markets),
            "rejection_counts": rejected,
        }
        return {"candidates": accepted, **self._last_discovery}

    @staticmethod
    def _looks_leveraged(base: str) -> bool:
        upper = base.upper()
        return upper.endswith(("3L", "3S", "5L", "5S", "BULL", "BEAR"))

    def health(self) -> dict[str, Any]:
        capabilities = {
            name: bool(self.exchange.has.get(name, False))
            for name in ("fetchMarkets", "fetchTickers", "fetchOHLCV", "fetchOrderBook", "fetchTime")
        }
        return {
            "read_only": True,
            "environment": "live_public_market_data",
            "credentials_loaded": False,
            "trading_authority": False,
            "exchange_id": str(getattr(self.exchange, "id", "unknown")),
            "capabilities": capabilities,
            "markets_loaded": self._markets_loaded,
            "last_discovery": dict(self._last_discovery),
            "candle_cache": {
                "entries": len(getattr(self, "_candle_cache", {})),
                "hits": getattr(self, "_candle_cache_hits", 0),
                "misses": getattr(self, "_candle_cache_misses", 0),
                "timeframe_aware": True,
            },
        }




class MicrostructureMarketFeed(MarketFeed):
    """MarketFeed plus public trades, as the historical fast service used it."""

    def public_trades(self, symbol: str, limit: int = 80) -> list[dict[str, Any]]:
        self._load_markets()
        if not self.exchange.has.get("fetchTrades", False):
            return []
        rows = self.exchange.fetch_trades(symbol, limit=max(1, min(200, int(limit))))
        return [dict(row) for row in rows or []]
