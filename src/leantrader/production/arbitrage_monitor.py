from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class CrossVenueQuoteCollector:
    """Read-only, cached cross-venue quote collector for arbitrage research."""

    VERSION = "1.0"

    def __init__(
        self,
        *,
        primary_feed: Any,
        venues: tuple[str, ...],
        enabled: bool = True,
        refresh_seconds: int = 60,
        assumed_slippage_bps: float = 3.0,
        exchange_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.primary_feed = primary_feed
        self.venues = tuple(dict.fromkeys(value.lower() for value in venues if value.strip()))
        self.enabled = enabled and len(self.venues) >= 2
        self.refresh_seconds = refresh_seconds
        self.assumed_slippage_bps = assumed_slippage_bps
        self.exchange_factory = exchange_factory or self._ccxt_exchange
        self.exchanges: dict[str, Any] = {}
        self.last_quotes: list[dict[str, Any]] = []
        self.last_errors: dict[str, str] = {}
        self.last_successful_venues: list[str] = []
        self.last_collection_epoch = 0.0
        self.collections = 0
        self.cache_hits = 0

    def start(self) -> None:
        primary = getattr(self.primary_feed, "exchange", None)
        primary_id = str(getattr(primary, "id", "")).lower()
        if primary is not None and primary_id in self.venues:
            self._reject_credentials(primary, primary_id)
            self.exchanges[primary_id] = primary

    def collect(self, symbols: list[str] | tuple[str, ...]) -> dict[str, Any]:
        now = time.time()
        if not self.enabled:
            return {
                "available": False,
                "reason": "configure_at_least_two_public_venues",
                **self.health(),
                "quotes": [],
            }
        if self.last_collection_epoch and now - self.last_collection_epoch < self.refresh_seconds:
            self.cache_hits += 1
            return {
                "available": len(self.last_successful_venues) >= 2,
                **self.health(),
                "quotes": list(self.last_quotes),
            }

        requested = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if str(symbol).strip()))
        quotes: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        successful: list[str] = []
        for venue in self.venues:
            try:
                exchange = self.exchanges.get(venue)
                if exchange is None:
                    exchange = self.exchange_factory(venue)
                    self._reject_credentials(exchange, venue)
                    self.exchanges[venue] = exchange
                markets = exchange.load_markets()
                supported = [symbol for symbol in requested if symbol in markets and markets[symbol].get("active") is not False]
                if not supported:
                    raise RuntimeError("none of the requested markets are active on this venue")
                tickers = exchange.fetch_tickers(supported)
                venue_rows = 0
                for symbol in supported:
                    ticker = tickers.get(symbol) or {}
                    bid = self._positive(ticker.get("bid"))
                    ask = self._positive(ticker.get("ask"))
                    if bid is None or ask is None or ask < bid:
                        continue
                    market = markets[symbol]
                    timestamp_ms = self._positive(ticker.get("timestamp"))
                    age_seconds = max(0.0, now - timestamp_ms / 1_000.0) if timestamp_ms else 0.0
                    if timestamp_ms and age_seconds > max(120.0, self.refresh_seconds * 2.0):
                        continue
                    quotes.append(
                        {
                            "venue": venue,
                            "symbol": symbol,
                            "bid": bid,
                            "ask": ask,
                            "bid_quantity": self._positive(ticker.get("bidVolume")) or 0.0,
                            "ask_quantity": self._positive(ticker.get("askVolume")) or 0.0,
                            "fee_bps": max(0.0, float(market.get("taker") or 0.0) * 10_000.0),
                            "slippage_bps": self.assumed_slippage_bps,
                            "observed_at_epoch": now,
                            "source_timestamp_ms": timestamp_ms,
                            "age_seconds": age_seconds,
                            "read_only": True,
                        }
                    )
                    venue_rows += 1
                if venue_rows:
                    successful.append(venue)
                else:
                    errors[venue] = "no valid fresh two-sided quotes"
            except Exception as exc:  # noqa: BLE001 - venue failures remain isolated and visible
                errors[venue] = f"{type(exc).__name__}: {exc}"
        self.collections += 1
        self.last_collection_epoch = now
        self.last_quotes = quotes
        self.last_errors = errors
        self.last_successful_venues = successful
        return {"available": len(successful) >= 2, **self.health(), "quotes": list(quotes)}

    @staticmethod
    def _reject_credentials(exchange: Any, venue: str) -> None:
        if getattr(exchange, "apiKey", "") or getattr(exchange, "secret", ""):
            raise RuntimeError(f"arbitrage research venue {venue} must not load credentials")

    @staticmethod
    def _positive(value: Any) -> float | None:
        try:
            number = float(value)
            return number if number > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ccxt_exchange(venue: str) -> Any:
        import ccxt  # type: ignore

        exchange_class = getattr(ccxt, venue, None)
        if exchange_class is None:
            raise ValueError(f"unknown CCXT arbitrage venue: {venue}")
        return exchange_class({"enableRateLimit": True, "timeout": 20_000})

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "venues": list(self.venues),
            "successful_venues": list(self.last_successful_venues),
            "quote_count": len(self.last_quotes),
            "last_errors": dict(self.last_errors),
            "last_collection_epoch": self.last_collection_epoch or None,
            "refresh_seconds": self.refresh_seconds,
            "assumed_slippage_bps": self.assumed_slippage_bps,
            "collections": self.collections,
            "cache_hits": self.cache_hits,
            "credentials_loaded": False,
            "execution_authority": False,
        }
