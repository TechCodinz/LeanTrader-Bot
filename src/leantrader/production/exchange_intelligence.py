from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .settings import BYBIT_KLINE_TIMEFRAMES


def timeframe_seconds(value: str) -> int:
    """Convert a CCXT timeframe label to seconds without losing m/M semantics."""
    match = re.fullmatch(r"([1-9][0-9]*)([smhdwMy])", value.strip())
    if match is None:
        raise ValueError(f"unsupported CCXT timeframe label: {value}")
    amount = int(match.group(1))
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 3_600,
        "d": 86_400,
        "w": 604_800,
        "M": 2_592_000,
        "y": 31_536_000,
    }[match.group(2)]
    return amount * multiplier


def sort_timeframes(values: set[str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    valid: list[tuple[int, str]] = []
    for value in values:
        try:
            valid.append((timeframe_seconds(str(value)), str(value)))
        except ValueError:
            continue
    return tuple(value for _, value in sorted(set(valid)))


class ExchangeIntelligence:
    """Read-only exchange capability and market-rule attestation.

    This layer discovers what a configured CCXT adapter actually supports. It
    never loads credentials and never grants order authority. Execution
    adapters must separately attest their environment, credentials and
    reconciliation semantics.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        state_path: Path,
        exchange_id: str,
        feed: Any,
        base_timeframe: str,
        requested_timeframes: tuple[str, ...],
    ) -> None:
        self.state_path = state_path
        self.exchange_id = exchange_id.lower()
        self.feed = feed
        self.base_timeframe = base_timeframe
        self.requested_timeframes = requested_timeframes
        self.profile: dict[str, Any] = {}
        self.inspections = 0

    def start(self) -> None:
        self.inspect()
        self.resolve_timeframes()

    def inspect(self) -> dict[str, Any]:
        self.inspections += 1
        exchange = getattr(self.feed, "exchange", None)
        if exchange is None:
            supported = (
                BYBIT_KLINE_TIMEFRAMES
                if self.exchange_id == "bybit"
                else self.requested_timeframes or (self.base_timeframe,)
            )
            self.profile = {
                "exchange_id": self.exchange_id,
                "adapter": "injected_feed",
                "environment": "public_market_data",
                "credentials_loaded": False,
                "execution_authority": False,
                "supported_timeframes": list(sort_timeframes(supported)),
                "market_types": {},
                "active_markets": 0,
                "capabilities": {},
                "sandbox_declared": False,
                "rule_coverage": {},
                "inspected_at": time.time(),
            }
            self._save()
            return dict(self.profile)

        loader = getattr(self.feed, "_load_markets", None)
        if callable(loader):
            loader()
        markets = getattr(exchange, "markets", {}) or {}
        advertised = getattr(exchange, "has", {}) or {}
        raw_timeframes = getattr(exchange, "timeframes", {}) or {}
        supported_timeframes = sort_timeframes(set(raw_timeframes))
        active = [market for market in markets.values() if market.get("active") is not False]
        market_types = {
            market_type: sum(bool(market.get(market_type)) for market in active)
            for market_type in ("spot", "margin", "swap", "future", "option")
        }
        capabilities = {
            name: bool(advertised.get(name, False))
            for name in (
                "fetchMarkets",
                "fetchTickers",
                "fetchTicker",
                "fetchOHLCV",
                "fetchOrderBook",
                "fetchTrades",
                "fetchTime",
                "fetchFundingRate",
                "fetchOpenInterest",
                "fetchBalance",
                "createOrder",
                "cancelOrder",
                "fetchOrder",
                "fetchOpenOrders",
                "fetchClosedOrders",
                "fetchMyTrades",
            )
        }
        self.profile = {
            "exchange_id": str(getattr(exchange, "id", self.exchange_id)).lower(),
            "adapter": "ccxt",
            "ccxt_version": self._ccxt_version(),
            "environment": "public_market_data",
            "credentials_loaded": bool(getattr(exchange, "apiKey", "") or getattr(exchange, "secret", "")),
            "execution_authority": False,
            "rate_limit_ms": int(getattr(exchange, "rateLimit", 0) or 0),
            "declared_timezone": str(getattr(exchange, "timezone", "UTC") or "UTC"),
            "supported_timeframes": list(supported_timeframes),
            "market_types": market_types,
            "active_markets": len(active),
            "quote_assets": sorted(
                {str(market.get("quote")).upper() for market in active if market.get("quote")}
            ),
            "capabilities": capabilities,
            "sandbox_declared": bool(self._flatten_urls((getattr(exchange, "urls", {}) or {}).get("test"))),
            "rule_coverage": {
                "precision": sum(bool(market.get("precision")) for market in active),
                "limits": sum(bool(market.get("limits")) for market in active),
                "maker_fee": sum(market.get("maker") is not None for market in active),
                "taker_fee": sum(market.get("taker") is not None for market in active),
                "contract_size": sum(market.get("contractSize") is not None for market in active),
            },
            "inspected_at": time.time(),
        }
        if self.profile["exchange_id"] != self.exchange_id:
            raise RuntimeError(
                f"configured exchange {self.exchange_id} resolved to {self.profile['exchange_id']}"
            )
        if self.profile["credentials_loaded"]:
            raise RuntimeError("public exchange intelligence must never load credentials")
        if not capabilities["fetchOHLCV"]:
            raise RuntimeError(f"{self.exchange_id} does not advertise OHLCV support")
        if not supported_timeframes:
            raise RuntimeError(f"{self.exchange_id} did not advertise any parseable candle timeframe")
        self._save()
        return dict(self.profile)

    def resolve_timeframes(self) -> tuple[str, ...]:
        if not self.profile:
            self.inspect()
        supported = set(self.profile.get("supported_timeframes") or [])
        requested = self.requested_timeframes or tuple(sort_timeframes(supported))
        if self.base_timeframe not in requested:
            requested = (*requested, self.base_timeframe)
        missing = sorted(set(requested) - supported)
        if missing:
            raise RuntimeError(
                f"{self.exchange_id} does not advertise requested candle timeframes: {', '.join(missing)}"
            )
        if self.exchange_id == "bybit":
            bybit_missing = sorted(set(BYBIT_KLINE_TIMEFRAMES) - set(requested))
            if bybit_missing:
                raise RuntimeError(
                    "Bybit runtime requires its complete verified timeframe matrix: "
                    + ", ".join(bybit_missing)
                )
        return sort_timeframes(requested)

    def market_rules(self, symbol: str) -> dict[str, Any]:
        exchange = getattr(self.feed, "exchange", None)
        markets = getattr(exchange, "markets", {}) if exchange is not None else {}
        market = (markets or {}).get(symbol)
        if not isinstance(market, dict):
            return {"available": False, "symbol": symbol, "reason": "market_not_loaded"}
        return {
            "available": True,
            "symbol": symbol,
            "market_id": market.get("id"),
            "type": market.get("type"),
            "spot": bool(market.get("spot")),
            "margin": bool(market.get("margin")),
            "swap": bool(market.get("swap")),
            "future": bool(market.get("future")),
            "option": bool(market.get("option")),
            "linear": market.get("linear"),
            "inverse": market.get("inverse"),
            "settle": market.get("settle"),
            "contract_size": market.get("contractSize"),
            "precision": market.get("precision") or {},
            "limits": market.get("limits") or {},
            "maker_fee": market.get("maker"),
            "taker_fee": market.get("taker"),
            "active": market.get("active") is not False,
            "execution_authority": False,
        }

    def health(self) -> dict[str, Any]:
        return {
            **self.profile,
            "inspections": self.inspections,
            "requested_timeframes": list(self.requested_timeframes) or ["AUTO"],
            "resolved_timeframes": list(self.resolve_timeframes()) if self.profile else [],
            "provider_rules_dynamic": True,
            "live_adapter_attested": False,
            "execution_authority": False,
            "state_path": str(self.state_path),
        }

    @staticmethod
    def _flatten_urls(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [url for nested in value.values() for url in ExchangeIntelligence._flatten_urls(nested)]
        if isinstance(value, (list, tuple)):
            return [url for nested in value for url in ExchangeIntelligence._flatten_urls(nested)]
        return []

    @staticmethod
    def _ccxt_version() -> str | None:
        try:
            import ccxt  # type: ignore

            return str(ccxt.__version__)
        except (ImportError, AttributeError):
            return None

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.profile, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
