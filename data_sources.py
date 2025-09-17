# data_sources.py
from __future__ import annotations

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd


# Mock data source classes for testing
class EconomicCalendarSource:
    """Mock economic calendar data source."""
    
    def __init__(self):
        self.events = []
    
    def fetch_events(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch economic calendar events."""
        return self.events
    
    def get_high_impact_events(self) -> List[Dict[str, Any]]:
        """Get high impact economic events."""
        return [e for e in self.events if e.get('impact') == 'high']


class FundingRateSource:
    """Mock funding rate data source."""
    
    def __init__(self):
        self.rates = {}
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch funding rate for a symbol."""
        return self.rates.get(symbol, 0.0001)
    
    def fetch_historical_rates(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch historical funding rates."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='8h')
        rates = [self.rates.get(symbol, 0.0001) for _ in range(len(dates))]
        return pd.DataFrame({'timestamp': dates, 'rate': rates})


class NewsSource:
    """Mock news data source."""
    
    def __init__(self):
        self.articles = []
    
    def fetch_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch latest news articles."""
        return self.articles[:limit]
    
    def fetch_news_by_symbol(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch news for a specific symbol."""
        return [a for a in self.articles if symbol in a.get('symbols', [])][:limit]


class OnChainSource:
    """Mock on-chain data source."""
    
    def __init__(self):
        self.metrics = {}
    
    def fetch_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch on-chain metrics for a symbol."""
        return self.metrics.get(symbol, {
            'tvl': 0,
            'volume_24h': 0,
            'active_addresses': 0,
            'hash_rate': 0
        })
    
    def fetch_whale_movements(self, symbol: str, threshold: float = 1000000) -> List[Dict[str, Any]]:
        """Fetch whale movements above threshold."""
        return []


class SentimentSource:
    """Mock sentiment data source."""
    
    def __init__(self):
        self.sentiment_data = {}
    
    def fetch_sentiment(self, symbol: str) -> Dict[str, float]:
        """Fetch sentiment scores for a symbol."""
        return self.sentiment_data.get(symbol, {
            'twitter': 0.5,
            'reddit': 0.5,
            'news': 0.5,
            'overall': 0.5
        })
    
    def fetch_fear_greed_index(self) -> float:
        """Fetch fear and greed index (0-100)."""
        return 50.0

import os  # noqa: F401
import random
from typing import Any, Dict, List, Optional

# ccxt is optional in FX-only setups
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/124.0 Safari/537.36"
)

# ---------------------------------------------------------------------
# Exchange fallbacks: try primary first, then these in order if it 403's,
# needs apiKey, or otherwise fails.  Gate.io included.
# Keys are ccxt ids.
EX_ROUTES: Dict[str, List[str]] = {
    "binanceus": ["gateio", "kraken", "okx", "kucoin"],
    "binance": ["kraken", "okx", "kucoin", "gateio"],
    "bybit": ["gateio", "okx", "kucoin", "kraken"],
    "kraken": ["okx", "gateio", "binance"],
    "okx": ["kraken", "gateio", "binance"],
    "kucoin": ["gateio", "okx", "kraken"],
    "gateio": ["kraken", "okx", "kucoin", "binance"],
}


def _has_ccxt() -> bool:
    return ccxt is not None


def _build_ex(ex_id: str):
    """
    Build a ccxt exchange with safe defaults for public market data.
    Uses env keys if present; otherwise public-only works for most venues.
    """
    if not _has_ccxt():
        raise RuntimeError("ccxt not installed")

    # Prefer project's ExchangeRouter when possible so that safety wrappers
    # (safe_fetch_ohlcv, safe_fetch_ticker) are used consistently.
    try:
        from router import ExchangeRouter

        router = ExchangeRouter()
        if getattr(router, "ex", None) and getattr(router.ex, "id", "").lower() == ex_id.lower():
            return router
    except Exception:
        pass

    klass = getattr(ccxt, ex_id)
    cfg = {
        "enableRateLimit": True,
        "timeout": 15000,
        "headers": {"User-Agent": USER_AGENT},
    }
    # If user provided keys for this venue, attach them
    env_key = os.getenv("API_KEY") or ""
    env_sec = os.getenv("API_SECRET") or ""
    env_id = (os.getenv("EXCHANGE_ID") or "").lower().strip()
    if env_id == ex_id and env_key and env_sec:
        cfg["apiKey"] = env_key
        cfg["secret"] = env_sec
    return klass(cfg)


def _map_symbol_for(ex_id: str, symbol: str) -> str:
    """
    Map 'BASE/USD' -> 'BASE/USDT' for venues that mainly quote in USDT.
    We *accept* USD input in your loops but translate to USDT where needed.
    """
    try:
        base, quote = symbol.split("/")
    except Exception:
        return symbol

    quote = quote.upper()
    # Most crypto venues want USDT
    if ex_id in ("binanceus", "binance", "bybit", "okx", "kucoin", "gateio") and quote == "USD":
        return f"{base}/USDT"
    return symbol


def _is_geo_or_cred_error(e: Exception) -> bool:
    s = str(e).lower()
    # common patterns: 403 forbidden cloudfront, requires apikey, not available in your country, etc.
    return any(
        k in s
        for k in [
            "403",
            "forbidden",
            "cloudfront",
            "your country",
            'requires "apikey"',
            "requires 'apikey'",
            "permission",
            "denied",
            "not available in your country",
        ]
    )


def fetch_ohlcv_router(
    ex_primary,
    symbol: str,
    timeframe: str,
    limit: int = 400,
    since: Optional[int] = None,
    backups: Optional[List[str]] = None,
) -> List[List[Any]]:
    """
    Try primary ccxt exchange first.  If it fails with geo/API-key issues,
    automatically fall back through EX_ROUTES (including Gate.io).
    Returns a standard ccxt OHLCV array: [ts, o, h, l, c, v] rows.
    """
    if not _has_ccxt():
        raise RuntimeError("ccxt not installed")

    tried: List[str] = []
    primary_id = getattr(ex_primary, "id", "").lower()
    routes = list(backups or EX_ROUTES.get(primary_id, []))

    # small shuffle after first two to spread load
    if len(routes) > 2:
        head, tail = routes[:2], routes[2:]
        random.shuffle(tail)
        routes = head + tail

    # Attempt 1: primary
    try:
        s = _map_symbol_for(primary_id, symbol)
        # If caller passed an ExchangeRouter-like object, prefer its safe wrapper
        if hasattr(ex_primary, "safe_fetch_ohlcv"):
            try:
                rows = ex_primary.safe_fetch_ohlcv(s, timeframe=timeframe, limit=limit, since=since)  # type: ignore[arg-type]
                if rows:
                    return rows
            except Exception as _e:
                print(f"[data_sources] safe_fetch_ohlcv primary failed: {_e}")
                # fall through to try direct fetch
        try:
            rows = ex_primary.fetch_ohlcv(s, timeframe=timeframe, limit=limit, since=since)
            if rows:
                return rows
            tried.append(primary_id)
        except Exception as _e:
            tried.append(primary_id)
            # Log the error and continue to fallback routes; do not re-raise to avoid crashing callers
            print(f"[data_sources] primary fetch_ohlcv failed for {primary_id}: {_e}")
    except Exception as _e:
        tried.append(primary_id)
        if not _is_geo_or_cred_error(_e):
            # Different error -> surface it
            raise

    # Fallbacks
    last_err = None
    for ex_id in routes:
        try:
            ex = _build_ex(ex_id)
            s = _map_symbol_for(ex_id, symbol)
            rows = ex.fetch_ohlcv(s, timeframe=timeframe, limit=limit, since=since)
            if rows:
                return rows
        except Exception as _e:  # pragma: no cover
            last_err = _e
            tried.append(ex_id)
            continue
    # If we reach here no route returned data; return empty list rather than raise so callers can handle gracefully
    print(f"[data_sources] fetch_ohlcv_router no data for {symbol}@{timeframe}; tried={tried}; last_err={last_err}")
    return []
