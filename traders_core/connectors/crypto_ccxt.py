from typing import Any, Dict
import pandas as pd
import os  # noqa: F401  # intentionally kept

from dotenv import load_dotenv

load_dotenv()

import ccxt  # type: ignore  # noqa: E402

def _mk_exchange(name: str, testnet: bool) -> Any:
    api_key = os.getenv(f"{name.upper()}_API_KEY") or os.getenv("CRYPTO_API_KEY") or ""
    secret = os.getenv(f"{name.upper()}_API_SECRET") or os.getenv("CRYPTO_API_SECRET") or ""
    password = os.getenv(f"{name.upper()}_API_PASSWORD") or os.getenv("CRYPTO_API_PASSWORD") or None

    # Prefer central ExchangeRouter when possible so safety checks (ALLOW_LIVE,
    # LIVE_CONFIRM, LIVE_ORDER_USD) are applied uniformly. We only use the
    # router when its configured exchange matches the requested name; otherwise
    # fall back to a direct ccxt instance.
    try:
        from router import ExchangeRouter  # noqa: E402

        router = ExchangeRouter()
        router_ex = getattr(router, "ex", None)
        if router_ex and getattr(router_ex, "id", "").lower() == name.lower():
            # ensure sandbox/testnet mode is set on the underlying exchange if supported
            try:
                if hasattr(router_ex, "set_sandbox_mode"):
                    router_ex.set_sandbox_mode(testnet)
            except Exception:
                pass
            return router
    except Exception:
        # if router isn't available or fails, fall back to direct ccxt exchange
        pass

    # Fallback: construct a plain ccxt exchange instance (unchanged behavior)
    klass = getattr(ccxt, name)
    ex = klass(
        {
            "apiKey": api_key,
            "secret": secret,
            "password": password,
            "enableRateLimit": True,
            "options": {},
        }
    )
    # testnet routing for binance-like exchanges
    if name == "binance":
        try:
            ex.set_sandbox_mode(testnet)
        except Exception:
            pass
    return ex

def ohlcv_df(
    exchange: str, symbol: str, timeframe: str, lookback_days: int, testnet: bool
) -> pd.DataFrame:
    ex = _mk_exchange(exchange, testnet)
    # ccxt timeframes like '5m','1m','1h'
    limit = min(5000, lookback_days * (24 * 60 // max(1, int(timeframe[:-1]))))
    # prefer safe wrapper if available, otherwise guarded direct fetch
    rows = []
    try:
        if hasattr(ex, "safe_fetch_ohlcv"):
            try:
                rows = ex.safe_fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception as _e:
                print(f"[traders_core.connectors.crypto_ccxt] safe_fetch_ohlcv failed: {_e}")
                rows = []
        else:
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception as _e:
                print(f"[traders_core.connectors.crypto_ccxt] fetch_ohlcv failed: {_e}")
                rows = []
    except Exception as _e:
        print(f"[traders_core.connectors.crypto_ccxt] unexpected error: {_e}")
        rows = []
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.set_index("time")

def ticker_price(exchange: str, symbol: str, testnet: bool) -> float:
    ex = _mk_exchange(exchange, testnet)
    try:
        # prefer safe wrapper but guard it
        if hasattr(ex, "safe_fetch_ticker"):
            try:
                t = ex.safe_fetch_ticker(symbol)
            except Exception as _e:
                print(f"[traders_core.connectors.crypto_ccxt] safe_fetch_ticker failed: {_e}")
                t = {}
        else:
            try:
                t = ex.fetch_ticker(symbol)
            except Exception as _e:
                print(f"[traders_core.connectors.crypto_ccxt] fetch_ticker failed: {_e}")
                t = {}
        try:
            return float(
                (t.get("last") if isinstance(t, dict) else None)
                or (t.get("close") if isinstance(t, dict) else None)
                or 0
            )
        except Exception:
            return 0.0
    except Exception:
        return 0.0

def market_buy(
    exchange: str,
    symbol: str,
    amount: float,
    testnet: bool,
) -> Dict[str, Any]:
    from src.leantrader.execution.router import (
        route_order,
    )

    payload = {
        "symbol": symbol,
        "side": "buy",
        "qty": amount,
        "order_type": "market",
        "exchange_id": exchange,
        "backend": "ccxt",
    }

    # Legacy parameter is only a
    # compatibility hint when no global
    # runtime mode has been selected.
    if (
        testnet
        and not os.getenv(
            "EXECUTION_MODE"
        )
    ):
        payload[
            "execution_mode"
        ] = "testnet"

    return route_order(
        payload
    )

def market_sell(
    exchange: str,
    symbol: str,
    amount: float,
    testnet: bool,
) -> Dict[str, Any]:
    from src.leantrader.execution.router import (
        route_order,
    )

    payload = {
        "symbol": symbol,
        "side": "sell",
        "qty": amount,
        "order_type": "market",
        "exchange_id": exchange,
        "backend": "ccxt",
    }

    # Legacy parameter is only a
    # compatibility hint when no global
    # runtime mode has been selected.
    if (
        testnet
        and not os.getenv(
            "EXECUTION_MODE"
        )
    ):
        payload[
            "execution_mode"
        ] = "testnet"

    return route_order(
        payload
    )

def market_info(exchange: str, symbol: str, testnet: bool) -> Dict[str, Any]:
    ex = _mk_exchange(exchange, testnet)
    try:
        if hasattr(ex, "load_markets"):
            ex.load_markets()
    except Exception:
        pass
    m = getattr(ex, "markets", {}).get(symbol) or {}
    return {
        "min_cost": float(
            m.get("limits", {}).get("cost", {}).get("min", 5.0)
        ),  # e.g., ~10 USDT on Binance
        "min_qty": float(m.get("limits", {}).get("amount", {}).get("min", 0.0001)),
        "step_qty": float(
            m.get("precision", {}).get("amount", 6)
        ),  # we’ll round with precision decimals
        "price_prec": int(m.get("precision", {}).get("price", 2)),
        "amount_prec": int(m.get("precision", {}).get("amount", 6)),
        "taker": float(m.get("taker", 0.001)),
        "maker": float(m.get("maker", 0.001)),
    }
