"""
Real Yahoo public-chart adapter for LeanTrader.

No random/synthetic fallback.
No yfinance dependency.
Bounded network calls.
"""

from urllib.parse import quote

import pandas as pd
from curl_cffi import requests as curl_requests

from curl_impersonate_compat import resolve_impersonation

# Resolved against the installed curl_cffi rather than hardcoded: the bare
# "chrome" alias is not present on every build, and a missing target takes
# down every FX and commodity fetch in this module.
EXPLICIT_IMPERSONATION = resolve_impersonation("chrome")


SYMBOL_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",

    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "GAS": "NG=F",
    "COPPER": "HG=F",
    "PLATINUM": "PL=F",
    "PALLADIUM": "PA=F",

    "XAU/USD": "GC=F",
    "XAG/USD": "SI=F",
}


def normalize_symbol(symbol):
    raw = str(
        symbol or ""
    ).strip()

    if not raw:
        raise ValueError(
            "symbol is required"
        )

    upper = raw.upper()

    if upper in SYMBOL_MAP:
        return SYMBOL_MAP[
            upper
        ]

    if upper.endswith(
        ("=X", "=F")
    ):
        return raw

    if "/" in upper:
        base, quote_currency = (
            upper.split(
                "/",
                1,
            )
        )

        fiat = {
            "USD",
            "EUR",
            "GBP",
            "JPY",
            "CHF",
            "AUD",
            "CAD",
            "NZD",
        }

        if (
            base in fiat
            and quote_currency
            in fiat
        ):
            return (
                base
                + quote_currency
                + "=X"
            )

        if quote_currency in {
            "USD",
            "USDT",
            "USDC",
        }:
            return (
                base
                + "-USD"
            )

    return raw


def normalize_range(value):
    raw = str(
        value or "1mo"
    ).strip().lower()

    if raw.endswith("d"):
        try:
            days = int(
                raw[:-1]
            )
        except ValueError:
            return "1mo"

        if days <= 5:
            return "5d"

        if days <= 31:
            return "1mo"

        if days <= 93:
            return "3mo"

        if days <= 186:
            return "6mo"

        if days <= 366:
            return "1y"

        if days <= 730:
            return "2y"

        return "5y"

    allowed = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }

    return (
        raw
        if raw in allowed
        else "1mo"
    )


def normalize_interval(value):
    raw = str(
        value or "1h"
    ).strip().lower()

    mapping = {
        "60m": "1h",
        "4h": "1h",
    }

    raw = mapping.get(
        raw,
        raw,
    )

    allowed = {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }

    return (
        raw
        if raw in allowed
        else "1h"
    )


def fetch_yahoo_rows(
    symbol,
    range_="1mo",
    interval="1h",
    timeout=12,
):
    yahoo_symbol = (
        normalize_symbol(
            symbol
        )
    )

    encoded = quote(
        yahoo_symbol,
        safe="",
    )

    url = (
        "https://query1.finance.yahoo.com/"
        f"v8/finance/chart/{encoded}"
        f"?range={normalize_range(range_)}"
        f"&interval={normalize_interval(interval)}"
        "&includePrePost=false"
        "&events=div%2Csplits"
    )

    session = (
        curl_requests.Session(
            impersonate=(
                EXPLICIT_IMPERSONATION
            )
        )
    )

    try:
        response = session.get(
            url,
            timeout=timeout,
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

    finally:
        try:
            session.close()
        except Exception:
            pass

    chart = payload.get(
        "chart",
        {},
    )

    if chart.get(
        "error"
    ):
        raise RuntimeError(
            str(
                chart["error"]
            )
        )

    results = (
        chart.get(
            "result"
        )
        or []
    )

    if not results:
        raise RuntimeError(
            f"No Yahoo chart data "
            f"for {symbol}"
        )

    result = results[0]

    timestamps = (
        result.get(
            "timestamp"
        )
        or []
    )

    quotes = (
        result.get(
            "indicators",
            {},
        ).get(
            "quote"
        )
        or []
    )

    if not quotes:
        raise RuntimeError(
            f"No quote series "
            f"for {symbol}"
        )

    q = quotes[0]

    opens = (
        q.get("open")
        or []
    )

    highs = (
        q.get("high")
        or []
    )

    lows = (
        q.get("low")
        or []
    )

    closes = (
        q.get("close")
        or []
    )

    volumes = (
        q.get("volume")
        or []
    )

    rows = []

    size = min(
        len(timestamps),
        len(closes),
    )

    for index in range(
        size
    ):
        close = closes[
            index
        ]

        if close is None:
            continue

        def value_or_close(
            values
        ):
            if (
                index < len(values)
                and values[index]
                is not None
            ):
                return float(
                    values[index]
                )

            return float(close)

        volume = (
            volumes[index]
            if (
                index
                < len(volumes)
                and volumes[index]
                is not None
            )
            else 0.0
        )

        rows.append(
            [
                int(
                    timestamps[
                        index
                    ]
                ) * 1000,
                value_or_close(
                    opens
                ),
                value_or_close(
                    highs
                ),
                value_or_close(
                    lows
                ),
                float(close),
                float(volume),
            ]
        )

    if not rows:
        raise RuntimeError(
            f"No usable Yahoo rows "
            f"for {symbol}"
        )

    return rows


def fetch_yahoo_dataframe(
    symbol,
    range_="1mo",
    interval="1h",
    timeout=12,
):
    rows = fetch_yahoo_rows(
        symbol,
        range_=range_,
        interval=interval,
        timeout=timeout,
    )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    df[
        "timestamp"
    ] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["symbol"] = symbol
    df["source"] = (
        "yahoo_chart_api"
    )

    return df
