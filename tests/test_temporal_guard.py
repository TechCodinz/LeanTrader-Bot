from __future__ import annotations

import datetime as dt
import time
from typing import ClassVar

import pandas as pd
import pytest

from leantrader.production.temporal_guard import MarketTemporalGuard


class ClockExchange:
    has: ClassVar[dict[str, bool]] = {"fetchTime": True}

    def __init__(self, offset_ms: float = 0.0) -> None:
        self.offset_ms = offset_ms

    def fetch_time(self) -> int:
        return int(time.time() * 1_000 + self.offset_ms)


class Feed:
    def __init__(self, offset_ms: float = 0.0) -> None:
        self.exchange = ClockExchange(offset_ms)


def test_exchange_clock_probe_is_measured_and_fail_closed_when_offset_is_excessive():
    safe = MarketTemporalGuard(Feed(), max_clock_offset_ms=5_000)
    safe.start()
    assert safe.health()["clock"]["verified"] is True
    assert safe.health()["clock"]["safe"] is True

    unsafe = MarketTemporalGuard(Feed(30_000), max_clock_offset_ms=5_000)
    unsafe.start()
    assert unsafe.health()["clock"]["safe"] is False
    assert unsafe.testnet_clock_safe() is False


def test_only_closed_fresh_candles_are_returned():
    guard = MarketTemporalGuard(Feed())
    now_ms = 1_800_000.0
    frame = pd.DataFrame(
        {
            "timestamp": [0.0, 900_000.0, 1_800_000.0],
            "open": [1.0, 1.0, 1.0],
            "high": [2.0, 2.0, 2.0],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5],
            "volume": [10.0, 10.0, 10.0],
        }
    )
    closed = guard.filter_closed_candles(frame, "15m", now_ms=now_ms)
    assert closed["timestamp"].tolist() == [0.0, 900_000.0]
    assert guard.health()["incomplete_candles_dropped"] == 1

    with pytest.raises(ValueError, match="stale"):
        guard.filter_closed_candles(frame.iloc[:1], "15m", now_ms=10_000_000.0)


def test_crypto_is_continuous_and_forex_calendar_is_dst_aware():
    guard = MarketTemporalGuard(object())
    saturday = dt.datetime(2026, 8, 15, 12, tzinfo=dt.UTC)
    assert guard.session_status("BTC/USDT", when=saturday)["allowed"] is True
    assert guard.session_status("EUR/USD", when=saturday)["allowed"] is False

    # 17:10 New York is 21:10 UTC in August and 22:10 UTC in January.
    summer_open = dt.datetime(2026, 8, 16, 21, 10, tzinfo=dt.UTC)
    winter_open = dt.datetime(2026, 1, 4, 22, 10, tzinfo=dt.UTC)
    summer = guard.session_status("EUR/USD", when=summer_open)
    winter = guard.session_status("EUR/USD", when=winter_open)
    assert summer["allowed"] is True and summer["dst_active"] is True
    assert winter["allowed"] is True and winter["dst_active"] is False

    friday_close = dt.datetime(2026, 8, 14, 20, 55, tzinfo=dt.UTC)
    assert guard.session_status("XAU/USD", when=friday_close)["reason"] == "forex_week_close"
