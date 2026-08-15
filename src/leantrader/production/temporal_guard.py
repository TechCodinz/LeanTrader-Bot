from __future__ import annotations

import datetime as dt
import math
import time
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import pandas as pd

from .exchange_intelligence import timeframe_seconds


class MarketTemporalGuard:
    """Exchange-clock, closed-candle, staleness and market-session safeguards."""

    VERSION = "1.0"
    FX_ROLLOVER_ZONE = ZoneInfo("America/New_York")
    FX_ASSETS: ClassVar[set[str]] = {
        "AUD",
        "CAD",
        "CHF",
        "CNH",
        "EUR",
        "GBP",
        "HKD",
        "JPY",
        "MXN",
        "NOK",
        "NZD",
        "SEK",
        "SGD",
        "TRY",
        "USD",
        "XAG",
        "XAU",
        "ZAR",
    }
    CRYPTO_QUOTES: ClassVar[set[str]] = {
        "BTC",
        "DAI",
        "ETH",
        "EURT",
        "FDUSD",
        "TUSD",
        "USDC",
        "USDD",
        "USDP",
        "USDT",
    }

    def __init__(
        self,
        feed: Any,
        *,
        max_clock_offset_ms: float = 5_000.0,
        clock_sync_seconds: int = 300,
        candle_stale_multiplier: float = 2.5,
    ) -> None:
        self.feed = feed
        self.max_clock_offset_ms = max_clock_offset_ms
        self.clock_sync_seconds = clock_sync_seconds
        self.candle_stale_multiplier = candle_stale_multiplier
        self.clock: dict[str, Any] = {
            "supported": False,
            "verified": False,
            "safe": False,
            "reason": "not_checked",
        }
        self.candles: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.sync_attempts = 0
        self.sync_failures = 0
        self.incomplete_candles_dropped = 0
        self.stale_rejections = 0

    def start(self) -> None:
        self.sync_clock(force=True)

    def sync_clock(self, *, force: bool = False) -> dict[str, Any]:
        last_sync = float(self.clock.get("checked_at_epoch") or 0.0)
        if not force and last_sync and time.time() - last_sync < self.clock_sync_seconds:
            return dict(self.clock)
        self.sync_attempts += 1
        exchange = getattr(self.feed, "exchange", None)
        advertised = getattr(exchange, "has", {}) or {}
        fetch_time = getattr(exchange, "fetch_time", None)
        if exchange is None or advertised.get("fetchTime") is not True or not callable(fetch_time):
            self.clock = {
                "supported": False,
                "verified": False,
                "safe": False,
                "reason": "exchange_server_time_unsupported",
                "checked_at_epoch": time.time(),
            }
            return dict(self.clock)
        before_ms = time.time() * 1_000.0
        try:
            server_ms = float(fetch_time())
            after_ms = time.time() * 1_000.0
            if not math.isfinite(server_ms) or server_ms <= 0:
                raise ValueError("exchange returned an invalid server timestamp")
            midpoint_ms = (before_ms + after_ms) / 2.0
            offset_ms = server_ms - midpoint_ms
            safe = abs(offset_ms) <= self.max_clock_offset_ms
            self.clock = {
                "supported": True,
                "verified": True,
                "safe": safe,
                "reason": "within_tolerance" if safe else "clock_offset_exceeds_limit",
                "server_epoch_ms": server_ms,
                "offset_ms": offset_ms,
                "round_trip_ms": after_ms - before_ms,
                "max_offset_ms": self.max_clock_offset_ms,
                "checked_at_epoch": after_ms / 1_000.0,
            }
        except Exception as exc:  # noqa: BLE001 - a failed public clock probe is reported and fail-closed
            self.sync_failures += 1
            self.clock = {
                "supported": True,
                "verified": False,
                "safe": False,
                "reason": "clock_probe_failed",
                "last_error": f"{type(exc).__name__}: {exc}",
                "checked_at_epoch": time.time(),
            }
        return dict(self.clock)

    def adjusted_now_ms(self) -> float:
        offset = float(self.clock.get("offset_ms") or 0.0) if self.clock.get("verified") else 0.0
        return time.time() * 1_000.0 + offset

    def filter_closed_candles(
        self,
        frame: pd.DataFrame,
        timeframe: str,
        *,
        now_ms: float | None = None,
        source_requires_timestamp: bool = True,
    ) -> pd.DataFrame:
        """Return only exchange-confirmed closed candles and reject stale series."""
        if "timestamp" not in frame.columns:
            if source_requires_timestamp:
                raise ValueError("exchange candle data is missing timestamp")
            return frame.copy()
        seconds = timeframe_seconds(timeframe)
        duration_ms = seconds * 1_000.0
        current_ms = self.adjusted_now_ms() if now_ms is None else float(now_ms)
        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
        if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
            raise ValueError(f"{timeframe} candle timestamps are invalid or unordered")
        closed_mask = timestamps + duration_ms <= current_ms
        filtered = frame.loc[closed_mask].copy()
        dropped = len(frame) - len(filtered)
        self.incomplete_candles_dropped += dropped
        if filtered.empty:
            raise ValueError(f"{timeframe} has no closed candles")
        latest_open_ms = float(pd.to_numeric(filtered["timestamp"], errors="raise").iloc[-1])
        age_since_close_ms = current_ms - (latest_open_ms + duration_ms)
        stale_limit_ms = duration_ms * self.candle_stale_multiplier
        stale = age_since_close_ms > stale_limit_ms
        self.candles[timeframe] = {
            "rows_received": len(frame),
            "rows_closed": len(filtered),
            "incomplete_dropped": dropped,
            "latest_open_epoch_ms": latest_open_ms,
            "latest_close_epoch_ms": latest_open_ms + duration_ms,
            "age_since_close_seconds": max(0.0, age_since_close_ms / 1_000.0),
            "stale_limit_seconds": stale_limit_ms / 1_000.0,
            "stale": stale,
            "checked_at_epoch": current_ms / 1_000.0,
        }
        if stale:
            self.stale_rejections += 1
            raise ValueError(
                f"{timeframe} closed candle is stale by {age_since_close_ms / 1_000.0:.1f} seconds"
            )
        return filtered.reset_index(drop=True)

    def session_status(
        self,
        symbol: str,
        *,
        market_rules: dict[str, Any] | None = None,
        when: dt.datetime | None = None,
    ) -> dict[str, Any]:
        current = when or dt.datetime.now(dt.UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt.UTC)
        current = current.astimezone(dt.UTC)
        base, quote = self._base_quote(symbol)
        rules = market_rules or {}
        if rules.get("available") and rules.get("active") is False:
            result = self._session_result(current, False, "inactive_instrument", "exchange_rules")
        elif base in self.FX_ASSETS and quote in self.FX_ASSETS and quote not in self.CRYPTO_QUOTES:
            result = self.forex_session_status(current)
        elif quote in self.CRYPTO_QUOTES or any(rules.get(name) for name in ("spot", "swap", "future", "option")):
            result = self._session_result(current, True, "continuous_market", "crypto_24_7")
        else:
            result = self._session_result(current, False, "market_calendar_unattested", "unknown")
        result["symbol"] = symbol
        result["base"] = base
        result["quote"] = quote
        self.sessions[symbol] = result
        return dict(result)

    @classmethod
    def forex_session_status(cls, when: dt.datetime) -> dict[str, Any]:
        utc = when.astimezone(dt.UTC)
        local = utc.astimezone(cls.FX_ROLLOVER_ZONE)
        minutes = local.hour * 60 + local.minute
        weekday = local.weekday()
        rollover_start = 16 * 60 + 55
        rollover_end = 17 * 60 + 10
        if weekday == 5:
            allowed, reason = False, "forex_weekend"
        elif weekday == 6:
            allowed, reason = (minutes >= rollover_end), "forex_week_open" if minutes >= rollover_end else "forex_weekend"
        elif weekday == 4 and minutes >= rollover_start:
            allowed, reason = False, "forex_week_close"
        elif rollover_start <= minutes < rollover_end:
            allowed, reason = False, "forex_daily_rollover_buffer"
        else:
            allowed, reason = True, "forex_session_open"
        return {
            **cls._session_result(utc, allowed, reason, "forex_new_york_rollover"),
            "market_timezone": str(cls.FX_ROLLOVER_ZONE),
            "market_local_time": local.isoformat(),
            "dst_active": bool(local.dst()),
        }

    @staticmethod
    def _session_result(when: dt.datetime, allowed: bool, reason: str, session: str) -> dict[str, Any]:
        return {
            "allowed": allowed,
            "reason": reason,
            "session": session,
            "utc_time": when.astimezone(dt.UTC).isoformat(),
        }

    @staticmethod
    def _base_quote(symbol: str) -> tuple[str, str]:
        raw = symbol.upper().split(":", 1)[0].replace("_", "/").replace("-", "/")
        if "/" in raw:
            base, quote = raw.split("/", 1)
            return base, quote
        compact = "".join(character for character in raw if character.isalnum())
        return (compact[:3], compact[3:6]) if len(compact) == 6 else (compact, "")

    def testnet_clock_safe(self) -> bool:
        exchange = getattr(self.feed, "exchange", None)
        if exchange is None:
            return True
        return bool(self.clock.get("supported") and self.clock.get("verified") and self.clock.get("safe"))

    def health(self) -> dict[str, Any]:
        return {
            "utc_internal": True,
            "dst_aware_forex": True,
            "forex_rollover_timezone": str(self.FX_ROLLOVER_ZONE),
            "closed_candles_only": True,
            "stale_candle_rejection": True,
            "clock": dict(self.clock),
            "clock_sync_seconds": self.clock_sync_seconds,
            "candle_stale_multiplier": self.candle_stale_multiplier,
            "candle_timeframes": {key: dict(value) for key, value in self.candles.items()},
            "sessions": {key: dict(value) for key, value in self.sessions.items()},
            "sync_attempts": self.sync_attempts,
            "sync_failures": self.sync_failures,
            "incomplete_candles_dropped": self.incomplete_candles_dropped,
            "stale_rejections": self.stale_rejections,
            "execution_authority": False,
        }
