from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd

from .fast_path import FastSwarmRuntime


class ReadOnlySwarmService:
    """Parallel market-scouting service with no trading authority.

    Uses a dedicated public/read-only feed so market scouting is not serialized
    behind the slower production intelligence/evidence cycle. The service only
    profiles and ranks opportunities plus activates observer specialists.
    Capital/tranche decisions remain gated elsewhere.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        feed: Any,
        runtime: FastSwarmRuntime,
        market_quote: str,
        min_quote_volume_usd: float,
        max_spread_bps: float,
        scan_batch_size: int = 12,
        candle_limit: int = 90,
        cadence_seconds: float = 15.0,
        discovery_refresh_seconds: float = 60.0,
        timeframe: str = "1m",
        timeframe_seconds: float = 60.0,
    ) -> None:
        if scan_batch_size < 1:
            raise ValueError("scan_batch_size must be positive")
        if candle_limit < 32:
            raise ValueError("candle_limit must be at least 32")
        if cadence_seconds < 1.0:
            raise ValueError("cadence_seconds must be at least one second")
        if discovery_refresh_seconds < cadence_seconds:
            raise ValueError("discovery refresh cannot be faster than service cadence")
        self.feed = feed
        self.runtime = runtime
        self.market_quote = str(market_quote).upper()
        self.min_quote_volume_usd = float(min_quote_volume_usd)
        self.max_spread_bps = float(max_spread_bps)
        self.scan_batch_size = int(scan_batch_size)
        self.candle_limit = int(candle_limit)
        self.cadence_seconds = float(cadence_seconds)
        self.discovery_refresh_seconds = float(discovery_refresh_seconds)
        self.timeframe = str(timeframe)
        self.timeframe_seconds = float(timeframe_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._candidates: list[dict[str, Any]] = []
        self._candidate_map: dict[str, dict[str, Any]] = {}
        self._cursor = 0
        self._last_discovery_at = 0.0
        self.cycles = 0
        self.full_sweeps = 0
        self.last_error: str | None = None
        self.last_step: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="leantrader-market-swarm", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, min(10.0, self.cadence_seconds + 1.0)))

    def _refresh_discovery(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and self._candidates and now - self._last_discovery_at < self.discovery_refresh_seconds:
            return
        payload = self.feed.discover_markets(
            quote=self.market_quote,
            min_quote_volume_usd=self.min_quote_volume_usd,
            max_spread_bps=self.max_spread_bps,
        )
        candidates = [dict(row) for row in payload.get("candidates") or []]
        if not candidates:
            raise RuntimeError("fast swarm discovery returned no eligible markets")
        previous = None
        if self._candidates:
            previous = str(self._candidates[self._cursor % len(self._candidates)].get("symbol") or "").upper()
        self._candidates = candidates
        self._candidate_map = {
            str(row.get("symbol") or "").upper(): row for row in candidates if row.get("symbol")
        }
        symbols = [str(row.get("symbol") or "").upper() for row in candidates]
        self._cursor = symbols.index(previous) if previous in symbols else 0
        self._last_discovery_at = now

    def _next_candidates(self) -> list[dict[str, Any]]:
        if not self._candidates:
            raise RuntimeError("fast swarm candidates are unavailable")
        take = min(self.scan_batch_size, len(self._candidates))
        selected = [self._candidates[(self._cursor + offset) % len(self._candidates)] for offset in range(take)]
        if self._cursor + take >= len(self._candidates):
            self.full_sweeps += 1
        self._cursor = (self._cursor + take) % len(self._candidates)
        return [dict(row) for row in selected]

    @staticmethod
    def _closed_candles(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("market feed candles must be a DataFrame")
        # Public OHLCV endpoints generally include the currently-forming bar.
        # The fast path never profiles that bar as completed evidence.
        if len(frame) < 2:
            return frame.iloc[0:0].copy()
        return frame.iloc[:-1].copy()

    def step(self) -> dict[str, Any]:
        started = time.time()
        self._refresh_discovery()
        selected = self._next_candidates()
        frames: dict[str, pd.DataFrame] = {}
        fetch_errors: dict[str, str] = {}
        for candidate in selected:
            symbol = str(candidate.get("symbol") or "").upper()
            if not symbol:
                continue
            try:
                raw = self.feed.candles(symbol, self.timeframe, self.candle_limit)
                frames[symbol] = self._closed_candles(raw)
            except Exception as exc:  # noqa: BLE001 - one symbol must not halt the scouting sweep
                fetch_errors[symbol] = f"{type(exc).__name__}: {exc}"

        result = self.runtime.evaluate_batch(
            candidates=selected,
            frames=frames,
            timeframe_seconds=self.timeframe_seconds,
        )
        result["fetch_errors"] = fetch_errors
        result["selected_symbols"] = [str(row.get("symbol") or "").upper() for row in selected]
        result["universe_candidates"] = len(self._candidates)
        result["full_sweeps"] = self.full_sweeps
        result["service_duration_seconds"] = max(0.0, time.time() - started)
        result["forming_candle_excluded"] = True
        result["dedicated_read_only_feed"] = True
        result["execution_authority"] = False
        result["testnet_authority"] = False
        result["live_authority"] = False
        with self._lock:
            self.cycles += 1
            self.last_step = result
            self.last_error = None
        return dict(result)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001 - optional service remains isolated from core runner
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.cadence_seconds - elapsed))

    def health(self, *, equity: float) -> dict[str, Any]:
        thread = self._thread
        with self._lock:
            return {
                "version": self.VERSION,
                "running": bool(thread is not None and thread.is_alive() and not self._stop.is_set()),
                "healthy": self.last_error is None,
                "cycles": self.cycles,
                "full_sweeps": self.full_sweeps,
                "universe_candidates": len(self._candidates),
                "cursor": self._cursor,
                "cadence_seconds": self.cadence_seconds,
                "timeframe": self.timeframe,
                "forming_candle_excluded": True,
                "dedicated_read_only_feed": True,
                "last_error": self.last_error,
                "last_step": dict(self.last_step),
                "runtime": self.runtime.health(equity=equity),
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
