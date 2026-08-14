from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class MarketUniverse:
    """Persistent, rotating coverage of every eligible market.

    A broad exchange universe cannot be evaluated in one synchronous cycle
    without creating stale decisions and rate-limit pressure.  This engine
    therefore keeps the entire eligible set and advances through it in fair
    batches. Existing positions are always included so exits are never delayed.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        state_path: Path,
        mode: str,
        configured_symbols: tuple[str, ...],
        quote: str,
        batch_size: int,
        refresh_seconds: int,
    ) -> None:
        if mode not in {"configured", "dynamic"}:
            raise ValueError("market universe mode must be configured or dynamic")
        if batch_size < 1:
            raise ValueError("market universe batch size must be positive")
        if refresh_seconds < 60:
            raise ValueError("market universe refresh must be at least 60 seconds")
        self.state_path = state_path
        self.mode = mode
        self.configured_symbols = configured_symbols
        self.quote = quote.upper()
        self.batch_size = batch_size
        self.refresh_seconds = refresh_seconds
        self.symbols: list[str] = []
        self.cursor = 0
        self.full_sweeps = 0
        self.last_refresh_epoch = 0.0
        self.last_scan: list[str] = []
        self.discovered_count = 0
        self.rejection_counts: dict[str, int] = {}
        self._load()

    def start(self) -> None:
        if self.mode == "configured":
            self.symbols = list(dict.fromkeys(self.configured_symbols))
            self.cursor = 0

    def needs_refresh(self) -> bool:
        return self.mode == "dynamic" and (
            not self.symbols or time.time() - self.last_refresh_epoch >= self.refresh_seconds
        )

    def refresh(
        self,
        candidates: list[dict[str, Any]],
        *,
        allowed_symbols: set[str] | None = None,
        rejection_counts: dict[str, int] | None = None,
    ) -> list[str]:
        ranked = [str(candidate["symbol"]).upper() for candidate in candidates]
        if allowed_symbols is not None:
            ranked = [symbol for symbol in ranked if symbol in allowed_symbols]
        ranked = list(dict.fromkeys(ranked))
        if not ranked:
            raise RuntimeError("market discovery returned no eligible symbols")

        previous_next = self.symbols[self.cursor] if self.symbols and self.cursor < len(self.symbols) else None
        self.symbols = ranked
        self.discovered_count = len(candidates)
        self.rejection_counts = dict(rejection_counts or {})
        self.last_refresh_epoch = time.time()
        self.cursor = self.symbols.index(previous_next) if previous_next in self.symbols else 0
        self._save()
        return list(self.symbols)

    def apply_testnet_intersection(self, allowed_symbols: set[str] | None) -> list[str]:
        if allowed_symbols is None:
            return list(self.symbols)
        supported = [symbol for symbol in self.symbols if symbol in allowed_symbols]
        if not supported:
            raise RuntimeError("configured markets are unavailable on Bybit Testnet")
        self.symbols = supported
        self.cursor %= len(self.symbols)
        self._save()
        return list(self.symbols)

    def next_batch(self, *, mandatory_symbols: set[str] | None = None) -> list[str]:
        if not self.symbols:
            raise RuntimeError("market universe has not been populated")
        # Existing positions remain mandatory even if a refreshed exchange
        # universe stops advertising the market; this keeps exit attempts and
        # the resulting operational error visible instead of orphaning risk.
        mandatory = sorted(mandatory_symbols or set())
        take = min(self.batch_size, len(self.symbols))
        selected = [self.symbols[(self.cursor + offset) % len(self.symbols)] for offset in range(take)]
        next_cursor = (self.cursor + take) % len(self.symbols)
        if self.cursor + take >= len(self.symbols):
            self.full_sweeps += 1
        self.cursor = next_cursor
        self.last_scan = list(dict.fromkeys(mandatory + selected))
        self._save()
        return list(self.last_scan)

    def health(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "quote": self.quote,
            "configured_symbols": list(self.configured_symbols),
            "eligible_symbols": len(self.symbols),
            "discovered_candidates": self.discovered_count,
            "scan_batch_size": self.batch_size,
            "last_scan_count": len(self.last_scan),
            "last_scan": list(self.last_scan),
            "cursor": self.cursor,
            "full_sweeps": self.full_sweeps,
            "last_refresh_epoch": self.last_refresh_epoch or None,
            "rejection_counts": dict(self.rejection_counts),
            "all_eligible_markets_rotate": self.mode == "dynamic",
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("mode") != self.mode or str(payload.get("quote", "")).upper() != self.quote:
                return
            self.symbols = [str(symbol).upper() for symbol in payload.get("symbols", [])]
            self.cursor = int(payload.get("cursor", 0))
            self.full_sweeps = int(payload.get("full_sweeps", 0))
            self.last_refresh_epoch = float(payload.get("last_refresh_epoch", 0.0))
            self.discovered_count = int(payload.get("discovered_count", 0))
            self.rejection_counts = {
                str(key): int(value) for key, value in payload.get("rejection_counts", {}).items()
            }
            if self.symbols:
                self.cursor %= len(self.symbols)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.symbols = []
            self.cursor = 0

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "mode": self.mode,
            "quote": self.quote,
            "symbols": self.symbols,
            "cursor": self.cursor,
            "full_sweeps": self.full_sweeps,
            "last_refresh_epoch": self.last_refresh_epoch,
            "discovered_count": self.discovered_count,
            "rejection_counts": self.rejection_counts,
            "updated_at": time.time(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
