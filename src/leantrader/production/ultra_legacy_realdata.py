from __future__ import annotations

import copy
import threading
import time
from typing import Any


class UltraLegacyRealDataHub:
    """
    Compatibility evidence hub for historical Ultra systems.

    It consumes observations already produced by LeanTrader's canonical
    real-data cycle. It never fabricates prices, balances, PnL, signals,
    whale flows, APY, or execution results.
    """

    VERSION = "1.62.2a"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest: dict[str, Any] = {}

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def observe_cycle(
        self,
        *,
        symbols: list[str],
        frames: dict[str, Any],
        sensor_snapshot: dict[str, Any],
        public_symbol_context: dict[str, Any],
        arbitrage_collection: dict[str, Any],
        advanced_market: dict[str, Any],
    ) -> dict[str, Any]:
        sensor_symbols = (
            sensor_snapshot.get("symbols", {})
            if isinstance(sensor_snapshot, dict)
            else {}
        )

        payload = {
            "timestamp": time.time(),
            "symbols": {
                str(symbol): {
                    "public_context": copy.deepcopy(
                        public_symbol_context.get(symbol, {})
                    ),
                    "sensor_context": copy.deepcopy(
                        sensor_symbols.get(symbol, {})
                    ),
                }
                for symbol in symbols
            },
            "moon_scout_ranking": copy.deepcopy(
                advanced_market.get("moon_scout_ranking", [])
                if isinstance(advanced_market, dict)
                else []
            )[:50],
            "arbitrage_opportunities": copy.deepcopy(
                advanced_market.get("arbitrage_opportunities", [])
                if isinstance(advanced_market, dict)
                else []
            )[:50],
            "cross_venue_quotes": copy.deepcopy(
                arbitrage_collection.get("quotes", [])
                if isinstance(arbitrage_collection, dict)
                else []
            )[:100],
            "frame_symbols": sorted(str(symbol) for symbol in frames),
            "real_data": True,
            "synthetic_market_data": False,
            "execution_authority": False,
            "live_authority": False,
        }

        with self._lock:
            self._latest = payload

        return self.snapshot()

    def symbol_context(self, symbol: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(
                (self._latest.get("symbols") or {}).get(
                    str(symbol).upper(),
                    {},
                )
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = copy.deepcopy(self._latest)

        return {
            "version": self.VERSION,
            "canonical_cycle": latest,
            "legacy_random_fallbacks": False,
            "fake_execution": False,
            "fake_backtest_results": False,
            "fake_account_balances": False,
            "testnet_is_system_identity": False,
            "system_identity": "leantrader",
            "execution_authority": False,
            "live_authority": False,
        }

    def health(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        cycle = snapshot.get("canonical_cycle") or {}

        return {
            "version": self.VERSION,
            "healthy": True,
            "symbols_observed": len(cycle.get("symbols") or {}),
            "real_data": True,
            "synthetic_market_data": False,
            "legacy_random_fallbacks": False,
            "execution_authority": False,
            "live_authority": False,
        }
