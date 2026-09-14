from __future__ import annotations

from dataclasses import asdict
import threading
import time
from typing import Any, Mapping

import pandas as pd

from .movement_profiler import MarketMovementProfiler, MovementProfile
from .opportunity_radar import OpportunityScore
from .swarm_orchestrator import MarketSwarmOrchestrator


class FastSwarmRuntime:
    """Fast read-only profiling seam for the market swarm.

    It transforms already-fetched discovery economics + recent candle frames
    into ranked opportunities and specialist observer agents. It deliberately
    does not place orders or allocate capital from movement alone; agent-specific
    directional evidence must independently qualify before any tranche join.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        swarm: MarketSwarmOrchestrator | None = None,
        profiler: MarketMovementProfiler | None = None,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        adverse_selection_bps: float = 2.0,
        max_ranked_opportunities: int = 12,
        max_observer_symbols: int = 4,
    ) -> None:
        if max_ranked_opportunities < 1:
            raise ValueError("max_ranked_opportunities must be positive")
        if max_observer_symbols < 1:
            raise ValueError("max_observer_symbols must be positive")
        self.swarm = swarm or MarketSwarmOrchestrator()
        self.profiler = profiler or MarketMovementProfiler()
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.adverse_selection_bps = float(adverse_selection_bps)
        self.max_ranked_opportunities = int(max_ranked_opportunities)
        self.max_observer_symbols = int(max_observer_symbols)
        self._activated_symbols: set[str] = set()
        self._lock = threading.RLock()
        self.cycles = 0
        self.last_started_at: float | None = None
        self.last_finished_at: float | None = None
        self.last_error: str | None = None
        self.last_ranked: list[dict[str, Any]] = []
        self.last_profiles: dict[str, dict[str, Any]] = {}
        self.last_rejections: dict[str, str] = {}

    @staticmethod
    def _candidate_map(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in candidates:
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                result[symbol] = dict(row)
        return result

    def evaluate_batch(
        self,
        *,
        candidates: list[dict[str, Any]],
        frames: Mapping[str, pd.DataFrame],
        timeframe_seconds: float = 60.0,
    ) -> dict[str, Any]:
        started = time.time()
        snapshots = []
        profiles: dict[str, MovementProfile] = {}
        rejections: dict[str, str] = {}
        candidate_map = self._candidate_map(candidates)

        for symbol, frame in frames.items():
            normalized = str(symbol).upper()
            candidate = candidate_map.get(normalized)
            if candidate is None:
                rejections[normalized] = "missing_discovery_economics"
                continue
            try:
                profile = self.profiler.profile(
                    symbol=normalized,
                    candles=frame,
                    quote_volume_usd=float(candidate.get("quote_volume_usd") or 0.0),
                    spread_bps=float(candidate.get("spread_bps") or 0.0),
                    nominal_price=float(candidate.get("last") or 0.0),
                    timeframe_seconds=timeframe_seconds,
                )
                profiles[normalized] = profile
                snapshots.append(
                    self.profiler.to_opportunity_snapshot(
                        profile,
                        fee_bps=self.fee_bps,
                        slippage_bps=self.slippage_bps,
                        adverse_selection_bps=self.adverse_selection_bps,
                    )
                )
            except (TypeError, ValueError) as exc:
                self.profiler.rejected += 1
                rejections[normalized] = f"{type(exc).__name__}: {exc}"

        ranked = self.swarm.scan(snapshots)[: self.max_ranked_opportunities]
        activated: list[str] = []
        observer_budget = self.max_observer_symbols
        for opportunity in ranked:
            if not opportunity.qualified or observer_budget <= 0:
                continue
            if opportunity.symbol in self._activated_symbols:
                continue
            self.swarm.activate_specialists(opportunity)
            self._activated_symbols.add(opportunity.symbol)
            activated.append(opportunity.symbol)
            observer_budget -= 1

        finished = time.time()
        with self._lock:
            self.cycles += 1
            self.last_started_at = started
            self.last_finished_at = finished
            self.last_error = None
            self.last_profiles = {symbol: profile.as_dict() for symbol, profile in profiles.items()}
            self.last_ranked = [score.as_dict() for score in ranked]
            self.last_rejections = rejections
        return {
            "profiles": dict(self.last_profiles),
            "ranked": list(self.last_ranked),
            "activated_observer_symbols": activated,
            "rejections": dict(rejections),
            "duration_seconds": max(0.0, finished - started),
            "movement_only_can_allocate_capital": False,
            "requires_independent_agent_qualification": True,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def health(self, *, equity: float) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "healthy": self.last_error is None,
                "cycles": self.cycles,
                "last_started_at": self.last_started_at,
                "last_finished_at": self.last_finished_at,
                "ranked_opportunities": list(self.last_ranked),
                "profiled_symbols": len(self.last_profiles),
                "rejections": dict(self.last_rejections),
                "activated_observer_symbols": sorted(self._activated_symbols),
                "profiler": self.profiler.health(),
                "swarm": self.swarm.health(equity=equity),
                "cadence_role": "seconds_to_minutes",
                "movement_only_can_allocate_capital": False,
                "requires_independent_agent_qualification": True,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
