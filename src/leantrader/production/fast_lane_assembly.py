"""Construct the mature fast market-intelligence service and velocity lane.

The historical runner built these inside a 1,480-line PaperRunner subclass that
also owned settings, supervision, a paper portfolio and the v1.60.x patch
chain. The pieces themselves never needed any of that: ReadOnlySwarmService
takes an injected feed, and the fast lane takes an injected service and an
injected executor. This module supplies those three seams and nothing else, so
the same components the profitable run used can be started from the current
orchestrator's lifecycle rather than from a second top-level runner.

Parameters below are the historical ones. Where a value is tunable by
environment variable the default is what the mature runner used; nothing is
loosened here to make trades happen.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..agents.capital_allocator import SwarmCapitalAllocator
from ..agents.fast_path import FastSwarmRuntime
from ..agents.micro_calibration import MicroCalibrationJournal
from ..agents.microstructure_sniper import MicroAgentFoundry, UltraMicrostructureSniper
from ..agents.shared_position_graph import PositionCoordinator
from ..agents.swarm_evidence import SwarmOutcomeJournal
from ..agents.swarm_orchestrator import MarketSwarmOrchestrator
from ..agents.swarm_service import ReadOnlySwarmService
from ..agents.swarm_shadow_portfolio import SwarmShadowPortfolio
from .capital_growth import CapitalGrowthGovernor
from .fast_lane_executor import CentralAuthorityExecutor
from .market_feed import MarketFeed, MicrostructureMarketFeed
from .velocity_sniper_testnet import VelocitySniperTestnetLane


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "") or default))
    except (TypeError, ValueError):
        return int(default)


def state_dir() -> Path:
    """Where the fast lane keeps its state, alongside the other runtime state."""
    shared = Path(os.getenv("LEANTRADER_DATA_DIR", "/app/data"))
    if shared.is_dir() and os.access(shared, os.W_OK):
        target = shared / "runtime"
    else:
        target = Path("runtime")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        target = Path("runtime")
        target.mkdir(parents=True, exist_ok=True)
    return target


def build_fast_swarm_service(
    *,
    data_venue: str = "",
    quote: str = "USDT",
    base_order_usd: float = 2.0,
) -> ReadOnlySwarmService:
    """The historical read-only market-intelligence service.

    ``data_venue`` is a public data venue and carries no credentials. It is
    intentionally separate from the execution venue: market intelligence must
    keep running at full cadence regardless of which environment holds order
    authority.
    """
    venue = (data_venue or os.getenv("MARKET_DATA_VENUE", "") or "bybit").strip().lower()

    feed = MarketFeed(venue)
    calibration_feed = MicrostructureMarketFeed(venue)
    microstream_feed = MicrostructureMarketFeed(venue)
    precision_scout_feed = MarketFeed(venue)

    reference_feed: Optional[MarketFeed] = None
    if venue != "okx":
        try:
            reference_feed = MarketFeed("okx")
        except Exception:
            reference_feed = None

    base = state_dir()

    coordinator = PositionCoordinator(
        max_symbol_exposure_fraction=_env_float("FAST_MAX_SYMBOL_EXPOSURE", 0.20),
        max_portfolio_exposure_fraction=_env_float("FAST_MAX_PORTFOLIO_EXPOSURE", 0.70),
        state_path=base / "fast_swarm_positions.json",
    )
    allocator = SwarmCapitalAllocator(
        coordinator,
        profit_reinvest_fraction=_env_float("CAPITAL_PROFIT_REINVEST_FRACTION", 0.50),
    )
    swarm = MarketSwarmOrchestrator(coordinator=coordinator, allocator=allocator)

    fee_bps = _env_float("FAST_FEE_BPS", 10.0)
    slippage_bps = _env_float("FAST_SLIPPAGE_BPS", 5.0)

    runtime = FastSwarmRuntime(
        swarm=swarm,
        fee_bps=2.0 * fee_bps,
        slippage_bps=2.0 * slippage_bps,
        adverse_selection_bps=0.0,
        max_ranked_opportunities=_env_int("FAST_MAX_RANKED_OPPORTUNITIES", 24),
        max_observer_symbols=_env_int("FAST_MAX_OBSERVER_SYMBOLS", 8),
    )

    shadow_portfolio = SwarmShadowPortfolio(
        base / "fast_swarm_shadow_portfolio.json",
        starting_equity=_env_float("FAST_STARTING_EQUITY", 50.0),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        principal_floor_fraction=_env_float("CAPITAL_PRINCIPAL_FLOOR_FRACTION", 0.70),
        profit_reinvest_fraction=_env_float("CAPITAL_PROFIT_REINVEST_FRACTION", 0.50),
    )

    round_trip_cost_bps = _env_float("FAST_ROUND_TRIP_COST_BPS", 30.0)

    return ReadOnlySwarmService(
        feed=feed,
        runtime=runtime,
        market_quote=str(quote).upper(),
        min_quote_volume_usd=_env_float("MARKET_MIN_QUOTE_VOLUME_USD", 5_000_000.0),
        max_spread_bps=_env_float("MARKET_MAX_SPREAD_BPS", 25.0),
        scan_batch_size=_env_int("MARKET_SCAN_BATCH_SIZE", 24),
        candle_limit=_env_int("FAST_CANDLE_LIMIT", 90),
        # The service's own scan cadence. The velocity lane below runs far
        # faster than this; the service is what keeps the ranked universe and
        # the microstream fresh underneath it.
        cadence_seconds=_env_float("FAST_SERVICE_CADENCE_SECONDS", 5.0),
        discovery_refresh_seconds=_env_float("MARKET_REFRESH_SECONDS", 120.0),
        timeframe="1m",
        timeframe_seconds=60.0,
        max_context_symbols=_env_int("FAST_MAX_CONTEXT_SYMBOLS", 4),
        shadow_portfolio=shadow_portfolio,
        outcome_journal=SwarmOutcomeJournal(base / "fast_swarm_outcomes.json"),
        base_order_usd=max(0.01, float(base_order_usd)),
        microstructure_sniper=UltraMicrostructureSniper(
            minimum_modeled_round_trip_cost_bps=round_trip_cost_bps,
            maximum_spread_bps=_env_float("MARKET_MAX_SPREAD_BPS", 25.0),
        ),
        micro_agent_foundry=MicroAgentFoundry(maximum_candidates_per_symbol=2),
        reference_feed=reference_feed,
        max_micro_symbols=_env_int("FAST_MAX_MICRO_SYMBOLS", 8),
        micro_calibration_journal=MicroCalibrationJournal(
            base / "fast_micro_calibration.json"
        ),
        micro_calibration_feed=calibration_feed,
        microstream_feed=microstream_feed,
        slow_calibration_journal=MicroCalibrationJournal(
            base / "fast_slow_calibration.json"
        ),
        precision_scout_feed=precision_scout_feed,
        precision_scout_refresh_seconds=_env_float(
            "FAST_PRECISION_SCOUT_REFRESH_SECONDS", 20.0
        ),
        precision_round_trip_cost_bps=round_trip_cost_bps,
    )


def build_velocity_lane(
    *,
    service_provider: Callable[[], Any],
    executor: Any,
    supervisory_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    order_usd: float = 2.0,
) -> VelocitySniperTestnetLane:
    """The sub-second velocity lane, with the mature runner's parameters.

    These are the historical values, not new ones: half-second cadence, a
    thirty-second maximum hold, 50 bps take-profit against a 30 bps stop, and a
    bounded daily entry count. The lane was fast by design -- replacing it with
    a one-minute scan is what made the current runtime stop behaving like the
    system that traded.
    """
    supervisory_provider = supervisory_provider or (lambda: {})

    return VelocitySniperTestnetLane(
        service_provider=service_provider,
        testnet=executor,
        state_path=state_dir() / "velocity_sniper_lane.json",
        supervisory_provider=supervisory_provider,
        order_usd=order_usd,
        round_trip_cost_bps=_env_float("FAST_ROUND_TRIP_COST_BPS", 30.0),
        cadence_seconds=_env_float("VELOCITY_CADENCE_SECONDS", 0.5),
        maximum_hold_seconds=_env_float("VELOCITY_MAX_HOLD_SECONDS", 30.0),
        take_profit_bps=_env_float("VELOCITY_TAKE_PROFIT_BPS", 50.0),
        stop_loss_bps=_env_float("VELOCITY_STOP_LOSS_BPS", 30.0),
        maximum_entries_per_day=_env_int("VELOCITY_MAX_ENTRIES_PER_DAY", 45),
        bootstrap_after_seconds=_env_float("VELOCITY_BOOTSTRAP_AFTER_SECONDS", 5.0),
        maximum_concurrent_positions=_env_int("VELOCITY_MAX_POSITIONS", 6),
        maximum_adaptive_positions=_env_int("VELOCITY_MAX_ADAPTIVE_POSITIONS", 24),
        maximum_entries_per_cycle=_env_int("VELOCITY_MAX_ENTRIES_PER_CYCLE", 3),
        maximum_adaptive_entries_per_cycle=_env_int(
            "VELOCITY_MAX_ADAPTIVE_ENTRIES_PER_CYCLE", 8
        ),
        candidate_scan_limit=_env_int("VELOCITY_CANDIDATE_SCAN_LIMIT", 24),
        reentry_cooldown_seconds=_env_float("VELOCITY_REENTRY_COOLDOWN_SECONDS", 2.0),
        maximum_order_usd=_env_float("VELOCITY_MAX_ORDER_USD", 5.0),
    )


class FastTradingLane:
    """Service, executor, lane and capital governor as one startable unit.

    This is an assembly, not a new engine: every component below already
    existed. It exists so the orchestrator has one thing to start and stop
    instead of four, and so the wiring is in one readable place.
    """

    def __init__(
        self,
        *,
        quote: str = "USDT",
        order_usd: float = 2.0,
        data_venue: str = "",
        executor: Optional[Any] = None,
        service: Optional[ReadOnlySwarmService] = None,
    ) -> None:
        self.quote = str(quote).upper()
        self.order_usd = max(0.01, float(order_usd))
        self.data_venue = data_venue

        self.executor = executor or CentralAuthorityExecutor(
            quote=self.quote,
            max_order_usd=_env_float("VELOCITY_MAX_ORDER_USD", 5.0),
        )
        self.service: Optional[ReadOnlySwarmService] = service
        self.lane: Optional[VelocitySniperTestnetLane] = None
        self.governor = CapitalGrowthGovernor(
            state_dir() / "fast_capital_growth.json",
            starting_equity=_env_float("FAST_STARTING_EQUITY", 50.0),
            principal_floor_fraction=_env_float(
                "CAPITAL_PRINCIPAL_FLOOR_FRACTION", 0.70
            ),
            profit_reinvest_fraction=_env_float(
                "CAPITAL_PROFIT_REINVEST_FRACTION", 0.50
            ),
        )
        self._started = False

    def build(self) -> "FastTradingLane":
        if self.service is None:
            self.service = build_fast_swarm_service(
                data_venue=self.data_venue,
                quote=self.quote,
                base_order_usd=self.order_usd,
            )
        if self.lane is None:
            self.lane = build_velocity_lane(
                service_provider=lambda: self.service,
                executor=self.executor,
                supervisory_provider=self.supervisory_state,
                order_usd=self.order_usd,
            )
        return self

    def supervisory_state(self) -> Dict[str, Any]:
        """What the capital governor currently permits.

        The lane consults this before entering. It is the same governor the
        historical runner used, reading real account equity rather than a
        remembered number.
        """
        snapshot = self.executor.safe_snapshot()
        free_quote = float(snapshot.get("free_quote") or 0.0)
        portfolio_equity = float(snapshot.get("portfolio_equity") or 0.0)
        equity = portfolio_equity if portfolio_equity > 0.0 else free_quote

        try:
            verdict = self.governor.evaluate(
                equity=equity,
                open_notional=0.0,
            )
        except Exception:
            return {}

        healthy = bool(snapshot.get("fresh"))

        return {
            "timestamp": time.time(),
            "healthy": healthy,
            "halt_reason": (
                None
                if healthy
                else "fast_executor_account_unreadable"
            ),
            "required_failures": (
                []
                if healthy
                else ["fast_executor_account_unreadable"]
            ),
            "symbols": {},
            "canonical_open_positions": sorted(
                (snapshot.get("positions") or {}).keys()
            ),
            "capital_growth": verdict,
            "capital": verdict,
            "environment": snapshot.get("environment"),
        }

    def start(self) -> None:
        """Start market intelligence, then the lane that consumes it.

        Both own their own threads, which is how the historical system ran
        them: the service keeps the ranked universe and microstream fresh
        while the lane runs its half-second cycle against whatever is current.
        Order matters -- a lane started before its service has nothing fresh
        to qualify, and the lane's bootstrap delay exists for exactly that.
        """
        self.build()
        if self._started:
            return
        if self.service is not None:
            self.service.start()
        if self.lane is not None:
            self.lane.start()
        self._started = True

    def stop(self) -> None:
        if self.lane is not None:
            try:
                self.lane.stop()
            except Exception:
                pass
        if self.service is not None:
            try:
                self.service.stop()
            except Exception:
                pass
        self._started = False

    def running(self) -> bool:
        """Whether both threads are actually alive, not merely constructed."""
        lane_thread = getattr(self.lane, "_thread", None)
        return bool(
            self._started
            and lane_thread is not None
            and lane_thread.is_alive()
        )

    def step(self) -> Dict[str, Any]:
        """One lane cycle. Returns whatever decision the lane reached."""
        if self.lane is None:
            self.build()
        return self.lane.step() or {}

    def health(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "started": self._started,
            "service": bool(self.service is not None),
            "lane": bool(self.lane is not None),
        }
        try:
            state["executor"] = self.executor.health()
        except Exception:
            state["executor"] = {}
        if self.service is not None:
            try:
                state["service_health"] = self.service.health(
                    equity=float(
                        (state.get("executor") or {}).get("free_quote") or 0.0
                    )
                )
            except Exception:
                state["service_health"] = {}
        return state
