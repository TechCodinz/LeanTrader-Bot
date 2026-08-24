from __future__ import annotations

import argparse
import copy
import json
import signal
import threading
import time
from typing import Any

from . import runner_v142 as _runner_v142
from .runner_v142 import *  # noqa: F401,F403
from .runner_v142 import MarketFeed, PaperRunner as _V142PaperRunner, configure_logging, preflight
from .settings import Settings
from .velocity_sniper_testnet import VelocitySniperTestnetLane as FastCollectiveTestnetLane
from ..agents.capital_allocator import SwarmCapitalAllocator
from ..agents.fast_path import FastSwarmRuntime
from ..agents.micro_calibration import MicroCalibrationJournal
from ..agents.microstructure_sniper import MicroAgentFoundry, UltraMicrostructureSniper
from ..agents.shared_position_graph import PositionCoordinator
from ..agents.swarm_evidence import SwarmOutcomeJournal, build_v142_swarm_manifests
from ..agents.swarm_orchestrator import MarketSwarmOrchestrator
from ..agents.swarm_service import ReadOnlySwarmService
from ..agents.swarm_shadow_portfolio import SwarmShadowPortfolio


class MicrostructureMarketFeed(MarketFeed):
    def public_trades(self, symbol: str, limit: int = 80) -> list[dict[str, Any]]:
        self._load_markets()
        if not self.exchange.has.get("fetchTrades", False):
            return []
        rows = self.exchange.fetch_trades(symbol, limit=max(1, min(200, int(limit))))
        return [dict(row) for row in rows or []]


class PaperRunner(_V142PaperRunner):
    """v1.43: v1.42 supervision plus parallel costed market-swarm shadow evidence."""

    VERSION = "1.60.3"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _runner_v142.BybitTestnetExecutionEngine = BybitTestnetExecutionEngine
        super().__init__(*args, **kwargs)
        self.fast_swarm_service: ReadOnlySwarmService | None = None
        base = self.settings.strategy_observatory_state_path
        self.swarm_outcome_journal = SwarmOutcomeJournal(base.with_name("vps_market_swarm_outcomes.json"))
        self.swarm_manifests_registered = False
        self.last_swarm_evidence_ingest: dict[str, Any] = {
            "submitted": 0,
            "episodes_recorded": 0,
            "qualification_refresh": "not_started",
        }
        self._full_market_cycle_completed = False
        self._health_refresh_stop = threading.Event()
        self._health_refresh_thread: threading.Thread | None = None
        self._fast_supervisory_lock = threading.RLock()
        self._fast_supervisory = self._load_fast_supervisory()

        self.fast_collective_testnet = (
            FastCollectiveTestnetLane(
                service_provider=lambda: self.fast_swarm_service,
                testnet=self.testnet,
                state_path=self.settings.testnet_state_path.with_name(
                    "vps_fast_collective_testnet.json"
                ),
                supervisory_provider=self._fast_supervisory_snapshot,
                order_usd=max(
                    0.50,
                    min(1.0, self.settings.order_usd),
                ),
                round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
                cadence_seconds=0.5,
                maximum_hold_seconds=30.0,
                take_profit_bps=50.0,
                stop_loss_bps=30.0,
                maximum_entries_per_day=45,
                bootstrap_after_seconds=5.0,
                # Six is now only the starting baseline. Actual concurrency
                # expands/contracts from capital, market opportunities,
                # daily executor room and current risk.
                maximum_concurrent_positions=6,
                maximum_adaptive_positions=max(
                    6,
                    min(
                        24,
                        self.settings.market_scan_batch_size,
                        max(
                            6,
                            self.settings.testnet_max_orders_per_day
                            // 4,
                        ),
                    ),
                ),
                maximum_entries_per_cycle=3,
                maximum_adaptive_entries_per_cycle=max(
                    3,
                    min(
                        8,
                        max(
                            3,
                            self.settings.market_scan_batch_size
                            // 3,
                        ),
                    ),
                ),
                candidate_scan_limit=max(
                    24,
                    min(
                        48,
                        self.settings.market_scan_batch_size
                        * 2,
                    ),
                ),
                reentry_cooldown_seconds=2.0,
                starting_equity=(
                    self.settings.starting_cash
                ),
                # v1.59.1: the CapitalGrowthGovernor determines the
                # dynamic amount. This local ceiling must not artificially
                # pin profitable growth at $2. The Testnet executor still
                # independently enforces order, position, daily-notional
                # and daily-order hard limits.
                maximum_order_usd=min(
                    self.settings.testnet_max_order_usd,
                    self.settings.testnet_max_position_usd,
                ),
            )
            if self.testnet is not None
            else None
        )

    @staticmethod
    def _extract_fast_supervisory(
        status: dict[str, Any],
    ) -> dict[str, Any]:
        engines = (
            status.get("engines")
            or {}
        )

        required_failures = [
            name
            for name, row in engines.items()
            if (
                isinstance(row, dict)
                and row.get("required") is True
                and row.get("healthy") is not True
            )
        ]

        decisions = (
            status.get("decisions")
            or {}
        )

        collective_symbols = (
            (
                status.get(
                    "collective_profit_fabric"
                )
                or {}
            ).get("symbols")
            or {}
        )

        symbols: dict[str, Any] = {}

        for symbol in (
            set(decisions)
            | set(collective_symbols)
        ):
            decision = (
                decisions.get(symbol)
                or {}
            )

            symbols[str(symbol)] = {
                "route": copy.deepcopy(
                    decision.get("route")
                    or {}
                ),
                "collective": copy.deepcopy(
                    collective_symbols.get(
                        symbol,
                        {},
                    )
                ),
            }

        return {
            "timestamp": float(
                status.get("timestamp")
                or 0.0
            ),
            "healthy": bool(
                status.get("healthy") is True
                and not required_failures
            ),
            "halt_reason": (
                status.get("halt_reason")
            ),
            "equity": float(
                status.get("equity")
                or 0.0
            ),
            "realized_pnl": float(
                status.get(
                    "realized_pnl"
                )
                or 0.0
            ),
            "capital_growth": copy.deepcopy(
                status.get(
                    "capital_growth"
                )
                or {}
            ),
            "required_failures": (
                required_failures
            ),
            "symbols": symbols,
            "canonical_open_positions": list(
                status.get("open_positions")
                or []
            ),
            "live_authority": False,
        }

    def _load_fast_supervisory(
        self,
    ) -> dict[str, Any]:
        path = self.settings.heartbeat_path

        if not path.is_file():
            return {}

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            return (
                self._extract_fast_supervisory(
                    payload
                )
            )

        except Exception:
            return {}

    def _fast_supervisory_snapshot(
        self,
    ) -> dict[str, Any]:
        lock = getattr(
            self,
            "_fast_supervisory_lock",
            None,
        )

        if lock is None:
            lock = threading.RLock()
            self._fast_supervisory_lock = lock

        with lock:
            return copy.deepcopy(
                getattr(
                    self,
                    "_fast_supervisory",
                    {},
                )
            )

    def _update_fast_supervisory(
        self,
        status: dict[str, Any],
    ) -> None:
        snapshot = (
            self._extract_fast_supervisory(
                status
            )
        )

        lock = getattr(
            self,
            "_fast_supervisory_lock",
            None,
        )

        if lock is None:
            lock = threading.RLock()
            self._fast_supervisory_lock = lock

        with lock:
            self._fast_supervisory = (
                snapshot
            )

    def _swarm_round_trip_cost_bps(self) -> float:
        return max(30.0, 2.0 * (self.settings.fee_bps + self.settings.slippage_bps))

    def _register_swarm_research_manifests(self) -> dict[str, Any]:
        manifests = build_v142_swarm_manifests(
            minimum_samples=self.settings.evolution_min_shadow_samples,
            round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
        )
        result = self.prospective_validation.observe_cycle(
            observatory_authority=self.prospective_validation.EVIDENCE_AUTHORITY,
            observed_round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
            strategy_episodes=[],
            foundry_manifests=manifests,
            market_rows={},
        )
        self.swarm_manifests_registered = True
        return {
            "requested": len(manifests),
            "registered_now": len(result.get("experiments_registered") or []),
            "prospective_only": True,
            "freeze_before_outcome": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def _build_fast_swarm_service(self) -> ReadOnlySwarmService:
        dedicated_feed = MarketFeed(self.settings.exchange)
        calibration_feed = MicrostructureMarketFeed(
            self.settings.exchange
        )
        microstream_feed = MicrostructureMarketFeed(
            self.settings.exchange
        )
        precision_scout_feed = MarketFeed(
            self.settings.exchange
        )
        reference_feed = None
        if (
            str(self.settings.exchange).lower() != "okx"
            and hasattr(dedicated_feed, "order_book")
            and hasattr(dedicated_feed, "exchange")
        ):
            try:
                reference_feed = MarketFeed("okx")
            except Exception:
                reference_feed = None
        base = self.settings.strategy_observatory_state_path
        symbol_cap = max(0.01, min(0.20, float(self.settings.max_position_pct)))
        portfolio_cap = max(
            symbol_cap,
            min(0.70, float(self.settings.max_position_pct) * max(1, int(self.settings.max_open_positions))),
        )
        coordinator = PositionCoordinator(
            max_symbol_exposure_fraction=symbol_cap,
            max_portfolio_exposure_fraction=portfolio_cap,
            state_path=base.with_name("vps_market_swarm_positions.json"),
        )
        allocator = SwarmCapitalAllocator(
            coordinator,
            profit_reinvest_fraction=self.settings.capital_profit_reinvest_fraction,
        )
        swarm = MarketSwarmOrchestrator(coordinator=coordinator, allocator=allocator)
        runtime = FastSwarmRuntime(
            swarm=swarm,
            fee_bps=2.0 * self.settings.fee_bps,
            slippage_bps=2.0 * self.settings.slippage_bps,
            adverse_selection_bps=0.0,
            max_ranked_opportunities=max(4, min(24, self.settings.market_scan_batch_size)),
            max_observer_symbols=max(2, min(8, self.settings.max_open_positions * 2)),
        )
        shadow_portfolio = SwarmShadowPortfolio(
            base.with_name("vps_market_swarm_shadow_portfolio.json"),
            starting_equity=self.settings.starting_cash,
            fee_bps=self.settings.fee_bps,
            slippage_bps=self.settings.slippage_bps,
            principal_floor_fraction=self.settings.capital_principal_floor_fraction,
            profit_reinvest_fraction=self.settings.capital_profit_reinvest_fraction,
        )
        cadence = max(
            3.0,
            min(
                8.0,
                float(
                    self.settings.poll_seconds
                )
                / 8.0,
            ),
        )
        return ReadOnlySwarmService(
            feed=dedicated_feed,
            runtime=runtime,
            market_quote=self.settings.market_quote,
            min_quote_volume_usd=self.settings.market_min_quote_volume_usd,
            max_spread_bps=self.settings.market_max_spread_bps,
            scan_batch_size=max(1, self.settings.market_scan_batch_size),
            candle_limit=max(48, min(120, self.settings.candle_limit)),
            cadence_seconds=cadence,
            discovery_refresh_seconds=max(60.0, min(300.0, float(self.settings.market_refresh_seconds))),
            timeframe="1m",
            timeframe_seconds=60.0,
            max_context_symbols=4,
            shadow_portfolio=shadow_portfolio,
            outcome_journal=self.swarm_outcome_journal,
            base_order_usd=self.settings.order_usd,
            microstructure_sniper=UltraMicrostructureSniper(
                minimum_modeled_round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
                maximum_spread_bps=min(25.0, float(self.settings.market_max_spread_bps)),
            ),
            micro_agent_foundry=MicroAgentFoundry(maximum_candidates_per_symbol=2),
            reference_feed=reference_feed,
            # Deep micro polling is latency-adaptive up to 12 symbols.
            # The ranked market universe keeps rotating, so excess
            # opportunities are not permanently excluded.
            max_micro_symbols=max(
                6,
                min(
                    12,
                    self.settings.market_scan_batch_size,
                ),
            ),
            micro_calibration_journal=MicroCalibrationJournal(
                base.with_name("vps_micro_calibration.json")
            ),
            micro_calibration_feed=calibration_feed,
            microstream_feed=microstream_feed,
            slow_calibration_journal=MicroCalibrationJournal(
                base.with_name("vps_slow_calibration.json"),
                accepted_horizons=(120, 300, 900),
                max_resolution_delay_seconds=5.0,
            ),
            precision_scout_feed=(
                precision_scout_feed
            ),
            precision_scout_refresh_seconds=20.0,
            precision_round_trip_cost_bps=(
                self._swarm_round_trip_cost_bps()
            ),
        )

    def start_fast_swarm(self) -> None:
        if not self.swarm_manifests_registered:
            self._register_swarm_research_manifests()
        if self.fast_swarm_service is None:
            self.fast_swarm_service = self._build_fast_swarm_service()
        self.fast_swarm_service.start()

    def stop_fast_swarm(self) -> None:
        if self.fast_swarm_service is not None:
            self.fast_swarm_service.stop()

    def _ingest_swarm_outcomes(self) -> dict[str, Any]:
        episodes = self.swarm_outcome_journal.pending(limit=1000)
        if not episodes:
            self.last_swarm_evidence_ingest = {
                "submitted": 0,
                "episodes_recorded": 0,
                "qualification_refresh": "no_new_closed_outcomes",
                "journal": self.swarm_outcome_journal.health(),
            }
            return dict(self.last_swarm_evidence_ingest)
        if not self.swarm_manifests_registered:
            self._register_swarm_research_manifests()
        result = self.prospective_validation.observe_cycle(
            observatory_authority=self.prospective_validation.EVIDENCE_AUTHORITY,
            observed_round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
            strategy_episodes=episodes,
            foundry_manifests=[],
            market_rows={},
        )
        ids = [str(row.get("episode_id")) for row in episodes if row.get("episode_id")]
        acknowledged = self.swarm_outcome_journal.acknowledge(ids)
        self.last_swarm_evidence_ingest = {
            "submitted": len(episodes),
            "episodes_recorded": int(result.get("episodes_recorded") or 0),
            "acknowledged": acknowledged,
            "overlapping_or_ineligible_episodes_are_not_retried": True,
            "qualification_refresh": "next_supervisory_cycle",
            "journal": self.swarm_outcome_journal.health(),
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
        return dict(self.last_swarm_evidence_ingest)

    @staticmethod
    def _inactive_swarm_status() -> dict[str, Any]:
        return {
            "version": "1.43.0",
            "configured": True,
            "running": False,
            "reason": "parallel_service_starts_with_continuous_runner",
            "cadence_role": "seconds_to_minutes",
            "shadow_portfolio": "separate_from_canonical_paper_ledger",
            "movement_only_can_allocate_capital": False,
            "requires_independent_agent_qualification": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def _swarm_health_contract(
        self,
        swarm: dict[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Keep fast execution alive when only the slow research sweep is busy."""

        now = (
            time.time()
            if now is None
            else float(now)
        )

        slow_worker_healthy = bool(
            swarm.get("running") is True
            and swarm.get("healthy") is True
            and swarm.get("stale") is not True
        )

        precision = (
            swarm.get("precision_scout")
            or {}
        )

        micro = (
            swarm.get("micro_calibration")
            or {}
        )

        try:
            started_at = float(
                swarm.get("started_at")
                or 0.0
            )
        except (TypeError, ValueError):
            started_at = 0.0

        startup_age = (
            max(0.0, now - started_at)
            if started_at > 0.0
            else 1_000_000.0
        )

        try:
            refresh_seconds = max(
                10.0,
                float(
                    precision.get(
                        "refresh_seconds"
                    )
                    or 20.0
                ),
            )
        except (TypeError, ValueError):
            refresh_seconds = 20.0

        try:
            scout_last = float(
                precision.get(
                    "last_refresh_at"
                )
                or 0.0
            )
        except (TypeError, ValueError):
            scout_last = 0.0

        try:
            micro_last = float(
                micro.get(
                    "microstream_last_observation_at"
                )
                or 0.0
            )
        except (TypeError, ValueError):
            micro_last = 0.0

        scout_fresh = bool(
            precision.get("running") is True
            and (
                (
                    scout_last > 0.0
                    and now - scout_last
                    <= max(
                        90.0,
                        refresh_seconds * 4.0,
                    )
                )
                or (
                    scout_last <= 0.0
                    and startup_age <= 90.0
                )
            )
        )

        micro_fresh = bool(
            micro.get(
                "microstream_running"
            ) is True
            and (
                (
                    micro_last > 0.0
                    and now - micro_last <= 15.0
                )
                or (
                    micro_last <= 0.0
                    and startup_age <= 30.0
                )
            )
        )

        fast_precision_path_operational = bool(
            getattr(
                self,
                "testnet",
                None,
            )
            is not None
            and scout_fresh
            and micro_fresh
        )

        return {
            "slow_worker_healthy": slow_worker_healthy,
            "precision_scout_fresh": scout_fresh,
            "microstream_fresh": micro_fresh,
            "fast_precision_path_operational": (
                fast_precision_path_operational
            ),
            "health_contract_healthy": bool(
                slow_worker_healthy
                or fast_precision_path_operational
            ),
            "slow_research_failure_blocks_fast_testnet": False,
            "live_authority": False,
        }

    def _write_health_state(self, status: dict[str, Any]) -> None:
        swarm = status.get("market_swarm") or {}
        swarm_required = self.fast_swarm_service is not None
        contract = (
            self._swarm_health_contract(
                swarm
            )
            if isinstance(swarm, dict)
            else {
                "health_contract_healthy": False,
                "slow_worker_healthy": False,
                "fast_precision_path_operational": False,
            }
        )

        swarm_ok = bool(
            contract.get(
                "health_contract_healthy"
            )
        )
        payload = {
            "timestamp": status.get("timestamp"),
            "runtime": status.get("runtime"),
            "healthy": bool(
                status.get("healthy") is True
                and (not swarm_required or swarm_ok)
            ),
            "errors": status.get("errors") or [],
            "testnet_execution": status.get("testnet_execution") or {},
            "startup_heartbeat": (
                status.get("startup_heartbeat") is True
            ),
            "full_market_cycle_complete": (
                status.get("full_market_cycle_complete") is True
            ),
            "market_swarm": {
                "required": swarm_required,
                "running": swarm.get("running") is True,
                "healthy": swarm_ok,
                "health_contract_healthy": swarm_ok,
                "slow_worker_healthy": bool(
                    contract.get(
                        "slow_worker_healthy"
                    )
                ),
                "fast_precision_path_operational": bool(
                    contract.get(
                        "fast_precision_path_operational"
                    )
                ),
                "precision_scout_fresh": bool(
                    contract.get(
                        "precision_scout_fresh"
                    )
                ),
                "microstream_fresh": bool(
                    contract.get(
                        "microstream_fresh"
                    )
                ),
                "stale": swarm.get("stale") is True,
                "cycles": int(swarm.get("cycles") or 0),
                "consecutive_failures": int(swarm.get("consecutive_failures") or 0),
                "last_success_at": swarm.get("last_success_at"),
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            },
        }
        self._write_json_atomic(
            self.settings.heartbeat_path.with_name("vps_health_state.json"),
            payload,
        )

    def _refresh_runtime_health_state(self) -> dict[str, Any]:
        """Refresh Docker/runtime liveness without rewriting market evidence."""

        now = time.time()
        service = self.fast_swarm_service

        equity = float(self.ledger.cash)
        for position in self.ledger.positions.values():
            equity += (
                float(position.quantity)
                * float(position.entry_price)
            )

        swarm_health = (
            service.health(equity=equity)
            if service is not None
            else self._inactive_swarm_status()
        )

        swarm_contract = (
            self._swarm_health_contract(
                swarm_health,
                now=now,
            )
        )

        swarm_health = {
            **swarm_health,
            **swarm_contract,
        }

        swarm_ok = bool(
            service is not None
            and swarm_contract.get(
                "health_contract_healthy"
            )
        )

        engines = self.engines.snapshot()

        required_failures = [
            name
            for name, row in engines.items()
            if isinstance(row, dict)
            and row.get("required") is True
            and row.get("healthy") is not True
        ]

        failures = list(required_failures)
        if not swarm_ok:
            failures.append("market_swarm")

        healthy = not failures

        status = {
            "timestamp": now,
            "runtime": (
                "verified-multi-engine-v12.11-"
                "continuous-evolution-fabric"
            ),
            "healthy": healthy,
            "errors": (
                []
                if healthy
                else [
                    "runtime_health_failure:"
                    + ",".join(failures)
                ]
            ),
            "testnet_execution": {
                "enabled": self.testnet is not None,
                "live_authority": False,
            },
            "startup_heartbeat": False,
            "full_market_cycle_complete": (
                self._full_market_cycle_completed
            ),
            "market_swarm": {
                **swarm_health,
                "required": True,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            },
        }

        # Deliberately writes only vps_health_state.json.
        self._write_health_state(status)
        return status

    def _run_health_state_refresher(self) -> None:
        interval_seconds = 60.0

        while not self._health_refresh_stop.wait(
            interval_seconds
        ):
            try:
                self._refresh_runtime_health_state()
            except Exception as exc:
                LOGGER.warning(
                    "runtime health-state refresh failed: %s",
                    exc,
                )

    def _start_health_state_refresher(self) -> None:
        thread = self._health_refresh_thread
        if thread is not None and thread.is_alive():
            return

        self._health_refresh_stop.clear()
        self._health_refresh_thread = threading.Thread(
            target=self._run_health_state_refresher,
            name="leantrader-runtime-health-refresh",
            daemon=True,
        )
        self._health_refresh_thread.start()

    def _stop_health_state_refresher(self) -> None:
        # Idempotent shutdown also supports one-shot/test runners that
        # intentionally bypass __init__ and never start the refresher.
        stop_event = getattr(
            self,
            "_health_refresh_stop",
            None,
        )
        if stop_event is None:
            return

        stop_event.set()

        thread = getattr(
            self,
            "_health_refresh_thread",
            None,
        )
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def _write_startup_heartbeat(self) -> dict[str, Any]:
        """Publish initialized paper-runtime health before the long first cycle."""

        now = time.time()
        service = self.fast_swarm_service

        # Before the first fresh market marks exist, value open paper
        # positions at their persisted entry prices. This is deterministic
        # startup accounting and does not mutate the ledger.
        startup_equity = float(self.ledger.cash)

        for position in self.ledger.positions.values():
            startup_equity += (
                float(position.quantity)
                * float(position.entry_price)
            )

        startup_open_notional = sum(
            float(position.quantity)
            * float(position.entry_price)
            for position in (
                self.ledger.positions.values()
            )
        )

        startup_realized_pnl = float(
            getattr(
                self.ledger,
                "realized_pnl",
                0.0,
            )
        )

        capital_growth = getattr(
            self,
            "capital_growth",
            None,
        )

        # Production runners always initialize CapitalGrowthGovernor.
        # Legacy unit fixtures intentionally bypass __init__ via
        # object.__new__, so they receive an empty, non-authoritative
        # startup snapshot rather than crashing.
        if capital_growth is None:
            startup_growth_state = {}
        else:
            startup_growth_state = (
                capital_growth.evaluate(
                    equity=startup_equity,
                    realized_pnl=(
                        startup_realized_pnl
                    ),
                    open_notional=(
                        startup_open_notional
                    ),
                )
            )

        swarm_health = (
            service.health(
                equity=startup_equity
            )
            if service is not None
            else self._inactive_swarm_status()
        )

        swarm_contract = (
            self._swarm_health_contract(
                swarm_health,
                now=now,
            )
        )

        swarm_health = {
            **swarm_health,
            **swarm_contract,
        }

        swarm_operational = bool(
            service is not None
            and swarm_contract.get(
                "health_contract_healthy"
            )
        )

        engines = dict(
            self.engines.snapshot()
        )

        engines["market_swarm"] = {
            "required": True,
            "healthy": swarm_operational,
            "state": (
                "starting"
                if swarm_operational
                else "degraded"
            ),
            "failures": int(
                swarm_health.get(
                    "consecutive_failures"
                )
                or 0
            ),
        }

        required_failures = [
            name
            for name, row in engines.items()
            if isinstance(row, dict)
            and row.get("required") is True
            and row.get("healthy") is not True
        ]

        healthy = bool(
            not required_failures
            and swarm_operational
        )

        testnet_enabled = bool(
            self.testnet is not None
        )

        errors = (
            []
            if healthy
            else [
                "startup_required_engine_failure:"
                + ",".join(required_failures)
            ]
        )

        status = {
            "timestamp": now,
            "runtime": (
                "verified-multi-engine-v12.11-"
                "continuous-evolution-fabric"
            ),
            "mode": "paper",
            "healthy": healthy,
            "errors": errors,
            "equity": startup_equity,
            "equity_mark_basis": (
                "persisted_entry_prices_until_first_full_cycle"
            ),
            "cash": float(
                getattr(
                    self.ledger,
                    "cash",
                    self.settings.starting_cash,
                )
            ),
            "open_positions": list(
                self.ledger.positions
            ),
            "realized_pnl": (
                startup_realized_pnl
            ),
            "capital_growth": (
                startup_growth_state
            ),
            "events": [],
            "engines": engines,
            "market_swarm": {
                **swarm_health,
                "required": True,
                "startup_phase": True,
                "full_market_cycle_complete": False,
                "canonical_paper_ledger_mutation_from_fast_thread": False,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            },
            "testnet_execution": {
                "enabled": testnet_enabled,
                "live_authority": False,
            },
            "startup_heartbeat": True,
            "full_market_cycle_complete": False,
            "automatic_promotion": False,
            "live_authority": False,
        }

        self._write_json_atomic(
            self.settings.heartbeat_path,
            status,
        )
        self._write_health_state(status)

        return status

    def cycle(self) -> dict[str, Any]:
        status = super().cycle()
        self._full_market_cycle_completed = True
        status["startup_heartbeat"] = False
        status["full_market_cycle_complete"] = True
        swarm_evidence = self._ingest_swarm_outcomes() if self.fast_swarm_service is not None else {
            "submitted": 0,
            "episodes_recorded": 0,
            "qualification_refresh": "parallel_service_not_running",
            "journal": self.swarm_outcome_journal.health(),
        }
        service = self.fast_swarm_service
        if service is None:
            swarm_health = self._inactive_swarm_status()
        else:
            swarm_health = service.health(equity=float(status.get("equity") or self.settings.starting_cash))
            swarm_health["configured"] = True
        swarm_contract = (
            self._swarm_health_contract(
                swarm_health,
                now=float(
                    status.get("timestamp")
                    or time.time()
                ),
            )
        )

        swarm_health = {
            **swarm_health,
            **swarm_contract,
        }

        status["market_swarm"] = swarm_health
        status["market_swarm"]["swarm_evidence_ingest"] = swarm_evidence
        status["market_swarm"]["slow_control_plane_blocking_fast_scout"] = False
        status["market_swarm"]["supervisory_evidence_version"] = "1.42"
        status["market_swarm"]["canonical_paper_ledger_mutation_from_fast_thread"] = False
        status["market_swarm"]["automatic_promotion"] = False
        status["market_swarm"]["execution_authority"] = False
        status["market_swarm"]["testnet_authority"] = False
        status["market_swarm"]["live_authority"] = False

        engines = status.setdefault("engines", {})
        if isinstance(engines, dict) and service is not None:
            swarm_operational = bool(
                status["market_swarm"].get(
                    "health_contract_healthy"
                )
                is True
            )
            engines["market_swarm"] = {
                "required": True,
                "healthy": swarm_operational,
                "state": "running" if swarm_operational else "degraded",
                "failures": int(status["market_swarm"].get("consecutive_failures") or 0),
            }
            portfolio = status["market_swarm"].get("shadow_portfolio")
            engines["swarm_shadow_portfolio"] = {
                "required": True,
                "healthy": isinstance(portfolio, dict),
                "state": "running" if isinstance(portfolio, dict) else "missing",
                "failures": 0 if isinstance(portfolio, dict) else 1,
            }
            journal = status["market_swarm"].get("swarm_outcome_journal")
            engines["swarm_outcome_journal"] = {
                "required": True,
                "healthy": isinstance(journal, dict),
                "state": "running" if isinstance(journal, dict) else "missing",
                "failures": 0 if isinstance(journal, dict) else 1,
            }

        self._update_fast_supervisory(
            status
        )

        fast_testnet_lane = getattr(
            self,
            "fast_collective_testnet",
            None,
        )

        if fast_testnet_lane is not None:
            status["fast_collective_testnet"] = (
                fast_testnet_lane.health()
            )

            fabric = status.setdefault(
                "collective_profit_fabric",
                {},
            )

            fabric["version"] = "1.60.3"
            fabric[
                "fast_testnet_exploration_lane"
            ] = True
            fabric[
                "seconds_scale_micro_mtf_execution"
            ] = True
            fabric[
                "subsecond_velocity_detection"
            ] = True
            fabric[
                "velocity_sniper_testnet_lane"
            ] = True
            fabric[
                "microstream_priority"
            ] = True
            fabric[
                "precision_mtf_context"
            ] = True
            fabric[
                "always_on_precision_scout"
            ] = True
            fabric[
                "dedicated_precision_mtf_cache"
            ] = True
            fabric[
                "precision_micro_priority_scheduler"
            ] = True
            fabric[
                "adaptive_opportunity_capacity"
            ] = True
            fabric[
                "split_slow_swarm_fast_path_health"
            ] = True
            fabric[
                "autonomous_testnet_reconciliation_recovery"
            ] = True
            fabric[
                "dynamic_position_concurrency"
            ] = True
            fabric[
                "latency_adaptive_micro_depth"
            ] = True
            fabric[
                "sub_dollar_mover_coverage"
            ] = True
            fabric[
                "principal_protected_profit_compounding"
            ] = True
            fabric[
                "profit_flow_telemetry"
            ] = True
            fabric[
                "bounded_testnet_exploration_authority"
            ] = True
            fabric[
                "multi_position_hyper_router"
            ] = True
            fabric[
                "per_position_hyper_speed_sentinel"
            ] = True
            fabric[
                "immediate_slot_reuse"
            ] = True
            fabric[
                "real_money_authority"
            ] = False
            fabric[
                "live_authority"
            ] = False

        self._write_json_atomic(
            self.settings.heartbeat_path,
            status,
        )
        self._write_health_state(status)
        return status

    def run(self, once: bool = False) -> None:
        if once:
            try:
                super().run(once=True)
            finally:
                self.stop_fast_swarm()
            return

        self.start_fast_swarm()

        startup_status = (
            self._write_startup_heartbeat()
        )

        # v1.59: do not leave the 0.5-second lane using yesterday/previous
        # cycle's supervisor cache while the first large canonical cycle
        # is still running.
        self._update_fast_supervisory(
            startup_status
        )

        if self.fast_collective_testnet is not None:
            self.fast_collective_testnet.start()

        self._start_health_state_refresher()

        try:
            while not self.stop_requested:
                started = time.monotonic()

                try:
                    status = self.cycle()

                    LOGGER.info(
                        "cycle equity=%.2f cash=%.2f "
                        "positions=%s events=%d errors=%d",
                        status["equity"],
                        status["cash"],
                        status["open_positions"],
                        len(status["events"]),
                        len(status["errors"]),
                    )

                except Exception:
                    LOGGER.exception(
                        "paper cycle failed"
                    )

                remaining = max(
                    1.0,
                    self.settings.poll_seconds
                    - (
                        time.monotonic()
                        - started
                    ),
                )

                time.sleep(remaining)

        finally:
            if self.fast_collective_testnet is not None:
                self.fast_collective_testnet.stop()

            self._stop_health_state_refresher()
            self.stop_fast_swarm()
            self.engines.stop_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="LeanTrader v1.43 paper runner with evidence-qualified parallel market swarm")
    parser.add_argument("--once", action="store_true", help="run one canonical market cycle and exit")
    parser.add_argument("--preflight", action="store_true", help="validate safe configuration without network access")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    if args.preflight:
        payload = preflight(settings)
        payload["market_swarm"] = {
            "version": "1.43.0",
            "parallel_read_only_market_feed": True,
            "separate_persistent_shadow_portfolio": True,
            "closed_outcome_evidence_journal": True,
            "canonical_paper_ledger_mutation_from_fast_thread": False,
            "timeframe": "1m",
            "modeled_round_trip_cost_floor_bps": max(30.0, 2.0 * (settings.fee_bps + settings.slippage_bps)),
            "prospective_minimum_samples": max(100, settings.evolution_min_shadow_samples),
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
        payload["collective_profit_fabric"] = {
            "version": "1.60.3",
            "canonical_pretrade_integration": True,
            "sources": [
                "adaptive_intelligence",
                "advanced_ultra_suite",
                "fast_market_swarm",
                "microstructure_sniper",
                "subsecond_velocity_sniper",
                "multi_timeframe_minds",
                "continuous_evolution",
                "market_sensor_fabric",
                "cross_venue_context",
                "alpha_tournament_feedback",
            ],
            "costed_edge_required_for_collective_origin": True,
            "sensor_only_origin_allowed": False,
            "paper_authority": True,
            "testnet_mirror_enabled": settings.testnet_enabled,
            "fast_testnet_exploration_lane": settings.testnet_enabled,
            "seconds_scale_micro_mtf_execution": settings.testnet_enabled,
            "subsecond_velocity_detection": settings.testnet_enabled,
            "velocity_sniper_testnet_lane": settings.testnet_enabled,
            "microstream_priority": settings.testnet_enabled,
            "precision_mtf_context": settings.testnet_enabled,
            "always_on_precision_scout": settings.testnet_enabled,
            "dedicated_precision_mtf_cache": settings.testnet_enabled,
            "precision_micro_priority_scheduler": settings.testnet_enabled,
            "adaptive_opportunity_capacity": settings.testnet_enabled,
            "split_slow_swarm_fast_path_health": settings.testnet_enabled,
            "autonomous_testnet_reconciliation_recovery": settings.testnet_enabled,
            "dynamic_position_concurrency": settings.testnet_enabled,
            "latency_adaptive_micro_depth": settings.testnet_enabled,
            "sub_dollar_mover_coverage": settings.testnet_enabled,
            "principal_protected_profit_compounding": settings.testnet_enabled,
            "profit_flow_telemetry": settings.testnet_enabled,
            "bounded_testnet_exploration_authority": settings.testnet_enabled,
            "multi_position_hyper_router": settings.testnet_enabled,
            "per_position_hyper_speed_sentinel": settings.testnet_enabled,
            "immediate_slot_reuse": settings.testnet_enabled,
            "real_money_authority": False,
            "live_authority": False,
        }
        print(json.dumps(payload, indent=2))
        return

    runner = PaperRunner(settings, MarketFeed(settings.exchange))

    def request_stop(_signum: int, _frame: Any) -> None:
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    runner.run(once=args.once)


if __name__ == "__main__":
    main()
