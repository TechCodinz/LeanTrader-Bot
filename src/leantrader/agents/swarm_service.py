from __future__ import annotations

import copy
import threading
import time
from typing import Any

import pandas as pd

from .fast_path import FastSwarmRuntime
from .micro_calibration import MicroCalibrationJournal
from .microstructure_sniper import MicroAgentFoundry, UltraMicrostructureSniper
from .opportunity_radar import OpportunityScore
from .shared_position_graph import AgentRole
from .swarm_evidence import SwarmOutcomeJournal
from .swarm_shadow_portfolio import SwarmShadowPortfolio
from .timeframe_mind import MultiTimeframeMind


class ReadOnlySwarmService:
    """Parallel market scouting plus isolated paper-shadow tranche measurement.

    The service never mutates the canonical PaperLedger. Independently-qualified
    timeframe agents may enter only the separate SwarmShadowPortfolio, whose
    completed net-of-cost outcomes are journaled for later v1.42 evidence intake.
    """

    VERSION = "1.60.6"
    KINEMATIC_SLOW_HORIZONS = (120, 300, 900)
    KINEMATIC_SLOW_COOLDOWN_SECONDS = 900.0
    ROLE_BY_TIMEFRAME = {
        "1m": AgentRole.SCALP,
        "5m": AgentRole.MOMENTUM,
        "15m": AgentRole.MOMENTUM,
        "1h": AgentRole.TREND,
        "4h": AgentRole.TREND,
    }

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
        timeframe_mind: MultiTimeframeMind | None = None,
        context_timeframes: tuple[str, ...] = ("5m", "15m", "1h", "4h"),
        max_context_symbols: int = 2,
        shadow_portfolio: SwarmShadowPortfolio | None = None,
        outcome_journal: SwarmOutcomeJournal | None = None,
        base_order_usd: float = 2.0,
        microstructure_sniper: UltraMicrostructureSniper | None = None,
        micro_agent_foundry: MicroAgentFoundry | None = None,
        reference_feed: Any | None = None,
        max_micro_symbols: int = 2,
        micro_calibration_journal: MicroCalibrationJournal | None = None,
        micro_calibration_feed: Any | None = None,
        microstream_feed: Any | None = None,
        slow_calibration_journal: MicroCalibrationJournal | None = None,
        precision_scout_feed: Any | None = None,
        precision_scout_refresh_seconds: float = 20.0,
        precision_round_trip_cost_bps: float = 30.0,
    ) -> None:
        if scan_batch_size < 1:
            raise ValueError("scan_batch_size must be positive")
        if candle_limit < 32:
            raise ValueError("candle_limit must be at least 32")
        if cadence_seconds < 1.0:
            raise ValueError("cadence_seconds must be at least one second")
        if discovery_refresh_seconds < cadence_seconds:
            raise ValueError("discovery refresh cannot be faster than service cadence")
        if max_context_symbols < 1:
            raise ValueError("max_context_symbols must be positive")
        if base_order_usd <= 0:
            raise ValueError("base_order_usd must be positive")
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
        self.timeframe_mind = timeframe_mind or MultiTimeframeMind()
        self.context_timeframes = tuple(dict.fromkeys(str(value) for value in context_timeframes if value))
        self.max_context_symbols = int(max_context_symbols)
        self.shadow_portfolio = shadow_portfolio
        self.outcome_journal = outcome_journal
        self.base_order_usd = float(base_order_usd)
        self.microstructure_sniper = microstructure_sniper or UltraMicrostructureSniper()
        self.micro_agent_foundry = micro_agent_foundry or MicroAgentFoundry()
        self.reference_feed = reference_feed
        self.max_micro_symbols = max(1, int(max_micro_symbols))
        self.micro_calibration_journal = micro_calibration_journal
        self.micro_calibration_feed = micro_calibration_feed
        self.microstream_feed = microstream_feed
        self.slow_calibration_journal = slow_calibration_journal

        self.precision_scout_feed = precision_scout_feed
        self.precision_scout_refresh_seconds = max(
            10.0,
            min(
                60.0,
                float(
                    precision_scout_refresh_seconds
                ),
            ),
        )
        self.precision_round_trip_cost_bps = max(
            30.0,
            float(
                precision_round_trip_cost_bps
            ),
        )

        self.precision_timeframe_mind = (
            MultiTimeframeMind(
                minimum_modeled_round_trip_cost_bps=(
                    self.precision_round_trip_cost_bps
                )
            )
        )

        self._precision_scout_thread: (
            threading.Thread | None
        ) = None
        self._precision_scout_symbols: list[str] = []
        self._precision_context_cache: dict[
            str,
            dict[str, Any],
        ] = {}

        self.precision_scout_refreshes = 0
        self.precision_scout_failures = 0
        self.precision_context_updates = 0
        self.precision_context_failures = 0
        self.precision_scout_last_refresh_at = 0.0
        self.precision_context_last_update_at = 0.0
        self.precision_scout_candidates = 0
        self.precision_scout_sub_dollar = 0

        # v1.60.2: precision capacity is adaptive. The scout universe may
        # be wider than the instantaneous deep-poll budget; symbols rotate
        # through the deep lane instead of being permanently excluded.
        self._precision_micro_cursor = 0
        self.precision_micro_last_queue: list[str] = []
        self.precision_micro_live_slots = 0
        self.precision_micro_due_slots = 0
        self._precision_micro_capacity = max(
            1,
            min(
                6,
                self.max_micro_symbols,
            ),
        )
        self._precision_micro_last_failure_count = 0

        # v1.60.4 continuity-preserving precision scheduler.
        #
        # The broad scout remains wide, but a bounded hot cohort stays
        # continuously sampled long enough to build genuine 5-second
        # temporal microstructure evidence.
        self._precision_hot_symbols: list[str] = []
        self._precision_hot_until = 0.0
        self._precision_explorer_symbol: str | None = None
        self._precision_explorer_until = 0.0
        self.precision_hot_hold_seconds = 4.0
        self.precision_explorer_hold_seconds = 3.0
        self.precision_hot_promotions = 0
        self.precision_explorer_rotations = 0

        # v1.60.6: execution freshness is a resource budget rather
        # than a fixed symbol count.
        self.microstream_target_loop_seconds = 1.50
        self.microstream_freshness_seconds = 2.0
        self.microstream_per_symbol_latency_seconds = 0.25
        self._execution_precision_pins: dict[str, float] = {}

        # v1.60.28: entry candidates waiting on a fresh execution
        # signal must not compete behind long-lived active-position
        # or research pins. They receive a distinct bounded freshness
        # priority without gaining any execution authority.
        self._execution_candidate_pins: dict[str, float] = {}

        self._microstream_thread: threading.Thread | None = None
        self._microstream_watchdog_thread: threading.Thread | None = None
        self._retired_microstream_threads: list[threading.Thread] = []
        self._microstream_generation = 0
        self._calibration_thread: threading.Thread | None = None

        # v1.60.26: a public CCXT order-book read can very rarely
        # remain blocked beyond the configured request timeout. The
        # dedicated precision worker therefore has its own liveness
        # watchdog. Recovery replaces only the read-only feed/worker;
        # it never grants execution authority or bypasses freshness.
        self.microstream_stall_seconds = 30.0
        self.microstream_watchdog_checks = 0
        self.microstream_stall_recoveries = 0
        self.microstream_stall_recovery_failures = 0
        self.microstream_last_recovery_at = 0.0
        self.microstream_last_attempt_started_at = 0.0
        self.microstream_last_attempt_symbol: str | None = None

        self.microstream_sample_attempts = 0
        self.microstream_sample_failures = 0
        # v1.60.29: count of cadence sleeps released early by an execution pin.
        self.microstream_pin_wakeups = 0
        self.microstream_trade_context_failures = 0
        self.microstream_warmup_labels_skipped = 0
        self.microstream_non_event_labels_skipped = 0
        self.microstream_kinematic_labels_registered = 0

        self.kinematic_slow_event_triggers_registered = 0
        self.kinematic_slow_labels_registered = 0
        self.kinematic_slow_labels_suppressed_by_cooldown = 0
        self._kinematic_slow_last_registered: dict[str, float] = {}

        self.microstream_last_loop_seconds = 0.0
        self.microstream_labels_resolved = 0
        self.microstream_observations = 0
        self.microstream_last_observation_at = 0.0
        self._microstream_symbols: list[str] = []
        self._microstream_snapshots: dict[
            str,
            dict[str, Any],
        ] = {}

        self.calibration_sample_attempts = 0
        self.calibration_sample_failures = 0
        self.calibration_labels_resolved = 0
        self.micro_assessments = 0
        self.micro_qualified = 0
        self.micro_fetch_failures = 0
        if (self.shadow_portfolio is None) != (self.outcome_journal is None):
            raise ValueError("shadow portfolio and outcome journal must be configured together")
        self._stop = threading.Event()
        # v1.60.29: dedicated wake so a newly pinned execution candidate can
        # interrupt the microstream sleep immediately. This is read-only market
        # data scheduling only: it grants no execution authority, does not change
        # the execution freshness threshold, and does not alter priority order.
        self._execution_pin_event = threading.Event()
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
        self.shadow_entries = 0
        self.shadow_exits = 0
        self.shadow_failures = 0
        self.started_at = 0.0
        self.last_success_at = 0.0
        self.last_failure_at = 0.0
        self.step_in_progress = False
        self.step_started_at = 0.0
        self.last_step_duration_seconds = 0.0
        self.consecutive_failures = 0
        self.ranked_opportunities_total = 0
        self.qualified_opportunities_total = 0

    def _microstream_stall_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = (
            time.time()
            if now is None
            else float(now)
        )

        with self._lock:
            thread = self._microstream_thread
            attempt_started = float(
                getattr(
                    self,
                    "microstream_last_attempt_started_at",
                    0.0,
                )
                or 0.0
            )
            attempt_symbol = (
                getattr(
                    self,
                    "microstream_last_attempt_symbol",
                    None,
                )
            )
            last_observation = float(
                getattr(
                    self,
                    "microstream_last_observation_at",
                    0.0,
                )
                or 0.0
            )
            stall_seconds = max(
                25.0,
                float(
                    getattr(
                        self,
                        "microstream_stall_seconds",
                        30.0,
                    )
                    or 30.0
                ),
            )
            generation = int(
                getattr(
                    self,
                    "_microstream_generation",
                    0,
                )
            )

        request_age = (
            max(0.0, now - attempt_started)
            if attempt_started > 0.0
            else 0.0
        )

        alive = bool(
            thread is not None
            and thread.is_alive()
            and not self._stop.is_set()
        )

        stalled = bool(
            alive
            and attempt_started > 0.0
            and request_age > stall_seconds
            and last_observation < attempt_started
        )

        return {
            "alive": alive,
            "stalled": stalled,
            "generation": generation,
            "attempt_symbol": attempt_symbol,
            "attempt_started_at": attempt_started,
            "request_age_seconds": request_age,
            "last_observation_at": last_observation,
            "stall_seconds": stall_seconds,
            "live_authority": False,
        }

    @staticmethod
    def _clone_public_microstream_feed(
        feed: Any,
    ) -> Any | None:
        exchange = getattr(
            feed,
            "exchange",
            None,
        )

        exchange_id = str(
            getattr(
                exchange,
                "id",
                "",
            )
            or ""
        ).strip()

        if not exchange_id:
            return None

        try:
            return type(feed)(
                exchange_id
            )
        except Exception:
            return None

    def _start_microstream_worker(
        self,
        *,
        feed: Any,
    ) -> threading.Thread:
        with self._lock:
            self._microstream_generation = (
                int(
                    getattr(
                        self,
                        "_microstream_generation",
                        0,
                    )
                )
                + 1
            )

            generation = (
                self._microstream_generation
            )

            thread = threading.Thread(
                target=self._run_microstream,
                args=(
                    generation,
                    feed,
                ),
                name=(
                    "leantrader-dedicated-"
                    f"microstream-{generation}"
                ),
                daemon=True,
            )

            self._microstream_thread = thread
            self.microstream_last_attempt_started_at = 0.0
            self.microstream_last_attempt_symbol = None

        thread.start()
        return thread

    def _recover_stalled_microstream(
        self,
        *,
        now: float | None = None,
    ) -> bool:
        now = (
            time.time()
            if now is None
            else float(now)
        )

        snapshot = (
            self._microstream_stall_snapshot(
                now=now
            )
        )

        if snapshot["stalled"] is not True:
            return False

        with self._lock:
            retired = [
                thread
                for thread in (
                    getattr(
                        self,
                        "_retired_microstream_threads",
                        [],
                    )
                    or []
                )
                if thread.is_alive()
            ]

            self._retired_microstream_threads = (
                retired
            )

            # Bound abandoned daemon workers. If the provider
            # repeatedly wedges, degrade instead of leaking
            # unlimited threads.
            if len(retired) >= 2:
                self.microstream_stall_recovery_failures += 1
                return False

            current = self._microstream_thread
            source_feed = self.microstream_feed

        replacement = (
            self._clone_public_microstream_feed(
                source_feed
            )
        )

        if replacement is None:
            with self._lock:
                self.microstream_stall_recovery_failures += 1
            return False

        with self._lock:
            if (
                current is not None
                and current.is_alive()
            ):
                self._retired_microstream_threads.append(
                    current
                )

            self.microstream_feed = replacement
            self.microstream_stall_recoveries += 1
            self.microstream_last_recovery_at = now

        self._start_microstream_worker(
            feed=replacement
        )

        return True

    def _run_microstream_watchdog(
        self,
    ) -> None:
        while not self._stop.wait(2.0):
            try:
                self.microstream_watchdog_checks += 1
                self._recover_stalled_microstream()
            except Exception:
                self.microstream_stall_recovery_failures += 1

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self.started_at = time.time()

        if self.precision_scout_feed is not None:
            self._precision_scout_thread = (
                threading.Thread(
                    target=self._run_precision_scout,
                    name=(
                        "leantrader-always-on-"
                        "precision-scout"
                    ),
                    daemon=True,
                )
            )
            self._precision_scout_thread.start()

        self._thread = threading.Thread(
            target=self._run,
            name="leantrader-market-swarm",
            daemon=True,
        )
        self._thread.start()

        if (
            self.micro_calibration_journal is not None
            and self.microstream_feed is not None
        ):
            self._start_microstream_worker(
                feed=self.microstream_feed
            )

            self._microstream_watchdog_thread = (
                threading.Thread(
                    target=(
                        self._run_microstream_watchdog
                    ),
                    name=(
                        "leantrader-microstream-"
                        "watchdog"
                    ),
                    daemon=True,
                )
            )
            self._microstream_watchdog_thread.start()

        if (
            self.slow_calibration_journal is not None
            and self.micro_calibration_feed is not None
        ):
            self._calibration_thread = threading.Thread(
                target=self._run_calibration_sampler,
                name="leantrader-slow-calibration",
                daemon=True,
            )
            self._calibration_thread.start()

    def stop(self) -> None:
        self._stop.set()
        # v1.60.29: also release any microstream sleep parked on the pin wake.
        # Absent on legacy/lightweight instances, so guard rather than raise.
        wake_event = getattr(self, "_execution_pin_event", None)
        if wake_event is not None:
            wake_event.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(
                timeout=max(
                    1.0,
                    min(10.0, self.cadence_seconds + 1.0),
                )
            )

        precision_thread = (
            self._precision_scout_thread
        )

        if (
            precision_thread is not None
            and precision_thread.is_alive()
        ):
            precision_thread.join(timeout=8.0)

        watchdog_thread = (
            self._microstream_watchdog_thread
        )

        if (
            watchdog_thread is not None
            and watchdog_thread.is_alive()
        ):
            watchdog_thread.join(timeout=3.0)

        microstream_thread = self._microstream_thread
        if (
            microstream_thread is not None
            and microstream_thread.is_alive()
        ):
            microstream_thread.join(timeout=4.0)

        calibration_thread = self._calibration_thread
        if (
            calibration_thread is not None
            and calibration_thread.is_alive()
        ):
            calibration_thread.join(timeout=4.0)

    @staticmethod
    def _precision_number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default

        return (
            number
            if number == number
            and abs(number) != float("inf")
            else default
        )

    @classmethod
    def _select_precision_scout(
        cls,
        candidates: list[dict[str, Any]],
        *,
        capacity: int,
    ) -> list[dict[str, Any]]:
        """Reserve coverage for executable sub-$1 movers.

        Nominal price is a coverage bucket only. Movement,
        liquidity and spread determine ordering.
        """

        # Operationally bounded, but not fixed at six. The caller decides
        # the active scout width from market/runtime capacity.
        bounded = max(
            1,
            min(24, int(capacity)),
        )

        clean = [
            dict(row)
            for row in candidates
            if (
                isinstance(row, dict)
                and str(
                    row.get("symbol") or ""
                ).strip()
                and cls._precision_number(
                    row.get("last")
                ) > 0.0
            )
        ]

        def key(
            row: dict[str, Any],
        ) -> tuple[float, float, float]:
            movement = abs(
                cls._precision_number(
                    row.get(
                        "percentage_24h"
                    )
                )
            )

            volume = max(
                0.0,
                cls._precision_number(
                    row.get(
                        "quote_volume_usd"
                    )
                ),
            )

            spread = max(
                0.0,
                cls._precision_number(
                    row.get("spread_bps")
                ),
            )

            return (
                movement,
                volume,
                -spread,
            )

        ranked = sorted(
            clean,
            key=key,
            reverse=True,
        )

        sub_dollar = [
            row
            for row in ranked
            if (
                cls._precision_number(
                    row.get("last")
                )
                < 1.0
            )
        ]

        sub_slots = min(
            len(sub_dollar),
            max(1, bounded // 2),
        )

        selected = list(
            sub_dollar[:sub_slots]
        )

        seen = {
            str(
                row.get("symbol") or ""
            ).upper()
            for row in selected
        }

        for row in ranked:
            symbol = str(
                row.get("symbol") or ""
            ).upper()

            if (
                not symbol
                or symbol in seen
            ):
                continue

            selected.append(row)
            seen.add(symbol)

            if len(selected) >= bounded:
                break

        return selected[:bounded]

    def _refresh_precision_scout_once(
        self,
    ) -> dict[str, Any]:
        feed = self.precision_scout_feed

        if feed is None:
            return {
                "available": False,
                "reason": (
                    "precision_scout_feed_unavailable"
                ),
            }

        payload = feed.discover_markets(
            quote=self.market_quote,
            min_quote_volume_usd=(
                self.min_quote_volume_usd
            ),
            max_spread_bps=(
                self.max_spread_bps
            ),
        )

        candidates = [
            dict(row)
            for row in (
                payload.get("candidates")
                or []
            )
            if isinstance(row, dict)
        ]

        # v1.60.51: execution-pinned markets must not receive only
        # sub-second micro data while MTF context is built for a
        # completely different 24h-ranked symbol set. Preserve the
        # existing scout width; reserve part of that same capacity for
        # active execution/candidate pins, then fill the remainder with
        # the existing movement/liquidity/spread ranking.
        now = time.time()

        with self._lock:
            execution_pins = [
                symbol
                for symbol, until in (
                    getattr(
                        self,
                        "_execution_precision_pins",
                        {},
                    )
                    or {}
                ).items()
                if float(until) > now
            ]

            candidate_pins = [
                symbol
                for symbol, until in reversed(
                    list(
                        (
                            getattr(
                                self,
                                "_execution_candidate_pins",
                                {},
                            )
                            or {}
                        ).items()
                    )
                )
                if float(until) > now
            ]

        pinned_symbols = self._unique_symbols(
            [
                *candidate_pins,
                *execution_pins,
            ]
        )

        candidate_map = {
            str(
                row.get("symbol")
                or ""
            ).upper(): row
            for row in candidates
            if str(
                row.get("symbol")
                or ""
            ).strip()
        }

        pinned_rows = [
            dict(candidate_map[symbol])
            for symbol in pinned_symbols
            if symbol in candidate_map
        ]

        movement_rows = (
            self._select_precision_scout(
                candidates,
                capacity=self.max_micro_symbols,
            )
        )

        selected_rows = []
        selected_seen = set()

        for row in [
            *pinned_rows,
            *movement_rows,
        ]:
            symbol = str(
                row.get("symbol")
                or ""
            ).upper()

            if (
                not symbol
                or symbol in selected_seen
            ):
                continue

            selected_rows.append(
                dict(row)
            )
            selected_seen.add(symbol)

            if (
                len(selected_rows)
                >= self.max_micro_symbols
            ):
                break

        selected = [
            str(
                row.get("symbol") or ""
            ).upper()
            for row in selected_rows
            if str(
                row.get("symbol") or ""
            ).strip()
        ]

        now = time.time()

        with self._lock:
            self._precision_scout_symbols = (
                selected
            )
            self._precision_pinned_context_symbols = [
                symbol
                for symbol in selected
                if symbol in set(pinned_symbols)
            ]
            self.precision_scout_refreshes += 1
            self.precision_scout_last_refresh_at = (
                now
            )
            self.precision_scout_candidates = len(
                candidates
            )
            self.precision_scout_sub_dollar = sum(
                1
                for row in selected_rows
                if (
                    self._precision_number(
                        row.get("last")
                    )
                    < 1.0
                )
            )

        context_updates = 0
        context_failures = 0

        for symbol in selected:
            if self._stop.is_set():
                break

            frames: dict[
                str,
                pd.DataFrame,
            ] = {}

            for timeframe in (
                "1m",
                "5m",
                "15m",
            ):
                try:
                    frames[timeframe] = (
                        self._closed_candles(
                            feed.candles(
                                symbol,
                                timeframe,
                                self.candle_limit,
                            )
                        )
                    )
                except Exception:
                    context_failures += 1

            if not frames:
                continue

            try:
                rows = (
                    self.precision_timeframe_mind.assess_many(
                        symbol=symbol,
                        frames=frames,
                        modeled_round_trip_cost_bps=(
                            self.precision_round_trip_cost_bps
                        ),
                    )
                )
            except Exception:
                context_failures += 1
                continue

            if not rows:
                continue

            observed_at = time.time()

            with self._lock:
                self._precision_context_cache[
                    symbol
                ] = {
                    "timestamp": observed_at,
                    "assessments": {
                        timeframe: row.as_dict()
                        for timeframe, row in (
                            rows.items()
                        )
                    },
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                }

                self.precision_context_last_update_at = (
                    observed_at
                )

            context_updates += 1

        cutoff = (
            time.time()
            - max(
                180.0,
                self.precision_scout_refresh_seconds
                * 6.0,
            )
        )

        with self._lock:
            stale = [
                symbol
                for symbol, row in (
                    self._precision_context_cache.items()
                )
                if self._precision_number(
                    row.get("timestamp")
                ) < cutoff
            ]

            for symbol in stale:
                self._precision_context_cache.pop(
                    symbol,
                    None,
                )

            self.precision_context_updates += (
                context_updates
            )
            self.precision_context_failures += (
                context_failures
            )

        return {
            "available": True,
            "selected_symbols": selected,
            "eligible_candidates": len(
                candidates
            ),
            "sub_dollar_selected": sum(
                1
                for row in selected_rows
                if (
                    self._precision_number(
                        row.get("last")
                    )
                    < 1.0
                )
            ),
            "context_updates": context_updates,
            "context_failures": context_failures,
            "nominal_price_is_profit_score": False,
            "sub_dollar_is_coverage_bucket": True,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def _run_precision_scout(
        self,
    ) -> None:
        while not self._stop.is_set():
            started = time.monotonic()

            try:
                self._refresh_precision_scout_once()
            except Exception:
                with self._lock:
                    self.precision_scout_failures += 1

            elapsed = (
                time.monotonic()
                - started
            )

            self._stop.wait(
                max(
                    0.0,
                    self.precision_scout_refresh_seconds
                    - elapsed,
                )
            )

    def _refresh_discovery(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and self._candidates and now - self._last_discovery_at < self.discovery_refresh_seconds:
            return
        previous_map = dict(self._candidate_map)
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
        candidate_map = {str(row.get("symbol") or "").upper(): row for row in candidates if row.get("symbol")}
        if self.shadow_portfolio is not None:
            for symbol in self.shadow_portfolio.open_symbols():
                if symbol not in candidate_map and symbol in previous_map:
                    retained = dict(previous_map[symbol])
                    retained["_risk_only"] = True
                    candidates.append(retained)
                    candidate_map[symbol] = retained
        self._candidates = candidates
        self._candidate_map = candidate_map
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
        mandatory: list[dict[str, Any]] = []
        if self.shadow_portfolio is not None:
            for symbol in sorted(self.shadow_portfolio.open_symbols()):
                row = self._candidate_map.get(symbol)
                if row is not None:
                    mandatory.append(dict(row))
                else:
                    mandatory.append({"symbol": symbol, "_risk_only": True})
        with self._lock:
            sticky_micro_symbols = list(
                dict.fromkeys(
                    [
                        *self._precision_scout_symbols,
                        *self._microstream_symbols,
                    ]
                )
            )

        for symbol in sticky_micro_symbols:
            row = self._candidate_map.get(symbol)
            if row is not None:
                mandatory.append(dict(row))

        dedup: dict[str, dict[str, Any]] = {}
        for row in mandatory + selected:
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                dedup[symbol] = dict(row)
        return list(dedup.values())

    def precision_context_symbols(
        self,
    ) -> set[str]:
        """Symbols receiving both 0.5s micro and slower MTF context."""
        with self._lock:
            symbols = [
                *(
                    getattr(
                        self,
                        "_precision_scout_symbols",
                        [],
                    )
                    or []
                ),
                *(
                    getattr(
                        self,
                        "_microstream_symbols",
                        [],
                    )
                    or []
                ),
            ]

        return {
            str(symbol).upper()
            for symbol in symbols
            if str(symbol or "").strip()
        }

    @staticmethod
    def _closed_candles(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("market feed candles must be a DataFrame")
        if len(frame) < 2:
            return frame.iloc[0:0].copy()
        return frame.iloc[:-1].copy()

    @staticmethod
    def _last_close(frame: pd.DataFrame) -> float:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "close" not in frame.columns:
            return 0.0
        try:
            value = float(frame["close"].iloc[-1])
        except (TypeError, ValueError):
            return 0.0
        return value if value > 0 else 0.0

    @staticmethod
    def _score_from_dict(row: dict[str, Any]) -> OpportunityScore:
        return OpportunityScore(
            symbol=str(row.get("symbol") or "").upper(),
            score=float(row.get("score") or 0.0),
            net_capture_bps=float(row.get("net_capture_bps") or 0.0),
            modeled_round_trip_cost_bps=max(30.0, float(row.get("modeled_round_trip_cost_bps") or 30.0)),
            movement_frequency_per_minute=float(row.get("movement_frequency_per_minute") or 0.0),
            quality_multiplier=float(row.get("quality_multiplier") or 0.0),
            nominal_price=max(1e-12, float(row.get("nominal_price") or 1.0)),
            qualified=row.get("qualified") is True,
            reason=str(row.get("reason") or "unknown"),
            source=str(row.get("source") or "fast_swarm"),
        )

    def _assess_context(
        self,
        *,
        ranked: list[dict[str, Any]],
        one_minute_frames: dict[str, pd.DataFrame],
        required_symbols: set[str] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
        assessments: dict[str, dict[str, Any]] = {}
        extension_candidates: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        score_map = {str(row.get("symbol") or "").upper(): row for row in ranked}
        queue: list[str] = []
        for symbol in sorted(required_symbols or set()):
            if symbol in one_minute_frames and symbol not in queue:
                queue.append(symbol)
        for score in ranked:
            if score.get("qualified") is not True:
                continue
            symbol = str(score.get("symbol") or "").upper()
            if symbol and symbol not in queue:
                queue.append(symbol)
            if len([value for value in queue if value not in (required_symbols or set())]) >= self.max_context_symbols:
                break

        for symbol in queue:
            base_frame = one_minute_frames.get(symbol)
            if base_frame is None:
                continue
            context: dict[str, pd.DataFrame] = {self.timeframe: base_frame}
            for timeframe in self.context_timeframes:
                if timeframe == self.timeframe:
                    continue
                try:
                    context[timeframe] = self._closed_candles(self.feed.candles(symbol, timeframe, self.candle_limit))
                except Exception as exc:  # noqa: BLE001
                    errors[f"{symbol}:{timeframe}"] = f"{type(exc).__name__}: {exc}"
            measured_cost = max(30.0, float((score_map.get(symbol) or {}).get("modeled_round_trip_cost_bps") or 30.0))
            rows = self.timeframe_mind.assess_many(symbol=symbol, frames=context, modeled_round_trip_cost_bps=measured_cost)
            assessments[symbol] = {timeframe: row.as_dict() for timeframe, row in rows.items()}
            anchor = rows.get(self.timeframe)
            if anchor is None or not anchor.independently_qualified:
                continue
            for timeframe, row in rows.items():
                if timeframe == self.timeframe:
                    continue
                if not self.timeframe_mind.agrees_with_position(row, side=anchor.direction):
                    continue
                extension_candidates.append(
                    {
                        "symbol": symbol,
                        "anchor_timeframe": self.timeframe,
                        "anchor_side": anchor.direction,
                        "timeframe": timeframe,
                        "direction": row.direction,
                        "confidence": row.confidence,
                        "expected_edge_bps": row.expected_edge_bps,
                        "modeled_round_trip_cost_bps": row.modeled_round_trip_cost_bps,
                        "independently_qualified": True,
                        "shared_position_join_candidate": True,
                        "capital_allocated": False,
                        "execution_authority": False,
                    }
                )
        return assessments, extension_candidates, errors

    def _manage_shadow_exits(
        self,
        *,
        marks: dict[str, float],
        assessments: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.shadow_portfolio is None or self.outcome_journal is None:
            return []
        events: list[dict[str, Any]] = []
        for decision in self.shadow_portfolio.exit_candidates(marks=marks, timeframe_assessments=assessments):
            try:
                preview = self.shadow_portfolio.preview_net_return(decision["tranche_id"], reference_price=decision["reference_price"])
                self.runtime.swarm.close_agent_tranche(
                    agent_id=decision["agent_id"],
                    tranche_id=decision["tranche_id"],
                    exit_price=decision["reference_price"],
                    net_realized_pnl=preview["net_pnl"],
                )
                episode = self.shadow_portfolio.close_tranche(
                    decision["tranche_id"],
                    reference_price=decision["reference_price"],
                    reason=decision["reason"],
                )
                episode_id = self.outcome_journal.append(episode)
                events.append({**episode, "episode_id": episode_id})
                self.shadow_exits += 1
            except Exception as exc:  # noqa: BLE001
                self.shadow_failures += 1
                events.append({"tranche_id": decision.get("tranche_id"), "error": f"{type(exc).__name__}: {exc}"})
        return events

    def _open_shadow_candidates(
        self,
        *,
        ranked: list[dict[str, Any]],
        assessments: dict[str, dict[str, Any]],
        marks: dict[str, float],
    ) -> list[dict[str, Any]]:
        if self.shadow_portfolio is None:
            return []
        events: list[dict[str, Any]] = []
        score_map = {str(row.get("symbol") or "").upper(): row for row in ranked if row.get("qualified") is True}
        for symbol, score_payload in score_map.items():
            mark = marks.get(symbol, 0.0)
            rows = assessments.get(symbol) or {}
            anchor = rows.get(self.timeframe)
            if mark <= 0 or not isinstance(anchor, dict) or anchor.get("independently_qualified") is not True:
                continue
            side = str(anchor.get("direction") or "").lower()
            if side not in {"long", "short"}:
                continue
            opportunity = self._score_from_dict(score_payload)
            for timeframe in (self.timeframe, *self.context_timeframes):
                assessment = rows.get(timeframe)
                if not isinstance(assessment, dict):
                    continue
                if assessment.get("independently_qualified") is not True or str(assessment.get("direction") or "").lower() != side:
                    continue
                role = self.ROLE_BY_TIMEFRAME.get(timeframe)
                if role is None:
                    continue
                agent = self.runtime.swarm.find_agent(symbol=symbol, timeframe=timeframe, role=role)
                if agent is None:
                    agent = self.runtime.swarm.spawn_agent(role=role, timeframe=timeframe, symbol=symbol)
                if self.shadow_portfolio.has_open_agent(agent.agent_id):
                    continue
                marks_view = dict(marks)
                equity = self.shadow_portfolio.equity(marks_view)
                upstream = self.shadow_portfolio.remaining_deployable_notional(marks_view)
                if equity <= 0 or upstream <= 0:
                    continue
                decision = self.runtime.swarm.consider_join(
                    agent_id=agent.agent_id,
                    opportunity=opportunity,
                    side=side,
                    entry_price=mark,
                    requested_notional=self.base_order_usd,
                    equity=equity,
                    upstream_remaining_deployable_notional=upstream,
                    confidence=float(assessment.get("confidence") or 0.0),
                    expected_edge_bps=float(assessment.get("expected_edge_bps") or 0.0),
                    independently_qualified=True,
                    evidence_multiplier=max(0.0, min(1.0, opportunity.quality_multiplier)),
                )
                if decision.get("allowed") is not True:
                    continue
                try:
                    record = self.shadow_portfolio.open_tranche(
                        tranche_id=str(decision["tranche_id"]),
                        agent_id=agent.agent_id,
                        role=role.value,
                        timeframe=timeframe,
                        symbol=symbol,
                        side=side,
                        reference_price=mark,
                        notional=float(decision["allocated_notional"]),
                        confidence=float(assessment.get("confidence") or 0.0),
                        expected_edge_bps=float(assessment.get("expected_edge_bps") or 0.0),
                        modeled_round_trip_cost_bps=float(assessment.get("modeled_round_trip_cost_bps") or opportunity.modeled_round_trip_cost_bps),
                        marks=marks_view,
                        regime=f"{side}_{timeframe}",
                    )
                    events.append({"action": "shadow_open", **record})
                    self.shadow_entries += 1
                except Exception as exc:  # noqa: BLE001
                    self.runtime.swarm.rollback_join(agent_id=agent.agent_id, tranche_id=str(decision["tranche_id"]))
                    self.shadow_failures += 1
                    events.append({"action": "shadow_open_failed", "agent_id": agent.agent_id, "symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        return events

    def _microstructure_assess(
        self,
        *,
        ranked: list[dict[str, Any]],
        frames: dict[str, pd.DataFrame],
        profiles: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
        output: dict[str, Any] = {}
        marks: dict[str, float] = {}
        proposals: list[dict[str, Any]] = []
        # v1.45.1: the sub-minute scout must not depend on the slower
        # 1-minute opportunity gate. Inspect the strongest ranked markets,
        # then require the microstructure engine's own independent,
        # cost-aware qualification before any shadow proposal is allowed.
        profile_map = profiles or {}

        with self._lock:
            sticky_micro_symbols = set(
                self._microstream_symbols
            )

        candidates = [
            row for row in ranked
            if isinstance(row, dict)
            and str(row.get("symbol") or "").strip()
        ]

        def micro_priority(
            row: dict[str, Any],
        ) -> tuple[
            float,
            float,
            float,
            float,
            float,
        ]:
            symbol = str(
                row.get("symbol") or ""
            ).upper()
            profile = profile_map.get(symbol) or {}

            sticky = (
                1.0
                if symbol in sticky_micro_symbols
                else 0.0
            )

            fill = float(
                profile.get("fill_probability")
                or 0.0
            )
            liquidity = float(
                profile.get("liquidity_score")
                or 0.0
            )
            spread = float(
                profile.get("spread_bps")
                or 1_000_000.0
            )
            movement_score = float(
                row.get("score") or 0.0
            )

            return (
                sticky,
                movement_score,
                fill,
                liquidity,
                -spread,
            )

        # Examine a bounded pool rather than allowing the first two
        # movement-ranked but shallow markets to consume all micro slots.
        pool_limit = max(
            self.max_micro_symbols,
            self.max_micro_symbols * 4,
        )
        selected = sorted(
            candidates,
            key=micro_priority,
            reverse=True,
        )[:pool_limit]

        structurally_viable = 0
        active_microstream_symbols: list[str] = []

        for score in selected:
            symbol = str(score.get("symbol") or "").upper()
            frame = frames.get(symbol)
            if not symbol or frame is None or frame.empty:
                continue
            try:
                book = self.feed.order_book(symbol, limit=10)
                if hasattr(self.feed, "public_trades"):
                    trades = self.feed.public_trades(symbol, limit=80)
                elif (
                    hasattr(self.feed, "exchange")
                    and self.feed.exchange.has.get("fetchTrades", False)
                ):
                    trades = [
                        dict(row)
                        for row in self.feed.exchange.fetch_trades(
                            symbol, limit=80
                        ) or []
                    ]
                else:
                    trades = []
                reference = None
                if self.reference_feed is not None:
                    try:
                        reference = self.reference_feed.order_book(symbol, limit=10)
                    except Exception:
                        reference = None
                features = self.microstructure_sniper.extract(
                    symbol=symbol,
                    order_book=book,
                    trades=trades,
                    candles=frame,
                    reference_order_book=reference,
                )
                assessments = self.microstructure_sniper.assess(
                    features,
                    modeled_round_trip_cost_bps=max(
                        30.0, float(score.get("modeled_round_trip_cost_bps") or 30.0)
                    ),
                )
                evidence_rankings = (
                    self.micro_calibration_journal.evidence_rankings()
                    if self.micro_calibration_journal is not None
                    else {}
                )
                proposed = self.micro_agent_foundry.propose(
                    assessments,
                    evidence_rankings=evidence_rankings,
                )

                # Only symbols the dedicated stream can actually follow
                # may create prospective micro labels. This prevents the
                # evidence producer from outrunning the measurement sampler.
                total_depth = (
                    features.bid_depth_usd
                    + features.ask_depth_usd
                )
                structurally_eligible = bool(
                    features.spread_bps
                    <= self.microstructure_sniper.maximum_spread_bps
                    and total_depth
                    >= self.microstructure_sniper.minimum_depth_usd
                )

                temporal_ready_for_label = bool(
                    features.temporal_samples >= 3
                )

                kinematic_rows = (
                    [
                        row.as_dict()
                        for row in assessments
                        if str(
                            row.specialist
                        ).startswith("kinematic_")
                    ]
                    if temporal_ready_for_label
                    else []
                )

                if structurally_eligible:
                    active_microstream_symbols.append(
                        symbol
                    )

                    if not temporal_ready_for_label:
                        self.microstream_warmup_labels_skipped += (
                            len(assessments)
                        )

                    else:
                        if (
                            self.micro_calibration_journal
                            is not None
                        ):
                            if kinematic_rows:
                                added = (
                                    self.micro_calibration_journal.register(
                                        symbol=symbol,
                                        midpoint=features.midpoint,
                                        assessments=kinematic_rows,
                                        observed_at=features.timestamp,
                                    )
                                )

                                self.microstream_kinematic_labels_registered += (
                                    added
                                )
                            else:
                                self.microstream_non_event_labels_skipped += (
                                    len(assessments)
                                )

                        # One persistent burst must not manufacture many
                        # overlapping 2m/5m/15m observations. Select one
                        # representative assessment and impose a full
                        # 15-minute per-symbol/specialist/direction cooldown.
                        if (
                            self.slow_calibration_journal
                            is not None
                            and kinematic_rows
                        ):
                            event_row = max(
                                kinematic_rows,
                                key=lambda row: (
                                    float(
                                        row.get("confidence")
                                        or 0.0
                                    ),
                                    float(
                                        row.get(
                                            "expected_edge_bps"
                                        )
                                        or 0.0
                                    ),
                                ),
                            )

                            specialist = str(
                                event_row.get("specialist")
                                or "kinematic_unknown_v154"
                            )
                            direction = str(
                                event_row.get("direction")
                                or ""
                            ).lower()
                            regime = str(
                                event_row.get("regime")
                                or "micro_unknown"
                            )

                            event_key = (
                                f"{symbol}|"
                                f"{specialist}|"
                                f"{direction}"
                            )

                            last_registered = float(
                                self._kinematic_slow_last_registered.get(
                                    event_key,
                                    0.0,
                                )
                            )

                            cooldown_elapsed = (
                                features.timestamp
                                - last_registered
                                >= self.KINEMATIC_SLOW_COOLDOWN_SECONDS
                            )

                            if cooldown_elapsed:
                                base_specialist = (
                                    specialist.removesuffix(
                                        "_v154"
                                    )
                                )

                                slow_rows = [
                                    {
                                        "horizon_seconds": horizon,
                                        "direction": direction,
                                        "confidence": float(
                                            event_row.get(
                                                "confidence"
                                            )
                                            or 0.0
                                        ),
                                        "pressure_score": float(
                                            event_row.get(
                                                "pressure_score"
                                            )
                                            or 0.0
                                        ),

                                        # Do not transplant the short-horizon
                                        # heuristic edge into the slow study.
                                        # Actual prospective outcome evidence
                                        # is authoritative.
                                        "expected_edge_bps": 0.0,

                                        "modeled_round_trip_cost_bps": max(
                                            30.0,
                                            float(
                                                event_row.get(
                                                    "modeled_round_trip_cost_bps"
                                                )
                                                or 30.0
                                            ),
                                        ),

                                        "independently_qualified": False,

                                        "reason": (
                                            "kinematic_event_"
                                            "followthrough_research"
                                        ),

                                        "specialist": (
                                            f"{base_specialist}"
                                            "_followthrough_v155"
                                        ),

                                        "regime": regime,

                                        "automatic_promotion": False,
                                        "execution_authority": False,
                                        "testnet_authority": False,
                                        "live_authority": False,
                                    }
                                    for horizon in (
                                        self.KINEMATIC_SLOW_HORIZONS
                                    )
                                ]

                                slow_added = (
                                    self.slow_calibration_journal.register(
                                        symbol=symbol,
                                        midpoint=features.midpoint,
                                        assessments=slow_rows,
                                        observed_at=features.timestamp,
                                    )
                                )

                                if slow_added:
                                    self._kinematic_slow_last_registered[
                                        event_key
                                    ] = features.timestamp

                                    self.kinematic_slow_event_triggers_registered += 1
                                    self.kinematic_slow_labels_registered += (
                                        slow_added
                                    )

                            else:
                                self.kinematic_slow_labels_suppressed_by_cooldown += (
                                    len(
                                        self.KINEMATIC_SLOW_HORIZONS
                                    )
                                )

                output[symbol] = {
                    "features": features.as_dict(),
                    "path_assessments": [
                        r.as_dict()
                        for r in assessments
                    ],
                    "foundry_proposals": proposed,
                    "microstream_tracked": structurally_eligible,
                    "temporal_ready_for_prospective_label": (
                        temporal_ready_for_label
                    ),
                    "automatic_promotion": False,
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                }

                marks[symbol] = features.midpoint
                proposals.extend(proposed)

                self.micro_assessments += len(assessments)
                self.micro_qualified += sum(
                    1
                    for r in assessments
                    if r.independently_qualified
                )

                if structurally_eligible:
                    structurally_viable += 1
                    if structurally_viable >= self.max_micro_symbols:
                        break

            except Exception as exc:
                self.micro_fetch_failures += 1
                output[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self._microstream_symbols = list(
                dict.fromkeys(active_microstream_symbols)
            )[: self.max_micro_symbols]

        return output, marks, proposals

    def _open_micro_shadow_candidates(
        self,
        *,
        proposals: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
        marks: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Open only independently-qualified micro proposals in shadow capital.

        This never mutates the canonical paper ledger and has no Testnet/live
        execution authority.
        """
        if self.shadow_portfolio is None:
            return []

        score_map = {
            str(row.get("symbol") or "").upper(): row
            for row in ranked
            if isinstance(row, dict)
        }
        events: list[dict[str, Any]] = []

        for proposal in proposals:
            if proposal.get("evidence_qualified") is not True:
                continue
            if proposal.get("independently_qualified") is not True:
                continue
            if proposal.get("execution_authority") is not False:
                continue

            symbol = str(proposal.get("symbol") or "").upper()
            side = str(proposal.get("side") or "").lower()
            horizon = int(proposal.get("horizon_seconds") or 0)
            mark = float(marks.get(symbol) or 0.0)

            if not symbol or side not in {"long", "short"}:
                continue
            if horizon not in {5, 15, 30, 60}:
                continue
            if mark <= 0:
                continue

            score = score_map.get(symbol) or {}
            modeled_cost = max(
                30.0,
                float(
                    proposal.get("modeled_round_trip_cost_bps")
                    or score.get("modeled_round_trip_cost_bps")
                    or 30.0
                ),
            )
            expected_edge = float(proposal.get("expected_edge_bps") or 0.0)
            confidence = float(proposal.get("confidence") or 0.0)

            # Recheck fail-closed evidence constraints at the
            # shadow-entry seam. The conservative prospective net edge,
            # not the original heuristic edge, is authoritative here.
            conservative_net = float(
                proposal.get("conservative_net_edge_bps") or 0.0
            )
            evidence_samples = int(
                proposal.get("evidence_samples") or 0
            )

            if conservative_net <= 0.0:
                continue
            if evidence_samples < 30:
                continue
            if expected_edge <= modeled_cost:
                continue
            if confidence <= 0.0:
                continue

            role = (
                AgentRole.REVERSAL
                if proposal.get("specialist") == "reversal_snapper"
                else AgentRole.SCALP
            )
            agent_timeframe = (
                f"micro-{horizon}s-"
                f"{str(proposal.get('specialist') or 'micro')}"
            )

            agent = self.runtime.swarm.find_agent(
                symbol=symbol,
                timeframe=agent_timeframe,
                role=role,
            )
            if agent is None:
                agent = self.runtime.swarm.spawn_agent(
                    role=role,
                    timeframe=agent_timeframe,
                    symbol=symbol,
                )

            if self.shadow_portfolio.has_open_agent(agent.agent_id):
                continue

            equity = self.shadow_portfolio.equity(marks)
            deployable = self.shadow_portfolio.remaining_deployable_notional(marks)
            if equity <= 0 or deployable <= 0:
                continue

            # Small bounded shadow allocation only.
            requested = min(
                self.base_order_usd,
                deployable,
                max(0.01, equity * 0.01),
            )
            if requested <= 0:
                continue

            # This is a distinct microstructure opportunity, not a claim that
            # the slower 1-minute radar qualified. Its edge is derived only
            # from the independently-qualified micro proposal after costs.
            net_micro_edge = max(0.0, expected_edge - modeled_cost)
            opportunity = OpportunityScore(
                symbol=symbol,
                score=(
                    net_micro_edge
                    * max(
                        1.0,
                        float(score.get("movement_frequency_per_minute") or 1.0),
                    )
                    * max(
                        confidence,
                        float(score.get("quality_multiplier") or 0.0),
                    )
                ),
                net_capture_bps=net_micro_edge,
                modeled_round_trip_cost_bps=modeled_cost,
                movement_frequency_per_minute=max(
                    0.0,
                    float(score.get("movement_frequency_per_minute") or 0.0),
                ),
                quality_multiplier=max(
                    0.0,
                    min(
                        1.0,
                        max(
                            confidence,
                            float(score.get("quality_multiplier") or 0.0),
                        ),
                    ),
                ),
                nominal_price=mark,
                qualified=True,
                reason="prospective_evidence_qualified_micro_edge",
                source="evidence_ranked_micro_v1.48",
            )

            decision = self.runtime.swarm.consider_join(
                agent_id=agent.agent_id,
                opportunity=opportunity,
                side=side,
                entry_price=mark,
                requested_notional=requested,
                equity=equity,
                upstream_remaining_deployable_notional=deployable,
                confidence=confidence,
                expected_edge_bps=expected_edge,
                independently_qualified=True,
                evidence_multiplier=max(0.0, min(1.0, confidence)),
            )

            if decision.get("allowed") is not True:
                continue

            try:
                record = self.shadow_portfolio.open_tranche(
                    tranche_id=str(decision["tranche_id"]),
                    agent_id=agent.agent_id,
                    role=role.value,
                    timeframe=f"{horizon}s",
                    symbol=symbol,
                    side=side,
                    reference_price=mark,
                    notional=float(decision["allocated_notional"]),
                    confidence=confidence,
                    expected_edge_bps=expected_edge,
                    modeled_round_trip_cost_bps=modeled_cost,
                    marks=marks,
                    regime=str(proposal.get("regime") or "micro_unknown"),
                )
                events.append(
                    {
                        "action": "micro_shadow_open",
                        "specialist": proposal.get("specialist"),
                        **record,
                    }
                )
                self.shadow_entries += 1
            except Exception as exc:
                self.runtime.swarm.rollback_join(
                    agent_id=agent.agent_id,
                    tranche_id=str(decision["tranche_id"]),
                )
                self.shadow_failures += 1
                events.append(
                    {
                        "action": "micro_shadow_open_failed",
                        "agent_id": agent.agent_id,
                        "symbol": symbol,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return events

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
            except Exception as exc:  # noqa: BLE001
                fetch_errors[symbol] = f"{type(exc).__name__}: {exc}"

        runtime_candidates = [row for row in selected if row.get("_risk_only") is not True]
        result = self.runtime.evaluate_batch(candidates=runtime_candidates, frames=frames, timeframe_seconds=self.timeframe_seconds)
        ranked_rows = list(result.get("ranked") or [])
        with self._lock:
            self.ranked_opportunities_total += len(ranked_rows)
            self.qualified_opportunities_total += sum(
                1 for row in ranked_rows if isinstance(row, dict) and row.get("qualified") is True
            )
        microstructure, micro_marks, micro_proposals = self._microstructure_assess(
            ranked=ranked_rows,
            frames=frames,
            profiles=dict(result.get("profiles") or {}),
        )
        result["microstructure"] = microstructure
        result["micro_agent_foundry_proposals"] = micro_proposals
        result["microstructure_marks"] = micro_marks
        result["microstructure_is_trade_authority"] = False
        required_symbols = set(
            self.shadow_portfolio.open_symbols()
            if self.shadow_portfolio is not None
            else set()
        )

        # v1.59: the same bounded six-symbol precision universe sampled
        # every 0.5s must also receive slower MTF confirmation. Previously
        # MTF context was mostly reserved for already-qualified slow-radar
        # symbols, leaving velocity-first candidates with mtf_confidence=0.
        precision_context_symbols = (
            self.precision_context_symbols()
        )
        required_symbols.update(
            precision_context_symbols
        )

        assessments, extension_candidates, context_errors = self._assess_context(
            ranked=list(result.get("ranked") or []),
            one_minute_frames=frames,
            required_symbols=required_symbols,
        )
        fetch_errors.update(context_errors)
        marks = {symbol: self._last_close(frame) for symbol, frame in frames.items() if self._last_close(frame) > 0}
        # Prefer the current micro midpoint for sub-minute shadow accounting.
        marks.update(micro_marks)

        slow_registered = 0
        if self.slow_calibration_journal is not None:
            slow_horizons = {
                "1m": 120,
                "5m": 300,
                "15m": 900,
            }

            for symbol, timeframe_rows in assessments.items():
                if not isinstance(timeframe_rows, dict):
                    continue

                midpoint = float(marks.get(symbol) or 0.0)
                if midpoint <= 0:
                    continue

                registration_rows: list[dict[str, Any]] = []

                for timeframe, horizon in slow_horizons.items():
                    row = timeframe_rows.get(timeframe)
                    if not isinstance(row, dict):
                        continue

                    direction = str(
                        row.get("direction") or ""
                    ).lower()

                    if direction not in {"long", "short"}:
                        continue

                    registration_rows.append(
                        {
                            "horizon_seconds": horizon,
                            "direction": direction,
                            "confidence": float(
                                row.get("confidence") or 0.0
                            ),
                            "pressure_score": float(
                                row.get(
                                    "directional_consistency"
                                ) or 0.0
                            ),
                            "expected_edge_bps": float(
                                row.get(
                                    "expected_edge_bps"
                                ) or 0.0
                            ),
                            "modeled_round_trip_cost_bps": max(
                                30.0,
                                float(
                                    row.get(
                                        "modeled_round_trip_cost_bps"
                                    ) or 30.0
                                ),
                            ),
                            "independently_qualified": (
                                row.get(
                                    "independently_qualified"
                                ) is True
                            ),
                            "reason": str(
                                row.get("reason") or "unknown"
                            ),
                            "specialist": (
                                f"timeframe_mind_{timeframe}"
                            ),
                            "regime": direction,
                        }
                    )

                if registration_rows:
                    slow_registered += (
                        self.slow_calibration_journal.register(
                            symbol=symbol,
                            midpoint=midpoint,
                            assessments=registration_rows,
                            observed_at=time.time(),
                        )
                    )

        micro_calibration_resolved = 0
        if (
            self.micro_calibration_journal is not None
            and self.micro_calibration_feed is None
        ):
            # Fallback only. When the dedicated 1-second sampler exists,
            # the slower swarm loop must not consume horizon labels.
            micro_calibration_resolved = (
                self.micro_calibration_journal.resolve(marks=marks)
            )

        shadow_close_events = self._manage_shadow_exits(
            marks=marks,
            assessments=assessments,
        )
        shadow_open_events = self._open_shadow_candidates(
            ranked=list(result.get("ranked") or []),
            assessments=assessments,
            marks=marks,
        )
        micro_shadow_events = self._open_micro_shadow_candidates(
            proposals=micro_proposals,
            ranked=ranked_rows,
            marks=marks,
        )

        result["micro_shadow_events"] = micro_shadow_events
        result["micro_shadow_is_canonical_paper"] = False
        result["micro_calibration_resolved"] = micro_calibration_resolved
        result["micro_calibration"] = (
            self.micro_calibration_journal.health()
            if self.micro_calibration_journal is not None
            else {}
        )
        result["slow_calibration_registered"] = slow_registered
        result["slow_calibration"] = (
            self.slow_calibration_journal.health()
            if self.slow_calibration_journal is not None
            else {}
        )
        result["timeframe_assessments"] = assessments
        result["precision_context_symbols"] = sorted(
            precision_context_symbols
        )
        result["precision_context_mtf_ready"] = sum(
            1
            for symbol in precision_context_symbols
            if bool(
                assessments.get(symbol)
            )
        )
        result["precision_context_is_execution_authority"] = False
        result["shared_position_extension_candidates"] = extension_candidates
        result["extension_candidates_are_trade_authority"] = False
        result["shadow_open_events"] = shadow_open_events
        result["shadow_close_events"] = shadow_close_events
        result["shadow_execution_is_canonical_paper"] = False
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
        if self.shadow_portfolio is not None:
            result["shadow_portfolio"] = self.shadow_portfolio.health(marks)
        if self.outcome_journal is not None:
            result["swarm_outcome_journal"] = self.outcome_journal.health()
        with self._lock:
            self.cycles += 1
            self.last_step = result
            self.last_error = None
            self.last_success_at = time.time()
            self.consecutive_failures = 0
        return dict(result)

    @staticmethod
    def _book_midpoint(book: dict[str, Any]) -> float:
        bids = list(book.get("bids") or [])
        asks = list(book.get("asks") or [])
        if not bids or not asks:
            return 0.0
        try:
            bid = float(bids[0][0])
            ask = float(asks[0][0])
        except (TypeError, ValueError, IndexError):
            return 0.0
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0
        return (bid + ask) / 2.0

    @staticmethod
    def _unique_symbols(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            symbol = str(
                value or ""
            ).upper()

            if (
                symbol
                and symbol not in result
            ):
                result.append(symbol)

        return result

    @classmethod
    def _build_microstream_queue(
        cls,
        *,
        scout_symbols: list[str],
        sticky_symbols: list[str],
        due_symbols: list[str],
        capacity: int,
        cursor: int,
    ) -> tuple[
        list[str],
        int,
        dict[str, int],
    ]:
        """Bound execution scouting and research without priority inversion."""

        cap = max(
            1,
            min(6, int(capacity)),
        )

        scouts = cls._unique_symbols(
            scout_symbols
        )
        sticky = cls._unique_symbols(
            sticky_symbols
        )
        due = cls._unique_symbols(
            due_symbols
        )

        if not scouts:
            queue: list[str] = []

            for symbol in [
                *due,
                *sticky,
            ]:
                if (
                    symbol not in queue
                    and len(queue) < cap
                ):
                    queue.append(symbol)

            return (
                queue,
                int(cursor),
                {
                    "scout_slots": 0,
                    "due_slots": sum(
                        1
                        for symbol in queue
                        if symbol in due
                    ),
                },
            )

        offset = (
            int(cursor)
            % len(scouts)
        )

        rotated = (
            scouts[offset:]
            + scouts[:offset]
        )

        # When research labels are due, preserve exactly one bounded
        # opportunity for them while keeping five of six deep slots
        # available to current market movers.
        scout_reserve = (
            cap
            if not due
            else max(
                1,
                cap - 1,
            )
        )

        scout_take = min(
            len(rotated),
            scout_reserve,
        )

        queue = list(
            rotated[:scout_take]
        )

        due_inserted = 0

        for symbol in due:
            if len(queue) >= cap:
                break

            if symbol not in queue:
                queue.append(symbol)
                due_inserted += 1
                break

        # If fewer than the reserved execution symbols exist, fill the
        # remainder with sticky micro markets. Then allow the remaining
        # scout rotation back in before any second research symbol.
        for symbol in [
            *sticky,
            *rotated[scout_take:],
        ]:
            if len(queue) >= cap:
                break

            if symbol not in queue:
                queue.append(symbol)

        if len(queue) < cap:
            for symbol in due:
                if len(queue) >= cap:
                    break

                if symbol not in queue:
                    queue.append(symbol)
                    due_inserted += 1

        next_cursor = (
            (
                offset
                + max(
                    1,
                    scout_take,
                )
            )
            % len(scouts)
        )

        return (
            queue[:cap],
            next_cursor,
            {
                "scout_slots": sum(
                    1
                    for symbol in queue[:cap]
                    if symbol in scouts
                ),
                "due_slots": due_inserted,
            },
        )

    def pin_execution_symbols(
        self,
        symbols: list[str] | tuple[str, ...] | set[str],
        *,
        ttl_seconds: float = 10.0,
    ) -> None:
        """Keep currently managed Testnet positions in deep market sampling."""

        now = time.time()
        expires = (
            now
            + max(
                3.0,
                float(ttl_seconds),
            )
        )

        normalized = self._unique_symbols(
            [
                str(symbol)
                for symbol in symbols
            ]
        )

        with self._lock:
            stale = [
                symbol
                for symbol, until in (
                    self._execution_precision_pins.items()
                )
                if float(until) <= now
            ]

            for symbol in stale:
                self._execution_precision_pins.pop(
                    symbol,
                    None,
                )

            for symbol in normalized:
                self._execution_precision_pins[
                    symbol
                ] = expires

    def pin_execution_candidate_symbols(
        self,
        symbols: list[str] | tuple[str, ...] | set[str],
        *,
        ttl_seconds: float = 6.0,
    ) -> None:
        """Prioritize entry candidates only until a fresh sample arrives.

        This is read-only market-data scheduling. It cannot submit an
        order and does not alter the execution freshness threshold.
        """

        now = time.time()
        expires = (
            now
            + max(
                2.0,
                float(ttl_seconds),
            )
        )

        normalized = self._unique_symbols(
            [
                str(symbol)
                for symbol in symbols
            ]
        )

        with self._lock:
            stale = [
                symbol
                for symbol, until in (
                    self._execution_candidate_pins.items()
                )
                if float(until) <= now
            ]

            for symbol in stale:
                self._execution_candidate_pins.pop(
                    symbol,
                    None,
                )

            for symbol in normalized:
                # Reinsert so the newest execution request has an
                # unambiguous bounded priority order.
                self._execution_candidate_pins.pop(
                    symbol,
                    None,
                )
                self._execution_candidate_pins[
                    symbol
                ] = expires

        # v1.60.29: wake the sleeping microstream so a pinned candidate can be
        # sampled without waiting out the remaining cadence. Ordering, capacity
        # and freshness rules are unchanged. Legacy/lightweight instances built
        # with object.__new__ never create the event; pinning must still work
        # for them, so the wake is skipped rather than raising.
        wake_event = getattr(self, "_execution_pin_event", None)
        if normalized and wake_event is not None:
            wake_event.set()

    @classmethod
    def _build_sticky_precision_queue(
        cls,
        *,
        scout_symbols: list[str],
        sticky_symbols: list[str],
        due_symbols: list[str],
        velocity_symbols: list[str],
        capacity: int,
        cursor: int,
        now: float,
        current_hot: list[str],
        hot_until: float,
        current_explorer: str | None,
        explorer_until: float,
        hot_hold_seconds: float = 4.0,
        explorer_hold_seconds: float = 3.0,
        priority_symbols: list[str] | None = None,
    ) -> tuple[
        list[str],
        int,
        dict[str, object],
    ]:
        """Build one continuity-preserving deep-micro schedule."""

        cap = max(
            1,
            min(
                12,
                int(capacity),
            ),
        )

        scouts = cls._unique_symbols(
            scout_symbols
        )

        sticky = cls._unique_symbols(
            sticky_symbols
        )

        urgent = cls._unique_symbols(
            priority_symbols or []
        )

        due = cls._unique_symbols(
            due_symbols
        )

        velocity = cls._unique_symbols(
            velocity_symbols
        )

        previous_hot = cls._unique_symbols(
            current_hot
        )

        valid_live = cls._unique_symbols(
            [
                *urgent,
                *velocity,
                *scouts,
                *sticky,
            ]
        )

        # At most one bounded research label-resolution slot.
        # Research must never consume half of a latency-constrained
        # execution cohort. It receives a slot only when at least four
        # simultaneous deep samples fit inside the freshness budget.
        due_slots = (
            1
            if due and cap >= 4
            else 0
        )

        # Preserve one exploration slot when possible. Unlike the old
        # rotation, the explorer remains for several seconds so it can
        # accumulate temporal evidence rather than receiving one sample.
        explorer_slots = (
            1
            if (
                len(scouts) > 1
                and cap - due_slots >= 2
            )
            else 0
        )

        hot_slots = max(
            1,
            cap
            - due_slots
            - explorer_slots,
        )

        hold_active = bool(
            previous_hot
            and now < float(
                hot_until or 0.0
            )
        )

        if hold_active:
            priority = cls._unique_symbols(
                [
                    # A symbol actively waiting for a fresh execution
                    # handoff outranks continuity/research scheduling.
                    *urgent,

                    # A genuinely moving sampled market may immediately
                    # promote into the cohort.
                    *velocity,
                    *[
                        symbol
                        for symbol in previous_hot
                        if symbol in valid_live
                    ],
                    *scouts,
                    *sticky,
                ]
            )
        else:
            priority = cls._unique_symbols(
                [
                    *urgent,
                    *velocity,
                    *scouts,
                    *sticky,
                    *previous_hot,
                ]
            )

        hot: list[str] = []

        for symbol in priority:
            if len(hot) >= hot_slots:
                break

            if symbol not in hot:
                hot.append(symbol)

        if not hot and due:
            hot.append(due[0])

        normalized_previous_hot = [
            symbol
            for symbol in previous_hot
            if symbol in valid_live
        ][:hot_slots]

        hot_changed = (
            hot != normalized_previous_hot
        )

        next_hot_until = (
            float(hot_until or 0.0)
            if (
                hold_active
                and not hot_changed
            )
            else now
            + max(
                2.0,
                float(hot_hold_seconds),
            )
        )

        explorer: str | None = None
        next_explorer_until = float(
            explorer_until or 0.0
        )
        next_cursor = int(cursor)

        exploration_pool = [
            symbol
            for symbol in scouts
            if symbol not in hot
        ]

        current_explorer_normalized = (
            str(
                current_explorer
                or ""
            ).upper()
        )

        explorer_rotated = False

        if (
            explorer_slots
            and exploration_pool
        ):
            if (
                current_explorer_normalized
                in exploration_pool
                and now
                < next_explorer_until
            ):
                explorer = (
                    current_explorer_normalized
                )

            else:
                index = (
                    int(cursor)
                    % len(
                        exploration_pool
                    )
                )

                explorer = (
                    exploration_pool[index]
                )

                next_cursor = (
                    index + 1
                ) % len(
                    exploration_pool
                )

                next_explorer_until = (
                    now
                    + max(
                        2.0,
                        float(
                            explorer_hold_seconds
                        ),
                    )
                )

                explorer_rotated = True

        queue: list[str] = []

        for symbol in hot:
            if (
                symbol
                and symbol not in queue
                and len(queue) < cap
            ):
                queue.append(symbol)

        if (
            explorer
            and explorer not in queue
            and len(queue) < cap
        ):
            queue.append(explorer)

        due_inserted = 0

        for symbol in due:
            if len(queue) >= cap:
                break

            if symbol not in queue:
                queue.append(symbol)
                due_inserted = 1
                break

        # Fill only when normal slot categories did not consume capacity.
        for symbol in [
            *sticky,
            *scouts,
        ]:
            if len(queue) >= cap:
                break

            if symbol not in queue:
                queue.append(symbol)

        promoted = [
            symbol
            for symbol in velocity
            if (
                symbol in hot
                and symbol
                not in normalized_previous_hot
            )
        ]

        return (
            queue[:cap],
            next_cursor,
            {
                "hot_symbols": list(hot),
                "hot_until": (
                    next_hot_until
                ),
                "explorer_symbol": (
                    explorer
                ),
                "explorer_until": (
                    next_explorer_until
                ),
                "hot_slots": len(hot),
                "explorer_slots": (
                    1 if explorer else 0
                ),
                "due_slots": due_inserted,
                "scout_slots": sum(
                    1
                    for symbol in queue[:cap]
                    if symbol in scouts
                ),
                "promoted_symbols": promoted,
                "explorer_rotated": (
                    explorer_rotated
                ),
            },
        )

    def _adaptive_microstream_capacity(
        self,
        *,
        scout_count: int,
    ) -> int:
        """Adapt deep sampling while keeping the queue inside freshness SLA."""

        maximum = max(
            1,
            min(
                12,
                int(
                    getattr(
                        self,
                        "max_micro_symbols",
                        6,
                    )
                ),
            ),
        )

        minimum = min(
            maximum,
            2,
        )

        current = max(
            minimum,
            min(
                maximum,
                int(
                    getattr(
                        self,
                        "_precision_micro_capacity",
                        min(6, maximum),
                    )
                ),
            ),
        )

        loop_seconds = max(
            0.0,
            float(
                getattr(
                    self,
                    "microstream_last_loop_seconds",
                    0.0,
                )
                or 0.0
            ),
        )

        previous_queue = (
            getattr(
                self,
                "precision_micro_last_queue",
                [],
            )
            or []
        )

        # Older fixtures do not create the queue. Using current capacity
        # gives the correct per-symbol estimate rather than pretending
        # the entire loop belonged to one symbol.
        previous_queue_size = max(
            1,
            len(previous_queue)
            or current,
        )

        previous_estimate = max(
            0.02,
            float(
                getattr(
                    self,
                    "microstream_per_symbol_latency_seconds",
                    0.25,
                )
            ),
        )

        if loop_seconds > 0.0:
            observed_per_symbol = (
                loop_seconds
                / previous_queue_size
            )

            estimate = (
                0.75 * previous_estimate
                + 0.25 * observed_per_symbol
            )
        else:
            estimate = previous_estimate

        estimate = max(
            0.02,
            estimate,
        )

        self.microstream_per_symbol_latency_seconds = (
            estimate
        )

        target_loop = max(
            0.75,
            min(
                1.50,
                float(
                    getattr(
                        self,
                        "microstream_target_loop_seconds",
                        1.50,
                    )
                ),
            ),
        )

        freshness_seconds = max(
            target_loop,
            float(
                getattr(
                    self,
                    "microstream_freshness_seconds",
                    2.0,
                )
            ),
        )

        latency_capacity = max(
            minimum,
            min(
                maximum,
                int(
                    target_loop
                    // estimate
                ),
            ),
        )

        previous_failures = int(
            getattr(
                self,
                "_precision_micro_last_failure_count",
                0,
            )
        )

        current_failures = int(
            getattr(
                self,
                "microstream_sample_failures",
                0,
            )
        )

        new_failures = max(
            0,
            current_failures
            - previous_failures,
        )

        self._precision_micro_last_failure_count = (
            current_failures
        )

        scout_count = max(
            0,
            int(scout_count),
        )

        desired = current

        if new_failures > 0:
            # Provider/API pressure wins immediately.
            desired = max(
                minimum,
                current - 1,
            )

        elif (
            loop_seconds > freshness_seconds
        ):
            # A queue that already violates the execution freshness SLA
            # contracts immediately, possibly by more than one slot.
            desired = min(
                max(
                    minimum,
                    current - 1,
                ),
                latency_capacity,
            )

        elif (
            loop_seconds > 0.0
            and loop_seconds < target_loop
            and scout_count > current
            and latency_capacity > current
        ):
            # Preserve the proven adaptive behavior: grow one deep market
            # per healthy loop rather than jumping immediately to the max.
            desired = min(
                maximum,
                current + 1,
                latency_capacity,
            )

        elif latency_capacity < current:
            desired = max(
                minimum,
                latency_capacity,
            )

        elif loop_seconds <= 0.0:
            desired = min(
                maximum,
                max(
                    minimum,
                    min(
                        6,
                        max(
                            1,
                            scout_count,
                        ),
                    ),
                ),
            )

        self._precision_micro_capacity = (
            desired
        )

        return desired

    def _run_microstream(
        self,
        generation: int | None = None,
        feed: Any | None = None,
    ) -> None:
        """Continuously sample bounded micro markets with no trade authority."""

        cadence_seconds = 0.5

        generation = (
            int(
                getattr(
                    self,
                    "_microstream_generation",
                    0,
                )
            )
            if generation is None
            else int(generation)
        )

        active_feed = (
            self.microstream_feed
            if feed is None
            else feed
        )

        while (
            not self._stop.is_set()
            and generation
            == int(
                getattr(
                    self,
                    "_microstream_generation",
                    generation,
                )
            )
        ):
            started = time.monotonic()

            try:
                now = time.time()
                journal = self.micro_calibration_journal

                if journal is not None:
                    journal.censor_expired(
                        observed_at=now
                    )

                velocity_rows = (
                    self.micro_velocity_candidates(
                        limit=max(
                            4,
                            self.max_micro_symbols,
                        ),
                        max_age_seconds=3.0,
                    )
                )

                velocity_symbols = [
                    str(
                        row.get("symbol")
                        or ""
                    ).upper()
                    for row in velocity_rows
                    if str(
                        row.get("symbol")
                        or ""
                    ).strip()
                ]

                with self._lock:
                    expired_pins = [
                        symbol
                        for symbol, until in (
                            self._execution_precision_pins.items()
                        )
                        if float(until) <= now
                    ]

                    for symbol in expired_pins:
                        self._execution_precision_pins.pop(
                            symbol,
                            None,
                        )

                    expired_candidate_pins = [
                        symbol
                        for symbol, until in (
                            self._execution_candidate_pins.items()
                        )
                        if float(until) <= now
                    ]

                    for symbol in expired_candidate_pins:
                        self._execution_candidate_pins.pop(
                            symbol,
                            None,
                        )

                    execution_pins = list(
                        self._execution_precision_pins
                    )

                    # Newest candidate request goes first.
                    candidate_pins = list(
                        reversed(
                            list(
                                self._execution_candidate_pins
                            )
                        )
                    )

                    scout_symbols = (
                        self._unique_symbols(
                            [
                                *candidate_pins,
                                *execution_pins,
                                *self._precision_scout_symbols,
                            ]
                        )
                    )

                    sticky_symbols = (
                        self._unique_symbols(
                            [
                                *candidate_pins,
                                *execution_pins,
                                *self._microstream_symbols,
                            ]
                        )
                    )

                    cursor = int(
                        self._precision_micro_cursor
                    )
                    current_hot = list(
                        self._precision_hot_symbols
                    )
                    hot_until = float(
                        self._precision_hot_until
                        or 0.0
                    )
                    current_explorer = (
                        self._precision_explorer_symbol
                    )
                    explorer_until = float(
                        self._precision_explorer_until
                        or 0.0
                    )

                due: list[str] = []

                if journal is not None:
                    due = journal.due_symbols(
                        observed_at=now,
                        lookahead_seconds=0.25,
                        limit=max(
                            2,
                            self.max_micro_symbols,
                        ),
                    )

                capacity = (
                    self._adaptive_microstream_capacity(
                        scout_count=len(
                            scout_symbols
                        ),
                    )
                )

                (
                    queue,
                    next_cursor,
                    schedule,
                ) = (
                    self._build_sticky_precision_queue(
                        scout_symbols=(
                            scout_symbols
                        ),
                        sticky_symbols=(
                            sticky_symbols
                        ),
                        due_symbols=due,
                        velocity_symbols=(
                            velocity_symbols
                        ),
                        capacity=capacity,
                        cursor=cursor,
                        now=now,
                        current_hot=(
                            current_hot
                        ),
                        hot_until=hot_until,
                        current_explorer=(
                            current_explorer
                        ),
                        explorer_until=(
                            explorer_until
                        ),
                        hot_hold_seconds=(
                            self.precision_hot_hold_seconds
                        ),
                        explorer_hold_seconds=(
                            self.precision_explorer_hold_seconds
                        ),
                        priority_symbols=(
                            candidate_pins
                        ),
                    )
                )

                with self._lock:
                    self._precision_micro_cursor = (
                        next_cursor
                    )
                    self.precision_micro_last_queue = (
                        list(queue)
                    )
                    self.precision_micro_live_slots = (
                        int(
                            schedule.get(
                                "scout_slots"
                            )
                            or 0
                        )
                    )
                    self.precision_micro_due_slots = (
                        int(
                            schedule.get(
                                "due_slots"
                            )
                            or 0
                        )
                    )

                    self._precision_hot_symbols = list(
                        schedule.get(
                            "hot_symbols"
                        )
                        or []
                    )

                    self._precision_hot_until = float(
                        schedule.get(
                            "hot_until"
                        )
                        or 0.0
                    )

                    self._precision_explorer_symbol = (
                        str(
                            schedule.get(
                                "explorer_symbol"
                            )
                            or ""
                        ).upper()
                        or None
                    )

                    self._precision_explorer_until = float(
                        schedule.get(
                            "explorer_until"
                        )
                        or 0.0
                    )

                    self.precision_hot_promotions += len(
                        schedule.get(
                            "promoted_symbols"
                        )
                        or []
                    )

                    if schedule.get(
                        "explorer_rotated"
                    ) is True:
                        self.precision_explorer_rotations += 1

                for symbol in queue:
                    if self._stop.is_set():
                        break

                    self.microstream_sample_attempts += 1

                    try:
                        # Precision lane deliberately uses only the
                        # top/depth order book. Public-trade REST history
                        # remains available to the slower full assessment,
                        # but cannot delay 5-60 second midpoint resolution.
                        trades = None

                        # Fetch the book last and timestamp it immediately.
                        # This timestamp is the authoritative future midpoint
                        # observation used by the prospective journal.
                        with self._lock:
                            self.microstream_last_attempt_started_at = (
                                time.time()
                            )
                            self.microstream_last_attempt_symbol = (
                                symbol
                            )

                        book = active_feed.order_book(
                            symbol,
                            limit=10,
                        )

                        # A retired worker that eventually returns
                        # after watchdog replacement must never
                        # overwrite fresher observations.
                        if (
                            generation
                            != int(
                                getattr(
                                    self,
                                    "_microstream_generation",
                                    generation,
                                )
                            )
                        ):
                            break

                        observed_at = time.time()

                        with self._lock:
                            self.microstream_last_attempt_started_at = 0.0
                            self.microstream_last_attempt_symbol = None

                        midpoint = self._book_midpoint(book)

                        if midpoint <= 0:
                            self.microstream_sample_failures += 1
                            continue

                        # Resolve first so optional feature processing can never
                        # distort the timing of the prospective label.
                        if journal is not None:
                            self.microstream_labels_resolved += (
                                journal.resolve(
                                    marks={
                                        symbol: midpoint
                                    },
                                    observed_at=observed_at,
                                )
                            )

                        observation = (
                            self.microstructure_sniper.observe_snapshot(
                                symbol=symbol,
                                order_book=book,
                                trades=trades,
                                now=observed_at,
                            )
                        )

                        # v1.58: expose the exact fresh precision observation
                        # to the Testnet velocity router. This is data only;
                        # the microstream itself still has no execution
                        # authority.
                        with self._lock:
                            self._microstream_snapshots[
                                symbol
                            ] = dict(observation)

                            cutoff = (
                                observed_at - 10.0
                            )

                            stale = [
                                key
                                for key, row in (
                                    self._microstream_snapshots.items()
                                )
                                if float(
                                    row.get("timestamp")
                                    or 0.0
                                ) < cutoff
                            ]

                            for key in stale:
                                self._microstream_snapshots.pop(
                                    key,
                                    None,
                                )

                        self.microstream_observations += 1
                        self.microstream_last_observation_at = (
                            observed_at
                        )

                    except Exception:
                        if (
                            generation
                            != int(
                                getattr(
                                    self,
                                    "_microstream_generation",
                                    generation,
                                )
                            )
                        ):
                            break

                        with self._lock:
                            self.microstream_last_attempt_started_at = 0.0
                            self.microstream_last_attempt_symbol = None

                        self.microstream_sample_failures += 1

                    # v1.60.42: a newly pinned execution candidate must not
                    # wait behind the remainder of an already-built sampling
                    # queue. Finish the current network call, then rebuild the
                    # queue immediately with the newest candidate first.
                    wake_event = getattr(
                        self,
                        "_execution_pin_event",
                        None,
                    )
                    if (
                        wake_event is not None
                        and wake_event.is_set()
                    ):
                        break

            except Exception:
                with self._lock:
                    self.microstream_last_attempt_started_at = 0.0
                    self.microstream_last_attempt_symbol = None

                self.microstream_sample_failures += 1

            elapsed = time.monotonic() - started
            self.microstream_last_loop_seconds = elapsed
            # v1.60.29: sleep until the cadence expires, a new execution
            # candidate is pinned, or shutdown is requested. stop() sets the pin
            # event too, so shutdown stays as responsive as the previous
            # self._stop.wait(). Cadence, capacity and freshness are unchanged.
            remaining = max(
                0.0,
                cadence_seconds - elapsed,
            )

            wake_event = getattr(
                self,
                "_execution_pin_event",
                None,
            )

            if wake_event is None:
                # Legacy/lightweight instances keep the original semantics.
                self._stop.wait(remaining)
            elif wake_event.wait(remaining):
                wake_event.clear()
                if not self._stop.is_set():
                    self.microstream_pin_wakeups = (
                        int(getattr(self, "microstream_pin_wakeups", 0)) + 1
                    )

    def _run_calibration_sampler(self) -> None:
        """Resolve only slower 2m/5m/15m prospective labels."""

        cadence_seconds = 1.0

        while not self._stop.is_set():
            started = time.monotonic()

            try:
                now = time.time()
                journal = self.slow_calibration_journal

                if journal is not None:
                    journal.censor_expired(
                        observed_at=now
                    )

                    symbols = journal.due_symbols(
                        observed_at=now,
                        lookahead_seconds=0.5,
                        limit=2,
                    )

                    for symbol in symbols:
                        if self._stop.is_set():
                            break

                        self.calibration_sample_attempts += 1

                        try:
                            book = (
                                self.micro_calibration_feed.order_book(
                                    symbol,
                                    limit=5,
                                )
                            )

                            observed_at = time.time()
                            midpoint = self._book_midpoint(
                                book
                            )

                            if midpoint <= 0:
                                self.calibration_sample_failures += 1
                                continue

                            self.calibration_labels_resolved += (
                                journal.resolve(
                                    marks={
                                        symbol: midpoint
                                    },
                                    observed_at=observed_at,
                                )
                            )

                        except Exception:
                            self.calibration_sample_failures += 1

            except Exception:
                self.calibration_sample_failures += 1

            elapsed = time.monotonic() - started
            self._stop.wait(
                max(
                    0.0,
                    cadence_seconds - elapsed,
                )
            )

    @staticmethod
    def _micro_velocity_score(
        snapshot: dict[str, Any],
    ) -> float:
        def number(
            key: str,
            default: float = 0.0,
        ) -> float:
            try:
                value = float(
                    snapshot.get(key, default)
                )
            except (TypeError, ValueError):
                return default
            return value

        spread = number(
            "spread_bps",
            1_000_000.0,
        )
        depth = (
            number("bid_depth_usd")
            + number("ask_depth_usd")
        )
        samples = int(
            number("temporal_samples")
        )

        if (
            spread > 25.0
            or depth < 10_000.0
            or samples < 2
        ):
            return 0.0

        trend = max(
            0.0,
            number(
                "recent_midpoint_trend_bps_5s"
            ),
        )
        velocity = max(
            0.0,
            number(
                "midpoint_velocity_bps_per_second"
            ),
        )
        acceleration = max(
            0.0,
            number(
                "midpoint_acceleration_bps_per_second2"
            ),
        )
        movement_range = max(
            0.0,
            number(
                "recent_midpoint_range_bps_5s"
            ),
        )
        imbalance = max(
            0.0,
            number("depth_imbalance"),
        )
        microprice = max(
            0.0,
            number("microprice_shift_bps"),
        )
        persistence = max(
            0.0,
            min(
                1.0,
                number("pressure_persistence"),
            ),
        )

        # v1.60.43: static book pressure is useful supporting evidence,
        # but must not masquerade as velocity. Keep the fast cohort
        # reserved for the same genuine positive bursts recognized by
        # the Velocity Sniper execution assessor.
        positive_burst = bool(
            (
                trend >= 8.0
                and velocity >= 0.50
            )
            or (
                velocity >= 2.0
                and acceleration >= 0.15
            )
            or (
                trend >= 15.0
                and microprice >= 0.0
            )
        )

        if not positive_burst:
            return 0.0

        return (
            trend
            + min(30.0, velocity * 5.0)
            + min(20.0, acceleration * 3.0)
            + min(10.0, movement_range * 0.20)
            + 8.0 * imbalance
            + min(6.0, microprice)
            + 6.0 * persistence
        )

    def micro_velocity_candidates(
        self,
        limit: int = 8,
        *,
        max_age_seconds: float = 2.0,
    ) -> list[dict[str, Any]]:
        bounded = max(
            1,
            min(24, int(limit)),
        )
        now = time.time()

        with self._lock:
            snapshots = copy.deepcopy(
                getattr(
                    self,
                    "_microstream_snapshots",
                    {},
                )
            )

        rows: list[dict[str, Any]] = []

        for symbol, snapshot in snapshots.items():
            timestamp = float(
                snapshot.get("timestamp")
                or 0.0
            )

            if timestamp <= 0.0:
                continue

            age = max(
                0.0,
                now - timestamp,
            )

            if age > max_age_seconds:
                continue

            score = self._micro_velocity_score(
                snapshot
            )

            if score <= 0.0:
                continue

            rows.append(
                {
                    **snapshot,
                    "symbol": str(
                        symbol
                    ).upper(),
                    "age_seconds": age,
                    "fresh": True,
                    "velocity_score": score,
                    "testnet_authority": False,
                    "live_authority": False,
                }
            )

        return sorted(
            rows,
            key=lambda row: (
                float(
                    row.get("velocity_score")
                    or 0.0
                ),
                float(
                    row.get(
                        "recent_midpoint_trend_bps_5s"
                    )
                    or 0.0
                ),
                float(
                    row.get(
                        "midpoint_velocity_bps_per_second"
                    )
                    or 0.0
                ),
            ),
            reverse=True,
        )[:bounded]

    def collective_candidates(
        self,
        limit: int = 8,
    ) -> list[str]:
        """Return execution-ready markets before slower fallback candidates."""

        bounded = max(
            1,
            min(
                48,
                int(limit),
            ),
        )

        now = time.time()

        freshness_seconds = max(
            0.50,
            float(
                getattr(
                    self,
                    "microstream_freshness_seconds",
                    2.0,
                )
            ),
        )

        velocity_rows = (
            self.micro_velocity_candidates(
                bounded,
                max_age_seconds=(
                    freshness_seconds
                ),
            )
        )

        with self._lock:
            snapshots = copy.deepcopy(
                getattr(
                    self,
                    "_microstream_snapshots",
                    {},
                )
            )

            hot = list(
                getattr(
                    self,
                    "_precision_hot_symbols",
                    [],
                )
            )

            explorer = str(
                getattr(
                    self,
                    "_precision_explorer_symbol",
                    "",
                )
                or ""
            ).upper()

            ranked = list(
                self.last_step.get(
                    "ranked"
                )
                or []
            )

            # v1.60.49: break the cold-start freshness deadlock.
            #
            # These rows are candidate-seeding authority only. They do not
            # become execution-ready here. The execution-first wrapper must
            # still pass normal exchange preflight and require a <=2 second
            # microstream signal before returning a symbol to the fast lane.
            precision_scout_symbols = list(
                getattr(
                    self,
                    "_precision_scout_symbols",
                    [],
                )
                or []
            )

            discovered_seed_rows = [
                dict(row)
                for row in (
                    getattr(
                        self,
                        "_candidates",
                        [],
                    )
                    or []
                )
                if isinstance(row, dict)
            ]

        fresh_rows: list[
            tuple[
                str,
                float,
                float,
                int,
            ]
        ] = []

        for symbol, snapshot in (
            snapshots.items()
        ):
            timestamp = float(
                snapshot.get(
                    "timestamp"
                )
                or 0.0
            )

            if timestamp <= 0.0:
                continue

            age = max(
                0.0,
                now - timestamp,
            )

            if (
                age
                > freshness_seconds
            ):
                continue

            fresh_rows.append(
                (
                    str(symbol).upper(),
                    age,
                    self._micro_velocity_score(
                        snapshot
                    ),
                    int(
                        float(
                            snapshot.get(
                                "temporal_samples"
                            )
                            or 0
                        )
                    ),
                )
            )

        fresh_rows.sort(
            key=lambda item: (
                item[2],
                item[3],
                -item[1],
            ),
            reverse=True,
        )

        fresh_symbols = {
            row[0]
            for row in fresh_rows
        }

        symbols: list[str] = []

        def add(symbol: str) -> None:
            normalized = str(
                symbol or ""
            ).upper()

            if (
                normalized
                and normalized not in symbols
                and len(symbols) < bounded
            ):
                symbols.append(
                    normalized
                )

        # Positive measured velocity has first execution priority.
        for row in velocity_rows:
            add(
                str(
                    row.get("symbol")
                    or ""
                )
            )

        # Continuity-managed hot/explorer markets follow, but only
        # while their actual book snapshot is still fresh.
        for symbol in hot:
            if str(symbol).upper() in fresh_symbols:
                add(symbol)

        if explorer in fresh_symbols:
            add(explorer)

        # Any other currently fresh deep observation remains eligible.
        for symbol, _age, _score, _samples in fresh_rows:
            add(symbol)

        # v1.60.49 cold-start seeding:
        #
        # Precision-scout symbols may be offered to the execution-first
        # wrapper before they have a fresh micro snapshot. The wrapper does
        # not return them for execution while stale: it performs the normal
        # Testnet round-trip preflight and requests bounded microstream
        # warming first.
        #
        # This breaks the previous circular wait:
        # no micro sample -> no raw candidate -> no execution pin -> no sample.
        for symbol in precision_scout_symbols:
            add(symbol)

        # If the precision scout itself has not completed its first refresh,
        # use the already exchange-filtered discovery universe strictly as
        # bounded warming seeds. These rows still receive no execution
        # authority here and remain subject to every downstream gate.
        if not precision_scout_symbols and not symbols:
            seed_rows = sorted(
                discovered_seed_rows,
                key=lambda row: (
                    abs(
                        float(
                            row.get("percentage_24h")
                            or 0.0
                        )
                    ),
                    float(
                        row.get("quote_volume_usd")
                        or 0.0
                    ),
                    -float(
                        row.get("spread_bps")
                        or 1_000_000.0
                    ),
                ),
                reverse=True,
            )

            for row in seed_rows:
                add(
                    str(
                        row.get("symbol")
                        or ""
                    )
                )

        # Broad completed full-swarm evidence remains the slower fallback.
        for row in ranked:
            if not isinstance(
                row,
                dict,
            ):
                continue

            add(
                str(
                    row.get("symbol")
                    or ""
                )
            )

        return symbols[:bounded]

    def collective_signal(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Return slower collective context plus the freshest precision mark."""
        normalized = str(
            symbol or ""
        ).upper()

        with self._lock:
            ranked = {}

            for row in list(
                self.last_step.get("ranked")
                or []
            ):
                if (
                    isinstance(row, dict)
                    and str(
                        row.get("symbol")
                        or ""
                    ).upper()
                    == normalized
                ):
                    ranked = dict(row)
                    break

            assessments = copy.deepcopy(
                (
                    self.last_step.get(
                        "timeframe_assessments"
                    )
                    or {}
                ).get(normalized)
                or {}
            )

            micro_proposals = [
                dict(row)
                for row in (
                    self.last_step.get(
                        "micro_agent_foundry_proposals"
                    )
                    or []
                )
                if (
                    isinstance(row, dict)
                    and str(
                        row.get("symbol")
                        or ""
                    ).upper()
                    == normalized
                )
            ]

            microstructure = copy.deepcopy(
                (
                    self.last_step.get(
                        "microstructure"
                    )
                    or {}
                ).get(normalized)
                or {}
            )

            micro_snapshot = copy.deepcopy(
                getattr(
                    self,
                    "_microstream_snapshots",
                    {},
                ).get(
                    normalized
                )
                or {}
            )

            precision_context = copy.deepcopy(
                getattr(
                    self,
                    "_precision_context_cache",
                    {},
                ).get(
                    normalized
                )
                or {}
            )

            last_success = float(
                self.last_success_at
                or 0.0
            )
            cycles = int(self.cycles)

        now = time.time()

        stale_after = max(
            60.0,
            self.cadence_seconds * 6.0,
        )

        slow_age = (
            max(
                0.0,
                now - last_success,
            )
            if last_success > 0.0
            else None
        )

        micro_timestamp = float(
            micro_snapshot.get("timestamp")
            or 0.0
        )

        micro_age = (
            max(
                0.0,
                now - micro_timestamp,
            )
            if micro_timestamp > 0.0
            else None
        )

        micro_fresh = bool(
            micro_age is not None
            and micro_age <= 2.0
        )

        precision_timestamp = float(
            precision_context.get(
                "timestamp"
            )
            or 0.0
        )

        precision_age = (
            max(
                0.0,
                now - precision_timestamp,
            )
            if precision_timestamp > 0.0
            else None
        )

        precision_fresh = bool(
            precision_age is not None
            and precision_age
            <= max(
                120.0,
                self.precision_scout_refresh_seconds
                * 6.0,
            )
        )

        if precision_fresh:
            precision_rows = (
                precision_context.get(
                    "assessments"
                )
                or {}
            )

            if isinstance(
                precision_rows,
                dict,
            ):
                assessments = {
                    **assessments,
                    **copy.deepcopy(
                        precision_rows
                    ),
                }

        slow_fresh = bool(
            last_success > 0.0
            and slow_age is not None
            and slow_age <= stale_after
        )

        if micro_snapshot:
            features = dict(
                microstructure.get("features")
                or {}
            )

            for key in (
                "timestamp",
                "midpoint",
                "spread_bps",
                "bid_depth_usd",
                "ask_depth_usd",
                "depth_imbalance",
                "microprice_shift_bps",
                "depth_imbalance_velocity",
                "microprice_velocity_bps_per_second",
                "spread_velocity_bps_per_second",
                "pressure_persistence",
                "temporal_samples",
                "midpoint_velocity_bps_per_second",
                "midpoint_acceleration_bps_per_second2",
                "total_depth_change_fraction_per_second",
                "recent_midpoint_range_bps_5s",
                "recent_midpoint_trend_bps_5s",
            ):
                if key in micro_snapshot:
                    features[key] = (
                        micro_snapshot[key]
                    )

            microstructure = {
                **microstructure,
                "features": features,
                "microstream_tracked": True,
                "precision_snapshot_fresh": (
                    micro_fresh
                ),
            }

        velocity_score = (
            self._micro_velocity_score(
                micro_snapshot
            )
            if micro_snapshot
            else 0.0
        )

        ages = [
            value
            for value in (
                slow_age,
                micro_age,
                precision_age,
            )
            if value is not None
        ]

        return {
            "symbol": normalized,
            "version": self.VERSION,
            "available": bool(
                last_success > 0.0
                or micro_snapshot
            ),
            "fresh": bool(
                slow_fresh
                or micro_fresh
            ),
            "age_seconds": (
                min(ages)
                if ages
                else None
            ),
            "cycles": cycles,
            "ranked_opportunity": ranked,
            "timeframe_assessments": (
                assessments
            ),
            "micro_proposals": micro_proposals,
            "microstructure": microstructure,
            "precision_context": {
                "fresh": precision_fresh,
                "age_seconds": (
                    precision_age
                ),
                "timeframes": sorted(
                    assessments
                ),
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            },
            "micro_velocity": {
                **micro_snapshot,
                "fresh": micro_fresh,
                "age_seconds": micro_age,
                "velocity_score": velocity_score,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            },
            "qualified_timeframe_paths": sum(
                1
                for row in assessments.values()
                if (
                    isinstance(row, dict)
                    and row.get(
                        "independently_qualified"
                    )
                    is True
                )
            ),
            "qualified_micro_proposals": sum(
                1
                for row in micro_proposals
                if (
                    row.get(
                        "evidence_qualified"
                    )
                    is True
                    and row.get(
                        "independently_qualified"
                    )
                    is True
                )
            ),
            "canonical_router_input": True,
            "direct_execution_authority": False,
            "live_authority": False,
        }

    @staticmethod
    def _step_liveness(
        *,
        running: bool,
        now: float,
        freshness_anchor: float,
        stale_after_seconds: float,
        step_in_progress: bool,
        step_started_at: float,
    ) -> dict[str, Any]:
        """Separate active long work from a genuinely stale/hung worker."""

        step_age = (
            max(0.0, now - step_started_at)
            if step_in_progress
            and step_started_at > 0
            else 0.0
        )

        # A normal fast-swarm step should usually finish far sooner.
        # Five minutes is a bounded grace ceiling, not an unlimited
        # suppression of stale detection.
        busy_ceiling_seconds = max(
            300.0,
            stale_after_seconds * 5.0,
        )

        busy = bool(
            running
            and step_in_progress
            and step_started_at > 0
            and step_age <= busy_ceiling_seconds
        )

        hung = bool(
            running
            and step_in_progress
            and step_started_at > 0
            and step_age > busy_ceiling_seconds
        )

        stale = bool(
            running
            and freshness_anchor > 0
            and now - freshness_anchor
            > stale_after_seconds
            and not busy
        )

        return {
            "busy": busy,
            "hung": hung,
            "stale": stale,
            "step_age_seconds": step_age,
            "busy_ceiling_seconds": (
                busy_ceiling_seconds
            ),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            wall_started = time.time()

            with self._lock:
                self.step_in_progress = True
                self.step_started_at = wall_started

            try:
                self.step()

            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.last_failure_at = time.time()
                    self.consecutive_failures += 1

            finally:
                with self._lock:
                    self.step_in_progress = False
                    self.last_step_duration_seconds = max(
                        0.0,
                        time.time() - wall_started,
                    )

            elapsed = time.monotonic() - started

            self._stop.wait(
                max(
                    0.0,
                    self.cadence_seconds - elapsed,
                )
            )

    def health(self, *, equity: float) -> dict[str, Any]:
        thread = self._thread
        with self._lock:
            now = time.time()
            running = bool(thread is not None and thread.is_alive() and not self._stop.is_set())
            stale_after_seconds = max(
                60.0,
                self.cadence_seconds * 6.0,
            )

            freshness_anchor = (
                self.last_success_at
                or self.started_at
            )

            liveness = self._step_liveness(
                running=running,
                now=now,
                freshness_anchor=freshness_anchor,
                stale_after_seconds=(
                    stale_after_seconds
                ),
                step_in_progress=(
                    self.step_in_progress
                ),
                step_started_at=(
                    self.step_started_at
                ),
            )

            stale = bool(
                liveness["stale"]
            )

            healthy = bool(
                running
                and not stale
                and not liveness["hung"]
                and self.consecutive_failures < 3
            )

            result = {
                "version": self.VERSION,
                "running": running,
                "healthy": healthy,
                "stale": stale,
                "busy": liveness["busy"],
                "hung": liveness["hung"],
                "step_in_progress": (
                    self.step_in_progress
                ),
                "step_started_at": (
                    self.step_started_at
                ),
                "step_age_seconds": (
                    liveness[
                        "step_age_seconds"
                    ]
                ),
                "step_busy_ceiling_seconds": (
                    liveness[
                        "busy_ceiling_seconds"
                    ]
                ),
                "last_step_duration_seconds": (
                    self.last_step_duration_seconds
                ),
                "stale_after_seconds": stale_after_seconds,
                "started_at": self.started_at,
                "last_success_at": self.last_success_at,
                "last_failure_at": self.last_failure_at,
                "consecutive_failures": self.consecutive_failures,
                "cycles": self.cycles,
                "full_sweeps": self.full_sweeps,
                "ranked_opportunities_total": self.ranked_opportunities_total,
                "qualified_opportunities_total": self.qualified_opportunities_total,
                "micro_assessments": self.micro_assessments,
                "micro_qualified": self.micro_qualified,
                "micro_fetch_failures": self.micro_fetch_failures,
                "microstructure_sniper": self.microstructure_sniper.health(),
                "micro_agent_foundry": self.micro_agent_foundry.health(),
                "precision_scout": {
                    "configured": (
                        self.precision_scout_feed
                        is not None
                    ),
                    "running": bool(
                        self._precision_scout_thread
                        is not None
                        and self._precision_scout_thread.is_alive()
                        and not self._stop.is_set()
                    ),
                    "refresh_seconds": (
                        self.precision_scout_refresh_seconds
                    ),
                    "refreshes": (
                        self.precision_scout_refreshes
                    ),
                    "failures": (
                        self.precision_scout_failures
                    ),
                    "eligible_candidates": (
                        self.precision_scout_candidates
                    ),
                    "symbols": list(
                        self._precision_scout_symbols
                    ),
                    "sub_dollar_selected": (
                        self.precision_scout_sub_dollar
                    ),
                    "last_refresh_at": (
                        self.precision_scout_last_refresh_at
                    ),
                    "context_updates": (
                        self.precision_context_updates
                    ),
                    "context_failures": (
                        self.precision_context_failures
                    ),
                    "context_last_update_at": (
                        self.precision_context_last_update_at
                    ),
                    "context_timeframes": [
                        "1m",
                        "5m",
                        "15m",
                    ],
                    "micro_priority_scheduler": {
                        "execution_scout_priority": True,
                        "maximum_slots": max(
                            1,
                            min(
                                12,
                                self.max_micro_symbols,
                            ),
                        ),
                        "effective_slots": int(
                            getattr(
                                self,
                                "_precision_micro_capacity",
                                min(
                                    6,
                                    self.max_micro_symbols,
                                ),
                            )
                        ),
                        "capacity_is_adaptive": True,
                        "last_queue": list(
                            self.precision_micro_last_queue
                        ),
                        "sticky_hot_cohort": list(
                            self._precision_hot_symbols
                        ),
                        "hot_cohort_until": (
                            self._precision_hot_until
                        ),
                        "hot_hold_seconds": (
                            self.precision_hot_hold_seconds
                        ),
                        "explorer_symbol": (
                            self._precision_explorer_symbol
                        ),
                        "explorer_until": (
                            self._precision_explorer_until
                        ),
                        "explorer_hold_seconds": (
                            self.precision_explorer_hold_seconds
                        ),
                        "hot_promotions": (
                            self.precision_hot_promotions
                        ),
                        "explorer_rotations": (
                            self.precision_explorer_rotations
                        ),
                        "continuity_preserving": True,
                        "broad_scout_remains_wide": True,
                        "freshness_budgeted_capacity": True,
                        "target_loop_seconds": (
                            getattr(
                                self,
                                "microstream_target_loop_seconds",
                                1.50,
                            )
                        ),
                        "execution_freshness_seconds": (
                            getattr(
                                self,
                                "microstream_freshness_seconds",
                                2.0,
                            )
                        ),
                        "estimated_per_symbol_latency_seconds": (
                            getattr(
                                self,
                                "microstream_per_symbol_latency_seconds",
                                0.25,
                            )
                        ),
                        "execution_pins": sorted(
                            getattr(
                                self,
                                "_execution_precision_pins",
                                {},
                            )
                        ),
                        "live_scout_slots": (
                            self.precision_micro_live_slots
                        ),
                        "research_due_slots": (
                            self.precision_micro_due_slots
                        ),
                        "rotating_scout_cursor": (
                            self._precision_micro_cursor
                        ),
                    },
                    "sub_dollar_is_coverage_bucket": True,
                    "nominal_price_is_profit_score": False,
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                },
                "micro_calibration": (
                    {
                        **self.micro_calibration_journal.health(),
                        "microstream_running": bool(
                            self._microstream_thread is not None
                            and self._microstream_thread.is_alive()
                            and not self._stop.is_set()
                        ),
                        "microstream_attempts": (
                            self.microstream_sample_attempts
                        ),
                        "microstream_failures": (
                            self.microstream_sample_failures
                        ),
                        "microstream_trade_context_failures": (
                            self.microstream_trade_context_failures
                        ),
                        "microstream_observations": (
                            self.microstream_observations
                        ),
                        "microstream_labels_resolved": (
                            self.microstream_labels_resolved
                        ),
                        "microstream_last_observation_at": (
                            self.microstream_last_observation_at
                        ),
                        "microstream_cadence_seconds": 0.5,
                        "microstream_last_loop_seconds": (
                            self.microstream_last_loop_seconds
                        ),
                        "microstream_liveness": {
                            **self._microstream_stall_snapshot(
                                now=now
                            ),
                            "watchdog_running": bool(
                                self._microstream_watchdog_thread
                                is not None
                                and self._microstream_watchdog_thread.is_alive()
                                and not self._stop.is_set()
                            ),
                            "watchdog_checks": (
                                self.microstream_watchdog_checks
                            ),
                            "stall_recoveries": (
                                self.microstream_stall_recoveries
                            ),
                            "stall_recovery_failures": (
                                self.microstream_stall_recovery_failures
                            ),
                            "last_recovery_at": (
                                self.microstream_last_recovery_at
                            ),
                            "retired_workers_alive": sum(
                                1
                                for thread in (
                                    self._retired_microstream_threads
                                )
                                if thread.is_alive()
                            ),
                            "read_only_recovery": True,
                            "execution_authority": False,
                            "testnet_authority": False,
                            "live_authority": False,
                        },
                        "microstream_warmup_labels_skipped": (
                            self.microstream_warmup_labels_skipped
                        ),
                        "microstream_non_event_labels_skipped": (
                            self.microstream_non_event_labels_skipped
                        ),
                        "microstream_kinematic_labels_registered": (
                            self.microstream_kinematic_labels_registered
                        ),
                        "kinematic_event_only_prospective_labels": True,
                        "sticky_microstream_watchlist": True,
                        "order_book_only_precision_lane": True,
                        "trade_context_blocks_precision_lane": False,
                        "microstream_symbol_capacity": int(
                            getattr(
                                self,
                                "_precision_micro_capacity",
                                min(
                                    6,
                                    self.max_micro_symbols,
                                ),
                            )
                        ),
                        "microstream_maximum_symbol_capacity": max(
                            1,
                            min(
                                12,
                                self.max_micro_symbols,
                            ),
                        ),
                        "microstream_symbols": list(
                            self._microstream_symbols
                        ),
                        "same_sample_warms_and_resolves": True,
                        "prospective_registration_is_capacity_bounded": True,
                    }
                    if self.micro_calibration_journal is not None
                    else {}
                ),
                "slow_calibration_sampler": {
                    "running": bool(
                        self._calibration_thread is not None
                        and self._calibration_thread.is_alive()
                        and not self._stop.is_set()
                    ),
                    "attempts": self.calibration_sample_attempts,
                    "failures": self.calibration_sample_failures,
                    "labels_resolved": (
                        self.calibration_labels_resolved
                    ),
                    "cadence_seconds": 1.0,

                    "kinematic_event_followthrough": True,
                    "kinematic_horizons_seconds": list(
                        self.KINEMATIC_SLOW_HORIZONS
                    ),
                    "kinematic_event_cooldown_seconds": (
                        self.KINEMATIC_SLOW_COOLDOWN_SECONDS
                    ),
                    "kinematic_event_triggers_registered": (
                        self.kinematic_slow_event_triggers_registered
                    ),
                    "kinematic_labels_registered": (
                        self.kinematic_slow_labels_registered
                    ),
                    "kinematic_labels_suppressed_by_cooldown": (
                        self.kinematic_slow_labels_suppressed_by_cooldown
                    ),

                    "automatic_promotion": False,
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                },
                "opportunity_qualification_rate": (
                    self.qualified_opportunities_total / self.ranked_opportunities_total
                    if self.ranked_opportunities_total
                    else 0.0
                ),
                "universe_candidates": len(self._candidates),
                "cursor": self._cursor,
                "cadence_seconds": self.cadence_seconds,
                "timeframe": self.timeframe,
                "context_timeframes": list(self.context_timeframes),
                "forming_candle_excluded": True,
                "dedicated_read_only_feed": True,
                "last_error": self.last_error,
                "last_step": dict(self.last_step),
                "runtime": self.runtime.health(equity=equity),
                "timeframe_mind": self.timeframe_mind.health(),
                "shadow_entries": self.shadow_entries,
                "shadow_exits": self.shadow_exits,
                "shadow_failures": self.shadow_failures,
                "canonical_paper_ledger_mutation": False,
                "shared_position_extension_candidates_are_trade_authority": False,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
            if self.shadow_portfolio is not None:
                result["shadow_portfolio"] = self.shadow_portfolio.health()
            if self.outcome_journal is not None:
                result["swarm_outcome_journal"] = self.outcome_journal.health()
            return result
