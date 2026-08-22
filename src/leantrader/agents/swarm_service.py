from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd

from .fast_path import FastSwarmRuntime
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

    VERSION = "1.2"
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
        if (self.shadow_portfolio is None) != (self.outcome_journal is None):
            raise ValueError("shadow portfolio and outcome journal must be configured together")
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
        self.shadow_entries = 0
        self.shadow_exits = 0
        self.shadow_failures = 0

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
        dedup: dict[str, dict[str, Any]] = {}
        for row in mandatory + selected:
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                dedup[symbol] = dict(row)
        return list(dedup.values())

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
        required_symbols = self.shadow_portfolio.open_symbols() if self.shadow_portfolio is not None else set()
        assessments, extension_candidates, context_errors = self._assess_context(
            ranked=list(result.get("ranked") or []),
            one_minute_frames=frames,
            required_symbols=required_symbols,
        )
        fetch_errors.update(context_errors)
        marks = {symbol: self._last_close(frame) for symbol, frame in frames.items() if self._last_close(frame) > 0}
        shadow_close_events = self._manage_shadow_exits(marks=marks, assessments=assessments)
        shadow_open_events = self._open_shadow_candidates(ranked=list(result.get("ranked") or []), assessments=assessments, marks=marks)

        result["timeframe_assessments"] = assessments
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
        return dict(result)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.cadence_seconds - elapsed))

    def health(self, *, equity: float) -> dict[str, Any]:
        thread = self._thread
        with self._lock:
            result = {
                "version": self.VERSION,
                "running": bool(thread is not None and thread.is_alive() and not self._stop.is_set()),
                "healthy": self.last_error is None,
                "cycles": self.cycles,
                "full_sweeps": self.full_sweeps,
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
