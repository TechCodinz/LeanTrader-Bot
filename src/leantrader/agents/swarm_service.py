from __future__ import annotations

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

    VERSION = "1.49.0"
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
        slow_calibration_journal: MicroCalibrationJournal | None = None,
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
        self.slow_calibration_journal = slow_calibration_journal
        self._calibration_thread: threading.Thread | None = None
        self.calibration_sample_attempts = 0
        self.calibration_sample_failures = 0
        self.calibration_labels_resolved = 0
        self.micro_assessments = 0
        self.micro_qualified = 0
        self.micro_fetch_failures = 0
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
        self.started_at = 0.0
        self.last_success_at = 0.0
        self.last_failure_at = 0.0
        self.consecutive_failures = 0
        self.ranked_opportunities_total = 0
        self.qualified_opportunities_total = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(
            target=self._run,
            name="leantrader-market-swarm",
            daemon=True,
        )
        self._thread.start()

        if (
            self.micro_calibration_journal is not None
            and self.micro_calibration_feed is not None
        ):
            self._calibration_thread = threading.Thread(
                target=self._run_calibration_sampler,
                name="leantrader-micro-calibration",
                daemon=True,
            )
            self._calibration_thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(
                timeout=max(1.0, min(10.0, self.cadence_seconds + 1.0))
            )
        calibration_thread = self._calibration_thread
        if calibration_thread is not None and calibration_thread.is_alive():
            calibration_thread.join(timeout=3.0)

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

        candidates = [
            row for row in ranked
            if isinstance(row, dict)
            and str(row.get("symbol") or "").strip()
        ]

        def micro_priority(row: dict[str, Any]) -> tuple[float, float, float, float]:
            symbol = str(row.get("symbol") or "").upper()
            profile = profile_map.get(symbol) or {}
            fill = float(profile.get("fill_probability") or 0.0)
            liquidity = float(profile.get("liquidity_score") or 0.0)
            spread = float(profile.get("spread_bps") or 1_000_000.0)
            movement_score = float(row.get("score") or 0.0)
            return (
                fill,
                liquidity,
                -spread,
                movement_score,
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

                if self.micro_calibration_journal is not None:
                    self.micro_calibration_journal.register(
                        symbol=symbol,
                        midpoint=features.midpoint,
                        assessments=[row.as_dict() for row in assessments],
                        observed_at=features.timestamp,
                    )

                output[symbol] = {
                    "features": features.as_dict(),
                    "path_assessments": [r.as_dict() for r in assessments],
                    "foundry_proposals": proposed,
                    "automatic_promotion": False,
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                }
                marks[symbol] = features.midpoint
                proposals.extend(proposed)
                self.micro_assessments += len(assessments)
                self.micro_qualified += sum(
                    1 for r in assessments
                    if r.independently_qualified
                )

                # A symbol only consumes one of the scarce micro scout slots
                # after its real book clears the structural liquidity gates.
                total_depth = (
                    features.bid_depth_usd
                    + features.ask_depth_usd
                )
                if (
                    features.spread_bps
                    <= self.microstructure_sniper.maximum_spread_bps
                    and total_depth
                    >= self.microstructure_sniper.minimum_depth_usd
                ):
                    structurally_viable += 1
                    if structurally_viable >= self.max_micro_symbols:
                        break

            except Exception as exc:
                self.micro_fetch_failures += 1
                output[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
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
        required_symbols = self.shadow_portfolio.open_symbols() if self.shadow_portfolio is not None else set()
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

    def _run_calibration_sampler(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                now = time.time()

                journals = [
                    journal
                    for journal in (
                        self.micro_calibration_journal,
                        self.slow_calibration_journal,
                    )
                    if journal is not None
                ]

                due_symbols: list[str] = []

                for journal in journals:
                    journal.censor_expired(observed_at=now)

                    for symbol in journal.due_symbols(
                        observed_at=now,
                        lookahead_seconds=0.5,
                        limit=4,
                    ):
                        if symbol not in due_symbols:
                            due_symbols.append(symbol)

                symbols = due_symbols[:6]
                for symbol in symbols:
                    if self._stop.is_set():
                        break
                    self.calibration_sample_attempts += 1
                    try:
                        book = self.micro_calibration_feed.order_book(
                            symbol,
                            limit=5,
                        )
                        observed_at = time.time()
                        midpoint = self._book_midpoint(book)
                        if midpoint <= 0:
                            self.calibration_sample_failures += 1
                            continue
                        for journal in journals:
                            self.calibration_labels_resolved += (
                                journal.resolve(
                                    marks={symbol: midpoint},
                                    observed_at=observed_at,
                                )
                            )
                    except Exception:
                        self.calibration_sample_failures += 1
            except Exception:
                self.calibration_sample_failures += 1

            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, 1.0 - elapsed))

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.last_failure_at = time.time()
                    self.consecutive_failures += 1
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.cadence_seconds - elapsed))

    def health(self, *, equity: float) -> dict[str, Any]:
        thread = self._thread
        with self._lock:
            now = time.time()
            running = bool(thread is not None and thread.is_alive() and not self._stop.is_set())
            stale_after_seconds = max(60.0, self.cadence_seconds * 6.0)
            freshness_anchor = self.last_success_at or self.started_at
            stale = bool(
                running
                and freshness_anchor > 0
                and now - freshness_anchor > stale_after_seconds
            )
            healthy = bool(running and not stale and self.consecutive_failures < 3)
            result = {
                "version": self.VERSION,
                "running": running,
                "healthy": healthy,
                "stale": stale,
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
                "micro_calibration": (
                    {
                        **self.micro_calibration_journal.health(),
                        "sampler_running": bool(
                            self._calibration_thread is not None
                            and self._calibration_thread.is_alive()
                            and not self._stop.is_set()
                        ),
                        "sampler_attempts": self.calibration_sample_attempts,
                        "sampler_failures": self.calibration_sample_failures,
                        "sampler_labels_resolved": (
                            self.calibration_labels_resolved
                        ),
                        "sampler_cadence_seconds": 1.0,
                    }
                    if self.micro_calibration_journal is not None
                    else {}
                ),
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
