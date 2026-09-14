from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


class FastCollectiveTestnetLane:
    """Seconds-scale collective exploration on Bybit Testnet only.

    This lane exists to gather real authenticated Testnet execution evidence
    without pretending an unvalidated signal is profitable.

    Entry authority is strictly Testnet exploration. Real-money authority is
    permanently absent.
    """

    VERSION = "1.58.0"

    SAFETY_REASON_PREFIXES = (
        "high_impact_news_blackout",
        "exchange_clock_not_verified",
        "exchange_protection:",
        "cognitive_governance:",
        "brain:cognitive:",
    )

    TERMINAL_ORDER_STATES = {
        "closed",
        "canceled",
        "rejected",
        "skipped",
    }

    def __init__(
        self,
        *,
        service_provider: Callable[[], Any | None],
        testnet: Any,
        state_path: Path,
        supervisory_provider: Callable[[], dict[str, Any]],
        order_usd: float,
        round_trip_cost_bps: float,
        cadence_seconds: float = 10.0,
        maximum_hold_seconds: float = 90.0,
        take_profit_bps: float = 60.0,
        stop_loss_bps: float = 40.0,
        maximum_entries_per_day: int = 6,
        bootstrap_after_seconds: float = 45.0,
    ) -> None:
        if testnet is None:
            raise ValueError("fast Testnet lane requires a Testnet executor")

        self.service_provider = service_provider
        self.testnet = testnet
        self.state_path = state_path
        self.supervisory_provider = supervisory_provider

        self.order_usd = max(
            0.50,
            min(float(order_usd), 2.0),
        )

        self.round_trip_cost_bps = max(
            30.0,
            float(round_trip_cost_bps),
        )

        self.cadence_seconds = max(
            0.25,
            min(15.0, float(cadence_seconds)),
        )

        self.maximum_hold_seconds = max(
            5.0,
            min(300.0, float(maximum_hold_seconds)),
        )

        self.take_profit_bps = max(
            self.round_trip_cost_bps + 10.0,
            float(take_profit_bps),
        )

        self.stop_loss_bps = max(
            20.0,
            float(stop_loss_bps),
        )

        self.maximum_entries_per_day = max(
            1,
            min(100, int(maximum_entries_per_day)),
        )

        self.bootstrap_after_seconds = max(
            5.0,
            float(bootstrap_after_seconds),
        )

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = 0.0

        self.state = self._load_state()

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default

        return (
            number
            if math.isfinite(number)
            else default
        )

    @staticmethod
    def _direction(value: Any) -> str:
        normalized = str(value or "").lower()

        if normalized in {
            "long",
            "buy",
            "bull",
            "bullish",
        }:
            return "long"

        if normalized in {
            "short",
            "sell",
            "bear",
            "bearish",
        }:
            return "short"

        return "flat"

    @classmethod
    def assess_candidate(
        cls,
        signal: dict[str, Any],
        supervisor_symbol: dict[str, Any] | None,
        *,
        relaxed: bool,
    ) -> dict[str, Any]:
        supervisor_symbol = (
            supervisor_symbol
            if isinstance(supervisor_symbol, dict)
            else {}
        )

        if signal.get("fresh") is not True:
            return {
                "allowed": False,
                "reason": "fast_signal_not_fresh",
            }

        micro = signal.get("microstructure") or {}

        if not isinstance(micro, dict):
            return {
                "allowed": False,
                "reason": "microstructure_unavailable",
            }

        if micro.get("microstream_tracked") is not True:
            return {
                "allowed": False,
                "reason": "microstructure_not_structurally_tracked",
            }

        features = micro.get("features") or {}

        price = cls._number(
            features.get("midpoint"),
        )

        spread_bps = cls._number(
            features.get("spread_bps"),
            1_000_000.0,
        )

        if price <= 0.0:
            return {
                "allowed": False,
                "reason": "invalid_micro_midpoint",
            }

        if spread_bps > 25.0:
            return {
                "allowed": False,
                "reason": "micro_spread_too_wide",
                "price": price,
                "spread_bps": spread_bps,
            }

        ranked = signal.get("ranked_opportunity") or {}

        quality = max(
            0.0,
            min(
                1.0,
                cls._number(
                    ranked.get("quality_multiplier")
                ),
            ),
        )

        timeframe_rows = [
            {
                **row,
                "timeframe": timeframe,
            }
            for timeframe, row in (
                signal.get("timeframe_assessments")
                or {}
            ).items()
            if isinstance(row, dict)
        ]

        long_mtf = [
            row
            for row in timeframe_rows
            if (
                cls._direction(row.get("direction"))
                == "long"
                and cls._number(row.get("confidence"))
                >= 0.50
                and cls._number(
                    row.get("expected_edge_bps")
                )
                > 0.0
            )
        ]

        short_mtf = [
            row
            for row in timeframe_rows
            if (
                cls._direction(row.get("direction"))
                == "short"
                and cls._number(row.get("confidence"))
                >= 0.50
                and cls._number(
                    row.get("expected_edge_bps")
                )
                > 0.0
            )
        ]

        path_rows = [
            row
            for row in (
                micro.get("path_assessments")
                or []
            )
            if isinstance(row, dict)
        ]

        long_micro = [
            row
            for row in path_rows
            if (
                cls._direction(row.get("direction"))
                == "long"
                and cls._number(
                    row.get("expected_edge_bps")
                )
                > 0.0
                and cls._number(
                    row.get("confidence")
                )
                >= 0.10
            )
        ]

        short_micro = [
            row
            for row in path_rows
            if (
                cls._direction(row.get("direction"))
                == "short"
                and cls._number(
                    row.get("expected_edge_bps")
                )
                > 0.0
                and cls._number(
                    row.get("confidence")
                )
                >= 0.10
            )
        ]

        mtf_confidence = max(
            [
                cls._number(row.get("confidence"))
                for row in long_mtf
            ]
            or [0.0]
        )

        micro_confidence = max(
            [
                cls._number(row.get("confidence"))
                for row in long_micro
            ]
            or [0.0]
        )

        short_mtf_confidence = max(
            [
                cls._number(row.get("confidence"))
                for row in short_mtf
            ]
            or [0.0]
        )

        short_micro_confidence = max(
            [
                cls._number(row.get("confidence"))
                for row in short_micro
            ]
            or [0.0]
        )

        cost_qualified_mtf = [
            row
            for row in long_mtf
            if row.get("independently_qualified") is True
        ]

        cost_qualified_micro = [
            row
            for row in (
                signal.get("micro_proposals")
                or []
            )
            if (
                isinstance(row, dict)
                and cls._direction(row.get("side"))
                == "long"
                and row.get("evidence_qualified") is True
                and row.get("independently_qualified") is True
                and cls._number(
                    row.get(
                        "conservative_net_edge_bps"
                    )
                )
                > 0.0
            )
        ]

        cost_qualified = bool(
            cost_qualified_mtf
            or cost_qualified_micro
        )

        strict_current_alignment = bool(
            long_mtf
            and long_micro
            and mtf_confidence >= 0.55
            and micro_confidence >= 0.22
        )

        bounded_bootstrap_alignment = bool(
            relaxed
            and long_mtf
            and long_micro
            and mtf_confidence >= 0.50
            and micro_confidence >= 0.10
            and quality >= 0.15
        )

        if not (
            cost_qualified
            or strict_current_alignment
            or bounded_bootstrap_alignment
        ):
            return {
                "allowed": False,
                "reason": "fresh_micro_mtf_alignment_not_ready",
                "price": price,
                "spread_bps": spread_bps,
                "mtf_confidence": mtf_confidence,
                "micro_confidence": micro_confidence,
                "quality": quality,
            }

        strong_short_conflict = bool(
            (
                short_micro_confidence
                >= max(
                    0.25,
                    micro_confidence + 0.05,
                )
                and short_mtf_confidence >= 0.50
            )
            or (
                len(short_mtf) >= 2
                and short_mtf_confidence
                > mtf_confidence
            )
        )

        if strong_short_conflict:
            return {
                "allowed": False,
                "reason": "fresh_short_conflict",
                "price": price,
                "spread_bps": spread_bps,
                "mtf_confidence": mtf_confidence,
                "micro_confidence": micro_confidence,
                "short_mtf_confidence": (
                    short_mtf_confidence
                ),
                "short_micro_confidence": (
                    short_micro_confidence
                ),
            }

        route = (
            supervisor_symbol.get("route")
            or {}
        )

        temporal = route.get("temporal_session") or {}

        if (
            isinstance(temporal, dict)
            and temporal
            and temporal.get("allowed") is False
        ):
            return {
                "allowed": False,
                "reason": "cached_temporal_safety_veto",
                "price": price,
            }

        protection = (
            route.get("exchange_protection")
            or {}
        )

        if (
            isinstance(protection, dict)
            and protection
            and protection.get("allowed") is False
        ):
            return {
                "allowed": False,
                "reason": "cached_exchange_protection_veto",
                "price": price,
            }

        route_reason = str(
            route.get("reason") or ""
        )

        if any(
            route_reason.startswith(prefix)
            for prefix in cls.SAFETY_REASON_PREFIXES
        ):
            return {
                "allowed": False,
                "reason": "cached_governance_safety_veto",
                "price": price,
            }

        cached_positive: list[str] = []
        cached_negative: list[str] = []

        base_score = cls._number(
            route.get("base_score")
        )
        base_confidence = cls._number(
            route.get("base_confidence")
        )

        if (
            base_score >= 0.10
            and base_confidence >= 0.20
        ):
            cached_positive.append("adaptive")
        elif (
            base_score <= -0.10
            and base_confidence >= 0.20
        ):
            cached_negative.append("adaptive")

        advanced_score = cls._number(
            route.get("advanced_score")
        )
        advanced_confidence = cls._number(
            route.get("advanced_confidence")
        )

        if (
            advanced_score >= 0.10
            and advanced_confidence >= 0.15
        ):
            cached_positive.append("ultra_ensemble")
        elif (
            advanced_score <= -0.10
            and advanced_confidence >= 0.15
        ):
            cached_negative.append("ultra_ensemble")

        collective = (
            supervisor_symbol.get("collective")
            or {}
        )

        cached_members: list[str] = []

        for row in (
            collective.get("groups")
            or []
        ):
            if not isinstance(row, dict):
                continue

            group = str(
                row.get("group") or ""
            )

            if group in {
                "fast_mtf",
                "microstructure",
            }:
                continue

            score = cls._number(
                row.get("score")
            )
            confidence = cls._number(
                row.get("confidence")
            )

            if (
                score >= 0.10
                and confidence >= 0.20
            ):
                cached_positive.append(group)
                cached_members.extend(
                    str(value)
                    for value in (
                        row.get("members")
                        or []
                    )
                )

            elif (
                score <= -0.10
                and confidence >= 0.20
            ):
                cached_negative.append(group)

        if (
            len(cached_negative) >= 2
            and len(cached_negative)
            > len(cached_positive)
        ):
            return {
                "allowed": False,
                "reason": "cached_collective_negative_consensus",
                "price": price,
                "cached_negative": (
                    sorted(set(cached_negative))
                ),
            }

        support_groups = [
            "microstructure_sniper",
            "multi_timeframe_minds",
            *cached_positive,
        ]

        current_confidence = (
            0.52 * micro_confidence
            + 0.43 * mtf_confidence
            + 0.05 * quality
        )

        cached_bonus = min(
            0.15,
            0.03 * len(set(cached_positive)),
        )

        decision_score = min(
            1.0,
            current_confidence
            + cached_bonus
            + (0.10 if cost_qualified else 0.0),
        )

        strongest_micro = sorted(
            long_micro,
            key=lambda row: (
                cls._number(row.get("confidence")),
                cls._number(
                    row.get("expected_edge_bps")
                ),
            ),
            reverse=True,
        )[:3]

        strongest_mtf = sorted(
            long_mtf,
            key=lambda row: (
                cls._number(row.get("confidence")),
                cls._number(
                    row.get("expected_edge_bps")
                ),
            ),
            reverse=True,
        )[:4]

        return {
            "allowed": True,
            "reason": (
                "cost_qualified_collective"
                if cost_qualified
                else "bounded_testnet_exploration"
            ),
            "entry_mode": (
                "cost_qualified"
                if cost_qualified
                else "exploration_probe"
            ),
            "price": price,
            "spread_bps": spread_bps,
            "quality": quality,
            "decision_score": decision_score,
            "micro_confidence": micro_confidence,
            "mtf_confidence": mtf_confidence,
            "support_groups": sorted(
                set(support_groups)
            ),
            "cached_negative_groups": sorted(
                set(cached_negative)
            ),
            "cached_contributors": sorted(
                set(cached_members)
            )[:20],
            "micro_support": strongest_micro,
            "mtf_support": strongest_mtf,
            "cost_qualified": cost_qualified,
            "modeled_round_trip_cost_bps": 30.0,
            "proven_positive_net_edge": (
                cost_qualified
            ),
            "testnet_exploration_authority": True,
            "live_authority": False,
        }

    @classmethod
    def strong_short_reversal(
        cls,
        signal: dict[str, Any],
    ) -> bool:
        micro = signal.get("microstructure") or {}

        path_rows = [
            row
            for row in (
                micro.get("path_assessments")
                or []
            )
            if isinstance(row, dict)
        ]

        mtf_rows = [
            row
            for row in (
                signal.get("timeframe_assessments")
                or {}
            ).values()
            if isinstance(row, dict)
        ]

        micro_short = max(
            [
                cls._number(row.get("confidence"))
                for row in path_rows
                if (
                    cls._direction(
                        row.get("direction")
                    )
                    == "short"
                    and cls._number(
                        row.get(
                            "expected_edge_bps"
                        )
                    )
                    > 0.0
                )
            ]
            or [0.0]
        )

        mtf_short = max(
            [
                cls._number(row.get("confidence"))
                for row in mtf_rows
                if (
                    cls._direction(
                        row.get("direction")
                    )
                    == "short"
                    and cls._number(
                        row.get(
                            "expected_edge_bps"
                        )
                    )
                    > 0.0
                )
            ]
            or [0.0]
        )

        return bool(
            micro_short >= 0.25
            and mtf_short >= 0.55
        )

    def start(self) -> None:
        thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
        ):
            return

        self.started_at = time.time()
        self._stop.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="leantrader-fast-collective-testnet",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

        thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
        ):
            thread.join(timeout=10.0)

    def _clear_transient_error_after_success(
        self,
    ) -> None:
        with self._lock:
            if (
                self.state.get("last_error")
                is None
                and self.state.get(
                    "last_error_at"
                )
                is None
            ):
                return

            self.state["last_error"] = None
            self.state["last_error_at"] = None
            self._save_locked()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()

            try:
                self.step()
                self._clear_transient_error_after_success()
            except Exception as exc:
                with self._lock:
                    self.state["last_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.state["last_error_at"] = (
                        time.time()
                    )
                    self._save_locked()

            elapsed = time.monotonic() - started

            self._stop.wait(
                max(
                    0.0,
                    self.cadence_seconds - elapsed,
                )
            )

    def step(
        self,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = (
            time.time()
            if now is None
            else float(now)
        )

        service = self.service_provider()

        if service is None:
            return self._decision(
                "waiting_for_fast_swarm"
            )

        self._refresh_day(now)

        snapshot = self.testnet.safe_snapshot()

        pending = self._pending()

        if pending is not None:
            return self._submit_pending(
                pending,
                now=now,
            )

        active = self._active_snapshot()

        if active:
            symbol = next(iter(active))
            record = active[symbol]

            return self._manage_active(
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        positions = {
            str(symbol): self._number(quantity)
            for symbol, quantity in (
                snapshot.get("positions") or {}
            ).items()
            if self._number(quantity) > 0.0
        }

        if positions:
            return self._decision(
                "existing_non_fast_testnet_position",
                details={
                    "positions": sorted(positions),
                },
            )

        if int(snapshot.get("open_orders") or 0) > 0:
            return self._decision(
                "testnet_order_in_flight"
            )

        if snapshot.get(
            "kill_switch_active"
        ) is True:
            return self._decision(
                "testnet_kill_switch"
            )

        if (
            snapshot.get(
                "last_reconciliation_errors"
            )
            or []
        ):
            return self._decision(
                "testnet_reconciliation_not_clear"
            )

        with self._lock:
            if (
                int(
                    self.state.get(
                        "entries_today"
                    )
                    or 0
                )
                >= self.maximum_entries_per_day
            ):
                return self._decision(
                    "fast_daily_entry_cap"
                )

        supervisor = (
            self.supervisory_provider()
            or {}
        )

        supervisor_gate = (
            self._supervisor_allows_entries(
                supervisor,
                now=now,
            )
        )

        if not supervisor_gate["allowed"]:
            return self._decision(
                supervisor_gate["reason"],
                details=supervisor_gate,
            )

        symbols = service.collective_candidates(
            limit=8
        )

        if not symbols:
            return self._decision(
                "no_fast_collective_candidates"
            )

        relaxed = bool(
            self.started_at > 0
            and (
                now - self.started_at
                >= self.bootstrap_after_seconds
            )
        )

        assessments: list[
            tuple[str, dict[str, Any]]
        ] = []

        supervisor_symbols = (
            supervisor.get("symbols")
            or {}
        )

        for symbol in symbols:
            try:
                signal = (
                    service.collective_signal(
                        symbol
                    )
                )

                assessment = (
                    self.assess_candidate(
                        signal,
                        supervisor_symbols.get(
                            symbol,
                            {},
                        ),
                        relaxed=relaxed,
                    )
                )

            except Exception as exc:
                assessment = {
                    "allowed": False,
                    "reason": (
                        f"candidate_error:"
                        f"{type(exc).__name__}"
                    ),
                }

            assessments.append(
                (symbol, assessment)
            )

        allowed = [
            (symbol, row)
            for symbol, row in assessments
            if row.get("allowed") is True
        ]

        if not allowed:
            best = sorted(
                assessments,
                key=lambda item: (
                    self._number(
                        item[1].get(
                            "decision_score"
                        )
                    ),
                    self._number(
                        item[1].get(
                            "mtf_confidence"
                        )
                    ),
                    self._number(
                        item[1].get(
                            "micro_confidence"
                        )
                    ),
                ),
                reverse=True,
            )[:3]

            return self._decision(
                "no_aligned_long_candidate",
                details={
                    "relaxed": relaxed,
                    "top": [
                        {
                            "symbol": symbol,
                            "reason": row.get(
                                "reason"
                            ),
                            "mtf_confidence": (
                                row.get(
                                    "mtf_confidence"
                                )
                            ),
                            "micro_confidence": (
                                row.get(
                                    "micro_confidence"
                                )
                            ),
                        }
                        for symbol, row in best
                    ],
                },
            )

        symbol, assessment = max(
            allowed,
            key=lambda item: (
                self._number(
                    item[1].get(
                        "decision_score"
                    )
                ),
                self._number(
                    item[1].get(
                        "quality"
                    )
                ),
            ),
        )

        price = self._number(
            assessment.get("price")
        )

        quantity = (
            self.order_usd
            / max(price, 1e-12)
        )

        event = self._new_event(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            reason=(
                "fast_collective_testnet_entry:"
                + str(
                    assessment.get(
                        "entry_mode"
                    )
                    or "exploration_probe"
                )
            ),
            now=now,
        )

        pending = {
            "kind": "entry",
            "event": event,
            "assessment": assessment,
            "created_at": now,
        }

        self._set_pending(pending)

        return self._submit_pending(
            pending,
            now=now,
        )

    def _manage_active(
        self,
        service: Any,
        snapshot: dict[str, Any],
        symbol: str,
        record: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        current_quantity = self._number(
            (
                snapshot.get("positions")
                or {}
            ).get(symbol)
        )

        if current_quantity <= 0.0:
            with self._lock:
                removed = (
                    self.state.get("active")
                    or {}
                ).pop(
                    symbol,
                    None,
                )

                self.state["last_action"] = {
                    "action": (
                        "fast_position_absent"
                    ),
                    "symbol": symbol,
                    "timestamp": now,
                    "record": removed,
                }

                self._save_locked()

            return self._decision(
                "active_position_already_absent"
            )

        signal = service.collective_signal(
            symbol
        )

        micro = (
            signal.get("microstructure")
            or {}
        )
        features = (
            micro.get("features")
            or {}
        )

        price = self._number(
            features.get("midpoint")
        )

        if (
            signal.get("fresh") is not True
            or price <= 0.0
        ):
            return self._decision(
                "waiting_for_fresh_exit_mark"
            )

        entry_price = self._number(
            record.get("entry_price")
        )

        entered_at = self._number(
            record.get("entered_at")
        )

        age_seconds = max(
            0.0,
            now - entered_at,
        )

        gross_bps = (
            (price / entry_price - 1.0)
            * 10_000.0
            if entry_price > 0.0
            else 0.0
        )

        reason = None

        if gross_bps >= self.take_profit_bps:
            reason = "take_profit"

        elif gross_bps <= -self.stop_loss_bps:
            reason = "stop_loss"

        elif self.strong_short_reversal(signal):
            reason = "micro_mtf_reversal"

        elif (
            age_seconds
            >= self.maximum_hold_seconds
        ):
            reason = "time_exit"

        if reason is None:
            return self._decision(
                "holding_fast_testnet_position",
                details={
                    "symbol": symbol,
                    "age_seconds": age_seconds,
                    "gross_bps": gross_bps,
                },
            )

        event = self._new_event(
            symbol=symbol,
            side="sell",
            quantity=current_quantity,
            price=price,
            reason=(
                "fast_collective_testnet_exit:"
                + reason
            ),
            now=now,
            remaining_quantity=0.0,
        )

        pending = {
            "kind": "exit",
            "event": event,
            "assessment": {
                "exit_reason": reason,
                "gross_bps_at_decision": (
                    gross_bps
                ),
                "age_seconds": age_seconds,
            },
            "created_at": now,
        }

        self._set_pending(pending)

        return self._submit_pending(
            pending,
            now=now,
        )

    def _submit_pending(
        self,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        event = copy.deepcopy(
            pending["event"]
        )

        results = self.testnet.mirror_events(
            [event]
        )

        result = (
            dict(results[0])
            if results
            else {}
        )

        snapshot = self.testnet.safe_snapshot()

        symbol = str(
            event["symbol"]
        ).upper()

        status = str(
            result.get("status")
            or ""
        ).lower()

        current_quantity = self._number(
            (
                snapshot.get("positions")
                or {}
            ).get(symbol)
        )

        with self._lock:
            kind = str(
                pending.get("kind")
                or ""
            )

            if kind == "entry":
                if current_quantity > 0.0:
                    entry_price = self._number(
                        result.get("average"),
                        self._number(
                            event.get("price")
                        ),
                    )

                    self.state.setdefault(
                        "active",
                        {},
                    )[symbol] = {
                        "symbol": symbol,
                        "quantity": (
                            current_quantity
                        ),
                        "entry_price": (
                            entry_price
                        ),
                        "entered_at": now,
                        "entry_event_id": (
                            event.get("event_id")
                        ),
                        "entry_mode": (
                            pending.get(
                                "assessment",
                                {},
                            ).get(
                                "entry_mode"
                            )
                        ),
                        "intelligence": (
                            pending.get(
                                "assessment",
                                {},
                            )
                        ),
                    }

                    self.state[
                        "entries_today"
                    ] = (
                        int(
                            self.state.get(
                                "entries_today"
                            )
                            or 0
                        )
                        + 1
                    )

                    self.state["last_action"] = {
                        "action": "buy",
                        "symbol": symbol,
                        "status": status,
                        "quantity": (
                            current_quantity
                        ),
                        "price": entry_price,
                        "timestamp": now,
                        "intelligence": (
                            pending.get(
                                "assessment",
                                {},
                            )
                        ),
                    }

                    self.state[
                        "pending_event"
                    ] = None

                elif (
                    status
                    in self.TERMINAL_ORDER_STATES
                ):
                    self.state["last_action"] = {
                        "action": (
                            "entry_not_opened"
                        ),
                        "symbol": symbol,
                        "status": status,
                        "skip_reason": (
                            result.get(
                                "skip_reason"
                            )
                        ),
                        "timestamp": now,
                    }

                    self.state[
                        "pending_event"
                    ] = None

            elif kind == "exit":
                active = (
                    self.state.setdefault(
                        "active",
                        {},
                    ).get(symbol)
                )

                if current_quantity <= 0.0:
                    active = (
                        self.state[
                            "active"
                        ].pop(
                            symbol,
                            active,
                        )
                        or {}
                    )

                    exit_price = self._number(
                        result.get("average"),
                        self._number(
                            event.get("price")
                        ),
                    )

                    entry_price = self._number(
                        active.get(
                            "entry_price"
                        )
                    )

                    gross_bps = (
                        (
                            exit_price
                            / entry_price
                            - 1.0
                        )
                        * 10_000.0
                        if entry_price > 0.0
                        else 0.0
                    )

                    net_bps = (
                        gross_bps
                        - self.round_trip_cost_bps
                    )

                    closed = {
                        "symbol": symbol,
                        "entry_price": (
                            entry_price
                        ),
                        "exit_price": (
                            exit_price
                        ),
                        "gross_bps": (
                            gross_bps
                        ),
                        "net_bps_after_model": (
                            net_bps
                        ),
                        "modeled_round_trip_cost_bps": (
                            self.round_trip_cost_bps
                        ),
                        "entered_at": (
                            active.get(
                                "entered_at"
                            )
                        ),
                        "exited_at": now,
                        "exit_reason": (
                            pending.get(
                                "assessment",
                                {},
                            ).get(
                                "exit_reason"
                            )
                        ),
                        "entry_mode": (
                            active.get(
                                "entry_mode"
                            )
                        ),
                        "intelligence": (
                            active.get(
                                "intelligence"
                            )
                        ),
                        "testnet_only": True,
                        "live_authority": False,
                    }

                    history = list(
                        self.state.get(
                            "closed"
                        )
                        or []
                    )

                    history.append(closed)

                    self.state["closed"] = (
                        history[-100:]
                    )

                    self.state[
                        "exits_today"
                    ] = (
                        int(
                            self.state.get(
                                "exits_today"
                            )
                            or 0
                        )
                        + 1
                    )

                    self.state["last_action"] = {
                        "action": "sell",
                        "symbol": symbol,
                        "status": status,
                        "price": exit_price,
                        "timestamp": now,
                        "outcome": closed,
                    }

                    self.state[
                        "pending_event"
                    ] = None

                elif (
                    status
                    in self.TERMINAL_ORDER_STATES
                ):
                    # A partial/odd provider fill remains visible
                    # and receives a fresh idempotent exit event on
                    # the next step.
                    self.state[
                        "pending_event"
                    ] = None

                    if active is not None:
                        active[
                            "quantity"
                        ] = current_quantity

            self.state["last_error"] = None
            self._save_locked()

        return self._decision(
            "testnet_event_processed",
            details={
                "kind": pending.get("kind"),
                "symbol": symbol,
                "status": status,
                "current_quantity": (
                    current_quantity
                ),
            },
        )

    def _supervisor_allows_entries(
        self,
        supervisor: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        if not supervisor:
            return {
                "allowed": False,
                "reason": (
                    "waiting_for_supervisory_snapshot"
                ),
            }

        timestamp = self._number(
            supervisor.get("timestamp")
        )

        age = (
            max(0.0, now - timestamp)
            if timestamp > 0.0
            else math.inf
        )

        if age > 1_800.0:
            return {
                "allowed": False,
                "reason": (
                    "supervisory_snapshot_stale"
                ),
                "age_seconds": age,
            }

        if supervisor.get("healthy") is not True:
            return {
                "allowed": False,
                "reason": (
                    "supervisory_runtime_not_healthy"
                ),
            }

        if supervisor.get("halt_reason"):
            return {
                "allowed": False,
                "reason": (
                    "canonical_risk_halt_active"
                ),
            }

        required_failures = list(
            supervisor.get(
                "required_failures"
            )
            or []
        )

        if required_failures:
            return {
                "allowed": False,
                "reason": (
                    "required_engine_failure"
                ),
                "required_failures": (
                    required_failures
                ),
            }

        return {
            "allowed": True,
            "reason": "supervisory_clear",
            "age_seconds": age,
        }

    def _new_event(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reason: str,
        now: float,
        remaining_quantity: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            sequence = (
                int(
                    self.state.get(
                        "sequence"
                    )
                    or 0
                )
                + 1
            )

            self.state["sequence"] = (
                sequence
            )

            event = {
                "event_id": (
                    f"fast57-{int(now * 1000)}-"
                    f"{sequence}-{side}"
                ),
                "timestamp": (
                    dt.datetime.fromtimestamp(
                        now,
                        tz=dt.UTC,
                    ).isoformat()
                ),
                "side": side,
                "symbol": str(symbol).upper(),
                "quantity": float(quantity),
                "price": float(price),
                "reason": reason,
                "fast_collective_testnet": True,
                "live_authority": False,
            }

            if remaining_quantity is not None:
                event[
                    "remaining_quantity"
                ] = float(
                    remaining_quantity
                )

            self._save_locked()

            return event

    def _decision(
        self,
        reason: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "timestamp": time.time(),
            "reason": reason,
            "details": details or {},
        }

        with self._lock:
            self.state[
                "last_decision"
            ] = payload
            self._save_locked()

        return payload

    def _pending(
        self,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.state.get(
                "pending_event"
            )

            return (
                copy.deepcopy(row)
                if isinstance(row, dict)
                else None
            )

    def _set_pending(
        self,
        pending: dict[str, Any],
    ) -> None:
        with self._lock:
            self.state[
                "pending_event"
            ] = copy.deepcopy(
                pending
            )

            self._save_locked()

    def _active_snapshot(
        self,
    ) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(
                self.state.get("active")
                or {}
            )

    def _refresh_day(
        self,
        now: float,
    ) -> None:
        day = (
            dt.datetime.fromtimestamp(
                now,
                tz=dt.UTC,
            )
            .date()
            .isoformat()
        )

        with self._lock:
            if self.state.get("day") == day:
                return

            self.state["day"] = day
            self.state["entries_today"] = 0
            self.state["exits_today"] = 0
            self._save_locked()

    def health(self) -> dict[str, Any]:
        thread = self._thread

        with self._lock:
            return {
                "version": self.VERSION,
                "configured": True,
                "running": bool(
                    thread is not None
                    and thread.is_alive()
                    and not self._stop.is_set()
                ),
                "cadence_seconds": (
                    self.cadence_seconds
                ),
                "maximum_hold_seconds": (
                    self.maximum_hold_seconds
                ),
                "take_profit_bps": (
                    self.take_profit_bps
                ),
                "stop_loss_bps": (
                    self.stop_loss_bps
                ),
                "modeled_round_trip_cost_bps": (
                    self.round_trip_cost_bps
                ),
                "maximum_entries_per_day": (
                    self.maximum_entries_per_day
                ),
                "entries_today": int(
                    self.state.get(
                        "entries_today"
                    )
                    or 0
                ),
                "exits_today": int(
                    self.state.get(
                        "exits_today"
                    )
                    or 0
                ),
                "active_positions": (
                    copy.deepcopy(
                        self.state.get(
                            "active"
                        )
                        or {}
                    )
                ),
                "pending_event": bool(
                    self.state.get(
                        "pending_event"
                    )
                ),
                "last_action": (
                    copy.deepcopy(
                        self.state.get(
                            "last_action"
                        )
                    )
                ),
                "last_decision": (
                    copy.deepcopy(
                        self.state.get(
                            "last_decision"
                        )
                    )
                ),
                "last_error": (
                    self.state.get(
                        "last_error"
                    )
                ),
                "recent_closed": (
                    copy.deepcopy(
                        list(
                            self.state.get(
                                "closed"
                            )
                            or []
                        )[-10:]
                    )
                ),
                "testnet_exploration_authority": True,
                "canonical_paper_authority": False,
                "real_money_authority": False,
                "live_authority": False,
            }

    def _default_state(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "day": (
                dt.datetime.now(
                    dt.UTC
                )
                .date()
                .isoformat()
            ),
            "sequence": 0,
            "entries_today": 0,
            "exits_today": 0,
            "active": {},
            "pending_event": None,
            "closed": [],
            "last_action": None,
            "last_decision": None,
            "last_error": None,
            "last_error_at": None,
        }

    def _load_state(
        self,
    ) -> dict[str, Any]:
        default = self._default_state()

        if not self.state_path.exists():
            return default

        try:
            payload = json.loads(
                self.state_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                payload.get(
                    "schema_version"
                )
                != 1
            ):
                return default

            return {
                **default,
                **payload,
            }

        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return default

    def _save_locked(
        self,
    ) -> None:
        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = (
            self.state_path.with_suffix(
                self.state_path.suffix
                + ".tmp"
            )
        )

        temporary.write_text(
            json.dumps(
                self.state,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary,
            self.state_path,
        )
