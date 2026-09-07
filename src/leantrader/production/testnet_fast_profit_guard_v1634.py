from __future__ import annotations

import copy
from typing import Any


VERSION = "1.61.9"

# A fast lane must have current micro confirmation.
MIN_MICRO_CONFIRMATION = 0.10

# Existing floor is >=30 bps. Demand additional edge before opening.
MIN_ENTRY_NET_MARGIN_BPS = 10.0

# Existing profitable-decay behavior may bank once costs + small margin
# are actually covered.
MIN_EXIT_NET_MARGIN_BPS = 5.0

# Do not let fee-only holds become permanent positions.
MAX_FEE_ONLY_EXIT_EXTENSION_SECONDS = 60.0

# v1.61.6: these restored engines operate on micro/short-horizon
# real market data. They may contribute expected capture evidence,
# but never execution authority and never without current micro
# confirmation plus the existing profitability gate.
RESTORED_FAST_SOURCE_PREFIXES = (
    "ultra_scalping.",
    "continuous_scalping",
    "continuous_momentum",
    "continuous_breakout",
    "legacy_swarm.scalping",
    "ultra_quantum.microstructure_decoder",
    "ultra_quantum.momentum",
    "ultra_god_mode.quantum_price",
    "ultra_fluid_mechanics",
    "modern_fast.smart_scalping",
    "modern_fast.technical_structure",
    "modern_fast.spectral_harmonics",
)


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _best_micro_edge_bps(row: dict[str, Any]) -> float:
    return max(
        [
            _n(item.get("expected_edge_bps"))
            for item in (row.get("micro_support") or [])
            if isinstance(item, dict)
        ]
        or [0.0]
    )


def _restored_fast_support(
    signal: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    assessments = (
        signal.get("timeframe_assessments")
        or {}
    )

    if not isinstance(assessments, dict):
        return output

    for timeframe, raw in assessments.items():
        if not isinstance(raw, dict):
            continue

        if raw.get("legacy_restoration") is not True:
            continue

        if (
            raw.get("legacy_economically_positive")
            is not True
        ):
            continue

        direction = str(
            raw.get("direction") or ""
        ).lower()

        if direction not in {
            "long",
            "buy",
            "bull",
            "bullish",
        }:
            continue

        source = str(
            raw.get("source") or ""
        )

        if not any(
            source.startswith(prefix)
            for prefix in RESTORED_FAST_SOURCE_PREFIXES
        ):
            continue

        confidence = max(
            0.0,
            min(
                1.0,
                _n(raw.get("confidence")),
            ),
        )

        gross_edge = max(
            0.0,
            _n(
                raw.get(
                    "legacy_gross_edge_bps"
                )
            ),
        )

        net_edge = max(
            0.0,
            _n(
                raw.get(
                    "legacy_conservative_net_edge_bps"
                ),
                _n(
                    raw.get(
                        "expected_edge_bps"
                    )
                ),
            ),
        )

        if (
            confidence < 0.10
            or gross_edge <= 0.0
            or net_edge <= 0.0
        ):
            continue

        output.append(
            {
                "source": source,
                "timeframe": str(
                    raw.get("timeframe")
                    or timeframe
                ),
                "confidence": confidence,
                "gross_edge_bps": gross_edge,
                "conservative_net_edge_bps": (
                    net_edge
                ),
                "modeled_round_trip_cost_bps": (
                    max(
                        0.0,
                        _n(
                            raw.get(
                                "legacy_modeled_round_trip_cost_bps"
                            )
                        ),
                    )
                ),
                "legacy_restoration": True,
                "economically_positive": True,
                "execution_authority": False,
                "live_authority": False,
            }
        )

    output.sort(
        key=lambda row: (
            _n(row.get("gross_edge_bps")),
            _n(row.get("confidence")),
        ),
        reverse=True,
    )

    return output[:8]


def _canonical_fast_context(
    supervisor_symbol: dict[str, Any] | None,
) -> dict[str, Any]:
    supervisor = (
        supervisor_symbol
        if isinstance(
            supervisor_symbol,
            dict,
        )
        else {}
    )

    advanced = (
        supervisor.get(
            "advanced_shadow"
        )
        or {}
    )

    signal_rows = [
        row
        for row in (
            advanced.get("signals")
            or []
        )
        if isinstance(row, dict)
    ]

    weighted_score = 0.0
    weight = 0.0
    positive_engines = []
    negative_engines = []

    for row in signal_rows:
        score = max(
            -1.0,
            min(
                1.0,
                _n(row.get("score")),
            ),
        )

        confidence = max(
            0.0,
            min(
                1.0,
                _n(
                    row.get(
                        "confidence"
                    )
                ),
            ),
        )

        if confidence <= 0.0:
            continue

        weighted_score += (
            score * confidence
        )

        weight += confidence

        engine = str(
            row.get("engine")
            or "unknown"
        )

        if (
            score >= 0.10
            and confidence >= 0.20
        ):
            positive_engines.append(
                engine
            )

        elif (
            score <= -0.10
            and confidence >= 0.20
        ):
            negative_engines.append(
                engine
            )

    advanced_score = (
        weighted_score / weight
        if weight > 0.0
        else 0.0
    )

    advanced_confidence = (
        min(
            1.0,
            weight
            / max(
                1.0,
                float(
                    len(signal_rows)
                ),
            ),
        )
        if signal_rows
        else 0.0
    )

    swarm = (
        advanced.get("swarm")
        or {}
    )

    swarm_score = max(
        -1.0,
        min(
            1.0,
            _n(swarm.get("score")),
        ),
    )

    swarm_confidence = max(
        0.0,
        min(
            1.0,
            _n(
                swarm.get(
                    "confidence"
                )
            ),
        ),
    )

    if swarm_confidence > 0.0:
        combined_weight = (
            advanced_confidence
            + swarm_confidence
        )

        canonical_score = (
            (
                advanced_score
                * advanced_confidence
                + swarm_score
                * swarm_confidence
            )
            / combined_weight
            if combined_weight > 0.0
            else 0.0
        )

        canonical_confidence = min(
            1.0,
            combined_weight / 2.0,
        )

    else:
        canonical_score = (
            advanced_score
        )

        canonical_confidence = (
            advanced_confidence
        )

    world = (
        supervisor.get(
            "market_world_model"
        )
        or {}
    )

    sensors = (
        supervisor.get(
            "sensor_context"
        )
        or {}
    )

    sensor_sources = sorted(
        str(name)
        for name, row in (
            sensors.items()
            if isinstance(
                sensors,
                dict,
            )
            else []
        )
        if (
            isinstance(row, dict)
            and str(
                row.get("status")
                or "available"
            )
            not in {
                "unsupported",
                "not_applicable",
            }
        )
    )

    return {
        "available": bool(
            advanced
            or world
            or sensors
        ),
        "canonical_score": (
            canonical_score
        ),
        "canonical_confidence": (
            canonical_confidence
        ),
        "advanced_positive_engines": (
            sorted(
                set(
                    positive_engines
                )
            )
        ),
        "advanced_negative_engines": (
            sorted(
                set(
                    negative_engines
                )
            )
        ),
        "world_state_confidence": (
            _n(
                world.get(
                    "state_confidence"
                )
            )
        ),
        "world_rare_scope_score": (
            _n(
                (
                    world.get("senses")
                    or {}
                ).get(
                    "rare_scope_score"
                )
            )
        ),
        "sensor_sources": sensor_sources,
        "rich_context_bridge_version": (
            supervisor.get(
                "rich_context_bridge_version"
            )
        ),
        "ranking_only": True,
        "cannot_create_entry_authority": True,
        "cannot_bypass_profit_gate": True,
        "execution_authority": False,
        "live_authority": False,
    }



def fast_entry_profit_gate(
    row: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(row or {})

    if result.get("allowed") is not True:
        return result

    velocity = result.get("velocity") or {}

    economics = (
        result.get("dynamic_execution_economics")
        or {}
    )

    evidence_sufficient = (
        economics.get("evidence_sufficient")
        is True
    )

    requested_cost = max(
        0.0,
        _n(
            result.get("modeled_round_trip_cost_bps"),
            30.0,
        ),
    )

    # Below 30 bps is permitted only when authenticated, two-sided,
    # per-symbol Testnet execution evidence has qualified it.
    modeled_cost = (
        requested_cost
        if evidence_sufficient
        else max(30.0, requested_cost)
    )

    net_margin_bps = (
        max(
            5.0,
            _n(
                economics.get(
                    "recommended_net_margin_bps"
                ),
                MIN_ENTRY_NET_MARGIN_BPS,
            ),
        )
        if evidence_sufficient
        else MIN_ENTRY_NET_MARGIN_BPS
    )

    required_capture = (
        modeled_cost
        + net_margin_bps
    )

    micro_confidence = max(
        0.0,
        _n(result.get("micro_confidence")),
    )

    velocity_qualified = (
        velocity.get("qualified_long")
        is True
    )

    projected_capture = max(
        0.0,
        _n(
            velocity.get(
                "projected_capture_bps_5s"
            )
        ),
    )

    micro_edge = max(
        0.0,
        _best_micro_edge_bps(result),
    )

    restored_fast_support = [
        row
        for row in (
            result.get(
                "restored_fast_support"
            )
            or []
        )
        if (
            isinstance(row, dict)
            and row.get(
                "economically_positive"
            )
            is True
            and _n(
                row.get(
                    "conservative_net_edge_bps"
                )
            )
            > 0.0
        )
    ]

    restored_fast_gross_edge = max(
        [
            _n(
                row.get(
                    "gross_edge_bps"
                )
            )
            for row in restored_fast_support
        ]
        or [0.0]
    )

    restored_fast_net_edge = max(
        [
            _n(
                row.get(
                    "conservative_net_edge_bps"
                )
            )
            for row in restored_fast_support
        ]
        or [0.0]
    )

    restored_fast_sources = sorted(
        {
            str(
                row.get("source")
                or ""
            )
            for row in restored_fast_support
            if row.get("source")
        }
    )

    # v1.61.6:
    # - microstructure/velocity still provides CURRENT confirmation
    # - restored fast engines may provide the expected gross capture
    # - the existing cost + profit-margin requirement remains unchanged
    #
    # This reconnects the already-built fast intelligence without
    # granting old engines independent execution authority.
    fast_edge = max(
        projected_capture,
        micro_edge,
        restored_fast_gross_edge,
    )

    result["v1634_fast_profit_gate"] = {
        "modeled_round_trip_cost_bps": (
            modeled_cost
        ),
        "dynamic_execution_economics": (
            copy.deepcopy(economics)
        ),
        "dynamic_cost_evidence_sufficient": (
            evidence_sufficient
        ),
        "net_margin_bps": net_margin_bps,
        "required_capture_bps": (
            required_capture
        ),
        "projected_capture_bps_5s": (
            projected_capture
        ),
        "best_micro_edge_bps": (
            micro_edge
        ),
        "best_restored_fast_gross_edge_bps": (
            restored_fast_gross_edge
        ),
        "best_restored_fast_net_edge_bps": (
            restored_fast_net_edge
        ),
        "restored_fast_sources": (
            restored_fast_sources
        ),
        "restored_fast_support_count": (
            len(restored_fast_support)
        ),
        "restored_fast_edge_bridge_version": (
            "1.61.6"
        ),
        "restored_fast_evidence_is_execution_authority": (
            False
        ),
        "fast_edge_bps": fast_edge,
        "micro_confidence": (
            micro_confidence
        ),
        "velocity_qualified": (
            velocity_qualified
        ),
        "live_authority": False,
    }

    # This directly prevents another CSPR/JASMY-style fast entry
    # whose current micro confidence is zero.
    if not (
        velocity_qualified
        or micro_confidence
        >= MIN_MICRO_CONFIRMATION
    ):
        result["allowed"] = False
        result["reason"] = (
            "v1634_fast_micro_confirmation_required"
        )
        result[
            "proven_positive_net_edge"
        ] = False

        return result

    # A fast move must be large enough to cover the complete
    # modeled round trip plus an actual profit margin.
    if (
        fast_edge + 1e-12
        < required_capture
    ):
        result["allowed"] = False
        result["reason"] = (
            "v1634_fast_edge_below_cost_margin"
        )
        result[
            "proven_positive_net_edge"
        ] = False

        return result

    result[
        "v1634_fast_profit_gate"
    ]["passed"] = True

    return result


def fee_only_exit_deferral(
    pending: dict[str, Any],
    *,
    round_trip_cost_bps: float,
    stop_loss_bps: float,
    record: dict[str, Any] | None,
    dynamic_cost_trusted: bool = False,
) -> dict[str, Any] | None:
    if (
        str(pending.get("kind") or "")
        != "exit"
    ):
        return None

    assessment = (
        pending.get("assessment")
        or {}
    )

    reason = str(
        assessment.get("exit_reason")
        or ""
    )

    # These are the exit types that produced the repeated
    # nearly-flat XRP fee losses.
    if reason not in {
        "velocity_decay",
        "dynamic_time_exit",
        "time_exit",
    }:
        return None

    gross_bps = _n(
        assessment.get("gross_bps"),
        _n(
            assessment.get(
                "gross_bps_at_decision"
            )
        ),
    )

    age_seconds = max(
        0.0,
        _n(
            assessment.get(
                "age_seconds"
            )
        ),
    )

    record = record or {}

    target_hold = max(
        5.0,
        _n(
            assessment.get(
                "target_hold_seconds"
            ),
            _n(
                record.get(
                    "target_hold_seconds"
                ),
                30.0,
            ),
        ),
    )

    requested_cost = max(
        0.0,
        _n(
            round_trip_cost_bps,
            30.0,
        ),
    )

    modeled_cost = (
        requested_cost
        if dynamic_cost_trusted
        else max(30.0, requested_cost)
    )

    profit_floor = (
        modeled_cost
        + MIN_EXIT_NET_MARGIN_BPS
    )

    protective_stop = max(
        20.0,
        _n(
            assessment.get(
                "dynamic_stop_loss_bps"
            ),
            _n(
                stop_loss_bps,
                30.0,
            ),
        ),
    )

    # Never interfere with an actually profitable decay exit.
    if gross_bps >= profit_floor:
        return None

    # Never interfere with genuine protection.
    if gross_bps <= -protective_stop:
        return None

    extension_seconds = min(
        MAX_FEE_ONLY_EXIT_EXTENSION_SECONDS,
        max(
            20.0,
            target_hold * 2.0,
            target_hold + 5.0,
        ),
    )

    # Bound the extension. This is not an infinite hold.
    if age_seconds >= extension_seconds:
        return None

    return {
        "reason": (
            "holding_below_cost_decay_extension"
        ),
        "original_exit_reason": reason,
        "gross_bps": gross_bps,
        "modeled_round_trip_cost_bps": (
            modeled_cost
        ),
        "profit_floor_bps": (
            profit_floor
        ),
        "protective_stop_bps": (
            protective_stop
        ),
        "age_seconds": age_seconds,
        "extension_seconds": (
            extension_seconds
        ),
        "order_submitted": False,
        "position_remains_active": True,
        "live_authority": False,
    }


def install_testnet_fast_profit_guard_v1634() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        VelocitySniperTestnetLane,
        "_v1634_fast_profit_guard_installed",
        False,
    ):
        return

    original_assess_bound = (
        VelocitySniperTestnetLane.assess_candidate
    )

    original_submit = (
        HyperSpeedCollectiveTestnetLane._submit_pending
    )

    original_health = (
        VelocitySniperTestnetLane.health
    )

    def assess_candidate(
        cls: Any,
        signal: dict[str, Any],
        supervisor_symbol: (
            dict[str, Any] | None
        ),
        *,
        relaxed: bool,
    ) -> dict[str, Any]:
        row = original_assess_bound(
            signal,
            supervisor_symbol,
            relaxed=relaxed,
        )

        canonical_context = (
            _canonical_fast_context(
                supervisor_symbol
            )
        )

        row = {
            **row,
            "canonical_fast_context": (
                copy.deepcopy(
                    canonical_context
                )
            ),
        }

        # The rich canonical stack is allowed to rank an already
        # eligible Testnet candidate, never manufacture eligibility.
        # Fresh micro confirmation + cost-margin proof still decide
        # whether an order can be attempted.
        if row.get("allowed") is True:
            canonical_score = _n(
                canonical_context.get(
                    "canonical_score"
                )
            )

            canonical_confidence = max(
                0.0,
                min(
                    1.0,
                    _n(
                        canonical_context.get(
                            "canonical_confidence"
                        )
                    ),
                ),
            )

            ranking_adjustment = max(
                -0.08,
                min(
                    0.06,
                    canonical_score
                    * canonical_confidence
                    * 0.08,
                ),
            )

            row[
                "decision_score"
            ] = max(
                0.0,
                min(
                    1.0,
                    _n(
                        row.get(
                            "decision_score"
                        )
                    )
                    + ranking_adjustment,
                ),
            )

            row[
                "canonical_context_ranking_adjustment"
            ] = ranking_adjustment

            if (
                canonical_score >= 0.10
                and canonical_confidence
                >= 0.20
            ):
                groups = list(
                    row.get(
                        "support_groups"
                    )
                    or []
                )

                groups.append(
                    "canonical_advanced_real_data"
                )

                row[
                    "support_groups"
                ] = sorted(
                    set(groups)
                )

        economics = (
            signal.get(
                "dynamic_execution_economics"
            )
            or {}
        )

        restored_fast = (
            _restored_fast_support(
                signal
            )
        )

        row = {
            **row,
            "restored_fast_support": (
                copy.deepcopy(
                    restored_fast
                )
            ),
        }

        if economics:
            row = {
                **row,
                "dynamic_execution_economics": (
                    copy.deepcopy(economics)
                ),
                "modeled_round_trip_cost_bps": (
                    _n(
                        economics.get(
                            "effective_round_trip_cost_bps"
                        ),
                        row.get(
                            "modeled_round_trip_cost_bps"
                        )
                        or 30.0,
                    )
                ),
            }

        return fast_entry_profit_gate(
            row
        )

    def submit_pending(
        self: Any,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        event = (
            pending.get("event")
            or {}
        )

        symbol = str(
            event.get("symbol")
            or ""
        ).upper()

        with self._lock:
            record = copy.deepcopy(
                (
                    self.state.get("active")
                    or {}
                ).get(symbol)
                or {}
            )

        entry_intelligence = (
            record.get("intelligence")
            or {}
        )

        entry_economics = (
            entry_intelligence.get(
                "dynamic_execution_economics"
            )
            or {}
        )

        position_cost_bps = _n(
            entry_intelligence.get(
                "modeled_round_trip_cost_bps"
            ),
            getattr(
                self,
                "round_trip_cost_bps",
                30.0,
            ),
        )

        deferral = (
            fee_only_exit_deferral(
                pending,
                round_trip_cost_bps=(
                    position_cost_bps
                ),
                stop_loss_bps=getattr(
                    self,
                    "stop_loss_bps",
                    30.0,
                ),
                record=record,
                dynamic_cost_trusted=(
                    entry_economics.get(
                        "evidence_sufficient"
                    )
                    is True
                ),
            )
        )

        if deferral is not None:
            with self._lock:
                # The exit was never submitted, so remove only
                # the local pending latch and leave the real
                # Testnet position untouched.
                self.state[
                    "pending_event"
                ] = None

                self.state[
                    "v1634_fee_only_exit_deferrals"
                ] = (
                    int(
                        self.state.get(
                            "v1634_fee_only_exit_deferrals"
                        )
                        or 0
                    )
                    + 1
                )

                self.state[
                    "v1634_last_fee_only_exit_deferral"
                ] = {
                    **copy.deepcopy(
                        deferral
                    ),
                    "symbol": symbol,
                    "observed_at": now,
                }

                self._save_locked()

            return self._decision(
                "holding_below_cost_decay_extension",
                details={
                    **deferral,
                    "symbol": symbol,
                },
            )

        return original_submit(
            self,
            pending,
            now=now,
        )

    def health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_health(self)
        )

        with self._lock:
            deferrals = int(
                self.state.get(
                    "v1634_fee_only_exit_deferrals"
                )
                or 0
            )

            last_deferral = (
                copy.deepcopy(
                    self.state.get(
                        "v1634_last_fee_only_exit_deferral"
                    )
                    or {}
                )
            )

        payload[
            "fast_net_profit_guard"
        ] = {
            "version": VERSION,
            "enabled": True,
            "minimum_micro_confirmation": (
                MIN_MICRO_CONFIRMATION
            ),
            "minimum_entry_net_margin_bps": (
                MIN_ENTRY_NET_MARGIN_BPS
            ),
            "minimum_exit_net_margin_bps": (
                MIN_EXIT_NET_MARGIN_BPS
            ),
            "maximum_fee_only_exit_extension_seconds": (
                MAX_FEE_ONLY_EXIT_EXTENSION_SECONDS
            ),
            "mtf_only_fast_entry_allowed": (
                False
            ),
            "below_cost_velocity_entry_allowed": (
                False
            ),
            "fee_only_decay_exit_deferred": (
                True
            ),
            "protective_stop_preserved": (
                True
            ),
            "short_reversal_exit_preserved": (
                True
            ),
            "price_limit_protection_preserved": (
                True
            ),
            "fee_only_exit_deferrals": (
                deferrals
            ),
            "last_fee_only_exit_deferral": (
                last_deferral
            ),
            "live_authority": False,
        }

        payload["version"] = VERSION
        payload[
            "live_authority"
        ] = False

        return payload

    VelocitySniperTestnetLane.assess_candidate = (
        classmethod(
            assess_candidate
        )
    )

    HyperSpeedCollectiveTestnetLane._submit_pending = (
        submit_pending
    )

    VelocitySniperTestnetLane.health = (
        health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        VERSION
    )

    VelocitySniperTestnetLane.VERSION = (
        VERSION
    )

    VelocitySniperTestnetLane._v1634_fast_profit_guard_installed = (
        True
    )
