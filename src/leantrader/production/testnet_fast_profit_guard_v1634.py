from __future__ import annotations

import copy
from typing import Any


VERSION = "1.60.34"

# A fast lane must have current micro confirmation.
MIN_MICRO_CONFIRMATION = 0.10

# Existing floor is >=30 bps. Demand additional edge before opening.
MIN_ENTRY_NET_MARGIN_BPS = 10.0

# Existing profitable-decay behavior may bank once costs + small margin
# are actually covered.
MIN_EXIT_NET_MARGIN_BPS = 5.0

# Do not let fee-only holds become permanent positions.
MAX_FEE_ONLY_EXIT_EXTENSION_SECONDS = 60.0


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


def fast_entry_profit_gate(
    row: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(row or {})

    if result.get("allowed") is not True:
        return result

    velocity = result.get("velocity") or {}

    modeled_cost = max(
        30.0,
        _n(
            result.get("modeled_round_trip_cost_bps"),
            30.0,
        ),
    )

    required_capture = (
        modeled_cost
        + MIN_ENTRY_NET_MARGIN_BPS
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

    # Either real microstructure edge or current velocity capture
    # may prove the fast move, but MTF alone may not.
    fast_edge = max(
        projected_capture,
        micro_edge,
    )

    result["v1634_fast_profit_gate"] = {
        "modeled_round_trip_cost_bps": (
            modeled_cost
        ),
        "required_capture_bps": (
            required_capture
        ),
        "projected_capture_bps_5s": (
            projected_capture
        ),
        "best_micro_edge_bps": (
            micro_edge
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

    modeled_cost = max(
        30.0,
        _n(
            round_trip_cost_bps,
            30.0,
        ),
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

        deferral = (
            fee_only_exit_deferral(
                pending,
                round_trip_cost_bps=getattr(
                    self,
                    "round_trip_cost_bps",
                    30.0,
                ),
                stop_loss_bps=getattr(
                    self,
                    "stop_loss_bps",
                    30.0,
                ),
                record=record,
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
