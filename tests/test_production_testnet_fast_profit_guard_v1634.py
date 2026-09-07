from __future__ import annotations

from leantrader.production.testnet_fast_profit_guard_v1634 import (
    VERSION,
    _restored_fast_support,
    fast_entry_profit_gate,
    fee_only_exit_deferral,
)


def allowed_row(
    *,
    micro_confidence=0.0,
    projected_capture=0.0,
    velocity_qualified=False,
    micro_edge=0.0,
):
    return {
        "allowed": True,
        "reason": (
            "cost_qualified_collective"
        ),
        "cost_qualified": True,
        "micro_confidence": (
            micro_confidence
        ),
        "modeled_round_trip_cost_bps": (
            30.0
        ),
        "micro_support": (
            [
                {
                    "expected_edge_bps": (
                        micro_edge
                    )
                }
            ]
            if micro_edge > 0.0
            else []
        ),
        "velocity": {
            "qualified_long": (
                velocity_qualified
            ),
            "projected_capture_bps_5s": (
                projected_capture
            ),
        },
        "proven_positive_net_edge": (
            True
        ),
    }


def test_blocks_mtf_only_fast_entry():
    row = fast_entry_profit_gate(
        allowed_row(
            micro_confidence=0.0,
            projected_capture=80.0,
            velocity_qualified=False,
        )
    )

    assert row["allowed"] is False

    assert (
        row["reason"]
        == "v1634_fast_micro_confirmation_required"
    )


def test_blocks_edge_below_cost_margin():
    row = fast_entry_profit_gate(
        allowed_row(
            micro_confidence=0.20,
            projected_capture=20.0,
            micro_edge=25.0,
        )
    )

    assert row["allowed"] is False

    assert (
        row["reason"]
        == "v1634_fast_edge_below_cost_margin"
    )


def test_allows_micro_edge_above_cost():
    row = fast_entry_profit_gate(
        allowed_row(
            micro_confidence=0.20,
            projected_capture=22.0,
            micro_edge=52.0,
        )
    )

    assert row["allowed"] is True

    assert (
        row[
            "v1634_fast_profit_gate"
        ]["passed"]
        is True
    )


def test_allows_velocity_above_cost():
    row = fast_entry_profit_gate(
        allowed_row(
            micro_confidence=0.0,
            projected_capture=45.0,
            velocity_qualified=True,
        )
    )

    assert row["allowed"] is True


def test_defers_near_flat_velocity_decay():
    pending = {
        "kind": "exit",
        "assessment": {
            "exit_reason": (
                "velocity_decay"
            ),
            "gross_bps": 2.0,
            "age_seconds": 10.0,
            "target_hold_seconds": 30.0,
            "dynamic_stop_loss_bps": 20.0,
        },
    }

    result = fee_only_exit_deferral(
        pending,
        round_trip_cost_bps=30.0,
        stop_loss_bps=30.0,
        record={
            "target_hold_seconds": (
                30.0
            )
        },
    )

    assert result is not None
    assert (
        result["order_submitted"]
        is False
    )


def test_never_defers_profit_or_stop():
    profit = {
        "kind": "exit",
        "assessment": {
            "exit_reason": (
                "velocity_decay"
            ),
            "gross_bps": 36.0,
            "age_seconds": 10.0,
            "target_hold_seconds": 30.0,
        },
    }

    stop = {
        "kind": "exit",
        "assessment": {
            "exit_reason": (
                "velocity_stop_loss"
            ),
            "gross_bps": -25.0,
            "age_seconds": 10.0,
        },
    }

    assert (
        fee_only_exit_deferral(
            profit,
            round_trip_cost_bps=30.0,
            stop_loss_bps=30.0,
            record={},
        )
        is None
    )

    assert (
        fee_only_exit_deferral(
            stop,
            round_trip_cost_bps=30.0,
            stop_loss_bps=30.0,
            record={},
        )
        is None
    )


def test_extension_is_bounded():
    pending = {
        "kind": "exit",
        "assessment": {
            "exit_reason": (
                "dynamic_time_exit"
            ),
            "gross_bps": 0.0,
            "age_seconds": 60.0,
            "target_hold_seconds": 30.0,
        },
    }

    assert (
        fee_only_exit_deferral(
            pending,
            round_trip_cost_bps=30.0,
            stop_loss_bps=30.0,
            record={},
        )
        is None
    )


def test_version():
    assert VERSION == "1.60.44"


def test_restored_fast_support_extracts_only_cost_cleared_fast_families():
    signal = {
        "timeframe_assessments": {
            "fast": {
                "legacy_restoration": True,
                "legacy_economically_positive": True,
                "direction": "long",
                "confidence": 0.75,
                "expected_edge_bps": 25.0,
                "legacy_gross_edge_bps": 55.0,
                "legacy_conservative_net_edge_bps": 25.0,
                "legacy_modeled_round_trip_cost_bps": 30.0,
                "source": "continuous_momentum",
                "timeframe": "1m",
            },
            "slow": {
                "legacy_restoration": True,
                "legacy_economically_positive": True,
                "direction": "long",
                "confidence": 0.90,
                "expected_edge_bps": 70.0,
                "legacy_gross_edge_bps": 100.0,
                "legacy_conservative_net_edge_bps": 70.0,
                "legacy_modeled_round_trip_cost_bps": 30.0,
                "source": "legacy_multitimeframe",
                "timeframe": "1m+5m+15m",
            },
            "subcost": {
                "legacy_restoration": True,
                "legacy_economically_positive": False,
                "direction": "long",
                "confidence": 0.90,
                "expected_edge_bps": 0.0,
                "legacy_gross_edge_bps": 5.0,
                "legacy_conservative_net_edge_bps": 0.0,
                "source": "ultra_scalping.micro_momentum",
                "timeframe": "1m",
            },
        }
    }

    rows = _restored_fast_support(
        signal
    )

    assert len(rows) == 1
    assert (
        rows[0]["source"]
        == "continuous_momentum"
    )
    assert (
        rows[0]["gross_edge_bps"]
        == 55.0
    )
    assert (
        rows[0]["conservative_net_edge_bps"]
        == 25.0
    )
    assert (
        rows[0]["execution_authority"]
        is False
    )


def test_restored_fast_edge_can_prove_capture_only_with_current_micro_confirmation():
    row = allowed_row(
        micro_confidence=0.20,
        projected_capture=4.0,
        micro_edge=5.0,
    )

    row["restored_fast_support"] = [
        {
            "source": "continuous_momentum",
            "confidence": 0.75,
            "gross_edge_bps": 55.0,
            "conservative_net_edge_bps": 25.0,
            "economically_positive": True,
            "execution_authority": False,
        }
    ]

    result = fast_entry_profit_gate(
        row
    )

    assert result["allowed"] is True

    gate = result[
        "v1634_fast_profit_gate"
    ]

    assert (
        gate[
            "best_restored_fast_gross_edge_bps"
        ]
        == 55.0
    )

    assert (
        gate[
            "best_restored_fast_net_edge_bps"
        ]
        == 25.0
    )

    assert (
        gate["fast_edge_bps"]
        == 55.0
    )

    assert (
        gate[
            "restored_fast_evidence_is_execution_authority"
        ]
        is False
    )


def test_restored_fast_edge_never_bypasses_micro_confirmation():
    row = allowed_row(
        micro_confidence=0.0,
        projected_capture=0.0,
        velocity_qualified=False,
        micro_edge=0.0,
    )

    row["restored_fast_support"] = [
        {
            "source": (
                "ultra_scalping.micro_momentum"
            ),
            "confidence": 0.90,
            "gross_edge_bps": 100.0,
            "conservative_net_edge_bps": 70.0,
            "economically_positive": True,
            "execution_authority": False,
        }
    ]

    result = fast_entry_profit_gate(
        row
    )

    assert result["allowed"] is False

    assert (
        result["reason"]
        == "v1634_fast_micro_confirmation_required"
    )
