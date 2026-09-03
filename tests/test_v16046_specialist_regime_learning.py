from __future__ import annotations

from leantrader.production.fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)


def lane_fixture():
    lane = object.__new__(
        HyperSpeedCollectiveTestnetLane
    )
    import threading
    lane._lock = threading.RLock()
    lane.state = {
        "specialist_regime_learning": {},
        "last_specialist_attribution": {},
    }
    lane.round_trip_cost_bps = 30.0
    return lane


def test_learning_requires_three_authenticated_cycles():
    lane = lane_fixture()

    attribution = {
        "regime": "fast_momentum",
        "specialists": ["velocity_sniper"],
    }

    for i in range(2):
        lane._record_specialist_regime_outcome(
            attribution=attribution,
            realized_pnl_usd=0.01,
            entry_notional_usd=1.0,
            symbol="XRP/USDT",
            exit_reason="velocity_take_profit",
            closed_at=float(i + 1),
        )

    row = lane._specialist_learning_adjustment(
        attribution
    )

    assert row["adjustment"] == 0.0
    assert row["contributors"] == []


def test_learning_becomes_bounded_ranking_only_after_three_cycles():
    lane = lane_fixture()

    attribution = {
        "regime": "fast_momentum",
        "specialists": ["velocity_sniper"],
    }

    for i in range(3):
        lane._record_specialist_regime_outcome(
            attribution=attribution,
            realized_pnl_usd=0.01,
            entry_notional_usd=1.0,
            symbol="XRP/USDT",
            exit_reason="velocity_take_profit",
            closed_at=float(i + 1),
        )

    row = lane._specialist_learning_adjustment(
        attribution
    )

    assert 0.0 < row["adjustment"] <= 0.08
    assert row["ranking_only"] is True
    assert row["cannot_override_hard_gates"] is True


def test_negative_authenticated_cycles_reduce_ranking_only():
    lane = lane_fixture()

    attribution = {
        "regime": "volatile_microstructure",
        "specialists": ["qualified_micro_proposals"],
    }

    for i in range(3):
        lane._record_specialist_regime_outcome(
            attribution=attribution,
            realized_pnl_usd=-0.01,
            entry_notional_usd=1.0,
            symbol="SUI/USDT",
            exit_reason="velocity_stop_loss",
            closed_at=float(i + 1),
        )

    row = lane._specialist_learning_adjustment(
        attribution
    )

    assert -0.08 <= row["adjustment"] < 0.0
    assert row["ranking_only"] is True


def test_specialist_bundle_consumes_existing_qualified_branches():
    lane = lane_fixture()

    signal = {
        "fresh": True,
        "age_seconds": 0.3,
        "qualified_timeframe_paths": 2,
        "qualified_micro_proposals": 1,
        "micro_velocity": {
            "velocity_bps_per_second": 4.0,
            "acceleration_bps_per_second2": 1.0,
            "trend_5s_bps": 9.0,
        },
        "microstructure": {
            "features": {
                "spread_bps": 5.0,
            }
        },
    }

    assessment = {
        "support_groups": [
            "microstructure_sniper"
        ],
        "velocity_sniper": True,
        "cost_qualified": True,
        "velocity": {
            "qualified_long": True
        },
    }

    row = lane._specialist_regime_bundle(
        signal,
        assessment,
    )

    assert row["regime"] == "fast_momentum"
    assert "qualified_timeframe_paths" in row["specialists"]
    assert "qualified_micro_proposals" in row["specialists"]
    assert "velocity_sniper" in row["specialists"]
    assert "execution_economics" in row["specialists"]
    assert row["execution_authority"] is False


def test_executor_source_persists_authenticated_cycle_marker():
    from pathlib import Path

    source = Path(
        "src/leantrader/production/testnet_execution.py"
    ).read_text()

    assert '"last_closed_cycle"' in source
    assert "authenticated_bybit_testnet_fills" in source
