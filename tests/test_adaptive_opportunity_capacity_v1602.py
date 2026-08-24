from __future__ import annotations

import threading

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)
from leantrader.production.fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)


def lane():
    obj = object.__new__(
        HyperSpeedCollectiveTestnetLane
    )
    obj._lock = threading.RLock()
    obj.state = {
        "active": {},
    }
    obj.order_usd = 1.0
    obj.maximum_order_usd = 10.0
    obj.maximum_concurrent_positions = 6
    obj.maximum_adaptive_positions = 24
    obj.maximum_entries_per_day = 45
    obj.maximum_entries_per_cycle = 3
    obj.maximum_adaptive_entries_per_cycle = 8
    obj.candidate_scan_limit = 48
    return obj


def supervisor(
    remaining=30.0,
    risk=1.0,
):
    return {
        "capital_growth": {
            "remaining_deployable_notional": remaining,
            "risk_multiplier": risk,
            "new_entries_allowed": True,
        }
    }


def snapshot():
    return {
        "positions": {},
        "risk_limits": {
            "max_orders_per_day": 100,
            "max_daily_submitted_usd": 500.0,
        },
        "daily_order_count": 0,
        "daily_submitted_usd": 0.0,
    }


def test_position_capacity_can_expand_above_six():
    obj = lane()

    result = obj._adaptive_position_capacity(
        supervisor(
            remaining=30.0,
            risk=1.0,
        ),
        snapshot(),
        candidate_count=20,
        entries_today=0,
    )

    assert result["adaptive"] is True
    assert result["target_positions"] > 6
    assert result["target_positions"] <= 24


def test_position_capacity_contracts_with_risk():
    obj = lane()

    full = obj._adaptive_position_capacity(
        supervisor(30.0, 1.0),
        snapshot(),
        candidate_count=20,
        entries_today=0,
    )

    reduced = obj._adaptive_position_capacity(
        supervisor(30.0, 0.25),
        snapshot(),
        candidate_count=20,
        entries_today=0,
    )

    assert (
        reduced["target_positions"]
        < full["target_positions"]
    )


def test_position_capacity_respects_daily_executor_room():
    obj = lane()
    snap = snapshot()
    snap["daily_order_count"] = 98

    result = obj._adaptive_position_capacity(
        supervisor(100.0, 1.0),
        snap,
        candidate_count=20,
        entries_today=0,
    )

    assert result["available_slots"] <= 2


def test_entry_batch_expands_when_many_slots_exist():
    obj = lane()

    batch = obj._adaptive_entry_batch(
        slots=15,
        candidate_count=20,
        risk_multiplier=1.0,
    )

    assert batch > 3
    assert batch <= 8


def test_scout_selection_is_not_fixed_at_six():
    candidates = [
        {
            "symbol": f"S{i}/USDT",
            "last": 0.10 + i,
            "percentage_24h": float(
                30 - i
            ),
            "quote_volume_usd": (
                1_000_000 + i
            ),
            "spread_bps": 2.0,
        }
        for i in range(15)
    ]

    selected = (
        ReadOnlySwarmService._select_precision_scout(
            candidates,
            capacity=12,
        )
    )

    assert len(selected) == 12


def test_micro_depth_can_expand_and_contract():
    service = object.__new__(
        ReadOnlySwarmService
    )

    service.max_micro_symbols = 12
    service._precision_micro_capacity = 6
    service._precision_micro_last_failure_count = 0
    service.microstream_sample_failures = 0
    service.microstream_last_loop_seconds = 0.20

    expanded = (
        service._adaptive_microstream_capacity(
            scout_count=12,
        )
    )

    assert expanded == 7

    service.microstream_sample_failures = 1
    service.microstream_last_loop_seconds = 0.20

    contracted = (
        service._adaptive_microstream_capacity(
            scout_count=12,
        )
    )

    assert contracted == 6
