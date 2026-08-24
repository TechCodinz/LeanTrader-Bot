from __future__ import annotations

import threading

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)
from leantrader.production.fast_collective_testnet import (
    FastCollectiveTestnetLane,
)


def test_live_scout_gets_five_of_six_slots_when_research_due():
    scouts = [
        f"S{i}/USDT"
        for i in range(6)
    ]

    due = [
        f"R{i}/USDT"
        for i in range(6)
    ]

    queue, cursor, meta = (
        ReadOnlySwarmService._build_microstream_queue(
            scout_symbols=scouts,
            sticky_symbols=[],
            due_symbols=due,
            capacity=6,
            cursor=0,
        )
    )

    assert len(queue) == 6
    assert meta["scout_slots"] == 5
    assert meta["due_slots"] == 1
    assert cursor != 0


def test_six_scout_symbols_rotate_without_starvation():
    scouts = [
        f"S{i}/USDT"
        for i in range(6)
    ]

    q1, cursor, _ = (
        ReadOnlySwarmService._build_microstream_queue(
            scout_symbols=scouts,
            sticky_symbols=[],
            due_symbols=["RESEARCH/USDT"],
            capacity=6,
            cursor=0,
        )
    )

    q2, _, _ = (
        ReadOnlySwarmService._build_microstream_queue(
            scout_symbols=scouts,
            sticky_symbols=[],
            due_symbols=["RESEARCH/USDT"],
            capacity=6,
            cursor=cursor,
        )
    )

    observed = {
        symbol
        for symbol in [
            *q1,
            *q2,
        ]
        if symbol.startswith("S")
    }

    assert observed == set(scouts)


def test_without_due_research_all_six_scouts_are_sampled():
    scouts = [
        f"S{i}/USDT"
        for i in range(6)
    ]

    queue, _, meta = (
        ReadOnlySwarmService._build_microstream_queue(
            scout_symbols=scouts,
            sticky_symbols=[],
            due_symbols=[],
            capacity=6,
            cursor=0,
        )
    )

    assert queue == scouts
    assert meta["scout_slots"] == 6
    assert meta["due_slots"] == 0


def test_success_clears_stale_fast_lane_error():
    lane = object.__new__(
        FastCollectiveTestnetLane
    )

    lane._lock = threading.RLock()

    lane.state = {
        "last_error": (
            "RuntimeError: old reconciliation error"
        ),
        "last_error_at": 123.0,
    }

    lane._save_locked = lambda: None

    lane._clear_transient_error_after_success()

    assert lane.state["last_error"] is None
    assert lane.state["last_error_at"] is None
