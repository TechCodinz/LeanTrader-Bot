from __future__ import annotations

import threading

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)


def service():
    obj = object.__new__(
        ReadOnlySwarmService
    )

    obj.max_micro_symbols = 12
    obj.microstream_target_loop_seconds = 1.5
    obj.microstream_freshness_seconds = 2.0
    obj.microstream_per_symbol_latency_seconds = 0.25
    obj.microstream_last_loop_seconds = 0.0
    obj.microstream_sample_failures = 0
    obj._precision_micro_last_failure_count = 0
    obj._precision_micro_capacity = 6
    obj.precision_micro_last_queue = []
    obj._lock = threading.RLock()

    return obj


def test_slow_rest_loop_contracts_deep_cohort_below_six():
    obj = service()

    obj.precision_micro_last_queue = [
        f"S{i}/USDT"
        for i in range(6)
    ]

    obj.microstream_last_loop_seconds = 4.2

    capacity = (
        obj._adaptive_microstream_capacity(
            scout_count=12,
        )
    )

    assert capacity < 6
    assert capacity >= 2


def test_fast_rest_loop_can_expand_deep_cohort():
    obj = service()

    obj.precision_micro_last_queue = [
        "A/USDT",
        "B/USDT",
    ]

    obj.microstream_last_loop_seconds = 0.20
    obj.microstream_per_symbol_latency_seconds = 0.10

    capacity = (
        obj._adaptive_microstream_capacity(
            scout_count=12,
        )
    )

    assert capacity > 2
    assert capacity <= 12


def test_execution_symbols_are_pinned():
    obj = service()
    obj._execution_precision_pins = {}

    obj.pin_execution_symbols(
        {
            "abc/usdt",
            "XYZ/USDT",
        },
        ttl_seconds=10,
    )

    assert "ABC/USDT" in (
        obj._execution_precision_pins
    )

    assert "XYZ/USDT" in (
        obj._execution_precision_pins
    )


def test_fast_candidates_prioritize_only_fresh_samples():
    obj = service()

    import time

    now = time.time()

    obj._microstream_snapshots = {
        "FRESH/USDT": {
            "timestamp": now - 0.2,
            "spread_bps": 2.0,
            "bid_depth_usd": 20_000.0,
            "ask_depth_usd": 20_000.0,
            "temporal_samples": 4,
        },
        "STALE/USDT": {
            "timestamp": now - 9.0,
            "spread_bps": 2.0,
            "bid_depth_usd": 20_000.0,
            "ask_depth_usd": 20_000.0,
            "temporal_samples": 10,
        },
    }

    obj._precision_hot_symbols = [
        "FRESH/USDT",
        "STALE/USDT",
    ]

    obj._precision_explorer_symbol = None

    obj._precision_scout_symbols = [
        "UNSAMPLED/USDT",
    ]

    obj.last_step = {
        "ranked": [],
    }

    rows = obj.collective_candidates(
        limit=48
    )

    assert "FRESH/USDT" in rows
    assert "STALE/USDT" not in rows
    assert "UNSAMPLED/USDT" not in rows


def test_broad_ranked_fallback_remains_available():
    obj = service()

    obj._microstream_snapshots = {}
    obj._precision_hot_symbols = []
    obj._precision_explorer_symbol = None
    obj._precision_scout_symbols = [
        "DISCOVERY/USDT",
    ]

    obj.last_step = {
        "ranked": [
            {
                "symbol": "SLOW/USDT",
            }
        ],
    }

    rows = obj.collective_candidates(
        limit=48
    )

    assert "SLOW/USDT" in rows
    assert "DISCOVERY/USDT" not in rows


class _AliveThread:
    def is_alive(self):
        return True


def test_microstream_watchdog_detects_stuck_inflight_request():
    import time

    obj = service()

    obj._stop = threading.Event()
    obj._microstream_thread = _AliveThread()
    obj._microstream_generation = 4
    obj.microstream_stall_seconds = 30.0
    obj.microstream_last_observation_at = 100.0
    obj.microstream_last_attempt_started_at = 120.0
    obj.microstream_last_attempt_symbol = "MNT/USDT"

    row = obj._microstream_stall_snapshot(
        now=151.0
    )

    assert row["alive"] is True
    assert row["stalled"] is True
    assert row["generation"] == 4
    assert row["attempt_symbol"] == "MNT/USDT"
    assert row["request_age_seconds"] == 31.0
    assert row["live_authority"] is False


def test_microstream_watchdog_does_not_flag_idle_or_recent_request():
    obj = service()

    obj._stop = threading.Event()
    obj._microstream_thread = _AliveThread()
    obj._microstream_generation = 2
    obj.microstream_stall_seconds = 30.0
    obj.microstream_last_observation_at = 200.0

    obj.microstream_last_attempt_started_at = 0.0
    obj.microstream_last_attempt_symbol = None

    idle = obj._microstream_stall_snapshot(
        now=500.0
    )

    assert idle["stalled"] is False

    obj.microstream_last_attempt_started_at = 490.0
    obj.microstream_last_attempt_symbol = "XRP/USDT"

    recent = obj._microstream_stall_snapshot(
        now=500.0
    )

    assert recent["stalled"] is False


def test_execution_candidate_has_priority_over_held_hot_symbols():
    obj = service()

    queue, _cursor, schedule = (
        obj._build_sticky_precision_queue(
            scout_symbols=[
                "CSPR/USDT",
                "JASMY/USDT",
                "MNT/USDT",
            ],
            sticky_symbols=[
                "CSPR/USDT",
                "JASMY/USDT",
            ],
            due_symbols=[
                "LAB/USDT",
            ],
            velocity_symbols=[],
            capacity=2,
            cursor=0,
            now=100.0,
            current_hot=[
                "CSPR/USDT",
                "JASMY/USDT",
            ],
            hot_until=200.0,
            current_explorer=None,
            explorer_until=0.0,
            priority_symbols=[
                "MNT/USDT",
            ],
        )
    )

    assert queue[0] == "MNT/USDT"
    assert "MNT/USDT" in queue
    assert len(queue) <= 2


def test_execution_candidate_pin_is_separate_from_position_pins():
    obj = service()

    obj._execution_precision_pins = {
        "CSPR/USDT": 9999999999.0,
    }

    obj._execution_candidate_pins = {}

    obj.pin_execution_candidate_symbols(
        {
            "mnt/usdt",
        },
        ttl_seconds=6.0,
    )

    assert "MNT/USDT" in (
        obj._execution_candidate_pins
    )

    assert "MNT/USDT" not in (
        obj._execution_precision_pins
    )
