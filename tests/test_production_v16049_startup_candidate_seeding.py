
from __future__ import annotations

import threading

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)


def service_stub():
    service = object.__new__(
        ReadOnlySwarmService
    )

    service._lock = threading.RLock()
    service.microstream_freshness_seconds = 2.0
    service._microstream_snapshots = {}
    service._precision_hot_symbols = []
    service._precision_explorer_symbol = None
    service._precision_scout_symbols = [
        "SUI/USDT",
        "XRP/USDT",
        "ARB/USDT",
    ]
    service._candidates = []
    service.last_step = {}
    return service


def test_precision_scout_breaks_cold_start_candidate_deadlock():
    service = service_stub()

    rows = service.collective_candidates(
        limit=8
    )

    assert rows == [
        "SUI/USDT",
        "XRP/USDT",
        "ARB/USDT",
    ]


def test_discovery_universe_is_bounded_fallback_seed():
    service = service_stub()
    service._precision_scout_symbols = []

    service._candidates = [
        {
            "symbol": "LOW/USDT",
            "percentage_24h": 2.0,
            "quote_volume_usd": 100000.0,
            "spread_bps": 5.0,
        },
        {
            "symbol": "FAST/USDT",
            "percentage_24h": 12.0,
            "quote_volume_usd": 500000.0,
            "spread_bps": 2.0,
        },
    ]

    rows = service.collective_candidates(
        limit=8
    )

    assert rows == [
        "FAST/USDT",
        "LOW/USDT",
    ]


def test_fresh_micro_observation_keeps_priority():
    service = service_stub()
    service._microstream_snapshots = {
        "LIVE/USDT": {
            "timestamp": __import__("time").time(),
            "spread_bps": 1.0,
            "bid_depth_usd": 20000.0,
            "ask_depth_usd": 20000.0,
            "temporal_samples": 3,
            "recent_midpoint_trend_bps_5s": 10.0,
            "midpoint_velocity_bps_per_second": 3.0,
            "midpoint_acceleration_bps_per_second2": 1.0,
        }
    }

    rows = service.collective_candidates(
        limit=8
    )

    assert rows[0] == "LIVE/USDT"
    assert "SUI/USDT" in rows
