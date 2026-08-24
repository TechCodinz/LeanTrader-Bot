from __future__ import annotations

import threading
import time

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)


def test_precision_scout_reserves_sub_dollar_coverage():
    candidates = [
        {
            "symbol": "CHEAPA/USDT",
            "last": 0.08,
            "percentage_24h": 18.0,
            "quote_volume_usd": 2_000_000,
            "spread_bps": 4.0,
        },
        {
            "symbol": "CHEAPB/USDT",
            "last": 0.50,
            "percentage_24h": 12.0,
            "quote_volume_usd": 3_000_000,
            "spread_bps": 3.0,
        },
        {
            "symbol": "CHEAPC/USDT",
            "last": 0.90,
            "percentage_24h": -10.0,
            "quote_volume_usd": 4_000_000,
            "spread_bps": 2.0,
        },
        {
            "symbol": "BTC/USDT",
            "last": 60000.0,
            "percentage_24h": 20.0,
            "quote_volume_usd": 50_000_000,
            "spread_bps": 1.0,
        },
        {
            "symbol": "MID/USDT",
            "last": 4.0,
            "percentage_24h": 15.0,
            "quote_volume_usd": 8_000_000,
            "spread_bps": 2.0,
        },
        {
            "symbol": "OTHER/USDT",
            "last": 20.0,
            "percentage_24h": 8.0,
            "quote_volume_usd": 9_000_000,
            "spread_bps": 2.0,
        },
    ]

    selected = (
        ReadOnlySwarmService._select_precision_scout(
            candidates,
            capacity=6,
        )
    )

    assert len(selected) == 6

    assert sum(
        float(row["last"]) < 1.0
        for row in selected
    ) >= 3


def test_price_does_not_override_movement_ordering():
    candidates = [
        {
            "symbol": "CHEAP/USDT",
            "last": 0.01,
            "percentage_24h": 2.0,
            "quote_volume_usd": 1_000_000,
            "spread_bps": 4.0,
        },
        {
            "symbol": "FAST/USDT",
            "last": 0.80,
            "percentage_24h": 15.0,
            "quote_volume_usd": 1_000_000,
            "spread_bps": 4.0,
        },
    ]

    selected = (
        ReadOnlySwarmService._select_precision_scout(
            candidates,
            capacity=2,
        )
    )

    assert selected[0]["symbol"] == "FAST/USDT"


def test_precision_context_symbols_include_scout_and_slow():
    service = object.__new__(
        ReadOnlySwarmService
    )

    service._lock = threading.RLock()
    service._precision_scout_symbols = [
        "FAST/USDT",
        "CHEAP/USDT",
    ]
    service._microstream_symbols = [
        "BTC/USDT",
    ]

    assert (
        service.precision_context_symbols()
        == {
            "FAST/USDT",
            "CHEAP/USDT",
            "BTC/USDT",
        }
    )


def test_collective_signal_uses_fresh_precision_mtf_cache():
    service = object.__new__(
        ReadOnlySwarmService
    )

    now = time.time()

    service._lock = threading.RLock()
    service.cadence_seconds = 5.0
    service.precision_scout_refresh_seconds = 20.0
    service.last_success_at = 0.0
    service.cycles = 0
    service.VERSION = "1.60.0"

    service._microstream_snapshots = {
        "FAST/USDT": {
            "timestamp": now,
            "midpoint": 0.5,
            "spread_bps": 2.0,
            "bid_depth_usd": 20_000.0,
            "ask_depth_usd": 20_000.0,
            "temporal_samples": 5,
        }
    }

    service._precision_context_cache = {
        "FAST/USDT": {
            "timestamp": now,
            "assessments": {
                "5m": {
                    "direction": "long",
                    "confidence": 0.66,
                    "expected_edge_bps": 8.0,
                    "independently_qualified": False,
                }
            },
        }
    }

    service.last_step = {
        "ranked": [],
        "timeframe_assessments": {},
        "micro_agent_foundry_proposals": [],
        "microstructure": {},
    }

    signal = service.collective_signal(
        "FAST/USDT"
    )

    assert (
        signal["timeframe_assessments"][
            "5m"
        ]["confidence"]
        == 0.66
    )

    assert (
        signal["precision_context"][
            "fresh"
        ]
        is True
    )
