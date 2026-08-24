from __future__ import annotations

import threading
import time

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)
from leantrader.production.velocity_sniper_testnet import (
    VelocitySniperTestnetLane,
)


def velocity_signal(*, fresh=True):
    return {
        "fresh": True,
        "ranked_opportunity": {
            "symbol": "FAST/USDT",
            "quality_multiplier": 0.55,
        },
        "timeframe_assessments": {
            "1m": {
                "direction": "long",
                "confidence": 0.62,
                "expected_edge_bps": 10.0,
                "independently_qualified": False,
            }
        },
        "micro_proposals": [],
        "microstructure": {
            "microstream_tracked": True,
            "features": {
                "midpoint": 0.012,
                "spread_bps": 3.0,
            },
            "path_assessments": [],
        },
        "micro_velocity": {
            "fresh": fresh,
            "age_seconds": (
                0.20 if fresh else 5.0
            ),
            "midpoint": 0.012,
            "spread_bps": 3.0,
            "bid_depth_usd": 30_000.0,
            "ask_depth_usd": 25_000.0,
            "temporal_samples": 8,
            "midpoint_velocity_bps_per_second": 3.0,
            "midpoint_acceleration_bps_per_second2": 0.5,
            "recent_midpoint_trend_bps_5s": 18.0,
            "recent_midpoint_range_bps_5s": 24.0,
            "depth_imbalance": 0.18,
            "microprice_shift_bps": 1.2,
            "pressure_persistence": 0.8,
            "velocity_score": 55.0,
        },
    }


def supervisor():
    return {
        "route": {
            "base_score": 0.12,
            "base_confidence": 0.50,
            "advanced_score": 0.08,
            "advanced_confidence": 0.25,
            "temporal_session": {
                "allowed": True,
            },
            "exchange_protection": {
                "allowed": True,
            },
        },
        "collective": {
            "groups": [],
        },
    }


def test_velocity_probe_can_originate_only_testnet_exploration():
    result = (
        VelocitySniperTestnetLane.assess_candidate(
            velocity_signal(),
            supervisor(),
            relaxed=True,
        )
    )

    assert result["allowed"] is True
    assert result["velocity_sniper"] is True
    assert (
        result["entry_mode"]
        == "velocity_sniper_probe"
    )
    assert result["cost_qualified"] is False
    assert (
        result["proven_positive_net_edge"]
        is False
    )
    assert result["live_authority"] is False


def test_stale_velocity_cannot_originate():
    result = (
        VelocitySniperTestnetLane.assess_candidate(
            velocity_signal(
                fresh=False
            ),
            supervisor(),
            relaxed=True,
        )
    )

    assert result["allowed"] is False


def test_velocity_requires_collective_support():
    weak = velocity_signal()
    weak["timeframe_assessments"] = {}

    result = (
        VelocitySniperTestnetLane.assess_candidate(
            weak,
            {
                "route": {
                    "temporal_session": {
                        "allowed": True,
                    },
                    "exchange_protection": {
                        "allowed": True,
                    },
                },
                "collective": {
                    "groups": [],
                },
            },
            relaxed=True,
        )
    )

    assert result["allowed"] is False
    assert (
        result["reason"]
        == "velocity_without_collective_support"
    )


def test_micro_velocity_candidates_are_prioritized():
    service = object.__new__(
        ReadOnlySwarmService
    )

    service._lock = threading.RLock()
    service.cadence_seconds = 5.0
    service.last_success_at = time.time()
    service.cycles = 1
    service._microstream_symbols = [
        "FAST/USDT",
    ]
    service._microstream_snapshots = {
        "FAST/USDT": {
            **velocity_signal()[
                "micro_velocity"
            ],
            "timestamp": time.time(),
        }
    }
    service.last_step = {
        "ranked": [
            {
                "symbol": "SLOW/USDT",
                "score": 999.0,
            },
            {
                "symbol": "FAST/USDT",
                "score": 1.0,
            },
        ],
        "timeframe_assessments": {},
        "micro_agent_foundry_proposals": [],
        "microstructure": {},
    }

    candidates = (
        service.collective_candidates(
            limit=2
        )
    )

    assert candidates[0] == "FAST/USDT"


class FakeTestnet:
    def safe_snapshot(self):
        return {
            "positions": {},
            "open_orders": 0,
            "last_reconciliation_errors": [],
            "kill_switch_active": False,
        }


def test_velocity_lane_accepts_subsecond_cadence_and_large_test_cap(
    tmp_path,
):
    lane = VelocitySniperTestnetLane(
        service_provider=lambda: None,
        testnet=FakeTestnet(),
        state_path=tmp_path / "v158.json",
        supervisory_provider=lambda: {},
        order_usd=1.0,
        round_trip_cost_bps=30.0,
        cadence_seconds=0.5,
        maximum_hold_seconds=30.0,
        maximum_entries_per_day=45,
        bootstrap_after_seconds=5.0,
        maximum_concurrent_positions=6,
        maximum_entries_per_cycle=3,
        reentry_cooldown_seconds=2.0,
    )

    assert lane.cadence_seconds == 0.5
    assert lane.maximum_hold_seconds == 30.0
    assert lane.maximum_entries_per_day == 45
    assert lane.reentry_cooldown_seconds == 2.0

    health = lane.health()

    assert health["velocity_sniper"] is True
    assert health["subsecond_detection"] is True
    assert health["live_authority"] is False
