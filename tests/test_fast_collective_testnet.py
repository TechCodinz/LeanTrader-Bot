from __future__ import annotations

from pathlib import Path

from leantrader.production.fast_collective_testnet import (
    FastCollectiveTestnetLane,
)


def signal(
    *,
    micro_direction="long",
    micro_confidence=0.30,
    mtf_direction="long",
    mtf_confidence=0.65,
):
    return {
        "fresh": True,
        "ranked_opportunity": {
            "symbol": "AAA/USDT",
            "quality_multiplier": 0.60,
            "qualified": False,
        },
        "timeframe_assessments": {
            "1m": {
                "direction": mtf_direction,
                "confidence": mtf_confidence,
                "expected_edge_bps": 8.0,
                "modeled_round_trip_cost_bps": 30.0,
                "independently_qualified": False,
            },
            "5m": {
                "direction": mtf_direction,
                "confidence": mtf_confidence,
                "expected_edge_bps": 12.0,
                "modeled_round_trip_cost_bps": 30.0,
                "independently_qualified": False,
            },
        },
        "micro_proposals": [],
        "microstructure": {
            "features": {
                "midpoint": 100.0,
                "spread_bps": 2.0,
            },
            "microstream_tracked": True,
            "path_assessments": [
                {
                    "direction": micro_direction,
                    "confidence": micro_confidence,
                    "expected_edge_bps": 6.0,
                    "modeled_round_trip_cost_bps": 30.0,
                    "pressure_score": 0.40,
                    "specialist": "kinematic_test",
                    "horizon_seconds": 30,
                    "independently_qualified": False,
                }
            ],
        },
    }


def supervisor(now=1_000.0):
    return {
        "timestamp": now,
        "healthy": True,
        "halt_reason": None,
        "required_failures": [],
        "symbols": {
            "AAA/USDT": {
                "route": {
                    "base_score": 0.20,
                    "base_confidence": 0.65,
                    "advanced_score": 0.15,
                    "advanced_confidence": 0.55,
                    "temporal_session": {
                        "allowed": True,
                    },
                    "exchange_protection": {
                        "allowed": True,
                    },
                },
                "collective": {
                    "groups": [
                        {
                            "group": "evolution",
                            "score": 0.20,
                            "confidence": 0.55,
                            "members": [
                                "evolution:test"
                            ],
                        }
                    ]
                },
            }
        },
    }


class FakeService:
    def __init__(self):
        self.current = signal()

    def collective_candidates(self, limit=8):
        assert limit > 0
        return ["AAA/USDT"]

    def collective_signal(self, symbol):
        assert symbol == "AAA/USDT"
        return self.current


class FakeTestnet:
    def __init__(self):
        self.positions = {}
        self.events = []

    def safe_snapshot(self):
        return {
            "positions": dict(self.positions),
            "open_orders": 0,
            "last_reconciliation_errors": [],
            "kill_switch_active": False,
        }

    def mirror_events(self, events):
        output = []

        for event in events:
            self.events.append(dict(event))
            symbol = event["symbol"]

            if event["side"] == "buy":
                self.positions[symbol] = (
                    event["quantity"]
                )

            else:
                self.positions.pop(
                    symbol,
                    None,
                )

            output.append(
                {
                    "status": "closed",
                    "symbol": symbol,
                    "side": event["side"],
                    "filled": event["quantity"],
                    "average": event["price"],
                }
            )

        return output


def lane(
    tmp_path: Path,
    *,
    service=None,
    testnet=None,
    supervisory=None,
):
    service = service or FakeService()
    testnet = testnet or FakeTestnet()
    supervisory = (
        supervisory
        if supervisory is not None
        else supervisor()
    )

    instance = FastCollectiveTestnetLane(
        service_provider=lambda: service,
        testnet=testnet,
        state_path=tmp_path / "fast.json",
        supervisory_provider=lambda: supervisory,
        order_usd=2.0,
        round_trip_cost_bps=30.0,
        cadence_seconds=5.0,
        maximum_hold_seconds=90.0,
        bootstrap_after_seconds=45.0,
    )

    instance.started_at = 900.0

    return instance, service, testnet


def test_fast_lane_opens_and_time_exits_testnet_probe(
    tmp_path,
):
    instance, service, testnet = lane(
        tmp_path
    )

    first = instance.step(
        now=1_000.0
    )

    assert first["reason"] == (
        "testnet_event_processed"
    )

    health = instance.health()

    assert (
        health["entries_today"]
        == 1
    )

    assert (
        "AAA/USDT"
        in health["active_positions"]
    )

    assert (
        testnet.events[0]["side"]
        == "buy"
    )

    second = instance.step(
        now=1_091.0
    )

    assert second["reason"] == (
        "testnet_event_processed"
    )

    health = instance.health()

    assert (
        health["exits_today"]
        == 1
    )

    assert (
        health["active_positions"]
        == {}
    )

    assert [
        event["side"]
        for event in testnet.events
    ] == [
        "buy",
        "sell",
    ]

    assert (
        health["recent_closed"][-1][
            "modeled_round_trip_cost_bps"
        ]
        == 30.0
    )

    assert (
        health["live_authority"]
        is False
    )


def test_fast_lane_fails_closed_on_required_engine_failure(
    tmp_path,
):
    blocked = supervisor()

    blocked["required_failures"] = [
        "market_data"
    ]

    instance, _service, testnet = lane(
        tmp_path,
        supervisory=blocked,
    )

    result = instance.step(
        now=1_000.0
    )

    assert result["reason"] == (
        "required_engine_failure"
    )

    assert testnet.events == []


def test_fast_lane_blocks_fresh_short_conflict(
    tmp_path,
):
    service = FakeService()

    service.current = signal(
        micro_direction="short",
        micro_confidence=0.40,
        mtf_direction="short",
        mtf_confidence=0.70,
    )

    instance, _service, testnet = lane(
        tmp_path,
        service=service,
    )

    result = instance.step(
        now=1_000.0
    )

    assert result["reason"] == (
        "no_aligned_long_candidate"
    )

    assert testnet.events == []
