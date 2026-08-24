
from __future__ import annotations

from leantrader.production.fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)


def signal(symbol: str, price: float = 100.0) -> dict:
    return {
        "fresh": True,
        "ranked_opportunity": {
            "symbol": symbol,
            "quality_multiplier": 0.60,
            "qualified": False,
        },
        "timeframe_assessments": {
            "1m": {
                "direction": "long",
                "confidence": 0.65,
                "expected_edge_bps": 8.0,
                "modeled_round_trip_cost_bps": 30.0,
                "independently_qualified": False,
            },
            "5m": {
                "direction": "long",
                "confidence": 0.65,
                "expected_edge_bps": 12.0,
                "modeled_round_trip_cost_bps": 30.0,
                "independently_qualified": False,
            },
        },
        "micro_proposals": [],
        "microstructure": {
            "features": {
                "midpoint": price,
                "spread_bps": 2.0,
            },
            "microstream_tracked": True,
            "path_assessments": [
                {
                    "direction": "long",
                    "confidence": 0.30,
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


class MultiService:
    def __init__(self) -> None:
        self.symbols = [
            "AAA/USDT",
            "BBB/USDT",
            "CCC/USDT",
            "DDD/USDT",
        ]
        self.prices = {
            symbol: 100.0
            for symbol in self.symbols
        }

    def collective_candidates(self, limit: int = 18):
        return self.symbols[:limit]

    def collective_signal(self, symbol: str):
        return signal(
            symbol,
            self.prices[symbol],
        )


class FakeTestnet:
    TERMINAL = {"closed", "canceled", "rejected", "skipped"}

    def __init__(self) -> None:
        self.positions: dict[str, float] = {}
        self.events: list[dict] = []

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
            event = dict(event)
            self.events.append(event)
            symbol = event["symbol"]
            side = event["side"]
            requested = float(event["quantity"])

            if side == "buy":
                self.positions[symbol] = (
                    self.positions.get(symbol, 0.0)
                    + requested
                )
                filled = requested
            else:
                current = self.positions.get(symbol, 0.0)
                remaining = max(
                    0.0,
                    float(event.get("remaining_quantity", 0.0)),
                )
                fraction = (
                    1.0
                    if remaining <= 0.0
                    else requested / (requested + remaining)
                )
                filled = min(
                    current,
                    current * max(0.0, min(1.0, fraction)),
                )
                left = max(0.0, current - filled)
                if left > 1e-12:
                    self.positions[symbol] = left
                else:
                    self.positions.pop(symbol, None)

            output.append(
                {
                    "status": "closed",
                    "symbol": symbol,
                    "side": side,
                    "filled": filled,
                    "average": float(event["price"]),
                }
            )
        return output


def supervisor(now: float = 1_000.0):
    return {
        "timestamp": now,
        "healthy": True,
        "halt_reason": None,
        "required_failures": [],
        "canonical_open_positions": [],
        "symbols": {},
    }


def build_lane(tmp_path):
    service = MultiService()
    testnet = FakeTestnet()
    lane = HyperSpeedCollectiveTestnetLane(
        service_provider=lambda: service,
        testnet=testnet,
        state_path=tmp_path / "hyper.json",
        supervisory_provider=lambda: supervisor(),
        order_usd=1.0,
        round_trip_cost_bps=30.0,
        cadence_seconds=5.0,
        maximum_hold_seconds=90.0,
        take_profit_bps=60.0,
        stop_loss_bps=40.0,
        maximum_entries_per_day=24,
        bootstrap_after_seconds=10.0,
        maximum_concurrent_positions=6,
        maximum_entries_per_cycle=3,
        reentry_cooldown_seconds=20.0,
    )
    lane.started_at = 900.0
    return lane, service, testnet


def test_hyper_lane_opens_multiple_positions_in_one_cycle(tmp_path):
    lane, _service, testnet = build_lane(tmp_path)

    result = lane.step(now=1_000.0)

    assert result["reason"] == "fast_multi_route_cycle"
    assert len(lane.health()["active_positions"]) == 3
    assert [event["side"] for event in testnet.events] == [
        "buy",
        "buy",
        "buy",
    ]


def test_hyper_sentinels_close_multiple_and_reuse_slots(tmp_path):
    lane, service, testnet = build_lane(tmp_path)

    lane.step(now=1_000.0)
    opened = set(lane.health()["active_positions"])
    assert len(opened) == 3

    lane.step(now=1_091.0)

    recycled = lane.health()

    # Sentinels close the original positions, then the router may
    # immediately recycle the freed capacity into fresh opportunities.
    assert recycled["active_positions"]
    assert (
        len(recycled["active_positions"])
        <= lane.maximum_concurrent_positions
    )

    assert (
        sum(
            event.get("side") == "sell"
            for event in testnet.events
        )
        >= len(opened)
    )

    assert (
        sum(
            event.get("side") == "buy"
            for event in testnet.events
        )
        > len(opened)
    )
    assert lane.health()["exits_today"] == 3
    assert len(lane.health()["recent_closed"]) == 3
    assert [event["side"] for event in testnet.events].count("sell") == 3

    result = lane.step(now=1_121.0)
    assert result["reason"] == "fast_multi_route_cycle"

    active = lane.health()["active_positions"]

    # Hyper routing keeps previously recycled positions alive while
    # filling additional free slots with fresh independent opportunities.
    assert active
    assert len(active) <= lane.maximum_concurrent_positions
    assert len(active) >= 3


def test_fast_exit_does_not_sell_external_quantity(tmp_path):
    lane, _service, testnet = build_lane(tmp_path)

    lane.step(now=1_000.0)
    symbol = next(iter(lane.health()["active_positions"]))
    fast_qty = lane.state["active"][symbol]["quantity"]

    testnet.positions[symbol] += fast_qty
    lane.step(now=1_091.0)

    assert symbol in testnet.positions
    assert abs(testnet.positions[symbol] - fast_qty) < 1e-9
    assert symbol not in lane.health()["active_positions"]


def test_required_engine_failure_blocks_new_entries(tmp_path):
    service = MultiService()
    testnet = FakeTestnet()
    blocked = supervisor()
    blocked["required_failures"] = ["market_data"]

    lane = HyperSpeedCollectiveTestnetLane(
        service_provider=lambda: service,
        testnet=testnet,
        state_path=tmp_path / "blocked.json",
        supervisory_provider=lambda: blocked,
        order_usd=1.0,
        round_trip_cost_bps=30.0,
        maximum_entries_per_day=24,
    )
    lane.started_at = 900.0

    result = lane.step(now=1_000.0)

    assert result["reason"] == "required_engine_failure"
    assert testnet.events == []
