from leantrader.production.fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)


def test_dynamic_ticket_floor_allows_one_dollar_lane_with_150_available():
    lane = HyperSpeedCollectiveTestnetLane.__new__(
        HyperSpeedCollectiveTestnetLane
    )

    lane.maximum_order_usd = 5.0
    lane.order_usd = 1.0
    lane.maximum_adaptive_positions = 24
    lane.maximum_concurrent_positions = 6
    lane.maximum_entries_per_day = 50
    lane._fast_open_notional = lambda: 0.0

    supervisor = {
        "capital_growth": {
            "risk_multiplier": 1.0,
            "remaining_deployable_notional": 1.50,
        }
    }

    snapshot = {
        "positions": {},
        "risk_limits": {
            "max_orders_per_day": 100,
            "max_daily_submitted_usd": 500.0,
        },
        "daily_order_count": 0,
        "daily_submitted_usd": 0.0,
    }

    result = lane._adaptive_position_capacity(
        supervisor,
        snapshot,
        candidate_count=1,
        entries_today=0,
    )

    assert result["minimum_viable_notional_usd"] == 1.0
    assert result["capital_slots"] == 1
    assert result["available_slots"] == 1
    assert result["live_authority"] is False


def test_dynamic_ticket_floor_never_plans_below_fifty_cents():
    lane = HyperSpeedCollectiveTestnetLane.__new__(
        HyperSpeedCollectiveTestnetLane
    )

    lane.maximum_order_usd = 5.0
    lane.order_usd = 0.25
    lane.maximum_adaptive_positions = 24
    lane.maximum_concurrent_positions = 6
    lane.maximum_entries_per_day = 50
    lane._fast_open_notional = lambda: 0.0

    supervisor = {
        "capital_growth": {
            "risk_multiplier": 1.0,
            "remaining_deployable_notional": 0.75,
        }
    }

    snapshot = {
        "positions": {},
        "risk_limits": {
            "max_orders_per_day": 100,
            "max_daily_submitted_usd": 500.0,
        },
        "daily_order_count": 0,
        "daily_submitted_usd": 0.0,
    }

    result = lane._adaptive_position_capacity(
        supervisor,
        snapshot,
        candidate_count=1,
        entries_today=0,
    )

    assert result["minimum_viable_notional_usd"] == 0.50
    assert result["capital_slots"] == 1
    assert result["available_slots"] == 1
