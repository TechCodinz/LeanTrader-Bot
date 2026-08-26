from __future__ import annotations

import copy

from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
    seed,
)
from tests.test_production_testnet_exit_recycle_v1608 import (
    hyper_lane,
)
from tests.test_testnet_execution import (
    engine,
)


def _runtime(
    tmp_path,
    *,
    sell_limit: float,
):
    fake = PriceGuardBybit()
    fake.bid = 100.0
    fake.ask = 101.0
    fake.sell_limit = sell_limit

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )

    instance.start()

    seed(
        instance,
        fake,
        0.1,
        9.5,
    )

    instance.reconcile_required()

    lane, service, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    now = 1_000.0

    active = {
        "symbol": "BTC/USDT",
        "quantity": 0.1,
        "initial_quantity": 0.1,
        "entry_price": 95.0,
        "entry_notional_usd": 9.5,
        "peak_price": 101.0,
        "entered_at": 900.0,
        "target_hold_seconds": 30.0,
    }

    queue = {
        "symbol": "BTC/USDT",
        "kind": "exit_recovery",
        "source_event": {
            "event_id": "legacy-exit",
            "symbol": "BTC/USDT",
            "side": "sell",
            "price": 100.0,
            "quantity": 0.1,
            "reason": (
                "fast_collective_testnet_exit:"
                "velocity_take_profit:"
                "corrected_recycle:"
                "corrected_recycle:"
                "corrected_recycle"
            ),
        },
        "assessment": {
            "exit_reason": (
                "velocity_take_profit"
            ),
        },
        "recovery_attempt": 150,
        "deferrals": 40,
        "deferred_at": 900.0,
        "next_retry_at": 1_300.0,
        "reason": (
            "exit_recovery_waiting_for_executable_balance"
        ),
        "live_authority": False,
    }

    with lane._lock:
        lane.state.setdefault(
            "active",
            {},
        )["BTC/USDT"] = copy.deepcopy(
            active
        )

        lane.state.setdefault(
            "deferred_exit_recoveries",
            {},
        )["BTC/USDT"] = copy.deepcopy(
            queue
        )

        lane._save_locked()

    return (
        lane,
        service,
        instance,
        fake,
        now,
    )


def test_blocked_sell_boundary_watches_without_order_or_deferral_growth(
    tmp_path,
):
    (
        lane,
        service,
        instance,
        fake,
        now,
    ) = _runtime(
        tmp_path,
        sell_limit=105.0,
    )

    snapshot = (
        instance.safe_snapshot()
    )

    record = (
        lane._active_snapshot()
        ["BTC/USDT"]
    )

    result = lane._manage_active(
        service,
        snapshot,
        "BTC/USDT",
        record,
        now=now,
    )

    assert result["reason"] == (
        "price_limit_exit_watch"
    )

    assert fake.created == []

    queue = (
        lane.state[
            "deferred_exit_recoveries"
        ]["BTC/USDT"]
    )

    assert queue["deferrals"] == 40
    assert queue["recovery_attempt"] == 150

    watch = (
        lane.state[
            "v1615_price_limit_watch"
        ]["BTC/USDT"]
    )

    assert (
        watch["executable_boundary"]
        is False
    )

    assert watch["fresh_bid"] == 100.0
    assert watch["sell_limit"] == 105.0
    assert watch["live_authority"] is False


def test_boundary_clear_releases_existing_safe_exit_immediately(
    tmp_path,
):
    (
        lane,
        service,
        instance,
        fake,
        now,
    ) = _runtime(
        tmp_path,
        sell_limit=95.0,
    )

    with instance._io_lock:
        instance.state.setdefault(
            "v1611_price_limit_blocked_until",
            {},
        )["BTC/USDT"] = 1_300.0

        instance._save_state()

    snapshot = (
        instance.safe_snapshot()
    )

    record = (
        lane._active_snapshot()
        ["BTC/USDT"]
    )

    result = lane._manage_active(
        service,
        snapshot,
        "BTC/USDT",
        record,
        now=now,
    )

    assert len(fake.created) == 1
    assert fake.created[0]["side"] == "sell"

    assert (
        "BTC/USDT"
        not in instance.health()[
            "positions"
        ]
    )

    assert (
        "BTC/USDT"
        not in lane._active_snapshot()
    )

    assert (
        "BTC/USDT"
        not in (
            lane.state.get(
                "deferred_exit_recoveries"
            )
            or {}
        )
    )

    assert (
        "BTC/USDT"
        not in (
            instance.state.get(
                "v1611_price_limit_blocked_until"
            )
            or {}
        )
    )

    assert result["reason"] in {
        "testnet_event_processed",
        "fast_multi_route_cycle",
    }
