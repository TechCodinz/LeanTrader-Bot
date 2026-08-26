from __future__ import annotations

import time

from leantrader.production.testnet_roundtrip_candidate_router_v1616 import (
    _CandidateProxy,
)
from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
)
from tests.test_production_testnet_exit_recycle_v1608 import (
    hyper_lane,
)
from tests.test_testnet_execution import (
    engine,
)


def _pending(
    lane,
    *,
    event_id: str,
):
    event = lane._new_event(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.02,
        price=100.0,
        reason=(
            "fast_collective_testnet_entry:"
            "cost_qualified"
        ),
        now=time.time(),
    )

    event["event_id"] = event_id

    pending = {
        "kind": "entry",
        "event": event,
        "assessment": {
            "entry_mode": "cost_qualified",
            "order_notional_usd": 2.0,
        },
        "created_at": time.time(),
    }

    lane._set_pending(pending)

    return pending


def test_router_blocks_unexecutable_candidate_before_executor_order_record(
    tmp_path,
):
    fake = PriceGuardBybit()
    fake.sell_limit = 105.0

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )
    instance.start()

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    before_orders = len(
        instance.state.get("orders")
        or {}
    )

    pending = _pending(
        lane,
        event_id="v1616-blocked",
    )

    result = lane._submit_pending(
        pending,
        now=time.time(),
    )

    assert result["reason"] == (
        "entry_route_preflight_blocked"
    )

    assert (
        result["details"]["reason"]
        == "prospective_exit_price_limit_unexecutable"
    )

    assert fake.created == []

    assert len(
        instance.state.get("orders")
        or {}
    ) == before_orders

    assert lane._pending() is None

    assert (
        instance.state[
            "v1613_entry_blocked_until"
        ]["BTC/USDT"]
        > time.time()
    )

    health = lane.health()

    assert (
        health[
            "roundtrip_candidate_router"
        ]["preflight_blocks"]
        == 1
    )

    assert (
        health[
            "roundtrip_candidate_router"
        ][
            "blocked_candidate_executor_order_created"
        ]
        is False
    )


def test_roundtrip_executable_candidate_is_rechecked_and_submitted(
    tmp_path,
):
    fake = PriceGuardBybit()
    fake.sell_limit = 90.0

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )
    instance.start()

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    pending = _pending(
        lane,
        event_id="v1616-allowed",
    )

    result = lane._submit_pending(
        pending,
        now=time.time(),
    )

    assert result["reason"] == (
        "testnet_event_processed"
    )

    assert len(fake.created) == 1
    assert fake.created[0]["side"] == "buy"

    price_limit_calls = [
        row
        for row in fake.calls
        if row[0] == "price_limit"
    ]

    # Route check + authoritative
    # executor check immediately before
    # the actual Testnet submission.
    assert len(price_limit_calls) >= 2

    health = lane.health()

    assert (
        health[
            "roundtrip_candidate_router"
        ]["preflight_passes"]
        == 1
    )

    assert (
        health[
            "roundtrip_candidate_router"
        ][
            "executor_rechecks_before_real_order"
        ]
        is True
    )


def test_candidate_proxy_filters_canonical_v1613_cooldown_before_ranking(
    tmp_path,
):
    fake = PriceGuardBybit()

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )
    instance.start()

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    class Service:
        def collective_candidates(
            self,
            limit=8,
        ):
            return [
                "BTC/USDT",
                "ETH/USDT",
            ][:limit]

    instance.state.setdefault(
        "v1613_entry_blocked_until",
        {},
    )["BTC/USDT"] = (
        time.time() + 120.0
    )
    instance._save_state()

    proxy = _CandidateProxy(
        Service(),
        lane,
        time.time(),
    )

    assert proxy.collective_candidates(
        limit=8
    ) == ["ETH/USDT"]

    health = lane.health()

    assert (
        health[
            "roundtrip_candidate_router"
        ][
            "cooldown_candidates_filtered"
        ]
        >= 1
    )

    assert (
        health["live_authority"]
        is False
    )
