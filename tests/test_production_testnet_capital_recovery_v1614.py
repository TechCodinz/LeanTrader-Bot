from __future__ import annotations

import datetime as dt
import time

from leantrader.production.testnet_capital_recovery_v1614 import (
    _capital_recovery_once,
)
from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
    seed,
)
from tests.test_production_testnet_exit_recycle_v1608 import (
    hyper_lane,
)
from tests.test_testnet_execution import engine


def _instance(tmp_path, fake):
    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )
    instance.start()
    return instance


def _quote_starve(
    instance,
    fake,
    btc=0.1,
):
    fake.balance_total["USDT"] = 1.80
    fake.balance_free["USDT"] = 1.80
    fake.balance_total["BTC"] = btc
    fake.balance_free["BTC"] = btc
    instance.reconcile_required()


def test_quote_starved_profitable_orphan_recycles_to_quote(
    tmp_path,
):
    fake = PriceGuardBybit()
    instance = _instance(
        tmp_path,
        fake,
    )

    seed(
        instance,
        fake,
        0.1,
        9.50,
    )

    _quote_starve(
        instance,
        fake,
    )

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    result = _capital_recovery_once(
        lane,
        now=time.time(),
    )

    assert result["reason"] == (
        "capital_recovery_exit_processed"
    )

    assert (
        result["selected"]
        ["profit_ready"]
        is True
    )

    assert (
        result["result"]
        ["status"]
        == "closed"
    )

    assert fake.created[-1]["side"] == "sell"

    assert (
        "BTC/USDT"
        not in instance.health()["positions"]
    )

    recovery = (
        instance.health()
        ["capital_recovery"]
    )

    assert recovery["recovery_attempts"] == 1
    assert recovery["recovery_fills"] == 1
    assert recovery["recovered_quote_usd"] > 0.0
    assert recovery["live_authority"] is False


def test_price_limit_block_does_not_submit_recovery_order(
    tmp_path,
):
    fake = PriceGuardBybit()
    fake.sell_limit = 105.0

    instance = _instance(
        tmp_path,
        fake,
    )

    seed(
        instance,
        fake,
        0.1,
        9.50,
    )

    _quote_starve(
        instance,
        fake,
    )

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    result = _capital_recovery_once(
        lane,
        now=time.time(),
    )

    assert result["submitted"] is False
    assert result["quote_starved"] is True
    assert fake.created == []

    assert any(
        row["reason"]
        in {
            "bybit_market_price_limit_unexecutable",
            "bybit_market_price_limit_cooldown",
        }
        for row in result["assessments"]
    )


def test_stale_small_loss_can_recycle_only_when_quote_starved(
    tmp_path,
):
    fake = PriceGuardBybit()
    fake.bid = 99.6
    fake.ask = 99.7
    fake.sell_limit = 90.0

    instance = _instance(
        tmp_path,
        fake,
    )

    seed(
        instance,
        fake,
        0.1,
        10.0,
    )

    _quote_starve(
        instance,
        fake,
    )

    old = (
        dt.datetime.now(dt.UTC)
        - dt.timedelta(seconds=1200)
    ).isoformat()

    instance.state[
        "orders"
    ][
        "old-filled-buy"
    ] = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "closed",
        "filled": 0.1,
        "submitted_at": old,
    }

    instance._save_state()

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    result = _capital_recovery_once(
        lane,
        now=time.time(),
    )

    assert (
        result["selected"]
        ["controlled_recycle"]
        is True
    )

    assert (
        result["selected"]
        ["profit_ready"]
        is False
    )

    assert (
        result["selected"]
        ["net_bps_after_model"]
        >= -75.0
    )

    assert (
        result["result"]
        ["status"]
        == "closed"
    )

    assert (
        instance.health()
        ["capital_recovery"]
        ["controlled_loss_recycles"]
        == 1
    )


def test_quote_reserve_blocks_new_fast_entry_when_capital_starved(
    tmp_path,
):
    fake = PriceGuardBybit()

    instance = _instance(
        tmp_path,
        fake,
    )

    fake.balance_total["USDT"] = 1.80
    fake.balance_free["USDT"] = 1.80

    instance.reconcile_required()

    lane, service, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    result = lane.step(
        now=time.time()
    )

    assert result["reason"] == (
        "capital_recovery_quote_reserve"
    )

    assert service.assessed_symbols == []
    assert fake.created == []

    recovery = (
        instance.health()
        ["capital_recovery"]
    )

    assert recovery["quote_starved"] is True
    assert recovery["fast_quote_reserve_usd"] >= 6.0
    assert recovery["slower_lane_surplus_usd"] == 0.0
    assert recovery["micro_lane_capital_priority"] is True
    assert recovery["slower_lanes_use_surplus_only"] is True
    assert recovery["live_authority"] is False


def test_tradeable_dust_is_reactivated_then_recycled(
    tmp_path,
):
    fake = PriceGuardBybit()

    instance = _instance(
        tmp_path,
        fake,
    )

    seed(
        instance,
        fake,
        0.05,
        4.0,
    )

    fake.bid = 80.0
    fake.ask = 81.0

    preparation = instance.prepare_sell(
        "BTC/USDT",
        0.05,
        80.0,
    )

    assert preparation["status"] == "dust"

    fake.bid = 120.0
    fake.ask = 121.0
    fake.sell_limit = 90.0

    _quote_starve(
        instance,
        fake,
        btc=0.05,
    )

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
    )

    result = _capital_recovery_once(
        lane,
        now=time.time(),
    )

    assert (
        "BTC/USDT"
        in result["reactivated_dust"]
    )

    assert (
        result["result"]
        ["status"]
        == "closed"
    )

    recovery = (
        instance.health()
        ["capital_recovery"]
    )

    assert recovery["dust_reactivated"] == 1

    health = instance.health()

    assert (
        "BTC/USDT"
        not in health[
            "non_tradeable_dust"
        ]
    )

    assert (
        "BTC/USDT"
        not in health["positions"]
    )
