from __future__ import annotations

from leantrader.production.testnet_exit_recycle import (
    MODELED_ROUND_TRIP_COST_FLOOR_BPS,
)
from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
)
from tests.test_testnet_execution import buy_event, engine


class ThinAskBybit(PriceGuardBybit):
    def fetch_order_book(self, symbol, limit=5):
        self.calls.append(
            ("fetch_order_book", symbol, limit)
        )
        return {
            "bids": [[self.bid, 1000.0]],
            "asks": [[self.ask, 0.01]],
        }


class ZeroFillBuyBybit(PriceGuardBybit):
    def create_order(
        self,
        symbol,
        order_type,
        side,
        amount,
        price,
        params,
    ):
        self.calls.append(
            (
                "create_order",
                symbol,
                order_type,
                side,
                amount,
                price,
                params,
            )
        )

        client_id = params["orderLinkId"]

        observed = {
            "id": f"exchange-{len(self.created) + 1}",
            "clientOrderId": client_id,
            "symbol": symbol,
            "side": side,
            "status": "canceled",
            "filled": 0.0,
            "average": None,
            "fee": {"cost": 0.0},
            "info": {
                "orderLinkId": client_id,
                "rejectReason": (
                    "EC_NoImmediateQtyToFill"
                ),
                "cancelType": "UNKNOWN",
            },
        }

        self.created.append(observed)
        self.orders[observed["id"]] = observed
        return observed


def make_buy(
    event_id: str,
    *,
    timestamp: str = "2026-08-26T10:30:00+00:00",
):
    return {
        **buy_event(),
        "timestamp": timestamp,
        "event_id": event_id,
    }


def test_safe_minimum_over_order_cap_blocks_without_budget_use(
    tmp_path,
):
    fake = PriceGuardBybit()

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=5.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )

    instance.start()

    result = instance.mirror_events(
        [make_buy("cap-block")]
    )[0]

    assert result["status"] == "skipped"
    assert result["skip_reason"] == (
        "entry_round_trip:"
        "safe_minimum_exceeds_order_cap"
    )
    assert fake.created == []

    health = instance.health()

    assert health["daily_entry_order_count"] == 0
    assert health["daily_total_order_count"] == 0
    assert (
        health["entry_round_trip_guard"]
        ["preflight_blocks"]
        == 1
    )
    assert health["live_authority"] is False


def test_insufficient_immediate_ask_liquidity_blocks_before_order(
    tmp_path,
):
    fake = ThinAskBybit()

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )

    instance.start()

    result = instance.mirror_events(
        [make_buy("thin-book")]
    )[0]

    assert result["status"] == "skipped"
    assert result["skip_reason"] == (
        "entry_round_trip:"
        "insufficient_immediate_ask_liquidity"
    )
    assert fake.created == []
    assert instance.health()[
        "daily_entry_order_count"
    ] == 0


def test_current_sell_limit_blocks_prospective_entry(
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

    result = instance.mirror_events(
        [make_buy("sell-limit")]
    )[0]

    assert result["status"] == "skipped"
    assert result["skip_reason"] == (
        "entry_round_trip:"
        "prospective_exit_price_limit_unexecutable"
    )
    assert fake.created == []


def test_stressed_exit_must_remain_above_exchange_minimum(
    tmp_path,
):
    fake = PriceGuardBybit()

    # Entry reference remains 100, but the currently executable
    # bid has already deteriorated materially.
    fake.bid = 90.0
    fake.ask = 100.0
    fake.sell_limit = 80.0

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )

    instance.start()

    result = instance.mirror_events(
        [make_buy("stress-dust")]
    )[0]

    assert result["status"] == "skipped"
    assert result["skip_reason"] == (
        "entry_round_trip:"
        "prospective_position_not_sellable_under_stress"
    )
    assert fake.created == []


def test_round_trip_executable_buy_preserves_market_order_path(
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

    result = instance.mirror_events(
        [make_buy("allowed")]
    )[0]

    assert result["status"] == "closed"
    assert result["filled"] > 0.0
    assert len(fake.created) == 1
    assert fake.created[0]["side"] == "buy"

    health = instance.health()

    assert health["daily_entry_order_count"] == 1
    assert (
        health["entry_round_trip_guard"]
        ["modeled_round_trip_cost_floor_bps"]
        >= MODELED_ROUND_TRIP_COST_FLOOR_BPS
        >= 30.0
    )
    assert health["live_authority"] is False


def test_terminal_zero_fill_buy_quarantines_symbol(
    tmp_path,
):
    fake = ZeroFillBuyBybit()

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )

    instance.start()

    first = instance.mirror_events(
        [
            make_buy(
                "zero-1",
                timestamp=(
                    "2026-08-26T10:31:00+00:00"
                ),
            )
        ]
    )[0]

    second = instance.mirror_events(
        [
            make_buy(
                "zero-2",
                timestamp=(
                    "2026-08-26T10:31:01+00:00"
                ),
            )
        ]
    )[0]

    assert first["status"] == "canceled"

    assert second["status"] == "skipped"
    assert second["skip_reason"] == (
        "entry_round_trip:entry_cooldown"
    )

    assert len(fake.created) == 1

    health = instance.health()
    guard = health["entry_round_trip_guard"]

    assert guard["terminal_zero_fill_buys"] == 1
    assert guard["cooldown_skips"] == 1

    # Only the first real Bybit submission consumes budget.
    assert health["daily_entry_order_count"] == 1
    assert health["daily_total_order_count"] == 1


def test_late_authoritative_zero_fill_resolution_also_quarantines(
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

    record = {
        "client_order_id": "lt-authoritative-test",
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "submitting",
        "filled": 0.0,
        "applied_filled": 0.0,
        "filled_cost": 0.0,
        "applied_fill_cost": 0.0,
        "average": None,
        "fee": 0.0,
        "fee_currency": None,
        "applied_fee": 0.0,
        "fill_counted": False,
        "reference_price": 100.0,
        "reconciliation_resolution": (
            "native_bybit_authoritative_absence"
        ),
    }

    instance.state["orders"][
        "lt-authoritative-test"
    ] = record

    instance._merge_observed(
        record,
        {
            "id": None,
            "clientOrderId": (
                "lt-authoritative-test"
            ),
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "rejected",
            "filled": 0.0,
            "cost": 0.0,
            "info": {
                "orderLinkId": (
                    "lt-authoritative-test"
                ),
            },
        },
    )

    guard = instance.health()[
        "entry_round_trip_guard"
    ]

    assert guard["terminal_zero_fill_buys"] == 1
    assert (
        guard["last_terminal_zero_fill_buy"]
        ["detail"]
        ["reconciliation_resolution"]
        == "native_bybit_authoritative_absence"
    )
    assert (
        guard["blocked_until"]["BTC/USDT"]
        > 0.0
    )
    assert instance.health()["live_authority"] is False


def test_non_exchange_testnet_adapter_preserves_legacy_lane_path(
    tmp_path,
):
    from tests.test_production_testnet_exit_recycle_v1608 import (
        hyper_lane,
    )

    lane, service, testnet = hyper_lane(tmp_path)

    assert not hasattr(testnet, "exchange")

    result = lane.step(now=1_000.0)

    assert result["reason"] == "fast_multi_route_cycle"
    assert service.assessed_symbols == ["AAA/USDT"]
    assert "AAA/USDT" in testnet.positions
    assert lane.health()["live_authority"] is False
