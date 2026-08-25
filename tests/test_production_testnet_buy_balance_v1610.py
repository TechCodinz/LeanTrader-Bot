from __future__ import annotations

from tests.test_production_testnet_exit_recycle_v1608 import (
    BalanceBybit,
)
from tests.test_testnet_execution import (
    buy_event,
    engine,
)


def test_buy_preflight_uses_fresh_free_quote_balance_without_consuming_budget(
    tmp_path,
):
    fake = BalanceBybit()

    fake.balance_total["USDT"] = 4.0
    fake.balance_free["USDT"] = 4.0

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    result = instance.mirror_events(
        [buy_event()]
    )[0]

    assert result["status"] == "skipped"
    assert (
        result["skip_reason"]
        == "insufficient_free_quote_balance_preflight"
    )

    assert fake.created == []

    health = instance.health()

    assert (
        health["daily_entry_order_count"]
        == 0
    )

    assert (
        health[
            "daily_entry_submitted_usd"
        ]
        == 0.0
    )

    assert (
        health[
            "buy_balance_guard"
        ]["preflight_skips"]
        == 1
    )

    assert (
        health[
            "buy_balance_guard"
        ]["live_authority"]
        is False
    )


class RejectingBalanceBybit(
    BalanceBybit
):
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

        raise RuntimeError(
            'bybit {"retCode":170131,'
            '"retMsg":"Insufficient balance."}'
        )


def test_definitive_bybit_insufficient_balance_is_terminal_and_budget_is_restored(
    tmp_path,
):
    fake = RejectingBalanceBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    result = instance.mirror_events(
        [buy_event()]
    )[0]

    assert result["status"] == "rejected"
    assert (
        result["skip_reason"]
        == "insufficient_free_quote_balance"
    )

    health = instance.health()

    assert (
        health["daily_entry_order_count"]
        == 0
    )

    assert (
        health[
            "daily_entry_submitted_usd"
        ]
        == 0.0
    )

    assert (
        health[
            "daily_total_order_count"
        ]
        == 0
    )

    quality = health[
        "current_day_execution_quality"
    ]

    assert (
        quality[
            "buy_entries_submitted"
        ]
        == 0
    )

    assert (
        health[
            "buy_balance_guard"
        ][
            "definitive_insufficient_balance_rejections"
        ]
        == 1
    )

    record = next(
        iter(
            instance.state[
                "orders"
            ].values()
        )
    )

    assert record[
        "submitted_at"
    ] is None

    assert (
        record[
            "v1610_budget_rollback"
        ]
        is True
    )

    assert (
        record[
            "exchange_reject_code"
        ]
        == 170131
    )


def test_pre_v1610_submitting_balance_rejection_can_be_safely_migrated(
    tmp_path,
):
    fake = BalanceBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    event = {
        **buy_event(),
        "event_id": "pre-v1610-balance-reject",
        "timestamp": "2026-08-25T14:34:49+00:00",
    }

    client_id = instance._client_order_id(
        event
    )

    instance.state["orders"][
        client_id
    ] = {
        "client_order_id": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.05,
        "submitted_usd": 5.0,
        "reference_price": 100.0,
        "submitted_at": "2026-08-25T14:34:49+00:00",
        "status": "submitting",
        "order_id": None,
        "filled": 0.0,
        "average": None,
        "fee": 0.0,
    }

    instance.state[
        "daily_order_count"
    ] = 1

    instance.state[
        "daily_submitted_usd"
    ] = 5.0

    instance.state[
        "daily_entry_order_count"
    ] = 1

    instance.state[
        "daily_entry_submitted_usd"
    ] = 5.0

    instance._save_state()

    result = (
        instance.resolve_definitive_insufficient_balance(
            event
        )
    )

    assert result is not None
    assert result["status"] == "rejected"

    assert (
        result["skip_reason"]
        == "insufficient_free_quote_balance"
    )

    health = instance.health()

    assert (
        health[
            "daily_total_order_count"
        ]
        == 0
    )

    assert (
        health[
            "daily_entry_order_count"
        ]
        == 0
    )

    assert (
        health[
            "daily_entry_submitted_usd"
        ]
        == 0.0
    )

    record = (
        instance.state["orders"][
            client_id
        ]
    )

    assert (
        record["submitted_at"]
        is None
    )

    assert (
        record[
            "v1610_budget_rollback"
        ]
        is True
    )


def test_pre_v1610_migration_never_terminalizes_exchange_accepted_order(
    tmp_path,
):
    fake = BalanceBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    event = {
        **buy_event(),
        "event_id": "accepted-order-must-survive",
        "timestamp": "2026-08-25T14:35:00+00:00",
    }

    client_id = instance._client_order_id(
        event
    )

    instance.state["orders"][
        client_id
    ] = {
        "client_order_id": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.05,
        "submitted_usd": 5.0,
        "reference_price": 100.0,
        "submitted_at": "2026-08-25T14:35:00+00:00",
        "status": "submitting",
        "order_id": "exchange-accepted-1",
        "filled": 0.0,
        "average": None,
        "fee": 0.0,
    }

    instance.state[
        "daily_order_count"
    ] = 1

    instance.state[
        "daily_submitted_usd"
    ] = 5.0

    instance.state[
        "daily_entry_order_count"
    ] = 1

    instance.state[
        "daily_entry_submitted_usd"
    ] = 5.0

    instance._save_state()

    result = (
        instance.resolve_definitive_insufficient_balance(
            event
        )
    )

    assert result is None

    record = (
        instance.state["orders"][
            client_id
        ]
    )

    assert (
        record["status"]
        == "submitting"
    )

    assert (
        record["order_id"]
        == "exchange-accepted-1"
    )

    assert (
        instance.state[
            "daily_entry_order_count"
        ]
        == 1
    )

    assert (
        instance.state[
            "daily_entry_submitted_usd"
        ]
        == 5.0
    )
