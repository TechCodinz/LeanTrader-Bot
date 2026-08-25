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
