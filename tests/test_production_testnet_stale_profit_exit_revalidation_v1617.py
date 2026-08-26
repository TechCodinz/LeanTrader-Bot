from __future__ import annotations

from tests.test_production_testnet_price_limit_edge_exit_v1615 import (
    _runtime,
)


def test_stale_take_profit_is_retired_without_sell(
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
        sell_limit=85.0,
    )

    fake.bid = 90.0
    fake.ask = 91.0

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
        "stale_profit_exit_retired_for_reassessment"
    )

    assert fake.created == []

    assert (
        "BTC/USDT"
        in lane._active_snapshot()
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

    details = (
        result["details"]
    )

    assert (
        details["fresh_gross_bps"]
        < details[
            "minimum_valid_gross_bps"
        ]
    )

    assert (
        details[
            "order_submitted"
        ]
        is False
    )

    health = lane.health()

    guard = health[
        "stale_profit_exit_revalidation"
    ]

    assert (
        guard["retirements"]
        == 1
    )

    assert (
        guard[
            "stale_profit_order_submission_allowed"
        ]
        is False
    )

    assert (
        health["live_authority"]
        is False
    )


def test_protective_stop_loss_still_releases_when_boundary_clear(
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
        sell_limit=85.0,
    )

    fake.bid = 90.0
    fake.ask = 91.0

    with lane._lock:
        queue = lane.state[
            "deferred_exit_recoveries"
        ]["BTC/USDT"]

        queue[
            "assessment"
        ][
            "exit_reason"
        ] = "velocity_stop_loss"

        queue[
            "source_event"
        ][
            "reason"
        ] = (
            "fast_collective_testnet_exit:"
            "velocity_stop_loss"
        )

        lane._save_locked()

    snapshot = (
        instance.safe_snapshot()
    )

    record = (
        lane._active_snapshot()
        ["BTC/USDT"]
    )

    lane._manage_active(
        service,
        snapshot,
        "BTC/USDT",
        record,
        now=now,
    )

    assert len(fake.created) == 1
    assert (
        fake.created[0]["side"]
        == "sell"
    )

    assert (
        "BTC/USDT"
        not in instance.health()[
            "positions"
        ]
    )

    assert (
        lane.health()[
            "stale_profit_exit_revalidation"
        ][
            "protective_exit_reasons_preserved"
        ]
        is True
    )
