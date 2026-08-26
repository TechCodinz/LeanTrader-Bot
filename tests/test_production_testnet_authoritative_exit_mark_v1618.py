from __future__ import annotations

from tests.test_fast_collective_testnet import (
    signal,
)
from tests.test_production_testnet_price_limit_edge_exit_v1615 import (
    _runtime,
)


class ExitService:
    def collective_signal(
        self,
        symbol,
    ):
        assert (
            symbol
            == "BTC/USDT"
        )

        # Deliberately stale/internal mark.
        # v1.60.18 must replace midpoint=100
        # with the fresh Testnet bid.
        return signal()


def test_fresh_bybit_bid_becomes_authoritative_exit_mark_after_stale_profit_retirement(
    tmp_path,
):
    (
        lane,
        _service,
        instance,
        fake,
        now,
    ) = _runtime(
        tmp_path,
        sell_limit=95.0,
    )

    service = ExitService()

    # Internal micro midpoint remains 100,
    # but Testnet can currently sell only
    # around 90.
    fake.bid = 90.0
    fake.ask = 91.0

    snapshot = (
        instance.safe_snapshot()
    )

    record = (
        lane._active_snapshot()
        ["BTC/USDT"]
    )

    first = lane._manage_active(
        service,
        snapshot,
        "BTC/USDT",
        record,
        now=now,
    )

    assert first["reason"] == (
        "stale_profit_exit_retired_for_reassessment"
    )

    assert (
        lane.state.get(
            "v1617_stale_profit_exit_retirements"
        )
        == 1
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

    second = lane._manage_active(
        service,
        instance.safe_snapshot(),
        "BTC/USDT",
        lane._active_snapshot()[
            "BTC/USDT"
        ],
        now=now + 1.1,
    )

    active = (
        lane._active_snapshot()
        ["BTC/USDT"]
    )

    sentinel = (
        active.get(
            "last_sentinel"
        )
        or {}
    )

    assert (
        sentinel["price"]
        == 90.0
    )

    assert (
        sentinel["reason"]
        == "velocity_stop_loss"
    )

    # The sell boundary is still blocked,
    # therefore no exchange order is sent.
    assert fake.created == []

    queue = (
        lane.state.get(
            "deferred_exit_recoveries"
        )
        or {}
    )["BTC/USDT"]

    assert (
        queue[
            "assessment"
        ][
            "exit_reason"
        ]
        == "velocity_stop_loss"
    )

    assert second["reason"] in {
        "exit_recovery_deferred_nonblocking",
        "exit_waiting_for_executable_balance",
        "exit_recycle_cooldown",
    }

    # Next cycle sees the protective queue.
    # It must not regenerate or retire another
    # stale take-profit.
    lane._manage_active(
        service,
        instance.safe_snapshot(),
        "BTC/USDT",
        lane._active_snapshot()[
            "BTC/USDT"
        ],
        now=now + 2.2,
    )

    assert (
        lane.state.get(
            "v1617_stale_profit_exit_retirements"
        )
        == 1
    )

    queue = (
        lane.state.get(
            "deferred_exit_recoveries"
        )
        or {}
    )["BTC/USDT"]

    assert (
        queue[
            "assessment"
        ][
            "exit_reason"
        ]
        == "velocity_stop_loss"
    )

    health = lane.health()

    guard = health[
        "testnet_authoritative_exit_mark"
    ]

    last = guard[
        "last_mark"
    ]

    assert (
        guard[
            "active_exit_internal_midpoint_authority"
        ]
        is False
    )

    assert (
        last["internal_midpoint"]
        == 100.0
    )

    assert (
        last["fresh_bid"]
        == 90.0
    )

    assert (
        last["sentinel_price"]
        == 90.0
    )

    assert (
        last["sentinel_reason"]
        == "velocity_stop_loss"
    )

    assert (
        health["live_authority"]
        is False
    )
