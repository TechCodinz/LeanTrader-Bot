from __future__ import annotations

from tests.test_production_testnet_exit_recycle_v1608 import (
    LaneTestnet,
    hyper_lane,
)


def test_large_persisted_exit_recovery_isolated_without_new_order(
    tmp_path,
):
    testnet = LaneTestnet()
    testnet.positions["AAA/USDT"] = 1.0

    lane, _service, _ = hyper_lane(
        tmp_path,
        testnet=testnet,
    )

    with lane._lock:
        lane.state["active"] = {
            "AAA/USDT": {
                "symbol": "AAA/USDT",
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_price": 100.0,
                "entry_notional_usd": 100.0,
                "entered_at": 900.0,
                "entry_event_id": "entry-old",
            }
        }
        lane._save_locked()

    source_event = lane._new_event(
        symbol="AAA/USDT",
        side="sell",
        quantity=1.0,
        price=101.0,
        reason="fast_collective_testnet_exit:time",
        now=1_000.0,
        remaining_quantity=0.0,
    )

    pending = {
        "kind": "exit_recovery",
        "source_event": source_event,
        "assessment": {},
        "recovery_attempt": 111,
        "created_at": 1_000.0,
    }

    lane._set_pending(pending)

    before = len(testnet.events)

    result = lane._submit_pending(
        pending,
        now=1_000.0,
    )

    assert (
        result["reason"]
        == "exit_recovery_deferred_nonblocking"
    )

    assert lane._pending() is None
    assert len(testnet.events) == before

    health = lane.health()

    # This is a behavioral regression, not a permanent
    # assertion about the latest aggregate health version.
    assert (
        health["exit_recovery_isolation"]["enabled"]
        is True
    )
    assert (
        health[
            "deferred_exit_recovery_count"
        ]
        == 1
    )

    assert (
        health[
            "deferred_exit_recoveries"
        ][0]["recovery_attempt"]
        == 111
    )

    assert (
        health[
            "exit_recovery_isolation"
        ][
            "global_pending_slot_released_after_terminal_failure"
        ]
        is True
    )


def test_actual_negative_testnet_pnl_never_reports_compounding(
    tmp_path,
):
    testnet = LaneTestnet()
    testnet.realized_pnl_usd = -0.01

    lane, _service, _ = hyper_lane(
        tmp_path,
        testnet=testnet,
        order_usd=1.0,
    )

    with lane._lock:
        lane.state["last_sizing"] = {
            "compounding": True,
            "order_notional_usd": 2.0,
        }
        lane._save_locked()

    health = lane.health()

    assert (
        health[
            "actual_testnet_profit_compounding_eligible"
        ]
        is False
    )

    assert (
        health[
            "principal_protected_compounding"
        ]
        is False
    )
