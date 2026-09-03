from __future__ import annotations

from leantrader.production.fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)
from leantrader.production.testnet_execution import (
    BybitTestnetExecutionEngine,
)


def _lane():
    lane = HyperSpeedCollectiveTestnetLane.__new__(
        HyperSpeedCollectiveTestnetLane
    )
    lane.maximum_order_usd = 5.0
    lane.order_usd = 1.0
    lane.maximum_adaptive_entries_per_cycle = 8
    lane._fast_open_notional = lambda: 0.0
    return lane


def _growth(remaining=100.0):
    return {
        "capital_growth": {
            "new_entries_allowed": True,
            "risk_multiplier": 1.0,
            "remaining_deployable_notional": remaining,
            "state": "growth",
            "equity": 50.0,
            "peak_equity": 50.0,
            "protected_principal": 35.0,
            "locked_profit": 0.0,
            "reinvestable_realized_profit": 15.0,
        }
    }


def test_authenticated_free_quote_caps_order_budget():
    lane = _lane()

    sizing = lane._compound_order_notional(
        _growth(remaining=100.0),
        snapshot={
            "account_balance": {
                "free": {"USDT": 20.0},
            }
        },
        entries=4,
    )

    assert sizing["allowed"] is True
    assert sizing["capital_authority"] == (
        "governance_and_authenticated_quote"
    )
    assert 4.9 < sizing["order_notional_usd"] <= 5.0
    assert sizing["available_pool"] < 20.0


def test_paper_governance_can_reduce_but_not_invent_quote():
    lane = _lane()

    sizing = lane._compound_order_notional(
        _growth(remaining=8.0),
        snapshot={
            "account_balance": {
                "free": {"USDT": 20.0},
            }
        },
        entries=4,
    )

    assert sizing["allowed"] is True
    assert sizing["available_pool"] == 8.0
    assert sizing["order_notional_usd"] == 2.0


def test_missing_authenticated_free_quote_fails_closed():
    lane = _lane()

    sizing = lane._compound_order_notional(
        _growth(),
        snapshot={
            "authenticated": True,
            "account_balance": {},
        },
        entries=3,
    )

    assert sizing["allowed"] is False
    assert sizing["reason"] == (
        "authenticated_quote_balance_unavailable"
    )


def test_balance_snapshot_preserves_free_and_used():
    engine = BybitTestnetExecutionEngine.__new__(
        BybitTestnetExecutionEngine
    )
    engine.state = {
        "positions": {},
        "account_balance": {},
    }

    engine._update_balance_snapshot(
        {
            "total": {"USDT": 20.0},
            "free": {"USDT": 17.5},
            "used": {"USDT": 2.5},
        }
    )

    row = engine.state["account_balance"]

    assert row["assets"]["USDT"] == 20.0
    assert row["free"]["USDT"] == 17.5
    assert row["used"]["USDT"] == 2.5
