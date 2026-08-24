from __future__ import annotations

import pytest

from leantrader.production.velocity_sniper_testnet import (
    VelocitySniperTestnetLane,
)


class FakeTestnet:
    def safe_snapshot(self):
        return {
            "positions": {},
            "open_orders": 0,
            "last_reconciliation_errors": [],
            "kill_switch_active": False,
        }


def lane(tmp_path, *, maximum_order_usd=10.0):
    return VelocitySniperTestnetLane(
        service_provider=lambda: None,
        testnet=FakeTestnet(),
        state_path=tmp_path / "v1591.json",
        supervisory_provider=lambda: {},
        order_usd=1.0,
        round_trip_cost_bps=30.0,
        cadence_seconds=0.5,
        maximum_hold_seconds=30.0,
        maximum_entries_per_day=45,
        bootstrap_after_seconds=5.0,
        maximum_concurrent_positions=6,
        maximum_entries_per_cycle=3,
        reentry_cooldown_seconds=2.0,
        starting_equity=50.0,
        maximum_order_usd=maximum_order_usd,
    )


def growth(remaining):
    return {
        "state": "normal",
        "equity": 50.0 + max(0.0, remaining - 15.0),
        "peak_equity": 50.0 + max(0.0, remaining - 15.0),
        "protected_principal": 35.0,
        "locked_profit": 0.0,
        "reinvestable_realized_profit": max(
            0.0,
            remaining - 15.0,
        ) * 0.5,
        "remaining_deployable_notional": remaining,
        "risk_multiplier": 1.0,
        "new_entries_allowed": True,
    }


def test_compounding_can_scale_above_old_two_dollar_ceiling(
    tmp_path,
):
    l = lane(tmp_path)

    result = l._compound_order_notional(
        {"capital_growth": growth(30.0)},
        slots=6,
    )

    assert result["allowed"] is True
    assert result["compounding"] is True
    assert result["order_notional_usd"] == pytest.approx(5.0)
    assert result["order_notional_usd"] > 2.0


def test_compounding_never_exceeds_local_testnet_ceiling(
    tmp_path,
):
    l = lane(
        tmp_path,
        maximum_order_usd=10.0,
    )

    result = l._compound_order_notional(
        {"capital_growth": growth(100.0)},
        slots=1,
    )

    assert result["order_notional_usd"] == pytest.approx(10.0)


def test_legacy_closed_trade_gets_dollar_pnl_backfill(
    tmp_path,
):
    l = lane(tmp_path)

    row = {
        "quantity": 2.0,
        "entry_price": 5.0,
        "net_bps_after_model": -30.0,
    }

    assert l._closed_entry_notional_usd(row) == pytest.approx(10.0)

    assert l._closed_modeled_net_pnl_usd(
        row
    ) == pytest.approx(-0.03)


def test_explicit_new_trade_pnl_remains_authoritative(
    tmp_path,
):
    l = lane(tmp_path)

    row = {
        "quantity": 2.0,
        "entry_price": 5.0,
        "net_bps_after_model": -30.0,
        "modeled_net_pnl_usd": -0.031,
    }

    assert l._closed_modeled_net_pnl_usd(
        row
    ) == pytest.approx(-0.031)
