from __future__ import annotations

import threading

import pytest

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)
from leantrader.production.runner import (
    PaperRunner,
)
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


def make_lane(tmp_path):
    return VelocitySniperTestnetLane(
        service_provider=lambda: None,
        testnet=FakeTestnet(),
        state_path=tmp_path / "v159.json",
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
        maximum_order_usd=5.0,
    )


def growth(
    *,
    remaining=15.0,
    risk=1.0,
    allowed=True,
    equity=50.0,
):
    return {
        "state": "normal",
        "equity": equity,
        "peak_equity": max(50.0, equity),
        "protected_principal": 35.0,
        "locked_profit": 0.0,
        "reinvestable_realized_profit": max(
            0.0,
            equity - 50.0,
        ) * 0.5,
        "remaining_deployable_notional": remaining,
        "risk_multiplier": risk,
        "new_entries_allowed": allowed,
    }


def test_precision_symbols_are_explicit_mtf_context():
    service = object.__new__(
        ReadOnlySwarmService
    )
    service._lock = threading.RLock()
    service._microstream_symbols = [
        "fast/usdt",
        "xrp/usdt",
    ]

    assert (
        service.precision_context_symbols()
        == {
            "FAST/USDT",
            "XRP/USDT",
        }
    )


def test_compound_budget_uses_deployable_capital(tmp_path):
    lane = make_lane(tmp_path)

    sizing = lane._compound_order_notional(
        {
            "capital_growth": growth(
                remaining=15.0
            )
        },
        slots=6,
    )

    assert sizing["allowed"] is True
    assert sizing["compounding"] is True
    assert sizing["order_notional_usd"] == pytest.approx(
        2.5
    )
    assert sizing["martingale"] is False
    assert sizing["live_authority"] is False


def test_compound_budget_increases_with_capital(tmp_path):
    lane = make_lane(tmp_path)

    base = lane._compound_order_notional(
        {
            "capital_growth": growth(
                remaining=15.0
            )
        },
        slots=6,
    )

    grown = lane._compound_order_notional(
        {
            "capital_growth": growth(
                remaining=18.0,
                equity=56.0,
            )
        },
        slots=6,
    )

    assert (
        grown["order_notional_usd"]
        > base["order_notional_usd"]
    )


def test_drawdown_risk_multiplier_reduces_size(tmp_path):
    lane = make_lane(tmp_path)

    normal = lane._compound_order_notional(
        {
            "capital_growth": growth(
                remaining=15.0,
                risk=1.0,
            )
        },
        slots=6,
    )

    defensive = lane._compound_order_notional(
        {
            "capital_growth": growth(
                remaining=15.0,
                risk=0.5,
            )
        },
        slots=6,
    )

    assert (
        defensive["order_notional_usd"]
        < normal["order_notional_usd"]
    )


def test_capital_governor_can_block_fast_entries(tmp_path):
    lane = make_lane(tmp_path)

    sizing = lane._compound_order_notional(
        {
            "capital_growth": growth(
                allowed=False
            )
        },
        slots=6,
    )

    assert sizing["allowed"] is False
    assert (
        sizing["reason"]
        == "capital_growth_new_entries_blocked"
    )


def test_supervisor_carries_capital_snapshot():
    status = {
        "timestamp": 100.0,
        "healthy": True,
        "halt_reason": None,
        "equity": 51.0,
        "realized_pnl": 1.0,
        "engines": {},
        "decisions": {},
        "collective_profit_fabric": {
            "symbols": {},
        },
        "open_positions": [],
        "capital_growth": growth(
            remaining=16.0,
            equity=51.0,
        ),
    }

    supervisor = (
        PaperRunner._extract_fast_supervisory(
            status
        )
    )

    assert supervisor["equity"] == 51.0
    assert supervisor["realized_pnl"] == 1.0
    assert (
        supervisor["capital_growth"][
            "remaining_deployable_notional"
        ]
        == 16.0
    )
    assert supervisor["live_authority"] is False


def test_health_exposes_compounding_metrics(tmp_path):
    lane = make_lane(tmp_path)

    with lane._lock:
        lane.state["closed"] = [
            {
                "net_bps_after_model": 20.0,
                "modeled_net_pnl_usd": 0.01,
                "exited_at": 9999999999.0,
            },
            {
                "net_bps_after_model": -10.0,
                "modeled_net_pnl_usd": -0.005,
                "exited_at": 9999999999.0,
            },
        ]

    health = lane.health()

    assert (
        health[
            "principal_protected_compounding"
        ]
        is True
    )
    assert health["win_rate"] == 0.5
    assert health["average_net_bps"] == 5.0
    assert health["profit_factor"] == 2.0
    assert health["live_authority"] is False
