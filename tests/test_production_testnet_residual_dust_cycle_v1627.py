from __future__ import annotations

import copy

import pytest

from leantrader.production.testnet_residual_dust_cycle_v1627 import (
    _current_cycle_evidence,
)

from tests.test_production_testnet_exit_recycle_v1608 import (
    BalanceBybit,
)

from tests.test_production_testnet_price_limit_edge_exit_v1615 import (
    _runtime,
)

from tests.test_testnet_execution import (
    engine,
)


def _seed_residual_cycle(
    instance,
    fake,
    *,
    realized_sell_pnl: float,
    dust_cost_basis: float = 0.002,
):
    with instance._io_lock:
        instance.state[
            "positions"
        ]["BTC/USDT"] = 0.00002

        instance.state[
            "position_cost_usd"
        ]["BTC/USDT"] = (
            dust_cost_basis
        )

        instance.state[
            "position_cycle_pnl_usd"
        ]["BTC/USDT"] = (
            realized_sell_pnl
        )

        instance.state[
            "realized_pnl_usd"
        ] = realized_sell_pnl

        instance.state[
            "orders"
        ]["cycle-buy"] = {
            "client_order_id": (
                "cycle-buy"
            ),
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "closed",
            "submitted_at": (
                "2026-08-27T10:00:00+00:00"
            ),
            "filled": 0.1001,
            "filled_cost": 10.0,
            "average": 99.9001,
            "fee": 0.0001,
            "fee_currency": "BTC",
        }

        instance.state[
            "orders"
        ]["cycle-sell"] = {
            "client_order_id": (
                "cycle-sell"
            ),
            "symbol": "BTC/USDT",
            "side": "sell",
            "status": "closed",
            "submitted_at": (
                "2026-08-27T10:00:10+00:00"
            ),
            "filled": 0.1,
            "filled_cost": 9.999,
            "average": 99.99,
            "fee": 0.009999,
            "fee_currency": "USDT",
        }

        instance._save_state()

    fake.balance_total[
        "BTC"
    ] = 0.00002

    fake.balance_free[
        "BTC"
    ] = 0.00002


def test_small_realized_profit_larger_dust_is_loss(
    tmp_path,
):
    fake = BalanceBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    _seed_residual_cycle(
        instance,
        fake,
        realized_sell_pnl=0.001,
        dust_cost_basis=0.002,
    )

    closed_before = int(
        instance.state.get(
            "closed_positions"
        )
        or 0
    )

    wins_before = int(
        instance.state.get(
            "winning_positions"
        )
        or 0
    )

    result = instance.prepare_sell(
        "BTC/USDT",
        0.00002,
        100.0,
    )

    assert (
        result["status"]
        == "dust"
    )

    assert (
        result[
            "completed_executable_cycle"
        ]
        is True
    )

    assert (
        result[
            "actual_realized_sell_pnl_usd"
        ]
        == pytest.approx(
            0.001
        )
    )

    assert (
        result[
            "residual_dust_cost_basis_usd"
        ]
        == pytest.approx(
            0.002
        )
    )

    assert (
        result[
            "actual_cycle_net_after_dust_usd"
        ]
        == pytest.approx(
            -0.001
        )
    )

    assert (
        result[
            "winning_after_dust"
        ]
        is False
    )

    assert (
        instance.state[
            "closed_positions"
        ]
        == closed_before + 1
    )

    assert (
        instance.state[
            "winning_positions"
        ]
        == wins_before
    )

    # Dust stays separate from exchange-realized PnL.
    assert (
        instance.state[
            "realized_pnl_usd"
        ]
        == pytest.approx(
            0.001
        )
    )


def test_positive_after_dust_counts_win_once(
    tmp_path,
):
    fake = BalanceBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    _seed_residual_cycle(
        instance,
        fake,
        realized_sell_pnl=0.003,
        dust_cost_basis=0.002,
    )

    closed_before = int(
        instance.state.get(
            "closed_positions"
        )
        or 0
    )

    wins_before = int(
        instance.state.get(
            "winning_positions"
        )
        or 0
    )

    first = instance.prepare_sell(
        "BTC/USDT",
        0.00002,
        100.0,
    )

    assert (
        first[
            "actual_cycle_net_after_dust_usd"
        ]
        == pytest.approx(
            0.001
        )
    )

    assert (
        first[
            "winning_after_dust"
        ]
        is True
    )

    assert (
        instance.state[
            "closed_positions"
        ]
        == closed_before + 1
    )

    assert (
        instance.state[
            "winning_positions"
        ]
        == wins_before + 1
    )

    # Position was already moved to non-tradeable dust.
    # Running preparation again cannot count the cycle twice.
    second = instance.prepare_sell(
        "BTC/USDT",
        0.00002,
        100.0,
    )

    assert (
        second["status"]
        == "absent"
    )

    assert (
        instance.state[
            "closed_positions"
        ]
        == closed_before + 1
    )

    assert (
        instance.state[
            "winning_positions"
        ]
        == wins_before + 1
    )

    assert (
        len(
            instance.state.get(
                "v1627_completed_cycle_keys"
            )
            or []
        )
        == 1
    )


def test_cycle_evidence_aggregates_consecutive_scale_in_buys():
    state = {
        "orders": {
            "buy-a": {
                "symbol": "AAA/USDT",
                "side": "buy",
                "status": "closed",
                "submitted_at": (
                    "2026-08-27T10:00:00+00:00"
                ),
                "filled": 1.0,
                "filled_cost": 10.0,
                "average": 10.0,
                "fee": 0.0,
                "fee_currency": "AAA",
            },
            "buy-b": {
                "symbol": "AAA/USDT",
                "side": "buy",
                "status": "closed",
                "submitted_at": (
                    "2026-08-27T10:00:01+00:00"
                ),
                "filled": 1.0,
                "filled_cost": 11.0,
                "average": 11.0,
                "fee": 0.0,
                "fee_currency": "AAA",
            },
            "sell-a": {
                "symbol": "AAA/USDT",
                "side": "sell",
                "status": "closed",
                "submitted_at": (
                    "2026-08-27T10:00:10+00:00"
                ),
                "filled": 1.9,
                "filled_cost": 20.9,
                "average": 11.0,
                "fee": 0.0,
                "fee_currency": "USDT",
            },
        }
    }

    row = (
        _current_cycle_evidence(
            state,
            "AAA/USDT",
        )
    )

    assert (
        row[
            "buy_order_count"
        ]
        == 2
    )

    assert (
        row[
            "scale_in_buys_aggregated"
        ]
        is True
    )

    assert (
        row[
            "effective_buy_quantity"
        ]
        == pytest.approx(
            2.0
        )
    )

    assert (
        row[
            "entry_cost_usd"
        ]
        == pytest.approx(
            21.0
        )
    )

    assert (
        row[
            "executed_sell_quantity"
        ]
        == pytest.approx(
            1.9
        )
    )


def test_absent_executor_position_retires_stale_fast_state(
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
        sell_limit=105.0,
    )

    with lane._lock:
        lane.state.setdefault(
            "v1615_price_limit_watch",
            {},
        )["BTC/USDT"] = {
            "symbol": "BTC/USDT",
            "checks": 99,
        }

        lane._save_locked()

    with instance._io_lock:
        instance.state[
            "positions"
        ].pop(
            "BTC/USDT",
            None,
        )

        instance.state[
            "position_cost_usd"
        ].pop(
            "BTC/USDT",
            None,
        )

        instance._save_state()

    before_created = len(
        fake.created
    )

    record = copy.deepcopy(
        lane._active_snapshot()[
            "BTC/USDT"
        ]
    )

    result = lane._manage_active(
        service,
        instance.safe_snapshot(),
        "BTC/USDT",
        record,
        now=now,
    )

    assert result["reason"] == (
        "authoritative_executor_"
        "position_absent_retired"
    )

    assert (
        "BTC/USDT"
        not in lane._active_snapshot()
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

    assert (
        "BTC/USDT"
        not in (
            lane.state.get(
                "v1615_price_limit_watch"
            )
            or {}
        )
    )

    assert (
        lane.state[
            "last_exit_by_symbol"
        ]["BTC/USDT"]
        == now
    )

    assert (
        len(fake.created)
        == before_created
    )

    assert (
        result[
            "details"
        ][
            "order_submitted"
        ]
        is False
    )


def test_residual_dust_finalization_cleans_lane_and_uses_net_after_dust(
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
        sell_limit=105.0,
    )

    with instance._io_lock:
        instance.state[
            "positions"
        ]["BTC/USDT"] = (
            0.00002
        )

        instance.state[
            "position_cost_usd"
        ]["BTC/USDT"] = (
            0.002
        )

        instance.state[
            "position_cycle_pnl_usd"
        ]["BTC/USDT"] = (
            0.001
        )

        instance.state[
            "orders"
        ]["v1627-buy"] = {
            "client_order_id": (
                "v1627-buy"
            ),
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "closed",
            "submitted_at": (
                "2026-08-27T12:00:00+00:00"
            ),
            "filled": 0.1001,
            "filled_cost": 10.0,
            "average": 99.9001,
            "fee": 0.0001,
            "fee_currency": "BTC",
        }

        instance.state[
            "orders"
        ]["v1627-sell"] = {
            "client_order_id": (
                "v1627-sell"
            ),
            "symbol": "BTC/USDT",
            "side": "sell",
            "status": "closed",
            "submitted_at": (
                "2026-08-27T12:00:10+00:00"
            ),
            "filled": 0.1,
            "filled_cost": 9.999,
            "average": 99.99,
            "fee": 0.009999,
            "fee_currency": "USDT",
        }

        instance._save_state()

    fake.balance_total[
        "BTC"
    ] = 0.00002

    fake.balance_free[
        "BTC"
    ] = 0.00002

    with lane._lock:
        lane.state[
            "active"
        ][
            "BTC/USDT"
        ][
            "quantity"
        ] = 0.00002

        lane.state.setdefault(
            "v1615_price_limit_watch",
            {},
        )["BTC/USDT"] = {
            "symbol": "BTC/USDT",
            "checks": 99,
        }

        lane._save_locked()

    before_created = len(
        fake.created
    )

    record = copy.deepcopy(
        lane._active_snapshot()[
            "BTC/USDT"
        ]
    )

    result = lane._manage_active(
        service,
        instance.safe_snapshot(),
        "BTC/USDT",
        record,
        now=now,
    )

    assert (
        result["reason"]
        == "residual_dust_cycle_finalized"
    )

    assert (
        len(fake.created)
        == before_created
    )

    assert (
        "BTC/USDT"
        not in lane._active_snapshot()
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

    assert (
        "BTC/USDT"
        not in (
            lane.state.get(
                "v1615_price_limit_watch"
            )
            or {}
        )
    )

    closed = (
        lane.state[
            "closed"
        ][-1]
    )

    assert (
        closed[
            "actual_cycle_net_after_dust_usd"
        ]
        == pytest.approx(
            -0.001
        )
    )

    assert (
        closed[
            "winning_after_dust"
        ]
        is False
    )

    assert (
        closed[
            "actual_return_bps_after_dust"
        ]
        < 0.0
    )

    assert (
        closed[
            "net_bps_after_model"
        ]
        < 0.0
    )

    assert (
        closed[
            "residual_dust_counted_as_sale"
        ]
        is False
    )

    assert (
        lane.state[
            "last_exit_by_symbol"
        ][
            "BTC/USDT"
        ]
        == now
    )
