from __future__ import annotations

import pytest

from leantrader.production.testnet_realized_dust_integrity_v1633 import (
    realized_dust_accounting,
)
from tests.test_production_testnet_exit_recycle_v1608 import (
    growth,
    hyper_lane,
)
from tests.test_testnet_execution import (
    FakeBybit,
    engine,
)


def _sample_state():
    return {
        "realized_pnl_usd": (
            0.001374608599974514
        ),
        "dust_cost_basis_usd_total": (
            12.712359277400974
        ),
        "account_balance": {
            "free": {
                "CHIP": 149.92503489,
                "ENS": 0.8506485,
                "ZAMA": 87.81888321,
                "XRP": 0.0004795,
            }
        },
        "non_tradeable_dust": {
            "CHIP/USDT": {
                "quantity": 149.92503489,
                "free_quantity": 149.92503489,
                "cost_basis_usd": 5.255228351604,
                "estimated_value_usd": 4.29355314917982,
                "counted_as_executed_close": False,
                "reason": "startup_fresh_bid_below_exchange_executable_threshold",
            },
            "ENS/USDT": {
                "quantity": 0.8506485,
                "free_quantity": 0.8506485,
                "cost_basis_usd": 4.37458125,
                "estimated_value_usd": 3.0766254948,
                "counted_as_executed_close": False,
                "reason": "startup_fresh_bid_below_exchange_executable_threshold",
            },
            "ZAMA/USDT": {
                "quantity": 87.81888321,
                "free_quantity": 87.81888321,
                "cost_basis_usd": 3.081875499922,
                "estimated_value_usd": 2.41299945396117,
                "counted_as_executed_close": False,
                "reason": "startup_fresh_bid_below_exchange_executable_threshold",
            },
            "XRP/USDT": {
                "quantity": 8.81e-05,
                "free_quantity": 0.0004795,
                "cost_basis_usd": 0.0001238526379380156,
                "counted_as_executed_close": True,
            },
        },
        "v1627_completed_executable_cycles": [
            {
                "symbol": "XRP/USDT",
                "cycle_key": "cycle-a",
                "completed_executable_cycle": True,
                "counted_as_executed_close": True,
                "residual_dust_cost_basis_usd": (
                    1.435913243241771e-05
                ),
            },
            {
                "symbol": "XRP/USDT",
                "cycle_key": "cycle-b",
                "completed_executable_cycle": True,
                "counted_as_executed_close": True,
                "residual_dust_cost_basis_usd": (
                    0.0001238526379380156
                ),
            },
        ],
    }


def test_wallet_held_inventory_is_not_realized_loss():
    accounting = realized_dust_accounting(
        _sample_state()
    )

    assert accounting[
        "wallet_held_nonfinalized_cost_basis_usd"
    ] == pytest.approx(
        12.711685101526001
    )

    assert accounting[
        "finalized_residual_dust_cost_basis_usd"
    ] == pytest.approx(
        0.0001382117703704333
    )

    assert accounting[
        "realized_net_after_finalized_dust_usd"
    ] == pytest.approx(
        0.0012363968296040806
    )

    assert accounting[
        "legacy_realized_net_after_all_dust_usd"
    ] == pytest.approx(
        -12.710984668801
    )

    assert accounting[
        "wallet_held_inventory_is_not_realized_loss"
    ] is True


def test_finalized_residual_cycle_key_is_idempotent():
    state = _sample_state()

    state[
        "v1627_completed_executable_cycles"
    ].append(
        dict(
            state[
                "v1627_completed_executable_cycles"
            ][0]
        )
    )

    accounting = realized_dust_accounting(
        state
    )

    assert accounting[
        "finalized_cycle_count"
    ] == 2

    assert accounting[
        "finalized_residual_dust_cost_basis_usd"
    ] == pytest.approx(
        0.0001382117703704333
    )


def test_engine_snapshot_exposes_finalized_dust_only_to_compounding(
    tmp_path,
):
    fake = FakeBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    sample = _sample_state()

    with instance._io_lock:
        instance.state[
            "realized_pnl_usd"
        ] = sample[
            "realized_pnl_usd"
        ]

        instance.state[
            "dust_cost_basis_usd_total"
        ] = sample[
            "dust_cost_basis_usd_total"
        ]

        instance.state[
            "non_tradeable_dust"
        ] = sample[
            "non_tradeable_dust"
        ]

        instance.state[
            "v1627_completed_executable_cycles"
        ] = sample[
            "v1627_completed_executable_cycles"
        ]

        instance.state[
            "account_balance"
        ] = sample[
            "account_balance"
        ]

        instance.state.pop(
            "v1633_finalized_residual_dust_cost_basis_usd",
            None,
        )

        instance.state.pop(
            "v1633_finalized_residual_dust_cycle_keys",
            None,
        )

        instance._save_state()

    snapshot = instance.safe_snapshot()

    performance = snapshot[
        "performance"
    ]

    assert performance[
        "legacy_non_tradeable_dust_cost_basis_usd"
    ] == pytest.approx(
        12.712359277400974
    )

    assert performance[
        "non_tradeable_dust_cost_basis_usd"
    ] == pytest.approx(
        0.0001382117703704333
    )

    assert performance[
        "realized_net_after_dust_usd"
    ] == pytest.approx(
        0.0012363968296040806
    )

    # Historical audit value remains untouched in canonical state.
    assert instance.state[
        "dust_cost_basis_usd_total"
    ] == pytest.approx(
        12.712359277400974
    )


def test_existing_v1608_gate_uses_corrected_dust_input(
    tmp_path,
):
    fake = FakeBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    with instance._io_lock:
        instance.state[
            "realized_pnl_usd"
        ] = 0.010

        instance.state[
            "dust_cost_basis_usd_total"
        ] = 12.0

        instance.state[
            "non_tradeable_dust"
        ] = {
            "CHIP/USDT": {
                "quantity": 100.0,
                "free_quantity": 100.0,
                "cost_basis_usd": 12.0,
                "estimated_value_usd": 9.0,
                "counted_as_executed_close": False,
            }
        }

        instance.state[
            "v1627_completed_executable_cycles"
        ] = [
            {
                "symbol": "XRP/USDT",
                "cycle_key": "positive-cycle",
                "completed_executable_cycle": True,
                "counted_as_executed_close": True,
                "residual_dust_cost_basis_usd": 0.001,
            }
        ]

        instance.state.pop(
            "v1633_finalized_residual_dust_cost_basis_usd",
            None,
        )

        instance.state.pop(
            "v1633_finalized_residual_dust_cycle_keys",
            None,
        )

        instance._save_state()

    supervisory = {
        "capital_growth": growth(
            remaining=15.0,
            risk=1.0,
            equity=51.0,
        )
    }

    lane, _service, _testnet = (
        hyper_lane(
            tmp_path,
            testnet=instance,
            supervisory=supervisory,
            order_usd=2.0,
        )
    )

    with lane._lock:
        lane.state["closed"] = [
            {
                "entry_notional_usd": 1.0,
            }
        ]
        lane._save_locked()

    sizing = lane._compound_order_notional(
        supervisory,
        slots=1,
    )

    assert sizing[
        "actual_testnet_dust_cost_basis_usd"
    ] == pytest.approx(
        0.001
    )

    # $0.010 realized - $0.001 finalized residual
    # - $0.003 modeled reserve on $1 completed notional.
    assert sizing[
        "actual_testnet_net_after_modeled_cost_usd"
    ] == pytest.approx(
        0.006
    )

    assert sizing[
        "actual_testnet_profit_compounding_eligible"
    ] is True

    assert sizing[
        "compounding"
    ] is True

    assert instance.state[
        "dust_cost_basis_usd_total"
    ] == pytest.approx(
        12.0
    )


def test_lightweight_testnet_adapter_preserves_original_lane_health(
    tmp_path,
):
    """v1.60.33 must not require private executor state from adapters."""

    lane, _service, testnet = hyper_lane(
        tmp_path,
    )

    assert not hasattr(
        testnet,
        "_io_lock",
    )

    payload = lane.health()

    # Existing adapter telemetry survives and v1.60.33 does not try
    # to manufacture a persistent accounting ledger for it.
    assert isinstance(
        payload,
        dict,
    )

    assert (
        "realized_dust_accounting_integrity"
        not in payload
    )

    assert payload[
        "live_authority"
    ] is False
