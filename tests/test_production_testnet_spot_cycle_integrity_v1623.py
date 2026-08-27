from __future__ import annotations

import copy

import pytest

from tests.test_production_testnet_exit_recycle_v1608 import (
    BalanceBybit,
)
from tests.test_production_testnet_price_limit_edge_exit_v1615 import (
    _runtime,
)
from tests.test_testnet_execution import (
    buy_event,
    engine,
)


class QuoteCostBybit(BalanceBybit):
    def __init__(self):
        super().__init__()
        self.quote_cost_calls = []

    def create_market_buy_order_with_cost(
        self,
        symbol,
        cost,
        params=None,
    ):
        params = dict(params or {})

        self.quote_cost_calls.append(
            (
                symbol,
                float(cost),
                params,
            )
        )

        client_id = params["orderLinkId"]
        amount = float(cost) / 100.0

        observed = {
            "id": "quote-cost-1",
            "clientOrderId": client_id,
            "symbol": symbol,
            "side": "buy",
            "status": "closed",
            "filled": amount,
            "average": 100.0,
            "cost": float(cost),
            "fee": {"cost": 0.0},
            "info": {
                "orderLinkId": client_id,
            },
        }

        self.created.append(observed)
        self.orders[observed["id"]] = observed

        return observed


def test_spot_buy_routes_by_quote_cost(
    tmp_path,
):
    fake = QuoteCostBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    result = instance.mirror_events(
        [buy_event()]
    )[0]

    assert result["status"] == "closed"
    assert len(fake.quote_cost_calls) == 1

    _, cost, params = fake.quote_cost_calls[0]

    assert cost == pytest.approx(5.0)
    assert params["orderLinkId"]

    record = next(
        row
        for row in instance.state["orders"].values()
        if row.get("side") == "buy"
    )

    assert (
        record["submission_mode"]
        == "quote_cost_market_buy"
    )

    assert (
        instance.health()[
            "spot_market_buy_routing"
        ]["quote_cost_market_buy_attempts"]
        == 1
    )


def test_sync_provider_error_is_preserved(
    tmp_path,
):
    fake = QuoteCostBybit()

    def reject(
        _symbol,
        _cost,
        _params=None,
    ):
        raise RuntimeError(
            "retCode 12345 deterministic rejection"
        )

    fake.create_market_buy_order_with_cost = (
        reject
    )

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    with pytest.raises(
        RuntimeError,
        match="12345",
    ):
        instance.mirror_events(
            [buy_event()]
        )

    row = instance.state[
        "v1623_last_submission_exception"
    ]

    assert row["exception_type"] == "RuntimeError"
    assert "12345" in row["reason"]
    assert row["live_authority"] is False


def test_true_dust_recycles_before_price_boundary(
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
        instance.state["positions"][
            "BTC/USDT"
        ] = 0.00002

        instance.state["position_cost_usd"][
            "BTC/USDT"
        ] = 0.002

        instance.state["account_balance"].setdefault(
            "free",
            {},
        )["BTC"] = 0.00002

        instance._save_state()

    with lane._lock:
        lane.state["active"]["BTC/USDT"][
            "quantity"
        ] = 0.00002

        lane.state["active"]["BTC/USDT"][
            "initial_quantity"
        ] = 0.00002

        lane._save_locked()

    snapshot = instance.safe_snapshot()

    record = copy.deepcopy(
        lane._active_snapshot()[
            "BTC/USDT"
        ]
    )

    result = lane._manage_active(
        service,
        snapshot,
        "BTC/USDT",
        record,
        now=now,
    )

    assert result["reason"] == (
        "active_exit_reclassified_"
        "dust_preboundary"
    )

    assert fake.created == []

    assert (
        "BTC/USDT"
        not in instance.health()["positions"]
    )

    assert (
        "BTC/USDT"
        in instance.health()[
            "non_tradeable_dust"
        ]
    )

    assert (
        instance.health()["performance"][
            "closed_positions"
        ]
        == 0
    )

    assert (
        lane.health()[
            "spot_cycle_integrity"
        ]["exchange_price_limit_bypass"]
        is False
    )
