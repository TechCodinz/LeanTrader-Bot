from __future__ import annotations

import datetime as dt
import threading

from leantrader.production.testnet_execution import (
    BybitTestnetExecutionEngine,
)
from leantrader.production.testnet_fast_profit_guard_v1634 import (
    fast_entry_profit_gate,
)


class _Exchange:
    def market(self, _symbol: str):
        return {
            "taker": 0.001,
            "base": "XRP",
            "quote": "USDT",
        }


def _engine_with_rows(rows):
    engine = BybitTestnetExecutionEngine.__new__(
        BybitTestnetExecutionEngine
    )
    engine.exchange = _Exchange()
    engine.state = {"orders": rows}
    engine._io_lock = threading.RLock()
    return engine


def _submitted(seconds_ago: int) -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        - dt.timedelta(seconds=seconds_ago)
    ).isoformat()


def _order(
    side: str,
    average: float,
    *,
    seconds_ago: int,
    fee_currency: str | None = None,
):
    filled = 1.0
    fee_currency = fee_currency or (
        "XRP" if side == "buy" else "USDT"
    )
    fee = (
        0.001
        if fee_currency == "XRP"
        else average * 0.001
    )
    return {
        "symbol": "XRP/USDT",
        "side": side,
        "filled": filled,
        "average": average,
        "reference_price": 1.0,
        "filled_cost": average,
        "fee": fee,
        "fee_currency": fee_currency,
        "submitted_at": _submitted(seconds_ago),
    }


def _six_cycles():
    rows = {}
    for index in range(6):
        rows[f"buy-{index}"] = _order(
            "buy",
            1.0001,
            seconds_ago=120 - index * 10,
        )
        rows[f"sell-{index}"] = _order(
            "sell",
            0.9999,
            seconds_ago=115 - index * 10,
        )
    return rows


def test_dynamic_cost_requires_six_recent_paired_cycles():
    engine = _engine_with_rows(_six_cycles())

    row = engine.dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert row["evidence_sufficient"] is True
    assert row["completed_cycles"] == 6
    assert row["authenticated_fee_sides"] == 2
    assert row["below_30_authorized"] is True
    assert 20.0 <= row["effective_round_trip_cost_bps"] < 30.0


def test_unpaired_fills_do_not_unlock_relaxation():
    rows = {
        f"buy-{index}": _order(
            "buy",
            1.0001,
            seconds_ago=100 - index,
        )
        for index in range(11)
    }
    rows["sell-only"] = _order(
        "sell",
        0.9999,
        seconds_ago=30,
    )

    row = _engine_with_rows(
        rows
    ).dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert row["completed_cycles"] == 1
    assert row["evidence_sufficient"] is False
    assert row["effective_round_trip_cost_bps"] >= 30.0


def test_old_fills_do_not_authorize_current_relaxation():
    rows = {}
    for index in range(6):
        old = 90_000 + index
        rows[f"buy-{index}"] = _order(
            "buy",
            1.0001,
            seconds_ago=old,
        )
        rows[f"sell-{index}"] = _order(
            "sell",
            0.9999,
            seconds_ago=old,
        )

    row = _engine_with_rows(
        rows
    ).dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert row["samples"] == 0
    assert row["evidence_sufficient"] is False
    assert row["effective_round_trip_cost_bps"] >= 30.0


def test_order_key_order_does_not_change_recent_sample():
    rows = _six_cycles()
    reverse = dict(reversed(list(rows.items())))

    forward_profile = _engine_with_rows(
        rows
    ).dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )
    reverse_profile = _engine_with_rows(
        reverse
    ).dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert (
        forward_profile["estimated_round_trip_cost_bps"]
        == reverse_profile["estimated_round_trip_cost_bps"]
    )
    assert (
        forward_profile["completed_cycles"]
        == reverse_profile["completed_cycles"]
    )


def test_base_asset_buy_fee_is_normalized():
    row = _engine_with_rows(
        _six_cycles()
    ).dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert row["authenticated_fee_sides"] == 2
    assert 9.0 <= row["buy_fee_bps"] <= 11.0
    assert 9.0 <= row["sell_fee_bps"] <= 11.0


def test_trusted_dynamic_economics_changes_profit_hurdle():
    economics = {
        "evidence_sufficient": True,
        "effective_round_trip_cost_bps": 24.0,
        "recommended_net_margin_bps": 6.0,
    }

    allowed = fast_entry_profit_gate(
        {
            "allowed": True,
            "reason": "candidate",
            "modeled_round_trip_cost_bps": 24.0,
            "dynamic_execution_economics": economics,
            "micro_confidence": 0.20,
            "velocity": {
                "qualified_long": False,
                "projected_capture_bps_5s": 35.0,
            },
            "micro_support": [],
        }
    )

    assert allowed["allowed"] is True
    assert (
        allowed["v1634_fast_profit_gate"]["required_capture_bps"]
        == 30.0
    )

    fallback = fast_entry_profit_gate(
        {
            "allowed": True,
            "reason": "candidate",
            "modeled_round_trip_cost_bps": 24.0,
            "dynamic_execution_economics": {
                "evidence_sufficient": False,
            },
            "micro_confidence": 0.20,
            "velocity": {
                "qualified_long": False,
                "projected_capture_bps_5s": 35.0,
            },
            "micro_support": [],
        }
    )

    assert fallback["allowed"] is False
    assert fallback["reason"] == "v1634_fast_edge_below_cost_margin"
