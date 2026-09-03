from __future__ import annotations

from leantrader.production.testnet_execution import (
    BybitTestnetExecutionEngine,
)
from leantrader.production.testnet_fast_profit_guard_v1634 import (
    fast_entry_profit_gate,
)


class _Exchange:
    def market(self, _symbol: str):
        # 10 bps per side taker metadata.
        return {"taker": 0.001}


def _engine_with_rows(rows):
    engine = BybitTestnetExecutionEngine.__new__(
        BybitTestnetExecutionEngine
    )
    engine.exchange = _Exchange()
    engine.state = {"orders": rows}
    return engine


def _order(side: str, average: float):
    return {
        "symbol": "XRP/USDT",
        "side": side,
        "filled": 1.0,
        "average": average,
        "reference_price": 1.0,
        "filled_cost": average,
        "fee": average * 0.001,
        "fee_currency": "USDT",
    }


def test_dynamic_cost_can_move_below_30_only_after_two_sided_evidence():
    engine = _engine_with_rows(
        {
            "1": _order("buy", 1.0001),
            "2": _order("sell", 0.9999),
            "3": _order("buy", 1.0001),
            "4": _order("sell", 0.9999),
        }
    )

    row = engine.dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert row["evidence_sufficient"] is True
    assert row["below_30_authorized"] is True
    assert 20.0 <= row["effective_round_trip_cost_bps"] < 30.0


def test_insufficient_evidence_keeps_30bps_fallback():
    engine = _engine_with_rows(
        {
            "1": _order("buy", 1.0001),
            "2": _order("sell", 0.9999),
        }
    )

    row = engine.dynamic_execution_cost_profile(
        "XRP/USDT",
        spread_bps=4.0,
    )

    assert row["evidence_sufficient"] is False
    assert row["effective_round_trip_cost_bps"] >= 30.0


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
