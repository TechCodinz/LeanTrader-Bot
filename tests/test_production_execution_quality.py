from __future__ import annotations

import pytest

from leantrader.production.execution_quality import (
    ExecutionQualityIntelligence,
)


def _event(
    event_id: str,
    *,
    side: str,
    price: float,
    fee: float,
    realized_return: float | None = None,
) -> dict:
    row = {
        "event_id": event_id,
        "timestamp": "2026-08-21T00:00:00+00:00",
        "symbol": "BTC/USDT",
        "side": side,
        "price": price,
        "quantity": 2.0,
        "fee": fee,
        "reason": "test",
    }
    if realized_return is not None:
        row.update(
            {
                "realized_return": realized_return,
                "realized_pnl": realized_return * 200.0,
                "remaining_quantity": 0.0,
            }
        )
    return row


def test_execution_quality_enforces_cost_floor(tmp_path):
    with pytest.raises(ValueError, match="30-bps cost floor"):
        ExecutionQualityIntelligence(
            tmp_path / "execution.json",
            modeled_round_trip_cost_bps=29.99,
        )


def test_execution_quality_records_shortfall_fees_and_deduplicates(tmp_path):
    state_path = tmp_path / "execution.json"
    engine = ExecutionQualityIntelligence(
        state_path,
        modeled_round_trip_cost_bps=30.0,
    )
    events = [
        _event("buy-1", side="buy", price=100.05, fee=0.2001),
        _event(
            "sell-1",
            side="sell",
            price=99.95,
            fee=0.1999,
            realized_return=0.01,
        ),
    ]

    first = engine.observe(events=events, reference_prices={"BTC/USDT": 100.0})
    second = engine.observe(events=events, reference_prices={"BTC/USDT": 100.0})

    assert first["paper_fill_events_retained"] == 2
    assert first["realized_exit_outcomes_retained"] == 1
    assert first["adverse_shortfall_bps"]["samples"] == 2
    assert first["fee_bps"]["average"] == pytest.approx(10.0)
    assert first["single_leg_drag_bps"]["average"] > 14.9
    assert second["paper_fill_events_retained"] == 2
    assert second["last"]["skipped_events"] == 2
    assert second["actual_market_impact_unobservable_in_paper"] is True
    assert second["execution_authority"] is False
    assert second["live_authority"] is False


def test_execution_quality_cost_survival_is_conservative_and_persistent(tmp_path):
    state_path = tmp_path / "execution.json"
    engine = ExecutionQualityIntelligence(
        state_path,
        modeled_round_trip_cost_bps=30.0,
    )
    snapshot = engine.observe(
        events=[
            _event(
                "sell-win",
                side="sell",
                price=99.95,
                fee=0.1999,
                realized_return=0.01,
            ),
            {
                **_event(
                    "sell-small",
                    side="sell",
                    price=99.95,
                    fee=0.1999,
                    realized_return=0.005,
                ),
                "symbol": "ETH/USDT",
            },
        ],
        reference_prices={"BTC/USDT": 100.0, "ETH/USDT": 100.0},
    )

    low_cost = snapshot["cost_survival"]["30bps"]
    high_cost = snapshot["cost_survival"]["250bps"]
    assert low_cost["average_net_return"] > high_cost["average_net_return"]
    assert low_cost["cumulative_net_return"] > high_cost["cumulative_net_return"]

    reloaded = ExecutionQualityIntelligence(
        state_path,
        modeled_round_trip_cost_bps=30.0,
    )
    persisted = reloaded.snapshot()
    assert persisted["realized_return_statistics"]["samples"] == 2
    assert reloaded.health()["testnet_authority"] is False
    assert reloaded.health()["can_modify_orders"] is False
