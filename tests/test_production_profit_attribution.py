from __future__ import annotations

import pytest

from leantrader.production.profit_attribution import NetProfitAttribution


def _event(
    index: int,
    *,
    net_return: float,
    regime: str = "trend",
    remaining_quantity: float = 0.0,
) -> dict:
    return {
        "event_id": f"sell-{index}",
        "timestamp": "2026-08-21T00:00:00+00:00",
        "side": "sell",
        "symbol": "BTC/USDT" if index % 2 == 0 else "ETH/USDT",
        "quantity": 1.0,
        "price": 100.0,
        "fee": 0.1,
        "reason": "take_profit" if net_return > 0.0 else "atr_stop",
        "remaining_quantity": remaining_quantity,
        "realized_return": net_return,
        "realized_pnl": net_return * 100.0,
        "trade_realized_return_total": net_return,
        "trade_realized_pnl_total": net_return * 100.0,
        "position_metadata": {
            "regime": regime,
            "confidence": 0.72,
            "component_scores": {
                "trend": 0.4,
                "momentum": 0.2,
                "mean_reversion": -0.2,
            },
            "decision_route": {
                "reason": "evidence_gate_passed",
                "predicted_probability": 0.7,
            },
        },
    }


def _engine(tmp_path, **overrides) -> NetProfitAttribution:
    values = {
        "minimum_samples": 100,
        "minimum_regimes": 2,
        "modeled_round_trip_cost_bps": 30.0,
    }
    values.update(overrides)
    return NetProfitAttribution(tmp_path / "attribution.json", **values)


def test_profit_attribution_preserves_evidence_and_cost_floors(tmp_path):
    with pytest.raises(ValueError, match="100-sample"):
        _engine(tmp_path, minimum_samples=99)
    with pytest.raises(ValueError, match="at least two regimes"):
        _engine(tmp_path, minimum_regimes=1)
    with pytest.raises(ValueError, match="30-bps cost floor"):
        _engine(tmp_path, modeled_round_trip_cost_bps=29.0)


def test_profit_attribution_uses_only_unique_fully_closed_trades(tmp_path):
    engine = _engine(tmp_path)
    partial = _event(1, net_return=0.02, remaining_quantity=0.5)
    complete = _event(2, net_return=0.01)

    first = engine.observe(events=[partial, complete])
    second = engine.observe(events=[complete])

    assert first["closed_trade_samples"] == 1
    assert first["last"]["partial_exits_ignored"] == 1
    assert second["closed_trade_samples"] == 1
    assert second["last"]["duplicates_ignored"] == 1
    assert second["sample_unit"] == "fully_closed_costed_paper_trade"
    assert second["observational_not_causal"] is True
    assert second["execution_authority"] is False


def test_profit_attribution_requires_positive_adjusted_lower_bound(tmp_path):
    engine = _engine(tmp_path)
    events = [
        _event(
            index,
            net_return=0.01,
            regime="trend" if index % 2 == 0 else "range",
        )
        for index in range(100)
    ]
    snapshot = engine.observe(events=events)
    overall = snapshot["overall"]

    assert overall["samples"] == 100
    assert overall["regime_count"] == 2
    assert overall["evidence_mature"] is True
    assert overall["mean_return_lower_bound"] > 0.0
    assert overall["positive_edge_after_costs"] is True
    assert snapshot["profitability_claim_allowed"] is True
    assert "overall:all" in snapshot["positive_edge_candidates"]
    assert snapshot["multiple_testing_correction"] == "bonferroni"
    assert snapshot["paper_promotion_authority"] is False
    assert snapshot["live_authority"] is False


def test_profit_attribution_identifies_mature_negative_edge_and_persists(tmp_path):
    state_path = tmp_path / "attribution.json"
    engine = _engine(tmp_path)
    snapshot = engine.observe(
        events=[
            _event(
                index,
                net_return=-0.01,
                regime="trend" if index % 2 == 0 else "range",
            )
            for index in range(100)
        ]
    )

    assert snapshot["overall"]["negative_edge_after_costs"] is True
    assert snapshot["profitability_claim_allowed"] is False
    assert "overall:all" in snapshot["negative_edge_candidates"]

    reloaded = NetProfitAttribution(
        state_path,
        minimum_samples=100,
        minimum_regimes=2,
        modeled_round_trip_cost_bps=30.0,
    )
    assert reloaded.health()["closed_trade_samples"] == 100
    assert reloaded.health()["can_increase_risk"] is False
