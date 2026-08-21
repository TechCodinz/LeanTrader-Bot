from __future__ import annotations

import pytest

from leantrader.production.probability_calibration import (
    ProbabilityCalibrationLab,
)


def _event(
    index: int,
    *,
    probability: float | None,
    profitable: bool,
    regime: str,
    remaining_quantity: float = 0.0,
) -> dict:
    route = (
        {"predicted_probability": probability}
        if probability is not None
        else {}
    )
    net_return = 0.01 if profitable else -0.01
    return {
        "event_id": f"calibration-{index}",
        "timestamp": "2026-08-21T00:00:00+00:00",
        "side": "sell",
        "symbol": "BTC/USDT",
        "quantity": 1.0,
        "price": 100.0,
        "fee": 0.1,
        "remaining_quantity": remaining_quantity,
        "realized_return": net_return,
        "trade_realized_return_total": net_return,
        "position_metadata": {
            "regime": regime,
            "decision_route": route,
        },
    }


def _engine(tmp_path, **overrides) -> ProbabilityCalibrationLab:
    values = {
        "minimum_samples": 100,
        "minimum_regimes": 2,
        "minimum_class_samples": 20,
        "modeled_round_trip_cost_bps": 30.0,
    }
    values.update(overrides)
    return ProbabilityCalibrationLab(tmp_path / "calibration.json", **values)


def test_probability_calibration_preserves_all_research_floors(tmp_path):
    with pytest.raises(ValueError, match="100-sample"):
        _engine(tmp_path, minimum_samples=99)
    with pytest.raises(ValueError, match="at least two regimes"):
        _engine(tmp_path, minimum_regimes=1)
    with pytest.raises(ValueError, match="20 outcomes"):
        _engine(tmp_path, minimum_class_samples=19)
    with pytest.raises(ValueError, match="30-bps cost floor"):
        _engine(tmp_path, modeled_round_trip_cost_bps=29.0)


def test_probability_calibration_ignores_partial_missing_and_duplicate_rows(tmp_path):
    engine = _engine(tmp_path)
    partial = _event(
        1,
        probability=0.7,
        profitable=True,
        regime="trend",
        remaining_quantity=0.5,
    )
    missing = _event(
        2,
        probability=None,
        profitable=False,
        regime="range",
    )
    complete = _event(
        3,
        probability=0.6,
        profitable=True,
        regime="trend",
    )

    first = engine.observe(events=[partial, missing, complete])
    second = engine.observe(events=[complete])

    assert first["overall"]["samples"] == 1
    assert first["last"]["partial_exits_ignored"] == 1
    assert first["last"]["missing_predictions_ignored"] == 1
    assert second["overall"]["samples"] == 1
    assert second["last"]["duplicates_ignored"] == 1
    assert second["calibration_state"] == "waiting_for_samples"
    assert second["can_rewrite_probabilities"] is False
    assert second["execution_authority"] is False


def test_probability_calibration_recognizes_balanced_calibrated_evidence(tmp_path):
    engine = _engine(tmp_path)
    events = [
        _event(
            index,
            probability=0.5,
            profitable=index % 2 == 0,
            regime="trend" if index % 4 < 2 else "range",
        )
        for index in range(100)
    ]
    snapshot = engine.observe(events=events)

    assert snapshot["evidence_mature"] is True
    assert snapshot["regime_count"] == 2
    assert snapshot["overall"]["wins"] == 50
    assert snapshot["overall"]["losses"] == 50
    assert snapshot["overall"]["brier_score"] == pytest.approx(0.25)
    assert snapshot["expected_calibration_error"] == pytest.approx(0.0)
    assert snapshot["calibration_state"] == "calibrated"
    assert snapshot["suggested_probability_application"] == (
        "none_advisory_diagnostics_only"
    )
    assert snapshot["live_authority"] is False


def test_probability_calibration_flags_miscalibration_and_persists(tmp_path):
    state_path = tmp_path / "calibration.json"
    engine = _engine(tmp_path)
    snapshot = engine.observe(
        events=[
            _event(
                index,
                probability=0.9,
                profitable=index % 2 == 0,
                regime="trend" if index % 4 < 2 else "range",
            )
            for index in range(100)
        ]
    )

    assert snapshot["evidence_mature"] is True
    assert snapshot["expected_calibration_error"] == pytest.approx(0.4)
    assert snapshot["calibration_state"] == "miscalibrated"

    reloaded = ProbabilityCalibrationLab(
        state_path,
        minimum_samples=100,
        minimum_regimes=2,
        minimum_class_samples=20,
        modeled_round_trip_cost_bps=30.0,
    )
    assert reloaded.health()["closed_trade_samples"] == 100
    assert reloaded.health()["testnet_authority"] is False
    assert reloaded.health()["can_modify_routes"] is False
