from __future__ import annotations

import pytest

from leantrader.production.capital_stress import CapitalStressSimulator


def _engine(tmp_path, **overrides) -> CapitalStressSimulator:
    values = {
        "starting_equity": 50.0,
        "principal_floor_fraction": 0.90,
        "risk_per_trade_fraction": 0.005,
        "max_daily_loss_fraction": 0.02,
        "max_drawdown_fraction": 0.10,
        "modeled_round_trip_cost_bps": 30.0,
    }
    values.update(overrides)
    return CapitalStressSimulator(tmp_path / "capital.json", **values)


def test_capital_stress_rejects_unsafe_configuration(tmp_path):
    with pytest.raises(ValueError, match="starting equity"):
        _engine(tmp_path, starting_equity=0.0)
    with pytest.raises(ValueError, match="between zero and one"):
        _engine(tmp_path, max_drawdown_fraction=1.0)
    with pytest.raises(ValueError, match="30-bps cost floor"):
        _engine(tmp_path, modeled_round_trip_cost_bps=10.0)


def test_capital_stress_survives_cash_only_small_account(tmp_path):
    engine = _engine(tmp_path)
    result = engine.evaluate(
        equity=50.0,
        cash=50.0,
        peak_equity=50.0,
        positions={},
        execution_quality={},
    )

    assert result["gross_open_exposure"] == 0.0
    assert result["stress_losing_streak"] == 5
    assert result["worst_projected_equity"] == pytest.approx(48.0)
    assert result["principal_floor_survives_all"] is True
    assert result["survives_worst_scenario"] is True
    assert result["not_a_forecast"] is True
    assert result["execution_authority"] is False


def test_capital_stress_detects_concentration_and_principal_floor_risk(tmp_path):
    engine = _engine(tmp_path)
    result = engine.evaluate(
        equity=50.0,
        cash=10.0,
        peak_equity=50.0,
        positions={
            "BTC/USDT": {
                "notional": 40.0,
                "price": 100.0,
                "atr": 2.0,
            }
        },
        execution_quality={},
    )

    assert result["exposure_fraction"] == pytest.approx(0.8)
    assert result["largest_position_fraction"] == pytest.approx(0.8)
    assert result["position_concentration_hhi"] == pytest.approx(1.0)
    assert result["stress_state"] == "principal_floor_at_risk"
    assert result["principal_floor_survives_all"] is False
    assert result["cannot_override_capital_governor"] is True
    assert result["can_modify_sizing"] is False


def test_capital_stress_uses_atr_and_observed_losing_streak(tmp_path):
    state_path = tmp_path / "capital.json"
    engine = _engine(tmp_path)
    result = engine.evaluate(
        equity=50.0,
        cash=30.0,
        peak_equity=55.0,
        positions={
            "ETH/USDT": {
                "notional": 20.0,
                "price": 100.0,
                "atr": 10.0,
            }
        },
        execution_quality={
            "realized_return_statistics": {
                "samples": 100,
                "max_losing_streak": 9,
            }
        },
    )
    scenarios = {row["scenario"]: row for row in result["scenarios"]}

    assert scenarios["atr_volatility_shock"]["projected_loss"] == pytest.approx(5.2)
    assert result["stress_losing_streak"] == 9
    assert result["observed_execution_evidence_mature"] is True

    reloaded = CapitalStressSimulator(
        state_path,
        starting_equity=50.0,
        principal_floor_fraction=0.90,
        risk_per_trade_fraction=0.005,
        max_daily_loss_fraction=0.02,
        max_drawdown_fraction=0.10,
        modeled_round_trip_cost_bps=30.0,
    )
    assert reloaded.health()["evaluations"] == 1
    assert reloaded.health()["live_authority"] is False
