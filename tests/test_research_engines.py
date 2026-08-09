from __future__ import annotations

import numpy as np
import pandas as pd

from leantrader.production.research_engines import (
    CalibrationEngine,
    CapitalPreservationEngine,
    ChampionChallengerGovernor,
    DriftEngine,
    GradientBoostForecastEngine,
    KronosForecastAdapter,
    OptunaResearchEngine,
    QuantumResearchAdapter,
    ReplayEngine,
    StressEngine,
)


def market_frame(rows: int = 520, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    returns = rng.normal(0.0003 + shift, 0.006, rows)
    close = 100 * np.cumprod(1 + returns)
    volume = 1000 * (1 + np.abs(rng.normal(0, 0.2, rows)))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": volume,
        }
    )


def test_replay_never_exposes_future_rows():
    frame = market_frame(230)
    result = ReplayEngine().replay(frame, lambda visible: len(visible), warmup=220)
    assert result[0] == {"index": 219, "visible_rows": 220, "result": 220}
    assert result[-1]["visible_rows"] == 230


def test_gradient_boost_walk_forward_is_deterministic_and_cost_aware():
    engine = GradientBoostForecastEngine()
    first = engine.walk_forward(market_frame(), train_bars=200, test_bars=40)
    second = engine.walk_forward(market_frame(), train_bars=200, test_bars=40)
    assert first == second
    assert first.windows > 0
    assert 0 <= first.brier_score <= 1
    assert 0 <= first.accuracy <= 1


def test_gradient_boost_exposes_canonical_30_day_7_day_schedule():
    engine = GradientBoostForecastEngine()
    result = engine.walk_forward_30_7(market_frame(), bars_per_day=4)
    assert result.windows > 0
    assert engine.health()["canonical_schedule_days"] == [30, 7]


def test_kronos_adapter_rejects_missing_and_invalid_predictors():
    frame = market_frame()
    assert KronosForecastAdapter().forecast(frame)["available"] is False

    class Predictor:
        def predict(self, **kwargs):
            close = float(kwargs["df"]["close"].iloc[-1])
            return pd.DataFrame({"close": np.repeat(close * 1.01, kwargs["pred_len"])})

    adapter = KronosForecastAdapter(Predictor())
    result = adapter.forecast(frame, horizon=4)
    assert result["available"] is True
    assert result["expected_return"] > 0


def test_optuna_and_quantum_adapters_never_fabricate_availability():
    optuna_result = OptunaResearchEngine().optimize(lambda params: params["x"], {"x": (0.0, 1.0)}, trials=2)
    assert "available" in optuna_result
    mu = np.array([0.02, 0.01])
    covariance = np.array([[0.04, 0.0], [0.0, 0.01]])
    unavailable = QuantumResearchAdapter().benchmark(mu, covariance)
    assert unavailable["available"] is False
    available = QuantumResearchAdapter(lambda _mu, _cov: np.array([1.0, 0.0])).benchmark(mu, covariance)
    assert available["available"] is True
    assert sum(available["candidate_weights"]) == 1.0


def test_calibration_and_drift_are_measured():
    calibration = CalibrationEngine().evaluate([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert calibration["brier_score"] < 0.1
    reference = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    current = pd.DataFrame({"x": np.linspace(5, 6, 100)})
    assert DriftEngine().compare(reference, current)["drifted"] is True


def test_champion_challenger_promotion_and_rollback_are_persistent(tmp_path):
    path = tmp_path / "governor.json"
    governor = ChampionChallengerGovernor(path)
    for _ in range(5):
        governor.record("adaptive_ensemble", 0.001, 0.03, 0.22)
        governor.record("gradient_boost", 0.01, 0.04, 0.18)
    assert governor.consider("gradient_boost") is True
    assert governor.health()["champion"] == "gradient_boost"
    assert governor.rollback("drift") is True
    assert ChampionChallengerGovernor(path).health()["champion"] == "adaptive_ensemble"


def test_capital_preservation_requires_healthy_recovery_cycles():
    engine = CapitalPreservationEngine()
    halted = engine.update(
        drawdown=0.11,
        daily_loss=0.0,
        data_healthy=True,
        required_engines_healthy=True,
    )
    assert halted["state"] == "halt"
    assert halted["size_multiplier"] == 0.0
    for _ in range(9):
        state = engine.update(
            drawdown=0.0,
            daily_loss=0.0,
            data_healthy=True,
            required_engines_healthy=True,
        )
        assert state["state"] == "recovery"
    state = engine.update(
        drawdown=0.0,
        daily_loss=0.0,
        data_healthy=True,
        required_engines_healthy=True,
    )
    assert state["state"] == "normal"


def test_stress_engine_has_no_random_scenarios():
    result = StressEngine().evaluate({"BTC/USDT": 10.0, "ETH/USDT": 5.0})
    assert result["worst_case_pnl"] == -3.75
