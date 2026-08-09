from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: tuple[float, ...]
    outcomes: tuple[int, ...]
    returns: tuple[float, ...]
    brier_score: float
    accuracy: float
    net_return: float
    windows: int


class ReplayEngine:
    """Causal candle replay that never exposes future rows to the evaluator."""

    VERSION = "1.0"

    def replay(
        self,
        frame: pd.DataFrame,
        evaluator: Callable[[pd.DataFrame], Any],
        *,
        warmup: int = 220,
    ) -> list[dict[str, Any]]:
        if warmup < 2 or len(frame) <= warmup:
            raise ValueError("replay frame must exceed warmup")
        output = []
        for index in range(warmup - 1, len(frame)):
            visible = frame.iloc[: index + 1].copy()
            result = evaluator(visible)
            output.append(
                {
                    "index": int(index),
                    "visible_rows": len(visible),
                    "result": result,
                }
            )
        return output

    def health(self) -> dict[str, Any]:
        return {"causal": True, "future_rows_visible": False}


class GradientBoostForecastEngine:
    """Deterministic, cost-aware, walk-forward probability forecaster."""

    VERSION = "1.0"

    @staticmethod
    def features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        returns = close.pct_change()
        features = pd.DataFrame(
            {
                "ret_1": returns,
                "ret_3": close.pct_change(3),
                "ret_12": close.pct_change(12),
                "vol_12": returns.rolling(12).std(ddof=0),
                "vol_48": returns.rolling(48).std(ddof=0),
                "ema_gap": close.ewm(span=20, adjust=False).mean() / close.ewm(span=50, adjust=False).mean() - 1,
                "volume_ratio": volume / volume.rolling(30).median().replace(0, np.nan),
            }
        )
        forward_return = close.shift(-1) / close - 1.0
        labels = (forward_return > 0).astype(int)
        valid = features.notna().all(axis=1) & forward_return.notna()
        return features.loc[valid], labels.loc[valid], forward_return.loc[valid]

    def walk_forward(
        self,
        frame: pd.DataFrame,
        *,
        train_bars: int = 200,
        test_bars: int = 30,
        cost_bps: float = 15.0,
    ) -> WalkForwardResult:
        from sklearn.ensemble import GradientBoostingClassifier

        features, labels, forward_returns = self.features(frame)
        if len(features) < train_bars + test_bars:
            raise ValueError("insufficient observations for walk-forward forecast")
        probabilities: list[float] = []
        outcomes: list[int] = []
        strategy_returns: list[float] = []
        windows = 0
        start = train_bars
        while start < len(features):
            stop = min(len(features), start + test_bars)
            x_train = features.iloc[start - train_bars : start]
            y_train = labels.iloc[start - train_bars : start]
            if y_train.nunique() < 2:
                start = stop
                continue
            model = GradientBoostingClassifier(
                n_estimators=60,
                learning_rate=0.04,
                max_depth=2,
                random_state=17,
            )
            model.fit(x_train, y_train)
            probability = model.predict_proba(features.iloc[start:stop])[:, 1]
            actual = labels.iloc[start:stop].to_numpy(dtype=int)
            returns = forward_returns.iloc[start:stop].to_numpy(dtype=float)
            positions = (probability >= 0.55).astype(float)
            turnover = np.abs(np.diff(np.r_[0.0, positions]))
            net = positions * returns - turnover * cost_bps / 10_000
            probabilities.extend(probability.tolist())
            outcomes.extend(actual.tolist())
            strategy_returns.extend(net.tolist())
            windows += 1
            start = stop
        if not probabilities:
            raise ValueError("walk-forward produced no two-class training window")
        probs = np.asarray(probabilities)
        observed = np.asarray(outcomes)
        return WalkForwardResult(
            predictions=tuple(float(value) for value in probabilities),
            outcomes=tuple(int(value) for value in outcomes),
            returns=tuple(float(value) for value in strategy_returns),
            brier_score=float(np.mean((probs - observed) ** 2)),
            accuracy=float(np.mean((probs >= 0.5) == observed)),
            net_return=float(np.sum(strategy_returns)),
            windows=windows,
        )

    def walk_forward_30_7(
        self,
        frame: pd.DataFrame,
        *,
        bars_per_day: int,
        cost_bps: float = 15.0,
    ) -> WalkForwardResult:
        """Run the canonical 30-day training / 7-day causal test schedule."""
        if bars_per_day < 1:
            raise ValueError("bars_per_day must be positive")
        return self.walk_forward(
            frame,
            train_bars=30 * bars_per_day,
            test_bars=7 * bars_per_day,
            cost_bps=cost_bps,
        )

    def health(self) -> dict[str, Any]:
        return {
            "model": "gradient_boosting",
            "walk_forward": True,
            "canonical_schedule_days": [30, 7],
            "cost_aware": True,
            "random_state": 17,
        }


class KronosForecastAdapter:
    """Validated adapter for a supplied Kronos predictor; unavailable is explicit, never fabricated."""

    VERSION = "1.0"

    def __init__(self, predictor: Any | None = None) -> None:
        self.predictor = predictor

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = "NeoQuasar/Kronos-mini",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k",
        max_context: int = 512,
    ) -> KronosForecastAdapter:
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as exc:
            raise RuntimeError("official Kronos repository is not installed in the research environment") from exc
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        return cls(KronosPredictor(model, tokenizer, max_context=max_context))

    def forecast(self, frame: pd.DataFrame, horizon: int = 12) -> dict[str, Any]:
        if self.predictor is None:
            return {"available": False, "reason": "Kronos predictor is not configured"}
        required = ["open", "high", "low", "close"]
        if not set(required).issubset(frame.columns) or len(frame) < 2:
            raise ValueError("Kronos requires at least two OHLC rows")
        x_frame = frame[required + [column for column in ("volume", "amount") if column in frame]].copy()
        if "timestamp" in frame:
            raw_timestamps = frame["timestamp"]
            if pd.api.types.is_numeric_dtype(raw_timestamps):
                x_timestamp = pd.to_datetime(raw_timestamps, unit="ms", utc=True)
            else:
                x_timestamp = pd.to_datetime(raw_timestamps, utc=True)
        else:
            x_timestamp = pd.date_range("2020-01-01", periods=len(frame), freq="15min", tz="UTC")
        deltas = pd.Series(x_timestamp).diff().dropna()
        step = deltas.median() if not deltas.empty else pd.Timedelta(minutes=15)
        y_timestamp = pd.Series([x_timestamp[-1] + step * (index + 1) for index in range(horizon)])
        prediction = self.predictor.predict(
            df=x_frame,
            x_timestamp=pd.Series(x_timestamp),
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )
        if not isinstance(prediction, pd.DataFrame) or "close" not in prediction:
            raise ValueError("Kronos predictor must return a DataFrame with close")
        close_forecast = pd.to_numeric(prediction["close"], errors="coerce").to_numpy(dtype=float)
        if len(close_forecast) != horizon or not np.isfinite(close_forecast).all() or (close_forecast <= 0).any():
            raise ValueError("Kronos predictor returned invalid forecast")
        current_close = float(pd.to_numeric(frame["close"], errors="raise").iloc[-1])
        return {
            "available": True,
            "horizon": horizon,
            "forecast": close_forecast.tolist(),
            "expected_return": float(close_forecast[-1] / current_close - 1.0),
        }

    def health(self) -> dict[str, Any]:
        return {"configured": self.predictor is not None, "execution_authority": False}


class OptunaResearchEngine:
    """Seeded offline parameter search; explicitly unavailable when Optuna is not installed."""

    VERSION = "1.0"

    def optimize(
        self,
        objective: Callable[[dict[str, float]], float],
        search_space: dict[str, tuple[float, float]],
        *,
        trials: int = 50,
        seed: int = 17,
    ) -> dict[str, Any]:
        try:
            import optuna
        except ImportError:
            return {"available": False, "reason": "optuna extra is not installed"}

        def wrapped(trial: Any) -> float:
            params = {
                name: trial.suggest_float(name, float(bounds[0]), float(bounds[1]))
                for name, bounds in search_space.items()
            }
            return float(objective(params))

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(wrapped, n_trials=trials, show_progress_bar=False)
        return {
            "available": True,
            "best_value": float(study.best_value),
            "best_params": {key: float(value) for key, value in study.best_params.items()},
            "trials": len(study.trials),
        }

    def health(self) -> dict[str, Any]:
        try:
            import optuna  # noqa: F401

            available = True
        except ImportError:
            available = False
        return {"available": available, "offline_only": True, "seeded": True}


class QuantumResearchAdapter:
    """Validates and benchmarks an optional quantum optimizer against a classical baseline."""

    VERSION = "1.0"

    def __init__(self, solver: Callable[[np.ndarray, np.ndarray], Any] | None = None) -> None:
        self.solver = solver

    def benchmark(self, expected_returns: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
        mu = np.asarray(expected_returns, dtype=float).reshape(-1)
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (len(mu), len(mu)) or not np.isfinite(mu).all() or not np.isfinite(cov).all():
            raise ValueError("valid expected returns and square covariance required")
        inverse_variance = 1.0 / np.maximum(np.diag(cov), 1e-12)
        classical = inverse_variance / inverse_variance.sum()
        classical_objective = float(mu @ classical - 0.5 * classical @ cov @ classical)
        if self.solver is None:
            return {
                "available": False,
                "classical_weights": classical.tolist(),
                "classical_objective": classical_objective,
                "reason": "quantum solver is not configured",
            }
        candidate = np.asarray(self.solver(mu, cov), dtype=float).reshape(-1)
        if len(candidate) != len(mu) or not np.isfinite(candidate).all() or (candidate < 0).any():
            raise ValueError("quantum solver returned invalid weights")
        total = float(candidate.sum())
        if total <= 0:
            raise ValueError("quantum weights must have positive sum")
        candidate /= total
        candidate_objective = float(mu @ candidate - 0.5 * candidate @ cov @ candidate)
        return {
            "available": True,
            "classical_weights": classical.tolist(),
            "candidate_weights": candidate.tolist(),
            "classical_objective": classical_objective,
            "candidate_objective": candidate_objective,
            "candidate_improvement": candidate_objective - classical_objective,
        }

    def health(self) -> dict[str, Any]:
        return {"configured": self.solver is not None, "benchmark_only": True, "execution_authority": False}


class CalibrationEngine:
    """Brier score and expected calibration error for forecast honesty."""

    VERSION = "1.0"

    def evaluate(self, probabilities: list[float], outcomes: list[int], bins: int = 10) -> dict[str, float]:
        if len(probabilities) != len(outcomes) or not probabilities:
            raise ValueError("aligned non-empty probabilities and outcomes required")
        probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        actual = np.asarray(outcomes, dtype=float)
        brier = float(np.mean((probs - actual) ** 2))
        ece = 0.0
        for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
            upper = lower + 1.0 / bins
            mask = (probs >= lower) & (probs < upper if upper < 1 else probs <= upper)
            if mask.any():
                ece += float(mask.mean()) * abs(float(probs[mask].mean() - actual[mask].mean()))
        return {"brier_score": brier, "expected_calibration_error": ece, "samples": float(len(probs))}

    def health(self) -> dict[str, Any]:
        return {"brier": True, "expected_calibration_error": True}


class DriftEngine:
    """Feature-distribution drift using standardized mean and variance shifts."""

    VERSION = "1.0"

    def compare(self, reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 1.0) -> dict[str, Any]:
        shared = sorted(set(reference.columns) & set(current.columns))
        shifts: dict[str, float] = {}
        for column in shared:
            ref = pd.to_numeric(reference[column], errors="coerce").dropna()
            cur = pd.to_numeric(current[column], errors="coerce").dropna()
            if len(ref) < 10 or len(cur) < 10:
                continue
            scale = max(float(ref.std(ddof=0)), 1e-12)
            mean_shift = abs(float(cur.mean() - ref.mean())) / scale
            variance_shift = abs(math.log(max(float(cur.std(ddof=0)), 1e-12) / scale))
            shifts[column] = mean_shift + 0.5 * variance_shift
        maximum = max(shifts.values(), default=0.0)
        return {"drifted": maximum >= threshold, "maximum_shift": maximum, "feature_shifts": shifts}

    def health(self) -> dict[str, Any]:
        return {"distribution_drift": True, "automatic_promotion": False}


class ChampionChallengerGovernor:
    """Evidence-gated strategy promotion and reversible champion state."""

    VERSION = "1.0"

    def __init__(self, state_path: Path, champion: str = "adaptive_ensemble") -> None:
        self.state_path = state_path
        self.state = self._load() or {
            "champion": champion,
            "previous_champion": None,
            "records": {},
            "promotions": [],
        }

    def record(self, strategy: str, net_return: float, max_drawdown: float, brier_score: float) -> None:
        records = self.state["records"].setdefault(strategy, [])
        records.append(
            {
                "net_return": float(net_return),
                "max_drawdown": float(max_drawdown),
                "brier_score": float(brier_score),
            }
        )
        self.state["records"][strategy] = records[-100:]
        self._save()

    def consider(
        self,
        challenger: str,
        *,
        minimum_windows: int = 5,
        minimum_return_advantage: float = 0.002,
        maximum_drawdown: float = 0.10,
        maximum_brier: float = 0.25,
    ) -> bool:
        champion = str(self.state["champion"])
        challenger_rows = self.state["records"].get(challenger, [])[-minimum_windows:]
        champion_rows = self.state["records"].get(champion, [])[-minimum_windows:]
        if len(challenger_rows) < minimum_windows or len(champion_rows) < minimum_windows:
            return False
        challenger_return = float(np.mean([row["net_return"] for row in challenger_rows]))
        champion_return = float(np.mean([row["net_return"] for row in champion_rows]))
        safe = max(row["max_drawdown"] for row in challenger_rows) <= maximum_drawdown
        calibrated = float(np.mean([row["brier_score"] for row in challenger_rows])) <= maximum_brier
        if challenger_return < champion_return + minimum_return_advantage or not safe or not calibrated:
            return False
        self.state["previous_champion"] = champion
        self.state["champion"] = challenger
        self.state["promotions"].append({"from": champion, "to": challenger})
        self._save()
        return True

    def rollback(self, reason: str) -> bool:
        previous = self.state.get("previous_champion")
        if not previous:
            return False
        current = self.state["champion"]
        self.state["champion"] = previous
        self.state["previous_champion"] = current
        self.state["promotions"].append({"from": current, "to": previous, "rollback_reason": reason})
        self._save()
        return True

    def _load(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def health(self) -> dict[str, Any]:
        return {
            "champion": self.state["champion"],
            "previous_champion": self.state.get("previous_champion"),
            "promotion_count": len(self.state["promotions"]),
            "reversible": True,
        }


class CapitalPreservationEngine:
    """Hysteretic normal/defensive/recovery/halt state machine."""

    VERSION = "1.0"

    def __init__(self) -> None:
        self.state = "normal"
        self.healthy_cycles = 0

    def update(
        self,
        *,
        drawdown: float,
        daily_loss: float,
        data_healthy: bool,
        required_engines_healthy: bool,
    ) -> dict[str, Any]:
        if not data_healthy or not required_engines_healthy or drawdown >= 0.10 or daily_loss >= 0.03:
            self.state = "halt"
            self.healthy_cycles = 0
        elif drawdown >= 0.06 or daily_loss >= 0.015:
            self.state = "defensive"
            self.healthy_cycles = 0
        elif self.state in {"halt", "defensive", "recovery"}:
            self.healthy_cycles += 1
            self.state = "normal" if self.healthy_cycles >= 10 else "recovery"
        else:
            self.state = "normal"
            self.healthy_cycles += 1
        multiplier = {"normal": 1.0, "defensive": 0.35, "recovery": 0.15, "halt": 0.0}[self.state]
        return {"state": self.state, "size_multiplier": multiplier, "healthy_cycles": self.healthy_cycles}

    def health(self) -> dict[str, Any]:
        return {"state": self.state, "hysteresis_cycles": 10}


class StressEngine:
    """Deterministic portfolio shocks for capital-at-risk inspection."""

    VERSION = "1.0"

    def evaluate(self, notionals: dict[str, float]) -> dict[str, Any]:
        scenarios = {
            "crypto_crash_10pct": {symbol: -0.10 for symbol in notionals},
            "flash_crash_25pct": {symbol: -0.25 for symbol in notionals},
            "liquidity_gap_5pct": {symbol: -0.05 for symbol in notionals},
        }
        losses = {
            name: float(sum(notionals[symbol] * shocks[symbol] for symbol in notionals))
            for name, shocks in scenarios.items()
        }
        return {"scenario_pnl": losses, "worst_case_pnl": min(losses.values(), default=0.0)}

    def health(self) -> dict[str, Any]:
        return {"deterministic_scenarios": 3, "random_monte_carlo": False}


class ResearchEngineSuite:
    """Research, evolution, rollback, drift, and capital-preservation control plane."""

    VERSION = "2.0"

    def __init__(self, governor_path: Path) -> None:
        self.replay = ReplayEngine()
        self.gradient_boost = GradientBoostForecastEngine()
        self.kronos = KronosForecastAdapter()
        self.optuna = OptunaResearchEngine()
        self.quantum = QuantumResearchAdapter()
        self.calibration = CalibrationEngine()
        self.drift = DriftEngine()
        self.champion_challenger = ChampionChallengerGovernor(governor_path)
        self.capital_preservation = CapitalPreservationEngine()
        self.stress = StressEngine()

    def runtime_snapshot(
        self,
        *,
        notionals: dict[str, float],
        drawdown: float,
        daily_loss: float,
        data_healthy: bool,
        required_engines_healthy: bool,
    ) -> dict[str, Any]:
        return {
            "capital_preservation": self.capital_preservation.update(
                drawdown=drawdown,
                daily_loss=daily_loss,
                data_healthy=data_healthy,
                required_engines_healthy=required_engines_healthy,
            ),
            "stress": self.stress.evaluate(notionals),
            "champion_challenger": self.champion_challenger.health(),
        }

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "replay": self.replay.health(),
            "walk_forward_gradient_boost": self.gradient_boost.health(),
            "kronos": self.kronos.health(),
            "optuna": self.optuna.health(),
            "ibm_quantum_benchmark": self.quantum.health(),
            "calibration": self.calibration.health(),
            "drift": self.drift.health(),
            "champion_challenger": self.champion_challenger.health(),
            "capital_preservation": self.capital_preservation.health(),
            "stress": self.stress.health(),
            "live_model_promotion": False,
        }
