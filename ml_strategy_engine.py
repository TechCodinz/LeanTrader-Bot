import os  # noqa: F401  # intentionally kept
import random
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import gym  # optional
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:
    from sklearn.preprocessing import MinMaxScaler  # type: ignore
except Exception:  # pragma: no cover
    class MinMaxScaler:  # type: ignore
        def fit_transform(self, x):
            return x

        def inverse_transform(self, x):
            return x

try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
except Exception:  # pragma: no cover
    # These are constructed with arguments (LSTM(100, return_sequences=True,
    # input_shape=(60, 1))), and bare `object` takes none -- so without
    # TensorFlow installed the engine raised TypeError before it could be
    # built at all. The stubs now absorb their arguments like the Sequential
    # and PPO stubs below already do.
    class _KerasLayerStub:  # type: ignore
        def __init__(self, *_a, **_k):
            pass

    LSTM = Dense = Dropout = _KerasLayerStub  # type: ignore

    class Sequential:  # type: ignore
        def __init__(self, *_a, **_k):
            pass

        def compile(self, **_k):
            pass

        def predict(self, x):
            return x

try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover
    class PPO:  # type: ignore
        def __init__(self, *_a, **_k):
            pass

        def learn(self, **_k):
            pass

class MLStrategyEngine:
    def __init__(self):
        self.lstm_model = None
        self.rl_model = None
        self.scaler = MinMaxScaler()
        self.build_lstm()
        self.build_rl_env()

    def build_lstm(self):
        """Build LSTM model for time-series prediction."""
        self.lstm_model = Sequential(
            [
                LSTM(100, return_sequences=True, input_shape=(60, 1)),
                Dropout(0.2),
                LSTM(100, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ]
        )
        self.lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    def build_rl_env(self):
        """Custom RL environment for trading."""

        if gym is None:
            self.env = None
            self.rl_model = PPO("MlpPolicy", None, verbose=0)  # type: ignore
            return

        class TradingEnv(gym.Env):  # type: ignore
            def __init__(self, data):
                super().__init__()
                self.data = data
                self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
                self.observation_space = gym.spaces.Box(low=0, high=1, shape=(60,))
                self.current_step = 0

            def step(self, action):
                # Reward is the PnL the action would actually have produced on
                # the next bar: long gains when price rises, short when it
                # falls, flat earns nothing. This was random.uniform(-1, 1),
                # which trained the policy against noise.
                step_index = self.current_step
                self.current_step += 1
                done = self.current_step >= len(self.data) - 1

                reward = 0.0
                try:
                    price_now = float(self.data[step_index])
                    price_next = float(self.data[min(self.current_step, len(self.data) - 1)])
                    if price_now > 0:
                        move = (price_next - price_now) / price_now
                        # 0 = buy/long, 1 = sell/short, 2 = hold
                        if action == 0:
                            reward = move
                        elif action == 1:
                            reward = -move
                except (TypeError, ValueError, IndexError):
                    reward = 0.0

                obs = self.data[self.current_step : self.current_step + 60]
                return obs, reward, done, {}

            def reset(self):
                self.current_step = 0
                return self.data[:60]

        # Real price series when the caller supplies one. Training against
        # np.random.rand produced a policy fitted to noise.
        series = getattr(self, "price_series", None)
        if series is None or len(series) < 120:
            self.env = None
            self.rl_model = None
            self.rl_available = False
            self.rl_reason = "no_real_price_series"
            return

        self.env = TradingEnv(np.asarray(series, dtype=float))
        self.rl_model = PPO("MlpPolicy", self.env, verbose=0)
        self.rl_available = True
        self.rl_reason = "ready"

    def predict_price(self, data: np.array) -> float:
        """Predict next price using LSTM."""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X = scaled_data[-60:].reshape(1, 60, 1)
        pred = self.lstm_model.predict(X)
        return self.scaler.inverse_transform(pred)[0][0]

    def optimize_strategy(self, episodes=1000):
        """Train RL model for strategy optimization."""
        self.rl_model.learn(total_timesteps=episodes)

    def backtest_strategy(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Walk-forward statistics from the data actually supplied.

        The previous implementation drew 100 samples from
        random.uniform(-0.1, 0.1) and ignored ``historical_data`` entirely, so
        its Sharpe ratio and drawdown described a random walk rather than the
        series it was handed.
        """
        closes = None
        if historical_data is not None:
            for column in ("close", "Close", "c"):
                if hasattr(historical_data, "columns") and column in historical_data.columns:
                    closes = historical_data[column].astype(float).tolist()
                    break
            if closes is None and hasattr(historical_data, "tolist"):
                try:
                    closes = [float(v) for v in historical_data.tolist()]
                except (TypeError, ValueError):
                    closes = None

        if not closes or len(closes) < 3:
            return {
                "available": False,
                "reason": "insufficient_historical_data",
                "avg_pnl": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
            }

        results = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]
        if not results:
            return {
                "available": False,
                "reason": "no_usable_returns",
                "avg_pnl": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
            }

        equity, peak, max_drawdown = 1.0, 1.0, 0.0
        for r in results:
            equity *= 1.0 + r
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, (equity - peak) / peak if peak > 0 else 0.0)

        deviation = float(np.std(results))
        return {
            "available": True,
            "samples": len(results),
            "avg_pnl": float(np.mean(results)),
            "sharpe_ratio": float(np.mean(results) / deviation) if deviation > 0 else 0.0,
            "max_drawdown": float(max_drawdown),
            "total_return": float(equity - 1.0),
        }

# Usage: Integrate into UltraCore or BrainLoop for predictions and strategy calls.
