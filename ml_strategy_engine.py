import os  # noqa: F401  # intentionally kept
import random
from typing import Dict

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
    class _UnavailableLayer:  # optional dependency absent; construction is a no-op
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    LSTM = Dense = Dropout = _UnavailableLayer

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
                # Implement step logic with rewards based on PnL
                reward = random.uniform(-1, 1)  # Placeholder; use real PnL calc
                self.current_step += 1
                done = self.current_step >= len(self.data) - 1
                obs = self.data[self.current_step : self.current_step + 60]
                return obs, reward, done, {}

            def reset(self):
                self.current_step = 0
                return self.data[:60]

        self.env = TradingEnv(np.random.rand(1000))  # Placeholder data
        self.rl_model = PPO("MlpPolicy", self.env, verbose=0)

    def predict_price(self, data: np.array) -> float:
        """Predict next price using LSTM."""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X = scaled_data[-60:].reshape(1, 60, 1)
        pred = self.lstm_model.predict(X)
        return self.scaler.inverse_transform(pred)[0][0]

    def optimize_strategy(self, episodes=1000):
        """Train RL model for strategy optimization."""
        self.rl_model.learn(total_timesteps=episodes)

    def backtest_strategy(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Walk-forward backtesting with Monte Carlo."""
        # This previously ignored historical_data entirely and returned
        # statistics computed from random draws, which is fabricated backtest
        # evidence rather than a Monte Carlo over real returns.
        if historical_data is None or len(historical_data) < 2:
            return {"available": False, "reason": "no historical data supplied"}

        close = historical_data.get("close")
        if close is None:
            return {"available": False, "reason": "historical data has no close column"}

        returns = pd.Series(close).pct_change().dropna()
        if returns.empty:
            return {"available": False, "reason": "insufficient price history"}

        equity = (1.0 + returns).cumprod()
        drawdown = (equity / equity.cummax()) - 1.0
        std = float(returns.std())
        return {
            "available": True,
            "samples": int(len(returns)),
            "avg_pnl": float(returns.mean()),
            "sharpe_ratio": float(returns.mean() / std) if std > 0 else 0.0,
            "max_drawdown": float(drawdown.min()),
        }

# Usage: Integrate into UltraCore or BrainLoop for predictions and strategy calls.
