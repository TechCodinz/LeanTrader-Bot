import os  # noqa: F401  # intentionally kept
from typing import Dict

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
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
        """Build the PPO environment without fabricated market experience."""

        if gym is None:
            raise RuntimeError(
                "Gymnasium is required for the RL strategy engine"
            )

        class TradingEnv(gym.Env):
            """
            Trading environment backed only by an actual supplied price series.

            No random observations and no random rewards are permitted.
            Reward is the realized one-step market return multiplied by
            the selected directional exposure.
            """

            metadata = {"render_modes": []}

            def __init__(self, window=60):
                super().__init__()

                self.window = int(window)
                self.prices = np.asarray([], dtype=np.float64)
                self.current_step = self.window

                self.action_space = gym.spaces.Discrete(3)

                self.observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window,),
                    dtype=np.float32,
                )

            @property
            def has_real_data(self):
                return (
                    self.prices.ndim == 1
                    and len(self.prices) > self.window + 1
                    and np.all(np.isfinite(self.prices))
                    and np.all(self.prices > 0)
                )

            def set_prices(self, prices):
                values = np.asarray(
                    prices,
                    dtype=np.float64,
                ).reshape(-1)

                values = values[
                    np.isfinite(values)
                    & (values > 0)
                ]

                if len(values) <= self.window + 1:
                    raise ValueError(
                        "real market series must contain "
                        f"more than {self.window + 1} prices"
                    )

                self.prices = values
                self.current_step = self.window

            def _observation(self):
                if not self.has_real_data:
                    raise RuntimeError(
                        "real_market_data_required"
                    )

                start = self.current_step - self.window
                end = self.current_step + 1

                window_prices = self.prices[start:end]

                returns = np.diff(window_prices) / window_prices[:-1]

                return returns.astype(
                    np.float32
                )

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)

                if not self.has_real_data:
                    raise RuntimeError(
                        "real_market_data_required"
                    )

                self.current_step = self.window

                return self._observation(), {
                    "source": "real_market_series"
                }

            def step(self, action):
                if not self.has_real_data:
                    raise RuntimeError(
                        "real_market_data_required"
                    )

                if self.current_step >= len(self.prices) - 1:
                    return (
                        self._observation(),
                        0.0,
                        True,
                        False,
                        {
                            "source": "real_market_series",
                            "reason": "end_of_series",
                        },
                    )

                current_price = float(
                    self.prices[self.current_step]
                )

                next_price = float(
                    self.prices[self.current_step + 1]
                )

                market_return = (
                    next_price / current_price
                ) - 1.0

                # 0 = short, 1 = flat, 2 = long
                exposure = {
                    0: -1.0,
                    1: 0.0,
                    2: 1.0,
                }[int(action)]

                reward = float(
                    exposure * market_return
                )

                self.current_step += 1

                terminated = (
                    self.current_step >= len(self.prices) - 1
                )

                observation = self._observation()

                return (
                    observation,
                    reward,
                    terminated,
                    False,
                    {
                        "source": "real_market_series",
                        "market_return": market_return,
                        "exposure": exposure,
                        "price": current_price,
                        "next_price": next_price,
                    },
                )

        self.env = TradingEnv(window=60)

        self.rl_model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
        )

        self.rl_data_source = None

        # Bootstrap only from a real public exchange.
        # Failure does not invent substitute observations.
        try:
            self.bootstrap_real_market_data()
        except Exception as exc:
            self.rl_bootstrap_error = str(exc)

    def bootstrap_real_market_data(
        self,
        symbol=None,
        timeframe=None,
        limit=None,
        exchange_id=None,
    ):
        """Populate the RL environment from actual public OHLCV."""

        import ccxt

        try:
            from ccxt_exchange_compat import (
                resolve_exchange_class,
            )
        except Exception:
            resolve_exchange_class = None

        symbol = (
            symbol
            or os.getenv(
                "ML_RL_SYMBOL",
                "BTC/USDT",
            )
        )

        timeframe = (
            timeframe
            or os.getenv(
                "ML_RL_TIMEFRAME",
                "1m",
            )
        )

        limit = int(
            limit
            or os.getenv(
                "ML_RL_BOOTSTRAP_LIMIT",
                "500",
            )
        )

        preferred = (
            exchange_id
            or os.getenv(
                "ML_RL_EXCHANGE",
                "bybit",
            )
        )

        candidates = []

        for value in (
            preferred,
            "bybit",
            "binance",
            "okx",
        ):
            if value not in candidates:
                candidates.append(value)

        failures = []

        for candidate in candidates:
            try:
                if resolve_exchange_class:
                    cls = resolve_exchange_class(
                        ccxt,
                        candidate,
                    )
                else:
                    cls = getattr(
                        ccxt,
                        candidate,
                    )

                ex = cls({
                    "enableRateLimit": True,
                    "timeout": 20000,
                })

                bars = ex.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                )

                prices = [
                    float(row[4])
                    for row in bars
                    if (
                        isinstance(row, (list, tuple))
                        and len(row) >= 5
                        and row[4] is not None
                    )
                ]

                if len(prices) <= 61:
                    raise RuntimeError(
                        f"insufficient real OHLCV: {len(prices)}"
                    )

                self.env.set_prices(
                    prices
                )

                self.rl_data_source = {
                    "exchange": candidate,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "samples": len(prices),
                    "kind": "public_ohlcv",
                }

                self.rl_bootstrap_error = None

                return self.rl_data_source

            except Exception as exc:
                failures.append(
                    f"{candidate}:{type(exc).__name__}:{exc}"
                )

        raise RuntimeError(
            "real_market_data_bootstrap_failed: "
            + " | ".join(failures)
        )

    def set_real_market_data(self, prices, source="external_real_series"):
        """Inject an already-verified real market series."""

        self.env.set_prices(
            prices
        )

        self.rl_data_source = {
            "kind": source,
            "samples": int(
                len(self.env.prices)
            ),
        }

        self.rl_bootstrap_error = None

        return self.rl_data_source

    def predict_price(self, data: np.array) -> float:
        """Predict using supplied price history; never invent training data."""

        values = np.asarray(
            data,
            dtype=np.float64,
        ).reshape(-1)

        values = values[
            np.isfinite(values)
        ]

        if len(values) < 61:
            raise ValueError(
                "at least 61 real observations are required"
            )

        scaled_data = self.scaler.fit_transform(
            values.reshape(-1, 1)
        )

        X = []
        y = []

        for index in range(
            60,
            len(scaled_data),
        ):
            X.append(
                scaled_data[
                    index - 60:index,
                    0,
                ]
            )

            y.append(
                scaled_data[
                    index,
                    0,
                ]
            )

        X = np.asarray(
            X,
            dtype=np.float32,
        ).reshape(-1, 60, 1)

        y = np.asarray(
            y,
            dtype=np.float32,
        )

        if len(X) == 0:
            raise ValueError(
                "insufficient real observations for LSTM"
            )

        # Train against the actual series supplied to this prediction.
        self.lstm_model.fit(
            X,
            y,
            epochs=max(
                1,
                int(
                    os.getenv(
                        "ML_LSTM_EPOCHS",
                        "1",
                    )
                ),
            ),
            batch_size=min(
                32,
                max(1, len(X)),
            ),
            verbose=0,
        )

        pred = self.lstm_model.predict(
            X[-1:],
            verbose=0,
        )

        return float(
            self.scaler.inverse_transform(
                pred
            )[0][0]
        )

    def optimize_strategy(self, episodes=1000):
        """Train PPO only when an actual market series is attached."""

        if (
            self.env is None
            or not getattr(
                self.env,
                "has_real_data",
                False,
            )
        ):
            self.bootstrap_real_market_data()

        if not self.env.has_real_data:
            raise RuntimeError(
                "real_market_data_required"
            )

        self.rl_model.learn(
            total_timesteps=int(
                episodes
            )
        )

        return {
            "trained": True,
            "timesteps": int(episodes),
            "data_source": self.rl_data_source,
        }

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
