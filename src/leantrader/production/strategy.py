from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Decision:
    close: float
    atr: float
    enter_long: bool
    trend_up: bool


def decide(frame: pd.DataFrame) -> Decision:
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns) or len(frame) < 220:
        raise ValueError("at least 220 OHLC candles are required")

    data = frame.copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()

    moving_average = close.rolling(20).mean()
    deviation = close.rolling(20).std(ddof=0)
    upper = moving_average + 2.0 * deviation
    bandwidth = (4.0 * deviation) / moving_average.replace(0, float("nan"))
    threshold = bandwidth.rolling(120).quantile(0.5)

    previous_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(
        axis=1
    )
    atr = true_range.rolling(14).mean()

    trend_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
    was_squeezed = bool(bandwidth.iloc[-2] <= threshold.iloc[-2])
    breakout = bool(close.iloc[-1] > upper.iloc[-2])
    return Decision(
        close=float(close.iloc[-1]),
        atr=float(atr.iloc[-1]),
        enter_long=trend_up and was_squeezed and breakout,
        trend_up=trend_up,
    )
