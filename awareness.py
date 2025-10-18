"""Lightweight situational awareness for tests.

Provides minimal `AwarenessConfig` and `SituationalAwareness` to support
regime inference and decision gating used by tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional  # noqa: F401

import pandas as pd


@dataclass
class AwarenessConfig:
    atr_window: int = 14
    calm_vol_thresh: float = 0.01
    spike_mult: float = 4.0
    cooldown_bars: int = 0
    dd_limit_pct: float = 0.10  # block when drawdown from equity high >= this


@dataclass
class Decision:
    allow: bool
    reason: str
    size_frac: float
    stop_atr: float
    take_atr: float


class SituationalAwareness:
    def __init__(self, cfg: AwarenessConfig) -> None:
        self.cfg = cfg
        self._cooldown_left: int = 0
        self._equity_high: float = 0.0

    def set_cooldown(self, bars: int) -> None:
        self._cooldown_left = max(0, int(bars))

    # --- indicators ---
    def atr(self, close: pd.Series, high: pd.Series, low: pd.Series, n: int = 14) -> pd.Series:
        tr_hl = (high - low).abs()
        tr_hc = (high - close.shift()).abs()
        tr_lc = (low - close.shift()).abs()
        tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    def regime(self, close: pd.Series) -> str:
        if len(close) < 20:
            return "range"
        rets = close.pct_change().fillna(0.0)
        vol = rets.rolling(20).std().iloc[-1]
        slope = (close.iloc[-1] - close.iloc[-20]) / max(abs(close.iloc[-20]), 1e-9)
        # detect spike on last bar
        if abs(rets.iloc[-1]) > self.cfg.spike_mult * max(vol, 1e-9):
            return "spike"
        if slope > 0.0005:
            return "trend_up"
        if slope < -0.0005:
            return "trend_down"
        return "range"

    def decide(
        self,
        df: pd.DataFrame,
        equity: float,
        base_conf: float,
        win_rate: float,
        payoff: float,
        high_impact_event_soon: bool = False,
    ) -> Decision:
        # cooldown gate
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return Decision(False, "cooldown", 0.0, 0.0, 0.0)

        # circuit breaker on drawdown (simple)
        if self._equity_high <= 0:
            self._equity_high = equity
        dd = (self._equity_high - equity) / max(self._equity_high, 1e-9)
        if dd >= self.cfg.dd_limit_pct:
            return Decision(False, "circuit_breaker_dd", 0.0, 0.0, 0.0)
        self._equity_high = max(self._equity_high, equity)

        # compute ATR and regime
        atr = self.atr(df["close"], df["high"], df["low"], n=self.cfg.atr_window).iloc[-1]
        rg = self.regime(df["close"])  # not used directly in decision score

        # simple Kelly-like size
        kelly = max(0.0, min(0.25, win_rate * payoff - (1 - win_rate)))
        size = float(base_conf) * kelly

        if high_impact_event_soon:
            size = 0.0

        allow = size > 0.0
        reason = "ok_calm" if allow else ("no_size_spike" if rg == "spike" else "no_size")
        return Decision(allow, reason, size, float(atr or 0.0), float(atr or 0.0))


__all__ = ["AwarenessConfig", "SituationalAwareness", "Decision"]
