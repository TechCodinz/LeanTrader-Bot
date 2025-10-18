"""Analytics utilities for performance reporting.

Reads the trading `ledger.csv` and computes robust metrics:
  - cumulative return, CAGR
  - Sharpe, Sortino
  - Max drawdown, Calmar
  - Win rate, average win/loss, profit factor
  - Per-symbol breakdown
  - Equity curve series

Usage (programmatic):
  from reporting.analytics import load_ledger, compute_report
  df = load_ledger()
  report = compute_report(df)

This module is pure-CPU and has no network calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


LEDGER_PATH = Path("data/ledger.csv")


def load_ledger(path: Path | str = LEDGER_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[
            "ts_entry",
            "date",
            "venue",
            "market",
            "symbol",
            "tf",
            "side",
            "entry_px",
            "qty",
            "sl",
            "tp",
            "meta",
            "ts_exit",
            "exit_px",
            "pnl_raw",
            "pnl_r",
            "hold_min",
            "status",
        ])
    df = pd.read_csv(p)
    # coerce dtypes
    for c in ("pnl_raw", "pnl_r", "qty", "entry_px", "exit_px"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    return df


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = -np.inf
    dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = max(dd, (peak - v) / peak if peak > 0 else 0.0)
    return float(dd)


def equity_curve_from_trades(df: pd.DataFrame, start_equity: float = 10000.0) -> pd.Series:
    """Construct a naive equity curve by summing `pnl_raw` over time.

    Assumes `pnl_raw` is quoted in USD (or a consistent quote asset).
    """
    if df.empty:
        return pd.Series([start_equity])
    dff = df.copy()
    dff = dff.sort_values(["date", "ts_exit"], na_position="last")
    pnl = dff.get("pnl_raw", pd.Series([0.0] * len(dff))).fillna(0.0).to_numpy(dtype=float)
    eq = start_equity + np.cumsum(pnl)
    return pd.Series(eq)


@dataclass
class Summary:
    total_pnl: float
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe: float
    sortino: float
    mdd: float
    calmar: float
    cagr: float


def compute_summary(df: pd.DataFrame, start_equity: float = 10000.0, periods_per_year: int = 252) -> Summary:
    if df.empty:
        return Summary(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # basic trade stats
    closed = df[df.get("status", "open") != "open"].copy()
    pnls = closed.get("pnl_raw", pd.Series([0.0] * len(closed))).fillna(0.0).to_numpy(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / max(1, len(pnls)))
    avg_win = _safe_mean(wins)
    avg_loss = abs(_safe_mean(losses))
    profit_factor = float((wins.sum() / max(1e-9, abs(losses.sum())))) if losses.size else float("inf")

    # daily (or per-trade) return proxy from equity curve
    eq = equity_curve_from_trades(closed, start_equity=start_equity).to_numpy(dtype=float)
    if eq.size <= 1:
        return Summary(float(pnls.sum()), int(len(pnls)), win_rate, avg_win, avg_loss, profit_factor, 0.0, 0.0, 0.0, 0.0, 0.0)
    rets = np.diff(eq) / np.clip(eq[:-1], 1e-9, None)
    mu = _safe_mean(rets)
    sd = float(np.std(rets)) if rets.size else 0.0
    downside = float(np.std(np.clip(rets, a_max=0.0, a_min=None))) if rets.size else 0.0
    sharpe = float(mu / sd * math.sqrt(periods_per_year)) if sd > 1e-12 else 0.0
    sortino = float(mu / downside * math.sqrt(periods_per_year)) if downside > 1e-12 else 0.0
    mdd = _max_drawdown(eq)
    years = max(1e-9, rets.size / periods_per_year)
    cagr = float((eq[-1] / max(1e-9, eq[0])) ** (1 / years) - 1.0) if eq[0] > 0 else 0.0
    calmar = float(cagr / mdd) if mdd > 1e-12 else 0.0

    return Summary(float(pnls.sum()), int(len(pnls)), win_rate, avg_win, avg_loss, profit_factor, sharpe, sortino, mdd, calmar, cagr)


def per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "trades", "win_rate", "sum_pnl"])
    dff = df.copy()
    dff["pnl_raw"] = pd.to_numeric(dff.get("pnl_raw", 0.0), errors="coerce").fillna(0.0)
    grp = dff.groupby("symbol", dropna=False)["pnl_raw"]
    out = pd.DataFrame({
        "trades": grp.count(),
        "win_rate": dff.assign(win=dff["pnl_raw"] > 0).groupby("symbol")["win"].mean(),
        "sum_pnl": grp.sum(),
    }).reset_index()
    return out.sort_values("sum_pnl", ascending=False)


def compute_report(df: pd.DataFrame, start_equity: float = 10000.0) -> Dict[str, object]:
    return {
        "summary": compute_summary(df, start_equity=start_equity).__dict__,
        "per_symbol": per_symbol(df).to_dict(orient="records"),
    }


def print_report(df: pd.DataFrame, start_equity: float = 10000.0) -> str:
    s = compute_summary(df, start_equity=start_equity)
    lines = [
        "=== Performance Summary ===",
        f"Total PnL: {s.total_pnl:.2f}",
        f"Trades: {s.trades} | Win rate: {s.win_rate*100:.1f}%",
        f"Avg win: {s.avg_win:.2f} | Avg loss: {s.avg_loss:.2f} | PF: {s.profit_factor:.2f}",
        f"Sharpe: {s.sharpe:.2f} | Sortino: {s.sortino:.2f}",
        f"Max DD: {s.mdd*100:.2f}% | CAGR: {s.cagr*100:.2f}% | Calmar: {s.calmar:.2f}",
        "",
        "Top symbols:",
    ]
    sym = per_symbol(df).head(10)
    for _, r in sym.iterrows():
        lines.append(f"{r['symbol']}: trades={int(r['trades'])} win={float(r['win_rate'])*100:.1f}% pnl={float(r['sum_pnl']):.2f}")
    return "\n".join(lines)


