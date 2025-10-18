"""
Ultra Advanced Features (production-safe)
- Converts research-class detectors into deterministic, testable features.
- No randomness; no network calls; CPU-light; async-safe (pure numpy/pandas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

# ---------- helpers

def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x)) if len(x) else 0.0
    return s if s > 1e-12 else 1e-12


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


# ---------- microstructure (deterministic, lightweight)

def features_microstructure(order_book: Dict, topn: int = 10) -> Dict[str, float]:
    """
    Expect order_book dict with 'bids' and 'asks' as lists of [price, size].
    Returns stable microstructure signals (no DBSCAN randomness).
    """
    bids = order_book.get("bids", [])[:topn]
    asks = order_book.get("asks", [])[:topn]
    if not bids or not asks:
        return {
            "ob_spread_bps": 9999.0,
            "ob_imbalance": 0.0,
            "ob_depth_ratio": 0.0,
            "ob_wall_score": 0.0,
        }

    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4

    bid_sz = float(sum(s for _, s in bids))
    ask_sz = float(sum(s for _, s in asks))
    imb = (bid_sz - ask_sz) / max(bid_sz + ask_sz, 1e-9)

    # “wall” = any level whose size is > 4x median of top N
    all_sz = np.array([s for _, s in bids + asks], dtype=float)
    med = float(np.median(all_sz)) if all_sz.size else 0.0
    wall_score = float((all_sz > 4.0 * max(med, 1e-9)).mean())  # 0..1 fraction

    # depth ratio = top5 / top10 total
    def depth_ratio(side):
        lvls = np.array([s for _, s in side], dtype=float)
        top5 = lvls[:5].sum() if lvls.size >= 5 else lvls.sum()
        top10 = lvls[:10].sum() if lvls.size else 1.0
        return float(top5 / max(top10, 1e-9))

    return {
        "ob_spread_bps": float(spread_bps),
        "ob_imbalance": float(imb),
        "ob_depth_ratio": float((depth_ratio(bids) + depth_ratio(asks)) / 2.0),
        "ob_wall_score": wall_score,
    }


# ---------- spectral “quantum” momentum (deterministic)

def features_spectral(df: pd.DataFrame, lookback: int = 128) -> Dict[str, float]:
    close = df["close"].to_numpy()
    if close.size < lookback:
        return {"spec_energy": 0.0, "spec_tilt": 0.0, "spec_entropy": 0.0}

    x = (close[-lookback:] - close[-lookback:].mean()) / _safe_std(close[-lookback:])
    spec = np.abs(np.fft.rfft(x))
    prob = spec / max(spec.sum(), 1e-12)

    energy = float((spec ** 2).sum())
    tilt = float((np.arange(prob.size) * prob).sum() / max(prob.size - 1, 1))
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(prob.size))

    return {"spec_energy": energy, "spec_tilt": tilt, "spec_entropy": entropy}


# ---------- fractal resonance (lightweight approximation)

def features_fractal(df: pd.DataFrame) -> Dict[str, float]:
    close = df["close"].to_numpy()
    if close.size < 60:
        return {"fractal_dim": 1.5, "fractal_trend": 0.0}

    # simple “dimension” proxy: ratio of cumulative absolute returns to net move
    ret = np.diff(close) / close[:-1]
    cum_abs = float(np.abs(ret[-100:]).sum()) if ret.size >= 100 else float(np.abs(ret).sum())
    net = float(
        abs(close[-1] - close[max(len(close) - 101, 0)]) / max(close[-101], 1e-9)
        if close.size > 101
        else abs(close[-1] - close[0]) / max(close[0], 1e-9)
    )
    fractal_dim = float(np.clip((cum_abs / max(net, 1e-6)), 1.0, 2.0))  # 1..2

    # trend via EMA slope
    ema_fast, ema_slow = _ema(close, 12), _ema(close, 26)
    trend = float(np.tanh((ema_fast[-1] - ema_slow[-1]) / (_safe_std(close[-50:]) * 5)))
    return {"fractal_dim": fractal_dim, "fractal_trend": trend}


# ---------- regime classification (stable rules)

def features_regime(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 30:
        return {"adx_like": 0.0, "atr_pct": 0.0, "efficiency": 0.5}

    close = df["close"].to_numpy()
    tr = np.maximum.reduce(
        [
            df["high"].to_numpy()[1:] - df["low"].to_numpy()[1:],
            np.abs(df["high"].to_numpy()[1:] - df["close"].to_numpy()[:-1]),
            np.abs(df["low"].to_numpy()[1:] - df["close"].to_numpy()[:-1]),
        ]
    )
    atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
    atr_pct = float(atr / max(close[-1], 1e-9))

    # “ADX-like” (not true ADX): magnitude of EMA slope relative to noise
    ema_f, ema_s = _ema(close, 12), _ema(close, 26)
    slope = float(ema_f[-1] - ema_s[-1])
    adx_like = float(abs(slope) / (_safe_std(close[-50:]) * 3))

    # efficiency ratio
    net = abs(close[-1] - close[-20]) if len(close) >= 21 else abs(close[-1] - close[0])
    tot = float(np.abs(np.diff(close[-20:])).sum()) if len(close) >= 21 else float(np.abs(np.diff(close)).sum())
    efficiency = float(net / max(tot, 1e-9))

    return {"adx_like": adx_like, "atr_pct": atr_pct, "efficiency": efficiency}


# ---------- public API

@dataclass
class UltraAdvancedOutput:
    cols: List[str]
    values: np.ndarray  # shape (F, )


def compute_ultra_advanced(df: pd.DataFrame, order_book: Dict) -> UltraAdvancedOutput:
    """
    Pure function: dataframe (ohlcv) + order_book -> feature vector.
    Never calls network, never uses randomness.
    """
    ms = features_microstructure(order_book)
    sp = features_spectral(df)
    fr = features_fractal(df)
    rg = features_regime(df)

    # stable ordering of features
    feat = {
        **{f"ms_{k}": v for k, v in ms.items()},
        **sp,  # spec_energy, spec_tilt, spec_entropy
        **fr,  # fractal_dim, fractal_trend
        **rg,  # adx_like, atr_pct, efficiency
    }
    cols = list(feat.keys())
    vec = np.array([feat[c] for c in cols], dtype=float)
    return UltraAdvancedOutput(cols=cols, values=vec)


