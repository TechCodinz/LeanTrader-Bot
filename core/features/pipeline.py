from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from .ultra_advanced import compute_ultra_advanced


def step_ultra_advanced(df: pd.DataFrame, order_book: Dict) -> Dict[str, Any]:
    """
    Adapter step: returns a dict compatible with simple feature pipelines.
    Produces a dense feature matrix X (rows match len(df)) and feature names.
    """
    out = compute_ultra_advanced(df, order_book)
    X_row = out.values.reshape(1, -1)
    X = np.repeat(X_row, len(df), axis=0)
    return {"X": X, "cols": out.cols}


def run_pipeline(df: pd.DataFrame, order_book: Dict) -> tuple[np.ndarray, List[str]]:
    out = compute_ultra_advanced(df, order_book)
    return out.values, out.cols



