import numpy as np
import pandas as pd

from core.features.ultra_advanced import compute_ultra_advanced


def test_ultra_advanced_shapes():
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 200),
            "high": np.linspace(101, 111, 200),
            "low": np.linspace(99, 109, 200),
            "close": np.linspace(100, 110, 200)
            + np.sin(np.linspace(0, 10, 200)) * 0.5,
            "volume": np.random.default_rng(1).integers(100, 200, 200),
        }
    )
    ob = {
        "bids": [[109.5, 3], [109.4, 2], [109.3, 2]],
        "asks": [[109.6, 2.5], [109.7, 2], [109.8, 1.5]],
    }
    out = compute_ultra_advanced(df, ob)
    assert out.values.ndim == 1
    assert len(out.cols) == out.values.shape[0]
    assert "spec_entropy" in out.cols
    assert "ms_ob_spread_bps" in out.cols



