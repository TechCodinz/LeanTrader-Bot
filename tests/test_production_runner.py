from __future__ import annotations

import numpy as np
import pandas as pd

from leantrader.production.runner import PaperRunner
from leantrader.production.settings import Settings


class FakeFeed:
    def candles(self, _symbol: str, _timeframe: str, limit: int) -> pd.DataFrame:
        close = np.linspace(90.0, 110.0, limit)
        return pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "volume": np.ones(limit),
            }
        )


def test_one_cycle_writes_healthy_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env()
    result = PaperRunner(settings, FakeFeed()).cycle()
    assert result["mode"] == "paper"
    assert result["healthy"] is True
    assert result["errors"] == {}
    assert settings.heartbeat_path.exists()
