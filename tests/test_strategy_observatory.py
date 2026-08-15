from __future__ import annotations

import pytest

from leantrader.production.strategy_observatory import StrategyObservatory


def test_every_engine_and_timeframe_gets_ungated_net_cost_evidence(tmp_path):
    path = tmp_path / "observatory.json"
    observer = StrategyObservatory(path, round_trip_cost_bps=10)
    first = observer.observe(
        "BTC/USDT",
        100.0,
        [
            {"engine": "smart_scalping", "score": 0.8},
            {"engine": "mean_reversion", "score": -0.7},
        ],
        {"1m": 0.5, "1d": -0.4},
    )
    assert first["signals_observed"] == 4
    assert first["outcomes_recorded"] == 0

    second = observer.observe(
        "BTC/USDT",
        102.0,
        [
            {"engine": "smart_scalping", "score": 0.2},
            {"engine": "mean_reversion", "score": -0.1},
        ],
        {"1m": 0.1, "1d": -0.2},
    )
    assert second["outcomes_recorded"] == 4
    health = observer.health()
    assert health["router_gates_applied"] is False
    assert health["strategies_measured"] == 4
    assert health["strategies"]["engine:smart_scalping"]["last_net_return"] == pytest.approx(0.019)
    assert health["strategies"]["engine:mean_reversion"]["last_net_return"] == pytest.approx(-0.021)
    assert StrategyObservatory(path).health()["strategies_measured"] == 4
