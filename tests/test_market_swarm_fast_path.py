from __future__ import annotations

import pandas as pd
import pytest

from leantrader.agents.fast_path import FastSwarmRuntime
from leantrader.agents.movement_profiler import MarketMovementProfiler


def _trend_frame(start: float, move_fraction: float, rows: int = 48) -> pd.DataFrame:
    prices = [float(start)]
    for _ in range(rows - 1):
        prices.append(prices[-1] * (1.0 + move_fraction))
    return pd.DataFrame(
        {
            "timestamp": [index * 60_000 for index in range(rows)],
            "open": prices,
            "high": [value * 1.001 for value in prices],
            "low": [value * 0.999 for value in prices],
            "close": prices,
            "volume": [1_000.0] * rows,
        }
    )


def _candidate(symbol: str, *, last: float, volume: float = 5_000_000.0, spread: float = 2.0) -> dict:
    return {
        "symbol": symbol,
        "last": last,
        "quote_volume_usd": volume,
        "spread_bps": spread,
    }


def test_movement_profiler_measures_percentage_motion_independent_of_nominal_price() -> None:
    profiler = MarketMovementProfiler(capture_efficiency=0.50)
    tiny = profiler.profile(
        symbol="TINY/USDT",
        candles=_trend_frame(0.000004, 0.01),
        quote_volume_usd=5_000_000.0,
        spread_bps=2.0,
        nominal_price=0.000004,
    )
    expensive = profiler.profile(
        symbol="HIGH/USDT",
        candles=_trend_frame(40_000.0, 0.01),
        quote_volume_usd=5_000_000.0,
        spread_bps=2.0,
        nominal_price=40_000.0,
    )
    assert tiny.q75_abs_move_bps == pytest.approx(expensive.q75_abs_move_bps, rel=1e-9)
    assert tiny.expected_capture_bps == pytest.approx(expensive.expected_capture_bps, rel=1e-9)
    assert tiny.movement_frequency_per_minute == pytest.approx(expensive.movement_frequency_per_minute)
    assert profiler.health()["nominal_price_is_selection_factor"] is False
    assert profiler.health()["predictive_claim"] is False


def test_fast_path_qualifies_measured_fast_market_after_costs() -> None:
    runtime = FastSwarmRuntime(fee_bps=10.0, slippage_bps=5.0, adverse_selection_bps=2.0)
    result = runtime.evaluate_batch(
        candidates=[_candidate("FAST/USDT", last=1.0)],
        frames={"FAST/USDT": _trend_frame(1.0, 0.01)},
    )
    assert result["ranked"][0]["symbol"] == "FAST/USDT"
    assert result["ranked"][0]["qualified"] is True
    assert result["ranked"][0]["modeled_round_trip_cost_bps"] >= 30.0
    assert result["ranked"][0]["net_capture_bps"] > 0
    assert result["activated_observer_symbols"] == ["FAST/USDT"]
    assert runtime.swarm.active_agents == len(runtime.swarm.DEFAULT_SPECIALISTS)


def test_fast_path_rejects_motion_that_cannot_clear_modeled_cost() -> None:
    runtime = FastSwarmRuntime()
    result = runtime.evaluate_batch(
        candidates=[_candidate("SLOW/USDT", last=100.0)],
        frames={"SLOW/USDT": _trend_frame(100.0, 0.0005)},
    )
    score = result["ranked"][0]
    assert score["qualified"] is False
    assert score["reason"] == "non_positive_net_capture_after_costs"
    assert result["activated_observer_symbols"] == []
    assert runtime.swarm.active_agents == 0


def test_fast_path_requires_discovery_economics_and_enough_samples() -> None:
    runtime = FastSwarmRuntime()
    missing = runtime.evaluate_batch(
        candidates=[],
        frames={"FAST/USDT": _trend_frame(1.0, 0.01)},
    )
    assert missing["rejections"]["FAST/USDT"] == "missing_discovery_economics"

    short = runtime.evaluate_batch(
        candidates=[_candidate("SHORT/USDT", last=1.0)],
        frames={"SHORT/USDT": _trend_frame(1.0, 0.01, rows=12)},
    )
    assert "insufficient movement samples" in short["rejections"]["SHORT/USDT"]


def test_fast_path_does_not_duplicate_specialist_observers_each_cycle() -> None:
    runtime = FastSwarmRuntime()
    kwargs = {
        "candidates": [_candidate("FAST/USDT", last=1.0)],
        "frames": {"FAST/USDT": _trend_frame(1.0, 0.01)},
    }
    first = runtime.evaluate_batch(**kwargs)
    agents_after_first = runtime.swarm.active_agents
    second = runtime.evaluate_batch(**kwargs)
    assert first["activated_observer_symbols"] == ["FAST/USDT"]
    assert second["activated_observer_symbols"] == []
    assert runtime.swarm.active_agents == agents_after_first


def test_fast_path_never_turns_movement_into_execution_authority() -> None:
    runtime = FastSwarmRuntime()
    result = runtime.evaluate_batch(
        candidates=[_candidate("FAST/USDT", last=1.0)],
        frames={"FAST/USDT": _trend_frame(1.0, 0.01)},
    )
    health = runtime.health(equity=50.0)
    assert result["movement_only_can_allocate_capital"] is False
    assert result["requires_independent_agent_qualification"] is True
    assert result["execution_authority"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False
    assert health["automatic_promotion"] is False
    assert health["execution_authority"] is False
    assert health["testnet_authority"] is False
    assert health["live_authority"] is False
