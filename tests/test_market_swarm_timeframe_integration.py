from __future__ import annotations

import pandas as pd

from leantrader.agents.fast_path import FastSwarmRuntime
from leantrader.agents.swarm_service import ReadOnlySwarmService


def _frame(start: float, move: float, rows: int = 48) -> pd.DataFrame:
    closes = [start]
    for _ in range(rows - 1):
        closes.append(closes[-1] * (1.0 + move))
    return pd.DataFrame(
        {
            "timestamp": [index * 60_000 for index in range(rows)],
            "open": closes,
            "high": [value * 1.001 for value in closes],
            "low": [value * 0.999 for value in closes],
            "close": closes,
            "volume": [10_000.0] * rows,
        }
    )


class MultiTimeframeFeed:
    def __init__(self, *, opposite_1h: bool = False) -> None:
        self.opposite_1h = opposite_1h

    def discover_markets(self, *, quote: str, min_quote_volume_usd: float, max_spread_bps: float) -> dict:
        return {
            "candidates": [
                {
                    "symbol": "FAST/USDT",
                    "last": 1.0,
                    "quote_volume_usd": 8_000_000.0,
                    "spread_bps": 2.0,
                }
            ]
        }

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        move = -0.01 if self.opposite_1h and timeframe == "1h" else 0.01
        return _frame(1.0, move, rows=48)


def _service(feed: MultiTimeframeFeed) -> ReadOnlySwarmService:
    return ReadOnlySwarmService(
        feed=feed,
        runtime=FastSwarmRuntime(fee_bps=20.0, slippage_bps=10.0, adverse_selection_bps=0.0),
        market_quote="USDT",
        min_quote_volume_usd=250_000.0,
        max_spread_bps=75.0,
        scan_batch_size=1,
        candle_limit=48,
        cadence_seconds=1.0,
        discovery_refresh_seconds=60.0,
        context_timeframes=("5m", "15m", "1h", "4h"),
        max_context_symbols=1,
    )


def test_fast_scalp_anchor_can_surface_independently_confirmed_extension_timeframes() -> None:
    result = _service(MultiTimeframeFeed()).step()
    assessments = result["timeframe_assessments"]["FAST/USDT"]
    assert assessments["1m"]["direction"] == "long"
    assert assessments["1m"]["independently_qualified"] is True
    assert assessments["5m"]["independently_qualified"] is True
    assert assessments["15m"]["independently_qualified"] is True
    assert assessments["1h"]["independently_qualified"] is True
    assert assessments["4h"]["independently_qualified"] is True
    extension_timeframes = {row["timeframe"] for row in result["shared_position_extension_candidates"]}
    assert extension_timeframes == {"5m", "15m", "1h", "4h"}
    assert all(row["capital_allocated"] is False for row in result["shared_position_extension_candidates"])
    assert result["extension_candidates_are_trade_authority"] is False


def test_opposing_higher_timeframe_does_not_join_scalp_side() -> None:
    result = _service(MultiTimeframeFeed(opposite_1h=True)).step()
    assessments = result["timeframe_assessments"]["FAST/USDT"]
    assert assessments["1m"]["direction"] == "long"
    assert assessments["1h"]["direction"] == "short"
    assert assessments["1h"]["independently_qualified"] is True
    extension_timeframes = {row["timeframe"] for row in result["shared_position_extension_candidates"]}
    assert "1h" not in extension_timeframes
    assert {"5m", "15m", "4h"}.issubset(extension_timeframes)


def test_timeframe_extension_layer_never_allocates_or_escalates_authority() -> None:
    service = _service(MultiTimeframeFeed())
    result = service.step()
    health = service.health(equity=50.0)
    assert result["extension_candidates_are_trade_authority"] is False
    assert health["shared_position_extension_candidates_are_trade_authority"] is False
    assert health["timeframe_mind"]["independent_timeframe_qualification"] is True
    assert health["automatic_promotion"] is False
    assert health["execution_authority"] is False
    assert health["testnet_authority"] is False
    assert health["live_authority"] is False
