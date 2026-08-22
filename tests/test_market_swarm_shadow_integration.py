from __future__ import annotations

from pathlib import Path

import pandas as pd

from leantrader.agents.fast_path import FastSwarmRuntime
from leantrader.agents.swarm_evidence import SwarmOutcomeJournal
from leantrader.agents.swarm_service import ReadOnlySwarmService
from leantrader.agents.swarm_shadow_portfolio import SwarmShadowPortfolio


def _frame(start: float, *, level: float, rows: int = 48) -> pd.DataFrame:
    closes = [start * level]
    for _ in range(rows - 1):
        closes.append(closes[-1] * 1.01)
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


class AdvancingFeed:
    def __init__(self) -> None:
        self.level = 1.0

    def discover_markets(self, *, quote: str, min_quote_volume_usd: float, max_spread_bps: float) -> dict:
        return {
            "candidates": [
                {
                    "symbol": "FAST/USDT",
                    "last": 1.0 * self.level,
                    "quote_volume_usd": 10_000_000.0,
                    "spread_bps": 1.0,
                }
            ]
        }

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        return _frame(1.0, level=self.level)


def _service(tmp_path: Path, feed: AdvancingFeed) -> ReadOnlySwarmService:
    portfolio = SwarmShadowPortfolio(
        tmp_path / "shadow.json",
        starting_equity=50.0,
        fee_bps=10.0,
        slippage_bps=5.0,
        principal_floor_fraction=0.70,
        profit_reinvest_fraction=0.50,
    )
    journal = SwarmOutcomeJournal(tmp_path / "outcomes.json")
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
        shadow_portfolio=portfolio,
        outcome_journal=journal,
        base_order_usd=2.0,
    )


def test_fast_swarm_opens_and_closes_only_shadow_tranches_with_costed_evidence(tmp_path: Path) -> None:
    feed = AdvancingFeed()
    service = _service(tmp_path, feed)

    first = service.step()
    assert first["shadow_execution_is_canonical_paper"] is False
    assert len(first["shadow_open_events"]) >= 1
    assert first["shadow_portfolio"]["open_tranches"] >= 1
    assert first["swarm_outcome_journal"]["pending_closed_outcomes"] == 0

    feed.level = 1.03
    second = service.step()
    assert len(second["shadow_close_events"]) >= 1
    assert second["swarm_outcome_journal"]["pending_closed_outcomes"] >= 1
    pending = service.outcome_journal.pending()
    assert pending
    assert all(row["evidence_interval_complete"] is True for row in pending)
    assert all(row["evidence_authority"] == "costed_shadow_episode_v2" for row in pending)
    assert all(row["execution_authority"] is False for row in pending)
    assert all(row["testnet_authority"] is False for row in pending)
    assert all(row["live_authority"] is False for row in pending)

    health = service.health(equity=50.0)
    assert health["shadow_entries"] >= 1
    assert health["shadow_exits"] >= 1
    assert health["canonical_paper_ledger_mutation"] is False
