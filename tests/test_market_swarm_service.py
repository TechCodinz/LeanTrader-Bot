from __future__ import annotations

import pandas as pd

from leantrader.agents.fast_path import FastSwarmRuntime
from leantrader.agents.swarm_service import ReadOnlySwarmService


def _frame(start: float, rows: int = 48, move: float = 0.01) -> pd.DataFrame:
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


class FakeReadOnlyFeed:
    def __init__(self) -> None:
        self.discovery_calls = 0
        self.candle_calls: list[tuple[str, str, int]] = []
        self.candidates = [
            {"symbol": "AAA/USDT", "last": 1.0, "quote_volume_usd": 5_000_000.0, "spread_bps": 2.0},
            {"symbol": "BBB/USDT", "last": 2.0, "quote_volume_usd": 4_000_000.0, "spread_bps": 2.0},
            {"symbol": "CCC/USDT", "last": 3.0, "quote_volume_usd": 3_000_000.0, "spread_bps": 2.0},
        ]

    def discover_markets(self, *, quote: str, min_quote_volume_usd: float, max_spread_bps: float) -> dict:
        self.discovery_calls += 1
        assert quote == "USDT"
        assert min_quote_volume_usd > 0
        assert max_spread_bps > 0
        return {"candidates": list(self.candidates)}

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        self.candle_calls.append((symbol, timeframe, limit))
        return _frame(float(next(row["last"] for row in self.candidates if row["symbol"] == symbol)))


def _service(feed: FakeReadOnlyFeed, *, batch: int = 2) -> ReadOnlySwarmService:
    return ReadOnlySwarmService(
        feed=feed,
        runtime=FastSwarmRuntime(),
        market_quote="USDT",
        min_quote_volume_usd=250_000.0,
        max_spread_bps=75.0,
        scan_batch_size=batch,
        candle_limit=48,
        cadence_seconds=1.0,
        discovery_refresh_seconds=60.0,
    )


def test_service_rotates_market_universe_without_blocking_on_one_symbol() -> None:
    feed = FakeReadOnlyFeed()
    service = _service(feed)
    first = service.step()
    second = service.step()
    assert first["selected_symbols"] == ["AAA/USDT", "BBB/USDT"]
    assert second["selected_symbols"] == ["CCC/USDT", "AAA/USDT"]
    assert first["universe_candidates"] == 3
    assert second["full_sweeps"] >= 1
    assert feed.discovery_calls == 1


def test_service_excludes_current_forming_candle_from_profile_evidence() -> None:
    feed = FakeReadOnlyFeed()
    service = _service(feed, batch=1)
    result = service.step()
    profile = result["profiles"]["AAA/USDT"]
    # 48 fetched candles -> last forming candle removed -> 47 closed bars -> 46 returns.
    assert profile["samples"] == 46
    assert result["forming_candle_excluded"] is True


def test_service_only_activates_observers_and_has_no_execution_authority() -> None:
    feed = FakeReadOnlyFeed()
    service = _service(feed, batch=1)
    result = service.step()
    health = service.health(equity=50.0)
    assert result["activated_observer_symbols"] == ["AAA/USDT"]
    assert result["dedicated_read_only_feed"] is True
    assert result["movement_only_can_allocate_capital"] is False
    assert result["execution_authority"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False
    assert health["automatic_promotion"] is False
    assert health["execution_authority"] is False
    assert health["testnet_authority"] is False
    assert health["live_authority"] is False


def test_service_symbol_failure_is_isolated_from_other_markets() -> None:
    class PartiallyBrokenFeed(FakeReadOnlyFeed):
        def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
            if symbol == "AAA/USDT":
                raise RuntimeError("temporary market-data failure")
            return super().candles(symbol, timeframe, limit)

    feed = PartiallyBrokenFeed()
    service = _service(feed, batch=2)
    result = service.step()
    assert "AAA/USDT" in result["fetch_errors"]
    assert "BBB/USDT" in result["profiles"]
    assert any(row["symbol"] == "BBB/USDT" for row in result["ranked"])
