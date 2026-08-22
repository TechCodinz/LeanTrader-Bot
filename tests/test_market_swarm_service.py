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

def test_v152_microstream_resolves_due_label_from_exact_book_sample(
    tmp_path,
):
    import time

    from leantrader.agents.micro_calibration import (
        MicroCalibrationJournal,
    )

    class MicrostreamFeed:
        def __init__(self):
            self.service = None

        def public_trades(self, symbol: str, limit: int = 40):
            return []

        def order_book(self, symbol: str, limit: int = 10):
            # End the sampler after this one deterministic observation.
            if self.service is not None:
                self.service._stop.set()
            return {
                "bids": [[99.9, 200.0]],
                "asks": [[100.1, 200.0]],
            }

    feed = FakeReadOnlyFeed()
    stream = MicrostreamFeed()
    journal = MicroCalibrationJournal(
        tmp_path / "micro.json",
        accepted_horizons=(5,),
    )

    service = ReadOnlySwarmService(
        feed=feed,
        runtime=FastSwarmRuntime(),
        market_quote="USDT",
        min_quote_volume_usd=250_000.0,
        max_spread_bps=75.0,
        cadence_seconds=1.0,
        discovery_refresh_seconds=60.0,
        micro_calibration_journal=journal,
        microstream_feed=stream,
    )

    stream.service = service

    observed = time.time() - 5.25

    assert journal.register(
        symbol="AAA/USDT",
        midpoint=100.0,
        assessments=[
            {
                "horizon_seconds": 5,
                "direction": "long",
                "confidence": 0.7,
                "pressure_score": 0.8,
                "expected_edge_bps": 40.0,
                "modeled_round_trip_cost_bps": 30.0,
                "independently_qualified": False,
                "reason": "research",
                "specialist": "temporal_orderflow",
                "regime": "micro_balanced",
            }
        ],
        observed_at=observed,
    ) == 1

    service._run_microstream()

    resolved = journal.state["resolved"]

    assert len(resolved) == 1
    assert resolved[0]["timing_valid"] is True
    assert resolved[0]["timing_censored"] is False
    assert resolved[0]["exit_midpoint"] == 100.0
    assert service.microstream_labels_resolved == 1
    assert service.microstream_observations == 1
    assert service.microstream_sample_failures == 0

def test_v153_sticky_micro_symbols_survive_candidate_rotation():
    feed = FakeReadOnlyFeed()
    service = _service(feed, batch=1)

    service._refresh_discovery(force=True)
    service._cursor = 0
    service._microstream_symbols = [
        "CCC/USDT"
    ]

    rows = service._next_candidates()
    symbols = {
        str(row.get("symbol") or "").upper()
        for row in rows
    }

    # AAA is the rotating batch member.
    assert "AAA/USDT" in symbols

    # CCC remains present because it is already being
    # warmed by the dedicated microstream.
    assert "CCC/USDT" in symbols


def test_v153_precision_lane_never_calls_public_trades():
    class PrecisionFeed:
        def __init__(self):
            self.service = None
            self.book_calls = 0

        def public_trades(
            self,
            symbol: str,
            limit: int = 40,
        ):
            raise AssertionError(
                "precision lane must not fetch trades"
            )

        def order_book(
            self,
            symbol: str,
            limit: int = 10,
        ):
            self.book_calls += 1

            if self.service is not None:
                self.service._stop.set()

            return {
                "bids": [[99.9, 200.0]],
                "asks": [[100.1, 200.0]],
            }

    base_feed = FakeReadOnlyFeed()
    precision = PrecisionFeed()

    service = ReadOnlySwarmService(
        feed=base_feed,
        runtime=FastSwarmRuntime(),
        market_quote="USDT",
        min_quote_volume_usd=250_000.0,
        max_spread_bps=75.0,
        cadence_seconds=1.0,
        discovery_refresh_seconds=60.0,
        microstream_feed=precision,
    )

    precision.service = service

    service._microstream_symbols = [
        "AAA/USDT"
    ]

    service._run_microstream()

    assert precision.book_calls == 1
    assert service.microstream_sample_attempts == 1
    assert service.microstream_observations == 1
    assert service.microstream_sample_failures == 0
    assert (
        service.microstream_trade_context_failures
        == 0
    )
