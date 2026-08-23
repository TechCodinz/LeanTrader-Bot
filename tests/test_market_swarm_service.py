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

def test_v154_journal_accepts_only_kinematic_event_rows(tmp_path):
    from types import SimpleNamespace

    from leantrader.agents.micro_calibration import (
        MicroCalibrationJournal,
    )
    from leantrader.agents.microstructure_sniper import (
        MicroPathAssessment,
        MicrostructureFeatures,
    )

    class MicroFeed(FakeReadOnlyFeed):
        def order_book(
            self,
            symbol: str,
            limit: int = 10,
        ):
            return {
                "bids": [[100.00, 1_000.0]],
                "asks": [[100.02, 1_000.0]],
            }

        def public_trades(
            self,
            symbol: str,
            limit: int = 80,
        ):
            return []

    features = MicrostructureFeatures(
        symbol="AAA/USDT",
        timestamp=1000.0,
        midpoint=100.01,
        spread_bps=2.0,
        bid_depth_usd=100_000.0,
        ask_depth_usd=100_000.0,
        depth_imbalance=0.0,
        microprice_shift_bps=0.0,
        trade_imbalance=0.0,
        trade_intensity_per_second=0.0,
        short_momentum_bps=0.0,
        realized_volatility_bps_1m=1.0,
        q90_abs_move_bps=1.0,
        cross_venue_basis_bps=0.0,
        cross_venue_pressure=0.0,
        liquidity_vacuum_score=0.0,
        depth_imbalance_velocity=0.0,
        microprice_velocity_bps_per_second=0.0,
        spread_velocity_bps_per_second=0.0,
        trade_imbalance_velocity=0.0,
        pressure_persistence=0.5,
        temporal_samples=20,
        midpoint_velocity_bps_per_second=0.0,
        midpoint_acceleration_bps_per_second2=0.0,
        total_depth_change_fraction_per_second=0.0,
        recent_midpoint_range_bps_5s=0.5,
        recent_midpoint_trend_bps_5s=0.1,
    )

    noise = MicroPathAssessment(
        symbol="AAA/USDT",
        horizon_seconds=5,
        direction="long",
        specialist="temporal_orderflow",
        probability_favorable_first=0.51,
        probability_adverse_first=0.49,
        confidence=0.1,
        path_budget_bps=2.0,
        expected_edge_bps=0.5,
        modeled_round_trip_cost_bps=30.0,
        independently_qualified=False,
        reason="micro_not_rare_enough",
        pressure_score=0.1,
        regime="micro_balanced",
    )

    event = MicroPathAssessment(
        symbol="AAA/USDT",
        horizon_seconds=5,
        direction="long",
        specialist="kinematic_momentum_burst_v154",
        probability_favorable_first=0.60,
        probability_adverse_first=0.40,
        confidence=0.3,
        path_budget_bps=12.0,
        expected_edge_bps=5.0,
        modeled_round_trip_cost_bps=30.0,
        independently_qualified=False,
        reason="micro_predicted_magnitude_below_cost",
        pressure_score=0.4,
        regime="micro_trend",
    )

    class StubSniper:
        maximum_spread_bps = 25.0
        minimum_depth_usd = 10_000.0

        def __init__(self):
            self.rows = [noise]

        def extract(self, **kwargs):
            return features

        def assess(self, *args, **kwargs):
            return list(self.rows)

    sniper = StubSniper()
    journal = MicroCalibrationJournal(
        tmp_path / "v154_micro.json",
        accepted_horizons=(5, 15, 30, 60),
    )

    service = ReadOnlySwarmService(
        feed=MicroFeed(),
        runtime=FastSwarmRuntime(),
        market_quote="USDT",
        min_quote_volume_usd=250_000.0,
        max_spread_bps=75.0,
        scan_batch_size=1,
        candle_limit=48,
        cadence_seconds=1.0,
        discovery_refresh_seconds=60.0,
        microstructure_sniper=sniper,
        micro_agent_foundry=SimpleNamespace(
            propose=lambda assessments, evidence_rankings: []
        ),
        micro_calibration_journal=journal,
    )

    ranked = [{
        "symbol": "AAA/USDT",
        "score": 1.0,
        "modeled_round_trip_cost_bps": 30.0,
    }]
    frames = {
        "AAA/USDT": _frame(100.0),
    }

    # Ordinary micro noise remains observable but creates no
    # prospective calibration label.
    service._microstructure_assess(
        ranked=ranked,
        frames=frames,
        profiles={},
    )

    assert journal.health()["pending_labels"] == 0
    assert service.microstream_non_event_labels_skipped == 1
    assert service.microstream_kinematic_labels_registered == 0

    # A fresh v1.54 kinematic identity enters the clean journal,
    # even when it remains safely below execution qualification.
    sniper.rows = [event]

    service._microstructure_assess(
        ranked=ranked,
        frames=frames,
        profiles={},
    )

    assert journal.health()["pending_labels"] == 1
    assert service.microstream_kinematic_labels_registered == 1

    pending = journal.state["pending"]
    assert pending[-1]["specialist"] == (
        "kinematic_momentum_burst_v154"
    )
    assert pending[-1]["execution_authority"] is False
    assert pending[-1]["testnet_authority"] is False
    assert pending[-1]["live_authority"] is False
