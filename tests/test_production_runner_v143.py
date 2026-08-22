from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import leantrader.production.runner as runner_module
from leantrader.production.runner import PaperRunner


class DummyFeed:
    pass


class DummySwarmService:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def health(self, *, equity: float) -> dict:
        return {
            "version": "1.0",
            "running": self.started and not self.stopped,
            "equity_seen": equity,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        exchange="bybit",
        fee_bps=10.0,
        slippage_bps=5.0,
        market_scan_batch_size=18,
        max_open_positions=2,
        poll_seconds=60,
        market_quote="USDT",
        market_min_quote_volume_usd=250_000.0,
        market_max_spread_bps=75.0,
        candle_limit=320,
        market_refresh_seconds=3_600,
        starting_cash=50.0,
        heartbeat_path=tmp_path / "heartbeat.json",
    )


def test_v143_fast_service_uses_dedicated_feed_and_round_trip_costs(monkeypatch, tmp_path):
    created: list[str] = []

    def fake_market_feed(exchange: str):
        created.append(exchange)
        return DummyFeed()

    monkeypatch.setattr(runner_module, "MarketFeed", fake_market_feed)
    runner = object.__new__(PaperRunner)
    runner.settings = _settings(tmp_path)
    service = runner._build_fast_swarm_service()

    assert created == ["bybit"]
    assert isinstance(service.feed, DummyFeed)
    assert service.runtime.fee_bps == 20.0
    assert service.runtime.slippage_bps == 10.0
    assert service.runtime.radar.minimum_modeled_round_trip_cost_bps >= 30.0
    assert service.cadence_seconds == 15.0
    assert service.timeframe == "1m"


def test_v143_cycle_only_reads_fast_service_health_into_heartbeat(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner_module._V142PaperRunner,
        "cycle",
        lambda self: {"healthy": True, "mode": "paper", "equity": 51.25},
    )
    runner = object.__new__(PaperRunner)
    runner.settings = _settings(tmp_path)
    service = DummySwarmService()
    service.start()
    runner.fast_swarm_service = service
    writes: list[dict] = []
    runner._write_json_atomic = lambda path, payload: writes.append(dict(payload))

    status = PaperRunner.cycle(runner)
    swarm = status["market_swarm"]
    assert swarm["running"] is True
    assert swarm["equity_seen"] == 51.25
    assert swarm["slow_control_plane_blocking_fast_scout"] is False
    assert swarm["supervisory_evidence_version"] == "1.42"
    assert swarm["automatic_promotion"] is False
    assert swarm["execution_authority"] is False
    assert swarm["testnet_authority"] is False
    assert swarm["live_authority"] is False
    assert writes[-1]["market_swarm"]["running"] is True


def test_v143_one_shot_run_does_not_start_parallel_network_service(monkeypatch, tmp_path):
    called = {"super_run": 0, "start": 0, "stop": 0}
    monkeypatch.setattr(
        runner_module._V142PaperRunner,
        "run",
        lambda self, once=False: called.__setitem__("super_run", called["super_run"] + 1),
    )
    runner = object.__new__(PaperRunner)
    runner.settings = _settings(tmp_path)
    runner.fast_swarm_service = None
    runner.start_fast_swarm = lambda: called.__setitem__("start", called["start"] + 1)
    runner.stop_fast_swarm = lambda: called.__setitem__("stop", called["stop"] + 1)

    PaperRunner.run(runner, once=True)
    assert called == {"super_run": 1, "start": 0, "stop": 1}


def test_v143_inactive_swarm_status_never_claims_authority() -> None:
    status = PaperRunner._inactive_swarm_status()
    assert status["configured"] is True
    assert status["running"] is False
    assert status["movement_only_can_allocate_capital"] is False
    assert status["requires_independent_agent_qualification"] is True
    assert status["automatic_promotion"] is False
    assert status["execution_authority"] is False
    assert status["testnet_authority"] is False
    assert status["live_authority"] is False
