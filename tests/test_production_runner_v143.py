from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import leantrader.production.runner as runner_module
from leantrader.agents.swarm_evidence import SwarmOutcomeJournal
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
            "version": "1.2",
            "running": self.started and not self.stopped,
            "equity_seen": equity,
            "canonical_paper_ledger_mutation": False,
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
        max_position_pct=0.10,
        poll_seconds=60,
        market_quote="USDT",
        market_min_quote_volume_usd=250_000.0,
        market_max_spread_bps=75.0,
        candle_limit=320,
        market_refresh_seconds=3_600,
        starting_cash=50.0,
        order_usd=2.0,
        capital_principal_floor_fraction=0.70,
        capital_profit_reinvest_fraction=0.50,
        evolution_min_shadow_samples=100,
        strategy_observatory_state_path=tmp_path / "observatory.json",
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
    runner.swarm_outcome_journal = SwarmOutcomeJournal(tmp_path / "outcomes.json")
    service = runner._build_fast_swarm_service()

    assert created == ["bybit"]
    assert isinstance(service.feed, DummyFeed)
    assert service.runtime.fee_bps == 20.0
    assert service.runtime.slippage_bps == 10.0
    assert service.runtime.swarm.radar.minimum_modeled_round_trip_cost_bps >= 30.0
    assert service.cadence_seconds == 15.0
    assert service.timeframe == "1m"
    assert service.shadow_portfolio is not None
    assert service.outcome_journal is runner.swarm_outcome_journal
    assert service.shadow_portfolio.health()["canonical_paper_ledger_mutation"] is False


def test_v143_cycle_reads_health_and_evidence_status_into_heartbeat(monkeypatch, tmp_path):
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
    runner._ingest_swarm_outcomes = lambda: {
        "submitted": 2,
        "episodes_recorded": 1,
        "qualification_refresh": "next_supervisory_cycle",
    }
    writes: list[dict] = []
    runner._write_json_atomic = lambda path, payload: writes.append(dict(payload))

    status = PaperRunner.cycle(runner)
    swarm = status["market_swarm"]
    assert swarm["running"] is True
    assert swarm["equity_seen"] == 51.25
    assert swarm["swarm_evidence_ingest"]["submitted"] == 2
    assert swarm["swarm_evidence_ingest"]["episodes_recorded"] == 1
    assert swarm["slow_control_plane_blocking_fast_scout"] is False
    assert swarm["supervisory_evidence_version"] == "1.42"
    assert swarm["canonical_paper_ledger_mutation_from_fast_thread"] is False
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

def test_v152_startup_heartbeat_is_fail_closed_and_phase_labeled(
    tmp_path,
):
    runner = object.__new__(PaperRunner)
    runner.settings = _settings(tmp_path)

    service = DummySwarmService()
    service.start()

    # Startup health must be explicit; no market-cycle evidence is implied.
    service.health = lambda *, equity: {
        "version": "1.52.0",
        "running": True,
        "healthy": True,
        "stale": False,
        "cycles": 0,
        "consecutive_failures": 0,
        "equity_seen": equity,
        "automatic_promotion": False,
        "execution_authority": False,
        "testnet_authority": False,
        "live_authority": False,
    }

    runner.fast_swarm_service = service
    runner.testnet = None

    runner.ledger = SimpleNamespace(
        cash=40.0,
        positions={
            "AAA/USDT": SimpleNamespace(
                quantity=2.0,
                entry_price=5.0,
            )
        },
    )

    runner.engines = SimpleNamespace(
        snapshot=lambda: {
            "paper_ledger": {
                "required": True,
                "healthy": True,
                "state": "running",
            },
            "market_data": {
                "required": True,
                "healthy": True,
                "state": "running",
            },
        }
    )

    writes = []

    runner._write_json_atomic = (
        lambda path, payload: writes.append(
            (path, dict(payload))
        )
    )

    status = runner._write_startup_heartbeat()

    assert status["healthy"] is True
    assert status["mode"] == "paper"
    assert status["startup_heartbeat"] is True
    assert status["full_market_cycle_complete"] is False
    assert status["equity"] == 50.0
    assert status["testnet_execution"]["enabled"] is False
    assert status["market_swarm"]["required"] is True
    assert status["market_swarm"]["running"] is True
    assert status["market_swarm"]["healthy"] is True
    assert status["automatic_promotion"] is False
    assert status["live_authority"] is False


def test_v152_startup_heartbeat_rejects_required_engine_failure(
    tmp_path,
):
    runner = object.__new__(PaperRunner)
    runner.settings = _settings(tmp_path)

    service = DummySwarmService()
    service.start()
    service.health = lambda *, equity: {
        "running": True,
        "healthy": True,
        "stale": False,
        "cycles": 0,
        "consecutive_failures": 0,
    }

    runner.fast_swarm_service = service
    runner.testnet = None
    runner.ledger = SimpleNamespace(
        cash=50.0,
        positions={},
    )

    runner.engines = SimpleNamespace(
        snapshot=lambda: {
            "market_data": {
                "required": True,
                "healthy": False,
                "state": "degraded",
            }
        }
    )

    writes = []
    runner._write_json_atomic = (
        lambda path, payload: writes.append(
            (path, dict(payload))
        )
    )

    status = runner._write_startup_heartbeat()

    assert status["healthy"] is False
    assert "market_data" in status["errors"][0]
