from __future__ import annotations

import json
import time

from leantrader.production.healthcheck import (
    main as healthcheck_main,
)
from leantrader.production.runner import PaperRunner


def fresh_fast_swarm(now: float):
    return {
        "running": True,
        "healthy": False,
        "stale": True,
        "hung": True,
        "started_at": now - 600.0,
        "precision_scout": {
            "running": True,
            "refresh_seconds": 20.0,
            "last_refresh_at": now - 5.0,
        },
        "micro_calibration": {
            "microstream_running": True,
            "microstream_last_observation_at": (
                now - 1.0
            ),
        },
    }


def test_slow_research_sweep_does_not_kill_fresh_fast_path():
    now = 1_000.0

    runner = object.__new__(PaperRunner)
    runner.testnet = object()

    result = runner._swarm_health_contract(
        fresh_fast_swarm(now),
        now=now,
    )

    assert result[
        "slow_worker_healthy"
    ] is False

    assert result[
        "fast_precision_path_operational"
    ] is True

    assert result[
        "health_contract_healthy"
    ] is True


def test_fast_path_fails_closed_when_microstream_is_stale():
    now = 1_000.0

    runner = object.__new__(PaperRunner)
    runner.testnet = object()

    payload = fresh_fast_swarm(now)

    payload["micro_calibration"][
        "microstream_last_observation_at"
    ] = now - 60.0

    result = runner._swarm_health_contract(
        payload,
        now=now,
    )

    assert result[
        "fast_precision_path_operational"
    ] is False

    assert result[
        "health_contract_healthy"
    ] is False


def test_healthcheck_accepts_split_health_contract(
    tmp_path,
    monkeypatch,
    capsys,
):
    heartbeat = tmp_path / "vps_heartbeat.json"
    health_state = (
        tmp_path / "vps_health_state.json"
    )

    health_state.write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "runtime": (
                    "verified-multi-engine-v12.11-"
                    "continuous-evolution-fabric"
                ),
                "healthy": True,
                "errors": [],
                "testnet_execution": {
                    "enabled": True,
                    "live_authority": False,
                },
                "market_swarm": {
                    "required": True,
                    "running": True,
                    "healthy": True,
                    "stale": True,
                    "slow_worker_healthy": False,
                    "fast_precision_path_operational": True,
                    "health_contract_healthy": True,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "HEARTBEAT_PATH",
        str(heartbeat),
    )

    healthcheck_main()

    assert "healthy:" in capsys.readouterr().out
