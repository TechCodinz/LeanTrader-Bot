from __future__ import annotations

import json
import time

import pytest

from leantrader.production import healthcheck


def _write_heartbeat(path, **overrides):
    payload = {
        "timestamp": time.time(),
        "runtime": "verified-multi-engine-v12.11-continuous-evolution-fabric",
        "healthy": True,
        "errors": {},
        "testnet_execution": {"enabled": False},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_healthcheck_accepts_current_v12_runtime(tmp_path, monkeypatch, capsys):
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat)
    monkeypatch.setenv("HEARTBEAT_PATH", str(heartbeat))
    monkeypatch.setenv("EXPECTED_RUNTIME_ID", "verified-multi-engine-v12.11-continuous-evolution-fabric")

    healthcheck.main()

    assert "healthy: paper heartbeat" in capsys.readouterr().out


def test_healthcheck_rejects_fresh_stale_release_heartbeat(tmp_path, monkeypatch, capsys):
    heartbeat = tmp_path / "heartbeat.json"
    _write_heartbeat(heartbeat, runtime="verified-multi-engine-v11-exchange-protection")
    monkeypatch.setenv("HEARTBEAT_PATH", str(heartbeat))
    monkeypatch.setenv("EXPECTED_RUNTIME_ID", "verified-multi-engine-v12.11-continuous-evolution-fabric")

    with pytest.raises(SystemExit) as exc:
        healthcheck.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "heartbeat runtime mismatch" in out
    assert "verified-multi-engine-v11-exchange-protection" in out

def test_healthcheck_prefers_fresh_runtime_health_state_over_stale_cycle(
    tmp_path,
    monkeypatch,
    capsys,
):
    heartbeat = tmp_path / "vps_heartbeat.json"
    health_state = tmp_path / "vps_health_state.json"

    _write_heartbeat(
        heartbeat,
        timestamp=time.time() - 1_000,
    )

    _write_heartbeat(
        health_state,
        market_swarm={
            "required": True,
            "running": True,
            "healthy": True,
            "stale": False,
            "consecutive_failures": 0,
        },
    )

    monkeypatch.setenv(
        "HEARTBEAT_PATH",
        str(heartbeat),
    )
    monkeypatch.setenv(
        "EXPECTED_RUNTIME_ID",
        "verified-multi-engine-v12.11-continuous-evolution-fabric",
    )
    monkeypatch.setenv(
        "HEARTBEAT_MAX_AGE",
        "180",
    )

    healthcheck.main()

    assert (
        "healthy: paper heartbeat"
        in capsys.readouterr().out
    )
