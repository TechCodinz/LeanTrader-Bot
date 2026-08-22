from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ops.vps_bridge import privileged_helper, server


def test_server_redacts_common_secret_shapes():
    raw = "Authorization: Bearer abc.def token=topsecret api_key:xyz sk-abcdefghijklmnop"
    clean = server._redact(raw)
    assert "abc.def" not in clean
    assert "topsecret" not in clean
    assert "api_key:[REDACTED]" in clean
    assert "sk-abcdefghijklmnop" not in clean


def test_server_invokes_only_fixed_helper_argv(monkeypatch, tmp_path):
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(server, "AUDIT_LOG", audit)

    def fake_run(command, **kwargs):
        assert command == ["sudo", "-n", server.HELPER, "heartbeat", "summary"]
        assert kwargs["env"] == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}
        return subprocess.CompletedProcess(command, 0, '{"healthy": true}', "")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    result = server._invoke("heartbeat", "summary")
    assert result["ok"] is True
    assert result["data"] == {"healthy": True}
    event = json.loads(audit.read_text(encoding="utf-8"))
    assert event["action"] == "heartbeat"
    assert event["ok"] is True


def test_state_changing_tools_require_exact_confirmation(monkeypatch):
    monkeypatch.setattr(server, "_invoke", lambda *args, **kwargs: {"ok": True, "args": args})
    assert server.restart_leantrader("yes")["ok"] is False
    assert server.deploy_verified_paper_release("deploy")["ok"] is False
    assert server.restart_leantrader("RESTART_LEANTRADER")["ok"] is True
    assert server.deploy_verified_paper_release("DEPLOY_VERIFIED_PAPER_RELEASE")["ok"] is True


def test_ci_verified_deploy_requires_exact_confirmation(monkeypatch):
    calls = []

    def fake_invoke(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(server, "_invoke", fake_invoke)
    denied = server.deploy_ci_verified_paper_commit(
        "feature/v1.41-test",
        "a" * 40,
        "deploy",
    )
    assert denied["ok"] is False
    assert calls == []

    accepted = server.deploy_ci_verified_paper_commit(
        "feature/v1.41-test",
        "a" * 40,
        "DEPLOY_CI_VERIFIED_PAPER_COMMIT",
    )
    assert accepted["ok"] is True
    assert calls[0][0] == ("repo-write",)
    assert calls[0][1]["payload"] == {
        "operation": "deploy-verified-commit",
        "branch": "feature/v1.41-test",
        "commit": "a" * 40,
    }


def test_autodeploy_identity_and_path_policy_fail_closed():
    assert privileged_helper._validate_autodeploy_identity(
        "feature/v1.41-safe",
        "a" * 40,
    ) == ("feature/v1.41-safe", "a" * 40)

    for branch, commit in (
        ("main", "a" * 40),
        ("feature/../escape", "a" * 40),
        ("feature/safe", "short"),
    ):
        try:
            privileged_helper._validate_autodeploy_identity(branch, commit)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe deployment identity was accepted")

    assert privileged_helper._validate_autodeploy_paths(
        ["src/leantrader/production/new_engine.py", "tests/test_new_engine.py"]
    ) == ["src/leantrader/production/new_engine.py", "tests/test_new_engine.py"]

    for path in (
        "docker-compose.yml",
        "Dockerfile",
        ".github/workflows/supported-release.yml",
        "ops/vps_bridge/server.py",
        "scripts/bootstrap_verified_vps.sh",
        "runtime/orders.json",
    ):
        try:
            privileged_helper._validate_autodeploy_paths([path])
        except ValueError as exc:
            assert "protected path" in str(exc)
        else:
            raise AssertionError(f"protected deployment path was accepted: {path}")


def test_autodeploy_compose_policy_preserves_paper_and_research_floors():
    safe = {
        "TRADING_MODE": "paper",
        "ENABLE_LIVE": "false",
        "ALLOW_LIVE": "false",
        "LIVE_CONFIRM": "NO",
        "BYBIT_TESTNET_ENABLED": "false",
        "EVOLUTION_MIN_SHADOW_SAMPLES": "100",
        "BRAIN_MIN_STRATEGY_SAMPLES": "50",
        "BRAIN_QUARANTINE_MIN_SAMPLES": "100",
        "MARKET_EVIDENCE_MIN_SAMPLES": "8",
        "ADAPTIVE_MIN_SAMPLES": "5",
        "CAPITAL_PRINCIPAL_FLOOR_FRACTION": "0.70",
        "RISK_PER_TRADE_PCT": "0.005",
        "MAX_DAILY_LOSS_PCT": "0.02",
        "MAX_DRAWDOWN_PCT": "0.10",
        "MAX_POSITION_PCT": "0.10",
        "CAPITAL_PROFIT_REINVEST_FRACTION": "0.50",
        "PAPER_FEE_BPS": "10",
        "PAPER_SLIPPAGE_BPS": "5",
    }
    snapshot = privileged_helper._validate_compose_environment(safe)
    assert snapshot == {
        "mode": "paper",
        "live_enabled": False,
        "testnet_enabled": False,
        "round_trip_cost_bps": 30.0,
        "research_sample_floor": 100,
    }

    unsafe_cases = [
        {"ENABLE_LIVE": "true"},
        {"BYBIT_TESTNET_ENABLED": "true"},
        {"EVOLUTION_MIN_SHADOW_SAMPLES": "99"},
        {"RISK_PER_TRADE_PCT": "0.006"},
        {"PAPER_SLIPPAGE_BPS": "4"},
    ]
    for change in unsafe_cases:
        candidate = {**safe, **change}
        try:
            privileged_helper._validate_compose_environment(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe Compose change was accepted: {change}")


def test_autodeploy_accepts_only_successful_supported_github_check(monkeypatch):
    success = {
        "check_runs": [
            {
                "name": "supported-paper-runtime",
                "head_sha": "a" * 40,
                "status": "completed",
                "conclusion": "success",
                "completed_at": "2026-08-21T00:00:00Z",
                "app": {"slug": "github-actions"},
            }
        ]
    }

    monkeypatch.setattr(
        privileged_helper,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            json.dumps(success),
            "",
        ),
    )
    assert privileged_helper._verify_ci_success("a" * 40)["conclusion"] == "success"

    success["check_runs"][0]["conclusion"] = "failure"
    try:
        privileged_helper._verify_ci_success("a" * 40)
    except ValueError as exc:
        assert "successful" in str(exc)
    else:
        raise AssertionError("failed CI check was accepted")


def test_helper_heartbeat_summary_is_bounded_projection(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    heartbeat_path = app_dir / "runtime/vps_heartbeat.json"
    heartbeat_path.parent.mkdir(parents=True)
    heartbeat_path.write_text(
        json.dumps(
            {
                "healthy": False,
                "timestamp": "2026-08-15T12:00:00Z",
                "errors": ["engine unavailable"],
                "runtime": {"mode": "paper"},
                "secret_should_not_escape": "hidden",
                "engines": {
                    "required_bad": {"required": True, "healthy": False, "internal": "hidden"},
                    "optional": {"required": False, "healthy": False},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(privileged_helper, "APP_DIR", app_dir)
    monkeypatch.setattr(privileged_helper, "HEARTBEAT", heartbeat_path)
    monkeypatch.setattr(privileged_helper, "HALT_FILE", app_dir / "runtime/TESTNET_HALT")

    result = privileged_helper.heartbeat("summary")
    assert result["healthy"] is False
    assert result["required_engine_failures"] == ["required_bad"]
    assert "secret_should_not_escape" not in result


def test_helper_engine_projection_omits_unapproved_fields(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "engines": {
                    "router": {
                        "required": True,
                        "healthy": True,
                        "state": "ready",
                        "credential": "must-not-escape",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(privileged_helper, "HEARTBEAT", heartbeat_path)
    result = privileged_helper.heartbeat("engines")
    assert result == {"router": {"required": True, "healthy": True, "state": "ready"}}


def test_helper_swarm_projection_is_bounded_and_authority_free(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "market_swarm": {
                    "running": True,
                    "healthy": True,
                    "stale": False,
                    "cycles": 12,
                    "micro_assessments": 40,
                    "micro_qualified": 5,
                    "micro_fetch_failures": 1,
                    "microstructure_sniper": {
                        "version": "1.45.0",
                        "minimum_modeled_round_trip_cost_bps": 30.0,
                    },
                    "last_step": {
                        "micro_agent_foundry_proposals": [
                            {
                                "specialist": "micro_burst_hunter",
                                "symbol": "BTC/USDT",
                                "horizon_seconds": 15,
                            }
                        ],
                        "microstructure": {
                            "BTC/USDT": {
                                "path_assessments": [
                                    {"independently_qualified": True},
                                    {"independently_qualified": False},
                                ]
                            }
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(privileged_helper, "HEARTBEAT", heartbeat_path)

    result = privileged_helper.heartbeat("swarm")
    assert result["micro_assessments"] == 40
    assert result["micro_qualified"] == 5
    assert result["micro_qualification_rate"] == 0.125
    assert result["latest_specialists"] == {"micro_burst_hunter": 1}
    assert result["latest_horizons"] == {"15": 1}
    assert result["latest_qualified_by_symbol"] == {"BTC/USDT": 1}
    assert result["modeled_round_trip_cost_floor_bps"] == 30.0
    assert result["automatic_promotion"] is False
    assert result["execution_authority"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False



def test_helper_rejects_non_allowlisted_values():
    for section in ("../secrets", "all", "summary extra"):
        try:
            privileged_helper.heartbeat(section)
        except ValueError as exc:
            assert "unsupported" in str(exc)
        else:
            raise AssertionError("unsafe heartbeat section was accepted")

    try:
        privileged_helper.logs("201")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unsafe log count was accepted")


def test_helper_status_does_not_run_docker_when_app_is_absent(monkeypatch, tmp_path):
    app_dir = tmp_path / "missing"
    monkeypatch.setattr(privileged_helper, "APP_DIR", app_dir)
    monkeypatch.setattr(privileged_helper, "HALT_FILE", app_dir / "runtime/TESTNET_HALT")
    assert privileged_helper.status() == {
        "installed": False,
        "testnet_halt_active": False,
        "state": "not_installed",
    }


def test_sudoers_has_no_wildcard_or_shell_entry():
    sudoers = Path("ops/vps_bridge/leantrader-ops.sudoers").read_text(encoding="utf-8")
    assert "*" not in sudoers
    assert "/bin/bash" not in sudoers
    assert "/bin/sh" not in sudoers
    assert "NOPASSWD: ALL" not in sudoers


def test_installer_persists_tunnel_client_before_profile_initialization():
    installer = Path("scripts/install_vps_ops_bridge.sh").read_text(encoding="utf-8")
    install_binary = installer.index('"${WORK_DIR}/tunnel-client/tunnel-client" /usr/local/bin/tunnel-client')
    initialize_profile = installer.index("runuser -u leantunnel -- /usr/local/bin/tunnel-client init")
    assert install_binary < initialize_profile


def test_bootstrap_migrates_only_known_legacy_market_defaults():
    bootstrap = Path("scripts/bootstrap_verified_vps.sh").read_text(encoding="utf-8")
    migration_function = bootstrap.index("migrate_legacy_setting()")
    symbols_migration = bootstrap.index(
        'migrate_legacy_setting "PAPER_SYMBOLS" "BTC/USDT,ETH/USDT,SOL/USDT" "AUTO"'
    )
    timeframes_migration = bootstrap.index(
        'migrate_legacy_setting "CONFIRM_TIMEFRAMES" "1h,4h" "AUTO"'
    )
    safety_validation = bootstrap.index("for required_setting in")
    compose_start = bootstrap.index("docker compose up -d --build")

    assert 'if [[ "${current}" == "${legacy_value}" ]]' in bootstrap
    assert migration_function < symbols_migration < safety_validation < compose_start
    assert migration_function < timeframes_migration < safety_validation < compose_start
