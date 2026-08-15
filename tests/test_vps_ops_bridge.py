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
