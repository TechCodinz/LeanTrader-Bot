#!/usr/bin/env python3
"""Root-only fixed-command helper for LeanTrader MCP operations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

APP_DIR = Path("/opt/leantrader/app")
HEARTBEAT = APP_DIR / "runtime/vps_heartbeat.json"
HALT_FILE = APP_DIR / "runtime/TESTNET_HALT"
BOOTSTRAP = Path("/usr/local/sbin/leantrader-bootstrap-verified")
ALLOWED_LOG_LINES = {"20", "50", "100", "200"}
ALLOWED_HEARTBEAT_SECTIONS = {"summary", "engines", "runtime", "testnet"}
MAX_HEARTBEAT_BYTES = 1_048_576
MAX_COMMAND_OUTPUT = 65_536


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
    )


def _service_state(name: str) -> str:
    result = _run(["systemctl", "is-active", name], timeout=10)
    return result.stdout.strip() or "unknown"


def _memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[f"{key.lower()}_kb"] = int(raw.strip().split()[0])
    except (OSError, ValueError):
        pass
    return values


def host_health() -> dict[str, Any]:
    disk = shutil.disk_usage("/")
    try:
        uptime_seconds = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        uptime_seconds = 0.0
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "tunnel_service": _service_state("leantrader-tunnel.service"),
        "docker_service": _service_state("docker.service"),
        "time_sync_service": _service_state("systemd-timesyncd.service"),
        "uptime_seconds": round(uptime_seconds),
        "load_average": load,
        "memory": _memory(),
        "root_disk": {"total": disk.total, "used": disk.used, "free": disk.free},
    }


def status() -> dict[str, Any]:
    response: dict[str, Any] = {
        "installed": (APP_DIR / ".git").is_dir(),
        "testnet_halt_active": HALT_FILE.exists(),
    }
    if not response["installed"]:
        response["state"] = "not_installed"
        return response

    commit = _run(["git", "rev-parse", "HEAD"], cwd=APP_DIR, timeout=10)
    response["commit"] = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    container = _run(["docker", "compose", "ps", "-q", "leantrader"], cwd=APP_DIR, timeout=30)
    container_id = container.stdout.strip()
    if not container_id:
        response["container"] = {"present": False}
        return response

    inspected = _run(["docker", "inspect", container_id], timeout=30)
    try:
        raw = json.loads(inspected.stdout)[0]
        state = raw.get("State", {})
        health = state.get("Health") or {}
        response["container"] = {
            "present": True,
            "id": container_id[:12],
            "status": state.get("Status", "unknown"),
            "health": health.get("Status", "missing"),
            "started_at": state.get("StartedAt"),
            "restart_count": raw.get("RestartCount", 0),
            "image": (raw.get("Config") or {}).get("Image"),
        }
    except (json.JSONDecodeError, IndexError, TypeError):
        response["container"] = {"present": True, "id": container_id[:12], "status": "inspect_failed"}
    return response


def heartbeat(section: str) -> dict[str, Any]:
    if section not in ALLOWED_HEARTBEAT_SECTIONS:
        raise ValueError("unsupported heartbeat section")
    if not HEARTBEAT.is_file():
        return {"available": False, "path": str(HEARTBEAT)}
    if HEARTBEAT.stat().st_size > MAX_HEARTBEAT_BYTES:
        raise ValueError("heartbeat exceeds the one-megabyte safety limit")
    document = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    if section == "summary":
        return {
            "available": True,
            "healthy": document.get("healthy"),
            "timestamp": document.get("timestamp") or document.get("generated_at"),
            "errors": document.get("errors", []),
            "runtime": document.get("runtime", {}),
            "testnet_halt_active": HALT_FILE.exists(),
            "required_engine_failures": [
                name
                for name, value in (document.get("engines") or {}).items()
                if isinstance(value, dict) and value.get("required") is True and value.get("healthy") is not True
            ],
        }
    if section == "engines":
        engines = document.get("engines") or {}
        return {
            name: {key: value.get(key) for key in ("required", "healthy", "state", "failures", "error") if key in value}
            for name, value in engines.items()
            if isinstance(value, dict)
        }
    key = "runtime" if section == "runtime" else "testnet_execution"
    return {"available": True, section: document.get(key, {})}


def logs(lines: str) -> str:
    if lines not in ALLOWED_LOG_LINES:
        raise ValueError("unsupported log line count")
    if not (APP_DIR / "docker-compose.yml").is_file():
        return "LeanTrader is not installed."
    result = _run(
        ["docker", "compose", "logs", "--no-color", f"--tail={lines}", "leantrader"],
        cwd=APP_DIR,
        timeout=60,
    )
    output = (result.stdout + result.stderr)[-MAX_COMMAND_OUTPUT:]
    if result.returncode != 0:
        raise RuntimeError(output or "docker compose logs failed")
    return output


def restart() -> dict[str, Any]:
    if not (APP_DIR / "docker-compose.yml").is_file():
        raise FileNotFoundError("LeanTrader is not installed")
    result = _run(["docker", "compose", "restart", "leantrader"], cwd=APP_DIR, timeout=180)
    if result.returncode != 0:
        raise RuntimeError((result.stdout + result.stderr)[-MAX_COMMAND_OUTPUT:])
    return {"restarted": True, "status": status()}


def halt() -> dict[str, Any]:
    runtime = APP_DIR / "runtime"
    if not runtime.is_dir():
        raise FileNotFoundError("LeanTrader runtime directory is unavailable")
    HALT_FILE.touch(exist_ok=True)
    try:
        os.chown(HALT_FILE, 10001, 10001)
    except PermissionError:
        pass
    os.chmod(HALT_FILE, 0o640)
    return {"testnet_halt_active": True, "path": str(HALT_FILE)}


def deploy() -> dict[str, Any]:
    if not BOOTSTRAP.is_file():
        raise FileNotFoundError("verified bootstrap is unavailable")
    # Run the fixed bootstrap in a separate transient root unit. This prevents
    # package updates from inheriting the tunnel daemon's filesystem sandbox.
    _run(["systemctl", "reset-failed", "leantrader-verified-deploy.service"], timeout=15)
    result = _run(
        [
            "systemd-run",
            "--unit=leantrader-verified-deploy",
            "--property=Type=oneshot",
            "--wait",
            "--collect",
            str(BOOTSTRAP),
        ],
        timeout=2_400,
    )
    output = (result.stdout + result.stderr)[-MAX_COMMAND_OUTPUT:]
    if result.returncode != 0:
        raise RuntimeError(output or "verified bootstrap failed")
    return {"deployed": True, "bootstrap_tail": output, "status": status()}


def main(argv: list[str]) -> int:
    if os.geteuid() != 0:
        print("ERROR: privileged helper must run as root", file=sys.stderr)
        return 1
    if len(argv) < 2:
        print("ERROR: missing fixed action", file=sys.stderr)
        return 2
    action = argv[1]
    try:
        if action == "host-health" and len(argv) == 2:
            result: Any = host_health()
        elif action == "status" and len(argv) == 2:
            result = status()
        elif action == "heartbeat" and len(argv) == 3:
            result = heartbeat(argv[2])
        elif action == "logs" and len(argv) == 3:
            result = logs(argv[2])
        elif action == "restart" and len(argv) == 2:
            result = restart()
        elif action == "halt" and len(argv) == 2:
            result = halt()
        elif action == "deploy" and len(argv) == 2:
            result = deploy()
        else:
            print("ERROR: action or arguments are not allowlisted", file=sys.stderr)
            return 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
