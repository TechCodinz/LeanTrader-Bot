#!/usr/bin/env python3
"""Restricted MCP operations bridge for the supported LeanTrader VPS release."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

HELPER = os.environ.get("LEANTRADER_OPS_HELPER", "/usr/local/sbin/leantrader-ops-helper")
AUDIT_LOG = Path(os.environ.get("LEANTRADER_OPS_AUDIT", "/var/log/leantrader-ops/audit.jsonl"))
MAX_OUTPUT_CHARS = 65_536
MAX_PAYLOAD_CHARS = 131_072
LOG_LINE_CHOICES = (20, 50, 100, 200)

SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)\b(api[_-]?key|api[_-]?secret|token|password|private[_-]?key)"
        r"\s*([=:])\s*([^\s,;]+)"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b"),
)


def _redact(value: str) -> str:
    result = value
    result = SECRET_PATTERNS[0].sub(r"\1 [REDACTED]", result)
    result = SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]", result)
    result = SECRET_PATTERNS[2].sub("[REDACTED_OPENAI_KEY]", result)
    result = SECRET_PATTERNS[3].sub("[REDACTED_GITHUB_TOKEN]", result)
    if len(result) > MAX_OUTPUT_CHARS:
        result = result[:MAX_OUTPUT_CHARS] + "\n[output truncated at 65536 characters]"
    return result


def _audit(action: str, *, ok: bool, detail: str = "") -> None:
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": action,
        "ok": ok,
        "detail": _redact(detail)[:500],
    }
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        print("leantrader-ops: unable to append audit event", file=sys.stderr)


def _decode_output(output: str) -> Any:
    clean = _redact(output.strip())
    if not clean:
        return {}
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return clean


def _invoke(
    action: str,
    *args: str,
    timeout: int = 60,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = ["sudo", "-n", HELPER, action, *args]
    encoded_payload = None
    if payload is not None:
        encoded_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if len(encoded_payload) > MAX_PAYLOAD_CHARS:
            return {"ok": False, "action": action, "error": "request payload exceeds 131072 characters"}
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            input=encoded_payload,
            timeout=timeout,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        detail = _redact(str(exc))
        _audit(action, ok=False, detail=detail)
        return {"ok": False, "action": action, "error": detail}

    ok = completed.returncode == 0
    stdout = _decode_output(completed.stdout)
    stderr = _redact(completed.stderr.strip())
    _audit(action, ok=ok, detail=stderr if not ok else "completed")
    response: dict[str, Any] = {"ok": ok, "action": action, "data": stdout}
    if stderr:
        response["error" if not ok else "notice"] = stderr
    response["exit_code"] = completed.returncode
    return response


server = MCPServer(
    name="leantrader-vps-ops",
    title="LeanTrader VPS Operations",
    description="A restricted, audited operations surface for the supported paper/Testnet LeanTrader VPS.",
    instructions=(
        "Use read-only tools first. Never claim live-trading authority. Repository writes require exact "
        "operation-specific confirmations and never force-push. The server cannot read credentials, run "
        "arbitrary shell commands, enable live trading, loosen research gates, or remove an emergency halt."
    ),
    version="1.1.0",
)

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
STATE_CHANGE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
EXTERNAL_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
FAIL_SAFE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


@server.tool(annotations=READ_ONLY, structured_output=True)
def vps_health() -> dict[str, Any]:
    """Return bounded host, tunnel service, clock, memory, and disk health without network or secret data."""
    return _invoke("host-health")


@server.tool(annotations=READ_ONLY, structured_output=True)
def leantrader_status() -> dict[str, Any]:
    """Return the installed commit, container state, healthcheck, restart count, and halt state."""
    return _invoke("status")


@server.tool(annotations=READ_ONLY, structured_output=True)
def leantrader_heartbeat(
    section: Literal["summary", "engines", "runtime", "testnet"] = "summary",
) -> dict[str, Any]:
    """Read one bounded projection of the canonical heartbeat; raw credentials and private data are omitted."""
    return _invoke("heartbeat", section)


@server.tool(annotations=READ_ONLY, structured_output=True)
def leantrader_logs(lines: Literal[20, 50, 100, 200] = 50) -> dict[str, Any]:
    """Return a redacted tail of LeanTrader container logs. Only 20, 50, 100, or 200 lines are accepted."""
    if lines not in LOG_LINE_CHOICES:
        return {"ok": False, "action": "logs", "error": "lines must be one of 20, 50, 100, 200"}
    return _invoke("logs", str(lines))


@server.tool(annotations=READ_ONLY, structured_output=True)
def leantrader_repository_read(
    operation: Literal[
        "inventory",
        "history",
        "diff",
        "read-source",
        "source-inventory",
        "evidence-inventory",
    ],
    path: str = "",
    ref: str = "HEAD",
    scope: Literal["worktree", "staged", "unpushed"] = "worktree",
    start_line: int = 1,
    end_line: int = 400,
    limit: int = 20,
) -> dict[str, Any]:
    """Inspect the fixed LeanTrader repo and discovered sidecar sources through bounded, secret-denying operations."""
    payload = {
        "operation": operation,
        "path": path,
        "ref": ref,
        "scope": scope,
        "start_line": start_line,
        "end_line": end_line,
        "limit": limit,
    }
    return _invoke("repo-read", payload=payload, timeout=180)


WRITE_CONFIRMATIONS = {
    "test": "RUN_REPOSITORY_TESTS",
    "backup": "CREATE_RECONCILIATION_BACKUP",
    "stage": "STAGE_REVIEWED_PATHS",
    "commit": "COMMIT_REVIEWED_BASELINE",
    "push": "PUSH_RECONCILED_BRANCHES",
    "tag-v1.34": "TAG_KNOWN_GOOD_V1_34",
    "push-tag-v1.34": "PUSH_KNOWN_GOOD_V1_34_TAG",
    "import-source": "IMPORT_REVIEWED_VPS_SOURCE",
}


@server.tool(annotations=EXTERNAL_WRITE, structured_output=True)
def leantrader_repository_write(
    operation: Literal[
        "test",
        "backup",
        "stage",
        "commit",
        "push",
        "tag-v1.34",
        "push-tag-v1.34",
        "import-source",
    ],
    confirmation: str,
    suite: Literal["bridge", "all"] = "bridge",
    paths: list[str] | None = None,
    message: str = "",
    branches: list[str] | None = None,
    commit: str = "",
    source_path: str = "",
    destination_path: str = "",
    expected_sha256: str = "",
) -> dict[str, Any]:
    """Perform one fixed, audited repository operation. Every operation requires its documented exact confirmation."""
    expected = WRITE_CONFIRMATIONS.get(operation)
    if expected is None or confirmation != expected:
        return {
            "ok": False,
            "action": "repo-write",
            "error": f"explicit confirmation {expected or 'for a supported operation'} is required",
        }
    payload = {
        "operation": operation,
        "suite": suite,
        "paths": paths or [],
        "message": message,
        "branches": branches or [],
        "commit": commit,
        "source_path": source_path,
        "destination_path": destination_path,
        "expected_sha256": expected_sha256,
    }
    timeout = 2_400 if operation in {"test", "backup"} else 300
    return _invoke("repo-write", payload=payload, timeout=timeout)


@server.tool(annotations=STATE_CHANGE, structured_output=True)
def restart_leantrader(confirmation: str) -> dict[str, Any]:
    """Restart only the LeanTrader Compose service. Requires confirmation='RESTART_LEANTRADER'."""
    if confirmation != "RESTART_LEANTRADER":
        return {"ok": False, "action": "restart", "error": "explicit confirmation RESTART_LEANTRADER is required"}
    return _invoke("restart", timeout=180)


@server.tool(annotations=FAIL_SAFE, structured_output=True)
def activate_testnet_emergency_halt(reason: str = "operator requested fail-safe halt") -> dict[str, Any]:
    """Idempotently block new Testnet entries. This bridge intentionally cannot remove the halt."""
    safe_reason = _redact(reason).replace("\n", " ")[:200]
    response = _invoke("halt")
    _audit("halt-reason", ok=bool(response.get("ok")), detail=safe_reason)
    return response


@server.tool(annotations=STATE_CHANGE, structured_output=True)
def deploy_verified_paper_release(confirmation: str) -> dict[str, Any]:
    """Run the pinned, audited paper-authority VPS bootstrap. Requires confirmation='DEPLOY_VERIFIED_PAPER_RELEASE'."""
    if confirmation != "DEPLOY_VERIFIED_PAPER_RELEASE":
        return {
            "ok": False,
            "action": "deploy",
            "error": "explicit confirmation DEPLOY_VERIFIED_PAPER_RELEASE is required",
        }
    return _invoke("deploy", timeout=2_400)


if __name__ == "__main__":
    server.run(transport="stdio")
