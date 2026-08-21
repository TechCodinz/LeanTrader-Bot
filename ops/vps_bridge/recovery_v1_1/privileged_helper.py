#!/usr/bin/env python3
"""Root-only fixed-command helper for LeanTrader MCP operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

APP_DIR = Path("/opt/leantrader/app")
HEARTBEAT = APP_DIR / "runtime/vps_heartbeat.json"
HALT_FILE = APP_DIR / "runtime/TESTNET_HALT"
BOOTSTRAP = Path("/usr/local/sbin/leantrader-bootstrap-verified")
OPS_ROOT = Path("/opt/leantrader-ops")
BACKUP_ROOT = Path("/opt/leantrader/backups/reconciliation")
EVOLUTION_SERVICE = "leantrader-evolution-sidecar.service"
CANONICAL_REPOSITORY = "TechCodinz/LeanTrader-Bot"
ALLOWED_LOG_LINES = {"20", "50", "100", "200"}
ALLOWED_HEARTBEAT_SECTIONS = {"summary", "engines", "runtime", "testnet"}
MAX_HEARTBEAT_BYTES = 16 * 1_048_576
MAX_COMMAND_OUTPUT = 65_536
MAX_REQUEST_BYTES = 131_072
MAX_SOURCE_BYTES = 2 * 1_048_576
MAX_BACKUP_BYTES = 5 * 1024 * 1024 * 1024

SAFE_REF = re.compile(r"^(?:HEAD|[0-9a-f]{7,40}|refs/(?:heads|tags)/[A-Za-z0-9._/-]+|[A-Za-z0-9][A-Za-z0-9._/-]{0,127})$")
SAFE_BRANCH = re.compile(r"^(?:local|recovery|feature|codex|release)/[A-Za-z0-9][A-Za-z0-9._/-]{0,120}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".mjs",
    ".py",
    ".pyi",
    ".rst",
    ".service",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
DENIED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "backups",
    "cache",
    "credentials",
    "data",
    "datasets",
    "evidence",
    "keys",
    "logs",
    "models",
    "node_modules",
    "private",
    "research",
    "results",
    "runtime",
    "secrets",
    "state",
    "venv",
    ".venv",
    "wallets",
}
DENIED_EXACT_NAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "runtime_api_key",
}
DENIED_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db"}
STRICT_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 60,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
        timeout=timeout,
        env={
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )


def _git(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=APP_DIR, timeout=timeout)


def _checked(result: subprocess.CompletedProcess[str], message: str) -> str:
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError((output or message)[-MAX_COMMAND_OUTPUT:])
    return output


def _bounded(value: str) -> str:
    if len(value) <= MAX_COMMAND_OUTPUT:
        return value
    return value[:MAX_COMMAND_OUTPUT] + "\n[output truncated at 65536 characters]"


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
        "evolution_service": _service_state(EVOLUTION_SERVICE),
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
        "evolution_service": _service_state(EVOLUTION_SERVICE),
    }
    if not response["installed"]:
        response["state"] = "not_installed"
        return response

    commit = _git(["rev-parse", "HEAD"], timeout=10)
    branch = _git(["branch", "--show-current"], timeout=10)
    response["commit"] = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    response["branch"] = branch.stdout.strip() if branch.returncode == 0 else "unknown"
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
    heartbeat_bytes = HEARTBEAT.stat().st_size
    if heartbeat_bytes > MAX_HEARTBEAT_BYTES:
        raise ValueError("heartbeat exceeds the sixteen-megabyte bounded projection limit")
    document = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    if section == "summary":
        return {
            "available": True,
            "source_bytes": heartbeat_bytes,
            "healthy": document.get("healthy"),
            "timestamp": document.get("timestamp") or document.get("generated_at"),
            "errors": _safe_projection(list(document.get("errors", []))[:100]),
            "runtime": _safe_projection(document.get("runtime", {})),
            "testnet_halt_active": HALT_FILE.exists(),
            "required_engine_failures": [
                name
                for name, value in (document.get("engines") or {}).items()
                if isinstance(value, dict) and value.get("required") is True and value.get("healthy") is not True
            ][:200],
        }
    if section == "engines":
        engines = document.get("engines") or {}
        return {
            name: {key: value.get(key) for key in ("required", "healthy", "state", "failures", "error") if key in value}
            for name, value in list(engines.items())[:500]
            if isinstance(value, dict)
        }
    key = "runtime" if section == "runtime" else "testnet_execution"
    return {"available": True, "source_bytes": heartbeat_bytes, section: _safe_projection(document.get(key, {}))}


def _safe_projection(value: Any, *, depth: int = 0) -> Any:
    """Bound nested heartbeat values and remove credential/private-account-shaped keys."""
    if depth > 6:
        return "[depth limit]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, nested in list(value.items())[:500]:
            key = str(raw_key)
            lowered = key.lower()
            if any(marker in lowered for marker in ("secret", "token", "password", "credential", "private_key", "api_key", "wallet_seed")):
                continue
            result[key] = _safe_projection(nested, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [_safe_projection(item, depth=depth + 1) for item in value[:500]]
    if isinstance(value, str):
        clean = re.sub(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+", r"\1 [REDACTED]", value)
        clean = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", clean)
        return clean[:4000]
    return value


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


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request payload exceeds 131072 bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def _is_denied_path(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if lowered in DENIED_PARTS or lowered in DENIED_EXACT_NAMES:
            return True
        if lowered.startswith(".env"):
            return True
        if any(word in lowered for word in ("credential", "private_key", "secret_key", "api_key", "wallet_seed")):
            return True
    return path.suffix.lower() in DENIED_SUFFIXES


def _repo_relative(raw: str, *, must_exist: bool = False) -> tuple[str, Path]:
    if not raw or "\x00" in raw:
        raise ValueError("repository path is empty or invalid")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("repository path must be relative and cannot traverse parents")
    normalized = candidate.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or _is_denied_path(Path(normalized)):
        raise ValueError("repository path is denied by the secret/runtime policy")
    resolved = (APP_DIR / normalized).resolve(strict=False)
    if APP_DIR.resolve() not in resolved.parents:
        raise ValueError("repository path escapes the fixed repository")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"repository file does not exist: {normalized}")
    return normalized, resolved


def _validate_ref(ref: str) -> str:
    if not SAFE_REF.fullmatch(ref) or ".." in ref or "@{" in ref or ":" in ref:
        raise ValueError("unsupported Git ref")
    result = _git(["rev-parse", "--verify", f"{ref}^{{commit}}"], timeout=15)
    if result.returncode != 0:
        raise ValueError("Git ref does not resolve to a commit")
    return result.stdout.strip()


def _sanitize_remote(url: str) -> str:
    value = url.strip()
    value = re.sub(r"^(https?://)[^/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"([?&](?:token|access_token|key)=)[^&]+", r"\1[REDACTED]", value, flags=re.I)
    return value


def _canonical_remote_ok(url: str) -> bool:
    clean = re.sub(r"^https?://[^/@]+@", "https://", url.strip())
    accepted = {
        f"https://github.com/{CANONICAL_REPOSITORY}.git",
        f"https://github.com/{CANONICAL_REPOSITORY}",
        f"git@github.com:{CANONICAL_REPOSITORY}.git",
        f"ssh://git@github.com/{CANONICAL_REPOSITORY}.git",
    }
    return clean in accepted


def _filtered_status() -> dict[str, Any]:
    result = _git(["status", "--short", "--branch", "--untracked-files=all"], timeout=30)
    text = _checked(result, "git status failed")
    visible: list[str] = []
    suppressed = 0
    for line in text.splitlines():
        if line.startswith("##"):
            visible.append(line)
            continue
        path_text = line[3:] if len(line) > 3 else ""
        path_candidates = [item.strip() for item in path_text.split(" -> ")]
        if any(_is_denied_path(Path(item)) for item in path_candidates):
            suppressed += 1
        else:
            visible.append(line)
    return {"lines": visible[:1000], "suppressed_sensitive_paths": suppressed, "truncated": len(visible) > 1000}


def repo_inventory() -> dict[str, Any]:
    head = _checked(_git(["rev-parse", "HEAD"], timeout=10), "unable to resolve HEAD")
    branch = _checked(_git(["branch", "--show-current"], timeout=10), "unable to resolve current branch")
    remote_result = _git(["remote", "get-url", "origin"], timeout=10)
    remote = remote_result.stdout.strip() if remote_result.returncode == 0 else ""
    refs_result = _git(
        ["for-each-ref", "--format=%(refname)%09%(objectname)%09%(upstream:short)%09%(upstream:track)", "refs/heads", "refs/tags"],
        timeout=30,
    )
    refs = _checked(refs_result, "unable to list refs").splitlines()
    unpushed_result = _git(
        ["log", "--all", "--not", "--remotes", "--max-count=500", "--format=%H%x09%P%x09%aI%x09%s"],
        timeout=60,
    )
    unpushed = _checked(unpushed_result, "unable to list unpushed commits").splitlines()
    return {
        "repository": str(APP_DIR),
        "head": head,
        "branch": branch,
        "origin": _sanitize_remote(remote),
        "canonical_origin": _canonical_remote_ok(remote),
        "status": _filtered_status(),
        "refs": refs[:1000],
        "refs_truncated": len(refs) > 1000,
        "unpushed_commits": unpushed,
        "unpushed_count": len(unpushed),
    }


def repo_history(payload: dict[str, Any]) -> dict[str, Any]:
    limit = int(payload.get("limit", 20))
    if not 1 <= limit <= 100:
        raise ValueError("history limit must be between 1 and 100")
    ref = str(payload.get("ref", "HEAD"))
    commit = _validate_ref(ref)
    result = _git(
        ["log", commit, f"--max-count={limit}", "--format=%H%x09%P%x09%aI%x09%an%x09%s", "--stat", "--summary"],
        timeout=60,
    )
    return {"ref": ref, "resolved_commit": commit, "history": _bounded(_checked(result, "git history failed"))}


def repo_diff(payload: dict[str, Any]) -> dict[str, Any]:
    scope = str(payload.get("scope", "worktree"))
    path = str(payload.get("path", ""))
    path_args: list[str] = []
    if path:
        normalized, _ = _repo_relative(path)
        path_args = ["--", normalized]
    if scope == "worktree":
        args = ["diff", "--no-ext-diff", "--unified=3", *path_args]
    elif scope == "staged":
        args = ["diff", "--cached", "--no-ext-diff", "--unified=3", *path_args]
    elif scope == "unpushed":
        commits = _git(["rev-list", "--all", "--not", "--remotes", "--max-count=200"], timeout=60)
        commit_list = _checked(commits, "unable to resolve unpushed commits").splitlines()
        if not commit_list:
            return {"scope": scope, "path": path, "commits": [], "diff": ""}
        output_parts: list[str] = []
        for commit in commit_list:
            shown = _git(["show", "--format=fuller", "--stat", "--summary", "--name-status", commit, *path_args], timeout=60)
            output_parts.append(_checked(shown, f"unable to inspect commit {commit}"))
            if sum(len(item) for item in output_parts) > MAX_COMMAND_OUTPUT:
                break
        return {"scope": scope, "path": path, "commits": commit_list, "diff": _bounded("\n\n".join(output_parts))}
    else:
        raise ValueError("unsupported diff scope")
    result = _git(args, timeout=60)
    return {"scope": scope, "path": path, "diff": _bounded(_checked(result, "git diff failed"))}


def _systemd_value(property_name: str) -> str:
    result = _run(["systemctl", "show", EVOLUTION_SERVICE, f"--property={property_name}", "--value"], timeout=10)
    return result.stdout.strip() if result.returncode == 0 else ""


def _external_source_roots_and_files() -> tuple[set[Path], set[Path]]:
    roots: set[Path] = set()
    files: set[Path] = set()
    fragment = _systemd_value("FragmentPath")
    if fragment:
        fragment_path = Path(fragment).resolve(strict=False)
        if fragment_path.is_file() and fragment_path.name.startswith("leantrader") and fragment_path.suffix == ".service":
            files.add(fragment_path)
    main_pid = _systemd_value("MainPID")
    if main_pid.isdigit() and int(main_pid) > 0:
        proc = Path("/proc") / main_pid
        try:
            cwd = (proc / "cwd").resolve(strict=True)
            if cwd != Path("/") and str(cwd).startswith(("/opt/", "/srv/")):
                roots.add(cwd)
        except OSError:
            pass
        try:
            arguments = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            arguments = []
        for raw in arguments:
            if not raw.startswith(b"/"):
                continue
            try:
                candidate = Path(raw.decode("utf-8")).resolve(strict=True)
            except (UnicodeDecodeError, OSError):
                continue
            if candidate.is_file() and (
                str(candidate).startswith(("/opt/", "/srv/"))
                or (str(candidate).startswith("/usr/local/") and "leantrader" in candidate.name.lower())
            ):
                files.add(candidate)
                if str(candidate).startswith(("/opt/", "/srv/")):
                    roots.add(candidate.parent)
    return roots, files


def _allowed_external_file(path: Path) -> bool:
    resolved = path.resolve(strict=True)
    roots, files = _external_source_roots_and_files()
    if resolved in files:
        return True
    return any(root == resolved.parent or root in resolved.parents for root in roots)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_source_files() -> tuple[list[Path], int]:
    roots, exact_files = _external_source_roots_and_files()
    candidates: set[Path] = set(exact_files)
    skipped = 0
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name.lower() not in DENIED_PARTS and not name.startswith(".")]
            current_path = Path(current)
            for filename in filenames:
                path = current_path / filename
                if _is_denied_path(path.relative_to(root)) or path.suffix.lower() not in TEXT_SUFFIXES:
                    skipped += 1
                    continue
                try:
                    if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_SOURCE_BYTES:
                        candidates.add(path.resolve())
                except OSError:
                    skipped += 1
    return sorted(candidates), skipped


def source_inventory() -> dict[str, Any]:
    tracked = _git(["ls-files", "-z"], timeout=60)
    tracked_paths = [item for item in tracked.stdout.split("\0") if item]
    visible_tracked = [path for path in tracked_paths if not _is_denied_path(Path(path))]
    external, skipped = _iter_source_files()
    records: list[dict[str, Any]] = []
    for path in external[:250]:
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": _sha256_file(path),
                "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        )
    roots, exact_files = _external_source_roots_and_files()
    return {
        "repository_tracked_files": len(tracked_paths),
        "repository_visible_source_files": len(visible_tracked),
        "external_roots": sorted(str(path) for path in roots),
        "exact_service_files": sorted(str(path) for path in exact_files),
        "external_source_files": records,
        "external_source_count": len(external),
        "external_source_truncated": len(external) > 250,
        "skipped_non_source_or_denied": skipped,
    }


def read_source(payload: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(payload.get("path", ""))
    start_line = int(payload.get("start_line", 1))
    end_line = int(payload.get("end_line", 400))
    if start_line < 1 or end_line < start_line or end_line - start_line > 500:
        raise ValueError("line range must be positive and contain at most 501 lines")

    if Path(raw_path).is_absolute():
        path = Path(raw_path).resolve(strict=True)
        if not _allowed_external_file(path) or _is_denied_path(path) or path.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError("external source path is not allowlisted")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("source file exceeds the two-megabyte read limit")
        data = path.read_bytes()
        source = "WORKTREE"
        display_path = str(path)
    else:
        normalized, path = _repo_relative(raw_path, must_exist=True)
        tracked = _git(["ls-files", "--error-unmatch", "--", normalized], timeout=10)
        if tracked.returncode != 0:
            raise ValueError("repository source reads are limited to tracked files")
        ref = str(payload.get("ref", "HEAD"))
        if ref == "WORKTREE":
            data = path.read_bytes()
            source = "WORKTREE"
        else:
            commit = _validate_ref(ref)
            shown = _git(["show", f"{commit}:{normalized}"], timeout=30)
            if shown.returncode != 0:
                raise FileNotFoundError("file does not exist at the requested ref")
            data = shown.stdout.encode("utf-8")
            source = commit
        display_path = normalized
    if len(data) > MAX_SOURCE_BYTES or b"\0" in data[:8192]:
        raise ValueError("source is binary or exceeds the read limit")
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    selected = lines[start_line - 1 : end_line]
    return {
        "path": display_path,
        "source": source,
        "start_line": start_line,
        "end_line": min(end_line, len(lines)),
        "total_lines": len(lines),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content": "\n".join(selected),
    }


def _evidence_roots() -> list[Path]:
    roots = [APP_DIR / "runtime", Path("/var/lib/leantrader-evolution")]
    external_roots, _ = _external_source_roots_and_files()
    for root in external_roots:
        for name in ("runtime", "evidence", "results", "research", "data"):
            roots.append(root / name)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def evidence_inventory() -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for root in _evidence_roots():
        digest = hashlib.sha256()
        count = 0
        total = 0
        newest = 0.0
        hashed_bytes = 0
        truncated = False
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name.lower() not in {".git", "venv", ".venv", "__pycache__"}]
            for filename in sorted(filenames):
                path = Path(current) / filename
                try:
                    if not path.is_file() or path.is_symlink():
                        continue
                    stat = path.stat()
                except OSError:
                    continue
                relative = path.relative_to(root).as_posix()
                count += 1
                total += stat.st_size
                newest = max(newest, stat.st_mtime)
                digest.update(relative.encode("utf-8", errors="replace"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
                if stat.st_size <= 64 * 1_048_576 and hashed_bytes + stat.st_size <= 512 * 1_048_576:
                    digest.update(_sha256_file(path).encode("ascii"))
                    hashed_bytes += stat.st_size
                if count >= 20_000:
                    truncated = True
                    break
            if truncated:
                break
        summaries.append(
            {
                "root": str(root),
                "file_count": count,
                "total_bytes": total,
                "newest_modified": datetime.fromtimestamp(newest, UTC).isoformat() if newest else None,
                "manifest_sha256": digest.hexdigest(),
                "content_hashed_bytes": hashed_bytes,
                "truncated": truncated,
            }
        )
    return {"roots": summaries, "private_contents_exposed": False}


def repository_read(payload: dict[str, Any]) -> dict[str, Any]:
    operation = str(payload.get("operation", ""))
    if operation == "inventory":
        return repo_inventory()
    if operation == "history":
        return repo_history(payload)
    if operation == "diff":
        return repo_diff(payload)
    if operation == "read-source":
        return read_source(payload)
    if operation == "source-inventory":
        return source_inventory()
    if operation == "evidence-inventory":
        return evidence_inventory()
    raise ValueError("unsupported repository read operation")


def _pytest_python() -> str:
    candidates = [
        APP_DIR / ".venv/bin/python",
        APP_DIR / "venv/bin/python",
        OPS_ROOT / "venv/bin/python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = _run([str(candidate), "-c", "import pytest"], timeout=15)
        if probe.returncode == 0:
            return str(candidate)
    raise RuntimeError("pytest is unavailable in the fixed LeanTrader or operations environments")


def run_tests(payload: dict[str, Any]) -> dict[str, Any]:
    suite = str(payload.get("suite", "bridge"))
    python = _pytest_python()
    if suite == "bridge":
        targets = ["tests/test_vps_ops_bridge.py"]
        extra = APP_DIR / "tests/test_vps_repo_reconciliation_bridge.py"
        if extra.is_file():
            targets.append(str(extra.relative_to(APP_DIR)))
    elif suite == "all":
        targets = ["tests"]
    else:
        raise ValueError("unsupported test suite")
    result = _run([python, "-m", "pytest", "-q", *targets], cwd=APP_DIR, timeout=2_400)
    output = _bounded((result.stdout + result.stderr).strip())
    if result.returncode != 0:
        raise RuntimeError(output or "pytest failed")
    return {"suite": suite, "python": python, "passed": True, "output": output}


def create_backup() -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = BACKUP_ROOT / timestamp
    destination.mkdir(parents=True, mode=0o700)
    bundle = destination / "leantrader-all-refs.bundle"
    _checked(_git(["bundle", "create", str(bundle), "--all"], timeout=600), "git bundle backup failed")
    _checked(_run(["git", "bundle", "verify", str(bundle)], timeout=120), "git bundle verification failed")

    inventory = repo_inventory()
    evidence = evidence_inventory()
    (destination / "repository-inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    (destination / "evidence-inventory.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")

    archive = destination / "evidence-snapshot.tar.gz"
    archived_files = 0
    archived_bytes = 0
    with tarfile.open(archive, "w:gz") as handle:
        for root_index, root in enumerate(_evidence_roots()):
            for current, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name.lower() not in {".git", "venv", ".venv", "__pycache__"}]
                for filename in filenames:
                    path = Path(current) / filename
                    relative = path.relative_to(root)
                    if _is_denied_path(relative):
                        continue
                    try:
                        if not path.is_file() or path.is_symlink():
                            continue
                        size = path.stat().st_size
                    except OSError:
                        continue
                    if archived_bytes + size > MAX_BACKUP_BYTES:
                        raise RuntimeError("evidence backup exceeds the five-gigabyte safety limit")
                    handle.add(path, arcname=f"root-{root_index}/{relative.as_posix()}", recursive=False)
                    archived_files += 1
                    archived_bytes += size
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_bundle": str(bundle),
        "git_bundle_sha256": _sha256_file(bundle),
        "evidence_archive": str(archive),
        "evidence_archive_sha256": _sha256_file(archive),
        "evidence_files": archived_files,
        "evidence_uncompressed_bytes": archived_bytes,
    }
    (destination / "backup-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(destination, 0o700)
    return {"backup": str(destination), **manifest}


def _validate_paths(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 200:
        raise ValueError("paths must contain between 1 and 200 explicit repository paths")
    paths: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError("every repository path must be a string")
        normalized, resolved = _repo_relative(raw, must_exist=True)
        if resolved.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(f"staged path is not an allowlisted source/document type: {normalized}")
        if resolved.is_symlink() or resolved.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"staged source is a symlink or exceeds two megabytes: {normalized}")
        if b"\0" in resolved.read_bytes()[:8192]:
            raise ValueError(f"staged source appears to be binary: {normalized}")
        paths.append(normalized)
    return paths


def stage_paths(payload: dict[str, Any]) -> dict[str, Any]:
    paths = _validate_paths(payload.get("paths"))
    for path in paths:
        _checked(_git(["add", "--", path], timeout=30), f"failed to stage {path}")
    staged = _checked(_git(["diff", "--cached", "--name-status"], timeout=30), "unable to inspect staged paths")
    return {"staged_paths": paths, "staged_status": staged}


def _scan_staged_secrets(paths: Iterable[str]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        normalized, _ = _repo_relative(path, must_exist=True)
        staged = _git(["show", f":{normalized}"], timeout=30)
        if staged.returncode != 0:
            findings.append(f"{normalized}: unable to read staged blob for secret scan")
            continue
        data = staged.stdout.encode("utf-8")
        for pattern in STRICT_SECRET_PATTERNS:
            if pattern.search(data):
                findings.append(f"{normalized}: matched prohibited secret material")
                break
        text = data.decode("utf-8", errors="ignore")
        for match in re.finditer(
            r"(?i)\b(api[_-]?key|api[_-]?secret|token|password|private[_-]?key)\b\s*[:=]\s*['\"]([^'\"]{12,})['\"]",
            text,
        ):
            value = match.group(2).strip().lower()
            if not any(marker in value for marker in ("redacted", "placeholder", "example", "dummy", "your_", "test", "none")):
                findings.append(f"{normalized}: possible hard-coded {match.group(1)}")
                break
    return findings


def commit_staged(payload: dict[str, Any]) -> dict[str, Any]:
    message = str(payload.get("message", "")).strip()
    if not 8 <= len(message) <= 200 or "\n" in message or "\r" in message:
        raise ValueError("commit message must be a single line between 8 and 200 characters")
    names = _checked(_git(["diff", "--cached", "--name-only"], timeout=30), "unable to list staged files").splitlines()
    if not names:
        raise ValueError("there are no staged files to commit")
    validated: list[str] = []
    for name in names:
        normalized, _ = _repo_relative(name, must_exist=True)
        validated.append(normalized)
    findings = _scan_staged_secrets(validated)
    if findings:
        raise ValueError("secret scan blocked commit: " + "; ".join(findings[:20]))
    check = _git(["diff", "--cached", "--check"], timeout=30)
    _checked(check, "staged diff failed whitespace validation")
    result = _git(["commit", "-m", message], timeout=120)
    _checked(result, "git commit failed")
    head = _checked(_git(["rev-parse", "HEAD"], timeout=10), "unable to resolve committed HEAD")
    return {"committed": True, "commit": head, "message": message, "paths": validated}


def _validate_branches(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise ValueError("branches must contain between 1 and 100 explicit branch names")
    branches: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not SAFE_BRANCH.fullmatch(raw) or ".." in raw:
            raise ValueError("unsupported branch name")
        exists = _git(["show-ref", "--verify", "--quiet", f"refs/heads/{raw}"], timeout=10)
        if exists.returncode != 0:
            raise ValueError(f"local branch does not exist: {raw}")
        branches.append(raw)
    return branches


def push_branches(payload: dict[str, Any]) -> dict[str, Any]:
    branches = _validate_branches(payload.get("branches"))
    remote = _checked(_git(["remote", "get-url", "origin"], timeout=10), "origin remote is unavailable")
    if not _canonical_remote_ok(remote):
        raise ValueError("origin does not match the canonical TechCodinz/LeanTrader-Bot repository")
    pushed: list[dict[str, Any]] = []
    for branch in branches:
        result = _git(["push", "origin", f"refs/heads/{branch}:refs/heads/{branch}"], timeout=300)
        output = _bounded((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            raise RuntimeError(f"push failed for {branch}: {output}")
        pushed.append({"branch": branch, "output": output})
    return {"pushed": pushed, "force_used": False, "origin": _sanitize_remote(remote)}


def tag_v134(payload: dict[str, Any]) -> dict[str, Any]:
    commit = str(payload.get("commit", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("tag target must be a full 40-character commit SHA")
    _checked(_git(["cat-file", "-e", f"{commit}^{{commit}}"], timeout=15), "tag target is not a commit")
    clean = _git(["status", "--porcelain"], timeout=30)
    if clean.stdout.strip():
        raise ValueError("working tree must be clean before creating the v1.34 baseline tag")
    existing = _git(["show-ref", "--verify", "--quiet", "refs/tags/v1.34"], timeout=10)
    if existing.returncode == 0:
        raise ValueError("v1.34 tag already exists; this bridge never overwrites tags")
    message = "LeanTrader known-good v1.34 VPS baseline: paper authority, evidence preserved, live trading disabled"
    _checked(_git(["tag", "-a", "v1.34", commit, "-m", message], timeout=30), "unable to create v1.34 tag")
    return {"tag": "v1.34", "commit": commit, "annotated": True, "message": message}


def push_tag_v134() -> dict[str, Any]:
    remote = _checked(_git(["remote", "get-url", "origin"], timeout=10), "origin remote is unavailable")
    if not _canonical_remote_ok(remote):
        raise ValueError("origin does not match the canonical TechCodinz/LeanTrader-Bot repository")
    exists = _git(["show-ref", "--verify", "--quiet", "refs/tags/v1.34"], timeout=10)
    if exists.returncode != 0:
        raise ValueError("local v1.34 tag does not exist")
    result = _git(["push", "origin", "refs/tags/v1.34:refs/tags/v1.34"], timeout=300)
    output = _bounded((result.stdout + result.stderr).strip())
    if result.returncode != 0:
        raise RuntimeError(output or "v1.34 tag push failed")
    return {"tag": "v1.34", "pushed": True, "force_used": False, "output": output}


def import_source(payload: dict[str, Any]) -> dict[str, Any]:
    source_raw = str(payload.get("source_path", ""))
    destination_raw = str(payload.get("destination_path", ""))
    expected = str(payload.get("expected_sha256", "")).lower()
    if not Path(source_raw).is_absolute() or not SHA256.fullmatch(expected):
        raise ValueError("source_path must be absolute and expected_sha256 must be exact")
    source = Path(source_raw).resolve(strict=True)
    if not _allowed_external_file(source) or _is_denied_path(source) or source.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError("external source path is not allowlisted")
    if source.stat().st_size > MAX_SOURCE_BYTES or source.is_symlink():
        raise ValueError("external source is a symlink or exceeds two megabytes")
    actual = _sha256_file(source)
    if actual != expected:
        raise ValueError("external source hash changed after review")
    destination_name, destination = _repo_relative(destination_raw)
    if destination.exists():
        raise FileExistsError("destination already exists; import never overwrites repository files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)
    return {
        "source": str(source),
        "destination": destination_name,
        "sha256": actual,
        "bytes": destination.stat().st_size,
    }


def repository_write(payload: dict[str, Any]) -> dict[str, Any]:
    operation = str(payload.get("operation", ""))
    if operation == "test":
        return run_tests(payload)
    if operation == "backup":
        return create_backup()
    if operation == "stage":
        return stage_paths(payload)
    if operation == "commit":
        return commit_staged(payload)
    if operation == "push":
        return push_branches(payload)
    if operation == "tag-v1.34":
        return tag_v134(payload)
    if operation == "push-tag-v1.34":
        return push_tag_v134()
    if operation == "import-source":
        return import_source(payload)
    raise ValueError("unsupported repository write operation")


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
        elif action == "repo-read" and len(argv) == 2:
            result = repository_read(_read_payload())
        elif action == "repo-write" and len(argv) == 2:
            result = repository_write(_read_payload())
        else:
            print("ERROR: action or arguments are not allowlisted", file=sys.stderr)
            return 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
