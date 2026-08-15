from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class ErrorAttributionTracker:
    """Persist and classify intermittent optional-runtime failures.

    The tracker has no trading authority. Repeated failures on optional data
    enhancements receive a short cooldown so one flaky endpoint/timeframe does
    not create identical errors every cycle. Required engines are never
    suppressed here; EngineRegistry remains their fail-closed authority.
    """

    VERSION = "1.0"

    def __init__(
        self,
        state_path: Path,
        *,
        cooldown_after: int = 3,
        cooldown_seconds: int = 300,
    ) -> None:
        if cooldown_after < 1:
            raise ValueError("error cooldown threshold must be positive")
        if cooldown_seconds < 30:
            raise ValueError("error cooldown must be at least 30 seconds")
        self.state_path = state_path
        self.cooldown_after = cooldown_after
        self.cooldown_seconds = cooldown_seconds
        self.records: dict[str, dict[str, Any]] = {}
        self.total_failures = 0
        self.total_successes = 0
        self.last_error: str | None = None
        self._load()

    def start(self) -> None:
        self._load()

    def stop(self) -> None:
        self._save()

    def should_attempt(self, key: str, *, now: float | None = None) -> bool:
        record = self.records.get(key) or {}
        until = float(record.get("suppressed_until") or 0.0)
        return float(now or time.time()) >= until

    def failure(
        self,
        key: str,
        error: str,
        *,
        optional: bool,
        component: str,
        symbol: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        epoch = float(now or time.time())
        record = self.records.setdefault(
            key,
            {
                "component": component,
                "symbol": symbol,
                "optional": bool(optional),
                "failures": 0,
                "successes": 0,
                "consecutive_failures": 0,
                "suppressed_until": 0.0,
            },
        )
        record["component"] = component
        record["symbol"] = symbol
        record["optional"] = bool(optional)
        record["failures"] = int(record.get("failures", 0)) + 1
        record["consecutive_failures"] = int(record.get("consecutive_failures", 0)) + 1
        record["last_error"] = str(error)[:800]
        record["last_failure_at"] = epoch
        if optional and record["consecutive_failures"] >= self.cooldown_after:
            record["suppressed_until"] = epoch + self.cooldown_seconds
        self.total_failures += 1
        self._save()
        return dict(record)

    def success(self, key: str, *, now: float | None = None) -> dict[str, Any]:
        epoch = float(now or time.time())
        record = self.records.get(key)
        if record is None:
            return {}
        record["successes"] = int(record.get("successes", 0)) + 1
        record["consecutive_failures"] = 0
        record["suppressed_until"] = 0.0
        record["last_success_at"] = epoch
        self.total_successes += 1
        if self.total_successes % 20 == 0:
            self._save()
        return dict(record)

    def cycle_summary(self, errors: dict[str, str], *, now: float | None = None) -> dict[str, Any]:
        epoch = float(now or time.time())
        active = []
        for key in errors:
            record = self.records.get(key) or {}
            active.append(
                {
                    "key": key,
                    "component": record.get("component") or key.split(":", 1)[0],
                    "symbol": record.get("symbol"),
                    "optional": bool(record.get("optional", False)),
                    "consecutive_failures": int(record.get("consecutive_failures", 0)),
                    "cooldown_remaining_seconds": max(
                        0.0, float(record.get("suppressed_until") or 0.0) - epoch
                    ),
                }
            )
        return {
            "count": len(errors),
            "keys": list(errors),
            "active": active,
            "optional_count": sum(int(row["optional"]) for row in active),
            "required_count": sum(int(not row["optional"]) for row in active),
        }

    def health(self) -> dict[str, Any]:
        now = time.time()
        suppressed = {
            key: max(0.0, float(record.get("suppressed_until") or 0.0) - now)
            for key, record in self.records.items()
            if float(record.get("suppressed_until") or 0.0) > now
        }
        top = sorted(
            self.records.items(),
            key=lambda item: int(item[1].get("failures", 0)),
            reverse=True,
        )[:20]
        return {
            "healthy": self.last_error is None,
            "tracked_error_keys": len(self.records),
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "cooldown_after": self.cooldown_after,
            "cooldown_seconds": self.cooldown_seconds,
            "suppressed_optional_keys": suppressed,
            "top_failures": {key: value for key, value in top},
            "execution_authority": False,
            "required_engines_never_suppressed": True,
            "error": self.last_error,
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.records = dict(payload.get("records") or {})
            self.total_failures = int(payload.get("total_failures", 0))
            self.total_successes = int(payload.get("total_successes", 0))
            self.last_error = None
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.records = {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "records": self.records,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "updated_at": time.time(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
