from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EngineState(str, Enum):
    REGISTERED = "registered"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class EngineUnavailable(RuntimeError):
    """Raised when an engine circuit is open or the engine is not running."""


@dataclass
class EngineRecord:
    name: str
    engine: Any
    required: bool
    dependencies: tuple[str, ...]
    version: str
    state: EngineState = EngineState.REGISTERED
    failures: int = 0
    total_calls: int = 0
    last_error: str | None = None
    last_success: str | None = None
    circuit_open_until: float = 0.0


class EngineRegistry:
    """Lifecycle, dependency, health, and failure isolation for supported engines."""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 60.0) -> None:
        if failure_threshold < 1 or recovery_seconds < 0:
            raise ValueError("invalid circuit-breaker configuration")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._records: dict[str, EngineRecord] = {}

    def register(
        self,
        name: str,
        engine: Any,
        *,
        required: bool = True,
        dependencies: tuple[str, ...] = (),
        version: str = "1",
    ) -> None:
        if not name or name in self._records:
            raise ValueError(f"engine already registered or unnamed: {name!r}")
        self._records[name] = EngineRecord(name, engine, required, dependencies, version)

    def start_all(self) -> None:
        for name in self._start_order():
            record = self._records[name]
            try:
                unavailable = [
                    dependency
                    for dependency in record.dependencies
                    if self._records[dependency].state is not EngineState.RUNNING
                ]
                if unavailable:
                    raise EngineUnavailable(f"engine {name} dependencies unavailable: {', '.join(unavailable)}")
                start = getattr(record.engine, "start", None)
                if callable(start):
                    start()
                record.state = EngineState.RUNNING
                record.last_error = None
            except Exception as exc:
                self._record_failure(record, exc)
                if record.required:
                    self.stop_all()
                    raise EngineUnavailable(f"required engine {name} failed to start: {exc}") from exc

    def stop_all(self) -> None:
        for name in reversed(self._start_order()):
            record = self._records[name]
            try:
                stop = getattr(record.engine, "stop", None)
                if callable(stop) and record.state in {EngineState.RUNNING, EngineState.DEGRADED}:
                    stop()
            except Exception as exc:  # noqa: BLE001 - continue stopping the remaining engines
                record.failures += 1
                record.last_error = f"shutdown {type(exc).__name__}: {exc}"
            finally:
                record.state = EngineState.STOPPED

    def call(self, name: str, method: str, *args: Any, **kwargs: Any) -> Any:
        record = self._records[name]
        now = time.monotonic()
        if record.state is EngineState.REGISTERED:
            raise EngineUnavailable(f"engine {name} has not been started")
        if record.state is EngineState.STOPPED:
            raise EngineUnavailable(f"engine {name} is stopped")
        if record.circuit_open_until > now:
            remaining = record.circuit_open_until - now
            raise EngineUnavailable(f"engine {name} circuit open for {remaining:.1f}s")
        if record.state is EngineState.DEGRADED:
            record.state = EngineState.RUNNING

        record.total_calls += 1
        try:
            result = getattr(record.engine, method)(*args, **kwargs)
        except Exception as exc:
            self._record_failure(record, exc)
            raise
        record.failures = 0
        record.state = EngineState.RUNNING
        record.last_error = None
        record.last_success = dt.datetime.now(dt.UTC).isoformat()
        return result

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        output: dict[str, dict[str, Any]] = {}
        for name, record in self._records.items():
            health = getattr(record.engine, "health", None)
            detail: dict[str, Any] = {}
            if callable(health):
                try:
                    result = health()
                    detail = result if isinstance(result, dict) else {"detail": result}
                except Exception as exc:  # noqa: BLE001 - a health probe cannot break the registry snapshot
                    detail = {"health_error": f"{type(exc).__name__}: {exc}"}
            output[name] = {
                "version": record.version,
                "required": record.required,
                "dependencies": list(record.dependencies),
                "state": record.state.value,
                "healthy": record.state is EngineState.RUNNING and record.circuit_open_until <= now,
                "failures": record.failures,
                "total_calls": record.total_calls,
                "last_success": record.last_success,
                "last_error": record.last_error,
                "circuit_open_seconds": max(0.0, record.circuit_open_until - now),
                **detail,
            }
        return output

    def required_healthy(self) -> bool:
        return all(
            record.state is EngineState.RUNNING and record.circuit_open_until <= time.monotonic()
            for record in self._records.values()
            if record.required
        )

    def _record_failure(self, record: EngineRecord, exc: Exception) -> None:
        record.failures += 1
        record.last_error = f"{type(exc).__name__}: {exc}"
        record.state = EngineState.DEGRADED
        if record.failures >= self.failure_threshold:
            record.circuit_open_until = time.monotonic() + self.recovery_seconds

    def _start_order(self) -> list[str]:
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"engine dependency cycle at {name}")
            if name not in self._records:
                raise ValueError(f"unknown engine dependency: {name}")
            visiting.add(name)
            for dependency in self._records[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for engine_name in self._records:
            visit(engine_name)
        return order
