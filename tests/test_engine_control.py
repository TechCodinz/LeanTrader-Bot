from __future__ import annotations

import pytest

from leantrader.production.engine_control import EngineRegistry, EngineUnavailable


class Engine:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def start(self) -> None:
        self.events.append(f"start:{self.name}")

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")

    def echo(self, value: str) -> str:
        return value


class FailingEngine:
    def fail(self) -> None:
        raise RuntimeError("boom")


def test_registry_starts_dependencies_and_stops_in_reverse_order():
    events: list[str] = []
    registry = EngineRegistry()
    registry.register("feed", Engine(events, "feed"))
    registry.register("signals", Engine(events, "signals"), dependencies=("feed",))

    registry.start_all()
    assert registry.call("signals", "echo", "ok") == "ok"
    registry.stop_all()

    assert events == ["start:feed", "start:signals", "stop:signals", "stop:feed"]


def test_registry_opens_circuit_after_repeated_failure():
    registry = EngineRegistry(failure_threshold=2, recovery_seconds=60)
    registry.register("bad", FailingEngine())
    registry.start_all()

    with pytest.raises(RuntimeError, match="boom"):
        registry.call("bad", "fail")
    with pytest.raises(RuntimeError, match="boom"):
        registry.call("bad", "fail")
    with pytest.raises(EngineUnavailable, match="circuit open"):
        registry.call("bad", "fail")

    status = registry.snapshot()["bad"]
    assert status["healthy"] is False
    assert status["failures"] == 2
    assert status["last_error"] == "RuntimeError: boom"


def test_registry_rejects_dependency_cycles():
    registry = EngineRegistry()
    registry.register("one", object(), dependencies=("two",))
    registry.register("two", object(), dependencies=("one",))
    with pytest.raises(ValueError, match="dependency cycle"):
        registry.start_all()
