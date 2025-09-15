from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest


def test_missing_key_with_runtime_available(monkeypatch, capsys):
    # Fake qiskit_ibm_runtime installed
    class DummyService:
        def __init__(self):
            pass

        @staticmethod
        def save_account(**kwargs):
            return True

        def backends(self):
            return []

        def least_busy(self, **kwargs):
            return None

    class DummySession:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummySampler:
        def __init__(self, *args, **kwargs):
            pass

    fake_mod = SimpleNamespace(QiskitRuntimeService=DummyService, Session=DummySession, Sampler=DummySampler)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_mod)
    # Ensure no key in env
    monkeypatch.delenv("IBM_QUANTUM_API_KEY", raising=False)

    from cli.test_ibm_runtime import run_ibm_check

    res, code = run_ibm_check(api_key=None)
    assert code == 1
    assert res["ok"] is False
    assert "Missing IBM_QUANTUM_API_KEY" in res.get("error", "Missing")


def test_success_path(monkeypatch):
    # Build fake qiskit_ibm_runtime
    class BStatus:
        pending_jobs = 1

    class DummyBackend:
        name = "ibm_fake"
        num_qubits = 127
        simulator = False

        def status(self):
            return BStatus()

    class DummyService:
        def __init__(self):
            pass

        @staticmethod
        def save_account(**kwargs):
            return True

        def backends(self):
            return [DummyBackend()]

        def least_busy(self, **kwargs):
            return DummyBackend()

    class DummySession:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyJob:
        def result(self, timeout=None):
            return {"ok": True}

    class DummySampler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, circuits):
            return DummyJob()

    fake_mod = SimpleNamespace(QiskitRuntimeService=DummyService, Session=DummySession, Sampler=DummySampler)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", fake_mod)

    # Fake qiskit.QuantumCircuit
    class QC:
        def __init__(self, *args, **kwargs):
            pass

        def h(self, *_):
            pass

        def cx(self, *_):
            pass

        def measure(self, *_, **__):
            pass

    monkeypatch.setitem(sys.modules, "qiskit", SimpleNamespace(QuantumCircuit=QC))

    from cli.test_ibm_runtime import run_ibm_check

    res, code = run_ibm_check(api_key="DUMMY", min_qubits=1, resilience_level=1, timeout=1)
    assert code == 0
    assert res["ok"] is True
    assert res["sampler_ok"] is True
    assert res["backend"]["name"] == "ibm_fake"

