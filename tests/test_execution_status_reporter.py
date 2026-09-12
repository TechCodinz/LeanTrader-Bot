"""The status reporter must say where execution stopped, and leak nothing.

It is meant to be run inside the container by an operator, so it reads real
configuration. Credential values must never appear in its output -- only
whether each input is set.
"""

import io
import os
import contextlib

import pytest

from tools import execution_status


SECRET_KEY = "LEAKCANARY_KEY_a1b2c3"
SECRET_VAL = "LEAKCANARY_SECRET_d4e5f6"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "EXECUTION_TELEMETRY_PATH", str(tmp_path / "telemetry.json")
    )
    # The reporter now reads the lineage journal too, so that has to be
    # isolated as well or a journal left by the host leaks into the verdict.
    monkeypatch.setenv("EXECUTION_LINEAGE_PATH", str(tmp_path / "lineage.jsonl"))
    monkeypatch.setenv("INVENTORY_RECOVERY_PATH", str(tmp_path / "recovery.json"))
    from src.leantrader.execution import preflight

    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    preflight.reset_caches()
    preflight.reset_shared_brokers()


def _capture(fn, *args):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn(*args)
    return buffer.getvalue()


def test_environment_report_never_prints_credential_values(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "testnet")
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", SECRET_KEY)
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", SECRET_VAL)

    output = _capture(execution_status.report_environment)

    assert SECRET_KEY not in output
    assert SECRET_VAL not in output
    assert "BYBIT_TESTNET_API_KEY" in output
    assert "set" in output


def test_environment_report_names_a_missing_secret_file(monkeypatch, tmp_path):
    missing = tmp_path / "not-there.key"
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY_FILE", str(missing))

    output = _capture(execution_status.report_environment)

    assert "MISSING FILE" in output


def test_counters_report_says_why_nothing_reached_execution():
    """A quiet run is classified, not blamed on an assumed upstream break.

    This used to print "the break is upstream of preflight" whenever attempts
    were zero. With no counters and no lineage that claim is unsupported: the
    honest report of an empty evidence store is that it is empty.
    """
    output = _capture(execution_status.report_counters)

    assert "attempts        0" in output
    assert "EXECUTION_IDLE_REASON=" in output
    assert "NO_EVIDENCE_RECORDED" in output
    assert "break is upstream of preflight" not in output


def test_counters_report_ranks_blockers_by_frequency():
    from src.leantrader.execution import preflight

    for _ in range(3):
        preflight.record_blocker(preflight.BELOW_MIN_NOTIONAL, "BTC/USDT")
    preflight.record_blocker(preflight.NO_EXECUTION_AUTHORITY, "bybit")
    preflight.record_event("attempts", 4)

    output = _capture(execution_status.report_counters)

    assert preflight.BELOW_MIN_NOTIONAL in output
    assert preflight.NO_EXECUTION_AUTHORITY in output
    # The more frequent blocker is listed first.
    assert output.index(preflight.BELOW_MIN_NOTIONAL) < output.index(
        preflight.NO_EXECUTION_AUTHORITY
    )
    assert "break is upstream of preflight" not in output


def test_the_reporter_does_not_submit_orders():
    """A status command must never place an order.

    Checked against the parsed source rather than the text, so the prose that
    explains what a prepared order would do does not trip it.
    """
    import ast

    tree = ast.parse(open(execution_status.__file__, encoding="utf-8").read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)

    for forbidden in ("route_order", "create_order", "order", "market"):
        assert forbidden not in called, f"the reporter calls {forbidden}()"
