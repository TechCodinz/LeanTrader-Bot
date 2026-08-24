from __future__ import annotations

from leantrader.production.fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)
from leantrader.production.runner import (
    PaperRunner,
)


class RecoveringTestnet:
    def __init__(self):
        self.errors = [
            {"reason": "temporary"}
        ]
        self.calls = 0

    def safe_snapshot(self):
        return {
            "authenticated": True,
            "sandbox_endpoint_verified": True,
            "last_reconciliation_errors": (
                list(self.errors)
            ),
        }

    def reconcile_required(self):
        self.calls += 1
        self.errors = []
        return {
            "reconciled": True,
            "errors": [],
        }


class FailingTestnet(
    RecoveringTestnet
):
    def reconcile_required(self):
        self.calls += 1
        raise RuntimeError(
            "temporary provider failure"
        )


def lane(testnet):
    obj = object.__new__(
        HyperSpeedCollectiveTestnetLane
    )

    obj.testnet = testnet
    obj.fast_reconciliation_retry_seconds = 3.0
    obj._last_fast_reconciliation_attempt_at = 0.0
    obj.fast_reconciliation_attempts = 0
    obj.fast_reconciliation_successes = 0
    obj.fast_reconciliation_failures = 0
    obj.last_fast_reconciliation_error = None

    return obj


def test_fast_lane_recovers_reconciliation_without_canonical_cycle():
    testnet = RecoveringTestnet()
    obj = lane(testnet)

    result = obj._fast_reconciliation_gate(
        testnet.safe_snapshot(),
        now=100.0,
    )

    assert result["clear"] is True
    assert result["attempted"] is True
    assert testnet.calls == 1
    assert obj.fast_reconciliation_successes == 1


def test_fast_reconciliation_is_rate_bounded_when_provider_fails():
    testnet = FailingTestnet()
    obj = lane(testnet)

    first = obj._fast_reconciliation_gate(
        testnet.safe_snapshot(),
        now=100.0,
    )

    second = obj._fast_reconciliation_gate(
        testnet.safe_snapshot(),
        now=101.0,
    )

    assert first["clear"] is False
    assert second["clear"] is False
    assert first["attempted"] is True
    assert second["attempted"] is False
    assert testnet.calls == 1


def test_stale_supervisor_testnet_only_failure_can_recover():
    supervisor = {
        "healthy": False,
        "required_failures": [
            "bybit_testnet_execution"
        ],
    }

    snapshot = {
        "authenticated": True,
        "sandbox_endpoint_verified": True,
        "last_reconciliation_errors": [],
    }

    result = (
        HyperSpeedCollectiveTestnetLane
        ._reconciled_supervisor_snapshot(
            supervisor,
            snapshot,
        )
    )

    assert result["healthy"] is True
    assert result["required_failures"] == []
    assert (
        result[
            "fast_testnet_health_reconciled"
        ]
        is True
    )


def test_other_required_failure_is_never_overridden():
    supervisor = {
        "healthy": False,
        "required_failures": [
            "bybit_testnet_execution",
            "market_data",
        ],
    }

    snapshot = {
        "authenticated": True,
        "sandbox_endpoint_verified": True,
        "last_reconciliation_errors": [],
    }

    result = (
        HyperSpeedCollectiveTestnetLane
        ._reconciled_supervisor_snapshot(
            supervisor,
            snapshot,
        )
    )

    assert result["healthy"] is False
    assert "market_data" in (
        result["required_failures"]
    )


def test_runtime_health_overlay_uses_fresh_executor_state():
    runner = object.__new__(
        PaperRunner
    )

    runner.testnet = RecoveringTestnet()
    runner.testnet.errors = []

    engines = {
        "bybit_testnet_execution": {
            "required": True,
            "healthy": False,
            "state": "degraded",
            "failures": 1,
        },
        "market_data": {
            "required": True,
            "healthy": True,
            "state": "running",
            "failures": 0,
        },
    }

    result = (
        runner._direct_testnet_health_overlay(
            engines
        )
    )

    assert (
        result[
            "bybit_testnet_execution"
        ]["healthy"]
        is True
    )

    assert (
        result[
            "bybit_testnet_execution"
        ]["direct_executor_reconciled"]
        is True
    )
