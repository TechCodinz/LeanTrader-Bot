from __future__ import annotations

from leantrader.production.testnet_pending_liveness_v1632 import (
    PENDING_RECONCILE_INTERVAL_SECONDS,
    reconcile_pending_order_v1632,
)
from tests.test_production_testnet_terminal_pending_recovery_v1629 import (
    _xrp_pending_runtime,
)


def test_submitting_pending_is_authoritatively_reconciled(tmp_path):
    lane, _service, instance, fake, client_id, _event = (
        _xrp_pending_runtime(
            tmp_path,
            status="submitting",
            filled=0.0,
            remaining=0.0,
        )
    )

    before_created = len(fake.created)
    calls = []

    def reconcile_required():
        calls.append(True)

        with instance._io_lock:
            row = instance.state["orders"][client_id]
            row["status"] = "closed"
            row["filled"] = 0.6894
            row["filled_cost"] = 0.998919918
            row["average"] = 1.44897
            instance.state["last_reconciliation_errors"] = []
            instance._save_state()

        return {
            "reconciled": True,
            "checked": 1,
            "errors": [],
        }

    instance.reconcile_required = reconcile_required

    outcome = reconcile_pending_order_v1632(
        lane,
        now=2000.0,
    )

    assert outcome["attempted"] is True
    assert outcome["ok"] is True
    assert outcome["before_status"] == "submitting"
    assert outcome["after_status"] == "closed"
    assert outcome["order_submitted"] is False
    assert outcome["resubmission_allowed"] is False
    assert len(calls) == 1
    assert len(fake.created) == before_created


def test_ambiguous_pending_stays_fail_closed(tmp_path):
    lane, _service, instance, fake, _client_id, _event = (
        _xrp_pending_runtime(
            tmp_path,
            status="submitting",
            filled=0.0,
            remaining=0.1,
        )
    )

    before_created = len(fake.created)

    def reconcile_required():
        raise RuntimeError("still ambiguous")

    instance.reconcile_required = reconcile_required

    outcome = reconcile_pending_order_v1632(
        lane,
        now=2000.0,
    )

    assert outcome["attempted"] is True
    assert outcome["ok"] is False
    assert outcome["after_status"] == "submitting"
    assert outcome["order_submitted"] is False
    assert len(fake.created) == before_created


def test_pending_reconciliation_is_throttled(tmp_path):
    lane, _service, instance, _fake, _client_id, _event = (
        _xrp_pending_runtime(
            tmp_path,
            status="submitting",
            filled=0.0,
            remaining=0.1,
        )
    )

    calls = []

    def reconcile_required():
        calls.append(True)
        raise RuntimeError("still ambiguous")

    instance.reconcile_required = reconcile_required

    first = reconcile_pending_order_v1632(
        lane,
        now=2000.0,
    )

    second = reconcile_pending_order_v1632(
        lane,
        now=(
            2000.0
            + PENDING_RECONCILE_INTERVAL_SECONDS / 2
        ),
    )

    assert first["attempted"] is True
    assert second["attempted"] is False
    assert second["reason"] == "pending_reconciliation_throttled"
    assert len(calls) == 1
