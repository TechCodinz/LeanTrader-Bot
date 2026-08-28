from __future__ import annotations

import pytest

from tests.test_production_testnet_terminal_pending_recovery_v1629 import (
    _xrp_pending_runtime,
)


BUY_QTY = 0.7111881
BUY_FILLED = 0.7119
BUY_PRICE = 1.404422
BUY_COST = 0.9998016147


def _install_terminal_buy(
    lane,
    instance,
    fake,
    *,
    with_pending: bool,
    add_reconciliation: bool,
):
    event = {
        "timestamp": "2026-08-28T13:32:48+00:00",
        "symbol": "XRP/USDT",
        "side": "buy",
        "price": BUY_PRICE,
        "quantity": BUY_FILLED,
        "reason": "fast_collective_testnet_entry:velocity_sniper_probe",
        "event_id": "fast57-1787923968513-8123-buy",
    }
    client_id = instance._client_order_id(event)

    fake.balance_total["XRP"] = BUY_QTY
    fake.balance_free["XRP"] = BUY_QTY

    with instance._io_lock:
        instance.state["orders"][client_id] = {
            "client_order_id": client_id,
            "order_id": "terminal-buy-1",
            "symbol": "XRP/USDT",
            "side": "buy",
            "status": "closed",
            "filled": BUY_FILLED,
            "filled_cost": BUY_COST,
            "average": BUY_PRICE,
            "submitted_at": "2026-08-28T13:32:48+00:00",
            "reason": "fast_collective_testnet_entry:velocity_sniper_probe",
        }
        instance.state["positions"]["XRP/USDT"] = BUY_QTY
        instance.state["position_cost_usd"]["XRP/USDT"] = BUY_COST
        instance._save_state()

    with lane._lock:
        lane.state.setdefault("active", {}).pop("XRP/USDT", None)
        lane.state["pending_event"] = (
            {
                "kind": "entry",
                "event": dict(event),
                "assessment": {
                    "allowed": True,
                    "entry_mode": "velocity_sniper_probe",
                    "target_hold_seconds": 30.0,
                    "live_authority": False,
                },
            }
            if with_pending
            else None
        )
        if add_reconciliation:
            rows = list(lane.state.get("v1629_terminal_pending_reconciliations") or [])
            rows.append(
                {
                    "symbol": "XRP/USDT",
                    "client_order_id": client_id,
                    "event_id": event["event_id"],
                    "status": "closed",
                    "filled": BUY_FILLED,
                    "authoritative_remaining_quantity": BUY_QTY,
                    "position_retired": False,
                    "observed_at": 1787924235.0,
                    "order_submitted": False,
                    "live_authority": False,
                }
            )
            lane.state["v1629_terminal_pending_reconciliations"] = rows
        lane._save_locked()

    return event, client_id


def test_terminal_closed_pending_buy_restores_active_without_resubmission(tmp_path):
    lane, _service, instance, fake, _old_client_id, _old_event = _xrp_pending_runtime(
        tmp_path
    )
    event, client_id = _install_terminal_buy(
        lane,
        instance,
        fake,
        with_pending=True,
        add_reconciliation=False,
    )
    before_created = len(fake.created)
    before_entries = int(lane.state.get("entries_today") or 0)

    result = lane._submit_pending(lane.state["pending_event"], now=1787924235.0)

    assert result["reason"] == "terminal_pending_buy_reconciled"
    assert result["details"]["client_order_id"] == client_id
    assert result["details"]["order_submitted"] is False
    assert result["details"]["executor_state_mutated"] is False
    assert result["details"]["live_authority"] is False
    assert lane.state["pending_event"] is None
    assert len(fake.created) == before_created

    active = lane.state["active"]["XRP/USDT"]
    assert active["quantity"] == pytest.approx(BUY_QTY)
    assert active["entry_price"] == pytest.approx(BUY_PRICE)
    assert active["entry_event_id"] == event["event_id"]
    assert active["recovered_by"] == "1.60.31"
    assert active["live_authority"] is False
    assert lane.state["entries_today"] == before_entries + 1


def test_startup_recovery_repairs_existing_v1629_orphan_once(tmp_path):
    lane, _service, instance, fake, _old_client_id, _old_event = _xrp_pending_runtime(
        tmp_path
    )
    _event, client_id = _install_terminal_buy(
        lane,
        instance,
        fake,
        with_pending=False,
        add_reconciliation=True,
    )
    before_created = len(fake.created)
    before_entries = int(lane.state.get("entries_today") or 0)

    first = lane.recover_orphaned_terminal_buys_v1631(now=1787930000.0)

    assert first["ok"] is True
    assert first["recovered"] == 1
    assert first["recovered_symbols"] == ["XRP/USDT"]
    assert len(fake.created) == before_created
    assert lane.state["active"]["XRP/USDT"]["quantity"] == pytest.approx(BUY_QTY)
    assert client_id in lane.state["v1631_recovered_client_ids"]
    assert lane.state["entries_today"] == before_entries + 1

    second = lane.recover_orphaned_terminal_buys_v1631(now=1787930001.0)
    assert second["ok"] is True
    assert second["recovered"] == 0
    assert lane.state["entries_today"] == before_entries + 1
    assert len(fake.created) == before_created


def test_startup_recovery_requires_matching_buy_to_be_latest_real_fill(tmp_path):
    lane, _service, instance, fake, _old_client_id, _old_event = _xrp_pending_runtime(
        tmp_path
    )
    _event, client_id = _install_terminal_buy(
        lane,
        instance,
        fake,
        with_pending=False,
        add_reconciliation=True,
    )

    with instance._io_lock:
        instance.state["orders"]["later-sell"] = {
            "client_order_id": "later-sell",
            "order_id": "later-sell-1",
            "symbol": "XRP/USDT",
            "side": "sell",
            "status": "closed",
            "filled": 0.1,
            "filled_cost": 0.15,
            "average": 1.5,
            "submitted_at": "2026-08-28T13:33:48+00:00",
        }
        instance._save_state()

    outcome = lane.recover_orphaned_terminal_buys_v1631(now=1787930000.0)

    assert outcome["ok"] is True
    assert outcome["recovered"] == 0
    assert "XRP/USDT" not in lane.state["active"]
    assert client_id not in lane.state.get("v1631_recovered_client_ids", [])
    assert any(
        row.get("reason") == "matching_buy_not_latest_filled_order"
        for row in outcome["recent_skips"]
    )


def test_startup_recovery_fails_closed_when_symbol_has_unresolved_order(tmp_path):
    lane, _service, instance, fake, _old_client_id, _old_event = _xrp_pending_runtime(
        tmp_path
    )
    _event, client_id = _install_terminal_buy(
        lane,
        instance,
        fake,
        with_pending=False,
        add_reconciliation=True,
    )

    with instance._io_lock:
        instance.state["orders"]["unresolved-xrp"] = {
            "client_order_id": "unresolved-xrp",
            "symbol": "XRP/USDT",
            "side": "sell",
            "status": "open",
            "filled": 0.0,
            "submitted_at": "2026-08-28T13:33:00+00:00",
        }
        instance._save_state()

    outcome = lane.recover_orphaned_terminal_buys_v1631(now=1787930000.0)

    # The executor's own reconciliation catches the unresolved order before the
    # narrower symbol scan. That is the stronger fail-closed result.
    assert outcome["ok"] is False
    assert outcome["reason"] == "executor_reconciliation_ambiguous"
    assert outcome["recovered"] == 0
    assert "XRP/USDT" not in lane.state["active"]
    assert client_id not in lane.state.get("v1631_recovered_client_ids", [])
