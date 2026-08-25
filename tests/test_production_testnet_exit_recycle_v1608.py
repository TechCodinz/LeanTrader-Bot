from __future__ import annotations

import copy

import pytest

from leantrader.production.fast_collective_hyper import HyperSpeedCollectiveTestnetLane
from tests.test_fast_collective_testnet import signal, supervisor
from tests.test_testnet_execution import FakeBybit, buy_event, engine


class BalanceBybit(FakeBybit):
    def __init__(self):
        super().__init__()
        self.balance_total = {"USDT": 10_000.0, "BTC": 0.0, "ETH": 0.0}
        self.balance_free = dict(self.balance_total)

    def fetch_balance(self):
        self.calls.append(("fetch_balance",))
        return {"total": dict(self.balance_total), "free": dict(self.balance_free)}


class CandidateService:
    def __init__(self):
        self.assessed_symbols: list[str] = []

    def collective_candidates(self, limit=8):
        return ["PUBLIC/USDT", "AAA/USDT"][:limit]

    def collective_signal(self, symbol):
        self.assessed_symbols.append(symbol)
        if symbol != "AAA/USDT":
            raise AssertionError("non-Testnet candidate reached assessment")
        return signal()


class LaneTestnet:
    def __init__(self):
        self.positions: dict[str, float] = {}
        self.events: list[dict] = []
        self.realized_pnl_usd = 0.0
        self.dust_cost_basis_usd = 0.0
        self.preparations: list[dict] = []
        self.prepare_mode = "executable"
        self.created_event_ids: list[str] = []
        self._terminal_first = False

    def eligible_symbols(self, quote="USDT"):
        assert quote == "USDT"
        return {"AAA/USDT"}

    def safe_snapshot(self):
        return {
            "positions": dict(self.positions),
            "open_orders": 0,
            "last_reconciliation_errors": [],
            "kill_switch_active": False,
            "daily_order_count": 99,
            "daily_submitted_usd": 999.0,
            "daily_entry_order_count": 0,
            "daily_entry_submitted_usd": 0.0,
            "risk_limits": {"max_orders_per_day": 20, "max_daily_submitted_usd": 50.0},
            "performance": {
                "realized_pnl_usd": self.realized_pnl_usd,
                "non_tradeable_dust_cost_basis_usd": self.dust_cost_basis_usd,
                "realized_net_after_dust_usd": self.realized_pnl_usd - self.dust_cost_basis_usd,
            },
        }

    def reconcile_required(self):
        return {"reconciled": True, "errors": []}

    def prepare_sell(self, symbol, requested_quantity, reference_price):
        row = {
            "status": self.prepare_mode,
            "symbol": symbol,
            "requested_quantity": requested_quantity,
            "position_quantity": self.positions.get(symbol, 0.0),
            "free_quantity": self.positions.get(symbol, 0.0),
            "executable_available_quantity": self.positions.get(symbol, 0.0),
            "executable_quantity": min(requested_quantity, self.positions.get(symbol, 0.0)),
            "minimum_amount": 0.001,
            "minimum_cost_usd": 1.0,
            "live_authority": False,
        }
        if self.prepare_mode == "dust":
            row.update({
                "quantity": self.positions.get(symbol, 0.0),
                "cost_basis_usd": 0.0005,
                "reason": "residual_below_exchange_executable_threshold",
            })
            self.positions.pop(symbol, None)
        self.preparations.append(copy.deepcopy(row))
        return row

    def mirror_events(self, events):
        output = []
        for event in events:
            event = copy.deepcopy(event)
            self.events.append(event)
            event_id = str(event.get("event_id"))
            if event_id not in self.created_event_ids:
                self.created_event_ids.append(event_id)
            symbol = event["symbol"]
            if event["side"] == "buy":
                self.positions[symbol] = event["quantity"]
                output.append({"status": "closed", "symbol": symbol, "side": "buy", "filled": event["quantity"], "average": event["price"]})
                continue
            if self._terminal_first and len([row for row in self.events if row["side"] == "sell"]) == 1:
                output.append({"status": "canceled", "symbol": symbol, "side": "sell", "filled": 0.0, "average": None})
                continue
            self.positions.pop(symbol, None)
            output.append({"status": "closed", "symbol": symbol, "side": "sell", "filled": event["quantity"], "average": event["price"]})
        return output


def growth(*, remaining=15.0, risk=1.0, equity=50.0):
    return {
        "state": "normal",
        "equity": equity,
        "peak_equity": max(50.0, equity),
        "protected_principal": 35.0,
        "locked_profit": 0.0,
        "reinvestable_realized_profit": max(0.0, equity - 50.0) * 0.5,
        "remaining_deployable_notional": remaining,
        "risk_multiplier": risk,
        "new_entries_allowed": True,
    }


def hyper_lane(tmp_path, *, service=None, testnet=None, supervisory=None, order_usd=2.0):
    service = service or CandidateService()
    testnet = testnet or LaneTestnet()
    supervisory = supervisor() if supervisory is None else supervisory
    lane = HyperSpeedCollectiveTestnetLane(
        service_provider=lambda: service,
        testnet=testnet,
        state_path=tmp_path / "hyper-v1608.json",
        supervisory_provider=lambda: supervisory,
        order_usd=order_usd,
        round_trip_cost_bps=30.0,
        cadence_seconds=5.0,
        maximum_hold_seconds=90.0,
        maximum_entries_per_day=20,
        bootstrap_after_seconds=45.0,
        maximum_concurrent_positions=6,
        maximum_entries_per_cycle=3,
        reentry_cooldown_seconds=2.0,
        starting_equity=50.0,
        maximum_order_usd=5.0,
    )
    lane.started_at = 900.0
    return lane, service, testnet


def test_fast_candidates_are_intersected_with_testnet_eligible_markets(tmp_path):
    lane, service, testnet = hyper_lane(tmp_path)
    result = lane.step(now=1_000.0)
    assert result["reason"] == "fast_multi_route_cycle"
    assert service.assessed_symbols == ["AAA/USDT"]
    assert "AAA/USDT" in testnet.positions
    health = lane.health()
    assert health["testnet_eligible_market_intersection"] is True
    assert health["testnet_market_candidate_filter"]["filtered_out_count"] == 1
    assert health["modeled_round_trip_cost_floor_bps"] >= 30.0
    assert health["live_authority"] is False


def test_protective_sells_do_not_consume_buy_entry_budget(tmp_path):
    fake = BalanceBybit()
    instance, _ = engine(tmp_path, fake, max_order_usd=10.0, max_position_usd=20.0, max_daily_submitted_usd=10.0, max_orders_per_day=2)
    instance.start()
    first = instance.mirror_events([buy_event()])[0]
    assert first["status"] == "closed"
    fake.balance_total["BTC"] = 0.05
    fake.balance_free["BTC"] = 0.05
    sell = {
        "timestamp": "2026-08-24T20:00:01+00:00",
        "event_id": "exit-1",
        "symbol": "BTC/USDT",
        "side": "sell",
        "price": 100.0,
        "quantity": 0.05,
        "remaining_quantity": 0.0,
        "reason": "protective_exit",
    }
    exited = instance.mirror_events([sell])[0]
    assert exited["status"] == "closed"
    mid = instance.health()
    assert mid["daily_total_order_count"] == 2
    assert mid["daily_entry_order_count"] == 1
    assert mid["daily_total_submitted_usd"] == pytest.approx(10.0)
    assert mid["daily_entry_submitted_usd"] == pytest.approx(5.0)

    second_buy = {**buy_event(), "timestamp": "2026-08-24T20:00:02+00:00", "event_id": "entry-2", "symbol": "ETH/USDT"}
    second = instance.mirror_events([second_buy])[0]
    assert second["status"] == "closed"
    after = instance.health()
    assert after["daily_total_order_count"] == 3
    assert after["daily_entry_order_count"] == 2
    assert after["daily_total_submitted_usd"] == pytest.approx(15.0)
    assert after["daily_entry_submitted_usd"] == pytest.approx(10.0)
    assert after["entry_budget_excludes_protective_exits"] is True
    assert after["current_day_execution_quality"]["protective_exits_submitted"] == 1
    assert after["live_authority"] is False


def test_current_day_execution_quality_excludes_historical_rejections(tmp_path):
    fake = BalanceBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    today = instance.state["day"]
    instance.state["orders"]["historical-rejected"] = {
        "client_order_id": "historical-rejected",
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "rejected",
        "submitted_at": "2026-08-01T00:00:00+00:00",
        "submitted_usd": 5.0,
        "filled": 0.0,
    }
    instance.state["orders"]["today-skipped"] = {
        "client_order_id": "today-skipped",
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "skipped",
        "decision_at": f"{today}T00:00:00+00:00",
        "submitted_usd": 0.0,
        "filled": 0.0,
    }
    quality = instance.health()["current_day_execution_quality"]
    assert quality["status_counts"]["rejected"] == 0
    assert quality["status_counts"]["skipped"] == 1
    assert quality["historical_rows_excluded"] >= 1


def test_sell_quantity_uses_fresh_free_base_balance_and_exchange_limits(tmp_path):
    fake = BalanceBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    instance.state["positions"]["BTC/USDT"] = 0.05
    instance.state["position_cost_usd"]["BTC/USDT"] = 50.0
    fake.balance_total["BTC"] = 0.05
    fake.balance_free["BTC"] = 0.02
    preparation = instance.prepare_sell("BTC/USDT", 0.05, 1_000.0)
    assert preparation["status"] == "executable"
    assert preparation["position_quantity"] == pytest.approx(0.05)
    assert preparation["free_quantity"] == pytest.approx(0.02)
    assert preparation["executable_quantity"] == pytest.approx(0.02)
    assert preparation["balance_source"] == "exchange_free"

    event = {
        "timestamp": "2026-08-24T20:01:00+00:00",
        "event_id": "fresh-free-exit",
        "symbol": "BTC/USDT",
        "side": "sell",
        "price": 1_000.0,
        "quantity": 0.05,
        "remaining_quantity": 0.0,
        "reason": "fresh_free_balance_exit",
    }
    result = instance.mirror_events([event])[0]
    assert result["status"] == "closed"
    assert fake.created[-1]["side"] == "sell"
    assert fake.created[-1]["filled"] == pytest.approx(0.02)
    assert instance.health()["fresh_free_balance_exit_sizing"] is True


def test_non_tradeable_dust_closes_executor_risk_and_recycles_fast_slot(tmp_path):
    fake = BalanceBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    instance.state["positions"]["BTC/USDT"] = 0.00002
    instance.state["position_cost_usd"]["BTC/USDT"] = 0.002
    fake.balance_total["BTC"] = 0.00002
    fake.balance_free["BTC"] = 0.00002
    preparation = instance.prepare_sell("BTC/USDT", 0.00002, 100.0)
    assert preparation["status"] == "dust"
    assert preparation["counted_as_executed_close"] is False
    health = instance.health()
    assert "BTC/USDT" not in health["positions"]
    assert "BTC/USDT" in health["non_tradeable_dust"]
    assert health["non_tradeable_dust"]["BTC/USDT"]["removed_from_active_risk_capacity"] is True
    assert health["performance"]["closed_positions"] == 0
    assert health["performance"]["dust_positions_closed"] == 1

    lane_testnet = LaneTestnet()
    lane_testnet.positions["AAA/USDT"] = 0.00002
    lane_testnet.prepare_mode = "dust"
    lane, _service, _ = hyper_lane(tmp_path, testnet=lane_testnet)
    with lane._lock:
        lane.state["active"] = {
            "AAA/USDT": {
                "symbol": "AAA/USDT",
                "quantity": 0.00002,
                "initial_quantity": 0.00002,
                "entry_price": 100.0,
                "entered_at": 900.0,
                "entry_event_id": "entry-dust",
            }
        }
        lane._save_locked()
    event = lane._new_event(symbol="AAA/USDT", side="sell", quantity=0.00002, price=100.0, reason="fast_collective_testnet_exit:dust", now=1_000.0, remaining_quantity=0.0)
    pending = {"kind": "exit", "event": event, "assessment": {"exit_reason": "dust"}, "created_at": 1_000.0}
    lane._set_pending(pending)
    result = lane._submit_pending(pending, now=1_000.0)
    assert result["reason"] == "non_tradeable_dust_slot_recycled"
    assert lane._active_snapshot() == {}
    assert lane._pending() is None
    assert lane_testnet.events == []
    assert lane.health()["dust_recycle_count"] == 1


def test_zero_fill_canceled_exit_reconciles_and_cools_down_before_new_order(tmp_path):
    testnet = LaneTestnet()
    testnet.positions["AAA/USDT"] = 1.0
    testnet._terminal_first = True
    lane, _service, _ = hyper_lane(tmp_path, testnet=testnet)
    with lane._lock:
        lane.state["active"] = {
            "AAA/USDT": {
                "symbol": "AAA/USDT",
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_price": 100.0,
                "entry_notional_usd": 100.0,
                "entered_at": 900.0,
                "entry_event_id": "entry-cancel",
            }
        }
        lane._save_locked()
    event = lane._new_event(symbol="AAA/USDT", side="sell", quantity=1.0, price=101.0, reason="fast_collective_testnet_exit:time", now=1_000.0, remaining_quantity=0.0)
    pending = {"kind": "exit", "event": event, "assessment": {"exit_reason": "time"}, "created_at": 1_000.0}
    lane._set_pending(pending)
    first = lane._submit_pending(pending, now=1_000.0)
    assert first["reason"] == "zero_fill_exit_reconciled_for_corrected_retry"
    assert len(testnet.created_event_ids) == 1
    recovery = lane._pending()
    assert recovery["kind"] == "exit_recovery"
    assert recovery["source_event"]["event_id"] == event["event_id"]
    early = lane._submit_pending(recovery, now=1_005.0)
    assert early["reason"] == "exit_recycle_cooldown"
    assert len(testnet.created_event_ids) == 1
    retry_at = float(recovery["retry_not_before"]) + 0.01
    corrected = lane._submit_pending(lane._pending(), now=retry_at)
    assert corrected["reason"] == "testnet_event_processed"
    assert len(testnet.created_event_ids) == 2
    assert testnet.created_event_ids[0] != testnet.created_event_ids[1]
    assert lane._pending() is None
    assert lane._active_snapshot() == {}
    assert len(testnet.preparations) >= 3
    health = lane.health()
    assert health["zero_fill_terminal_exit_recycle"]["ambiguous_order_resubmission_allowed"] is False
    assert health["live_authority"] is False


def test_compounding_requires_positive_actual_testnet_net_after_cost_floor(tmp_path):
    testnet = LaneTestnet()
    supervisory = supervisor()
    supervisory["capital_growth"] = growth(remaining=15.0, equity=56.0)
    lane, _service, _ = hyper_lane(tmp_path, testnet=testnet, supervisory=supervisory, order_usd=1.0)
    with lane._lock:
        lane.state["closed"] = [{"quantity": 0.1, "entry_price": 100.0, "entry_notional_usd": 10.0, "net_bps_after_model": -10.0, "exited_at": 999.0}]
        lane._save_locked()
    testnet.realized_pnl_usd = -0.25
    losing = lane._compound_order_notional(supervisory, slots=6)
    assert losing["allowed"] is True
    assert losing["compounding"] is False
    assert losing["order_notional_usd"] == pytest.approx(1.0)
    assert losing["canonical_paper_compounding_available"] is True
    assert losing["actual_testnet_profit_compounding_eligible"] is False
    assert losing["modeled_round_trip_cost_floor_bps"] >= 30.0

    testnet.realized_pnl_usd = 1.0
    profitable = lane._compound_order_notional(supervisory, slots=6)
    assert profitable["allowed"] is True
    assert profitable["actual_testnet_profit_compounding_eligible"] is True
    assert profitable["actual_testnet_net_after_modeled_cost_usd"] > 0.0
    assert profitable["order_notional_usd"] > 1.0
    assert profitable["order_notional_usd"] <= profitable["canonical_paper_order_notional_usd"]
    assert profitable["live_authority"] is False
