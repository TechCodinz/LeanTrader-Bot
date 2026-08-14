from __future__ import annotations

import json

import pytest

from leantrader.production.testnet_execution import BybitTestnetExecutionEngine
from leantrader.production.testnet_execution import TestnetSafetyError as SandboxSafetyError


class FakeBybit:
    def __init__(self, _config=None, *, unsafe_urls: bool = False):
        self.calls = []
        self.created = []
        self.orders = {}
        self.unsafe_urls = unsafe_urls
        self.markets = {
            "BTC/USDT": {"symbol": "BTC/USDT", "spot": True, "active": True, "quote": "USDT"},
            "ETH/USDT": {"symbol": "ETH/USDT", "spot": True, "active": True, "quote": "USDT"},
            "BTC/USDC": {"symbol": "BTC/USDC", "spot": True, "active": True, "quote": "USDC"},
        }
        self.urls = {"api": {"public": "https://api.bybit.com", "private": "https://api.bybit.com"}}

    def set_sandbox_mode(self, enabled):
        self.calls.append(("sandbox", enabled))
        if not self.unsafe_urls:
            self.urls = {
                "api": {
                    "public": "https://api-testnet.bybit.com",
                    "private": "https://api-testnet.bybit.com",
                }
            }

    def load_markets(self):
        self.calls.append(("load_markets",))

    def market(self, _symbol):
        return {"limits": {"cost": {"min": 5.0}, "amount": {"min": 0.001}}}

    def fetch_balance(self):
        self.calls.append(("fetch_balance",))
        return {"total": {"USDT": 10_000.0}}

    def amount_to_precision(self, _symbol, amount):
        return f"{amount:.6f}"

    def create_order(self, symbol, order_type, side, amount, price, params):
        self.calls.append(("create_order", symbol, order_type, side, amount, price, params))
        client_id = params["orderLinkId"]
        observed = {
            "id": f"exchange-{len(self.created) + 1}",
            "clientOrderId": client_id,
            "symbol": symbol,
            "side": side,
            "status": "closed",
            "filled": amount,
            "average": 100.0,
            "fee": {"cost": 0.01},
            "info": {"orderLinkId": client_id},
        }
        self.created.append(observed)
        self.orders[observed["id"]] = observed
        return observed

    def fetch_order(self, order_id, _symbol):
        return self.orders[order_id]

    def fetch_open_orders(self, _symbol, _since, _limit, params):
        return [
            order
            for order in self.orders.values()
            if order["status"] == "open" and order["clientOrderId"] == params["orderLinkId"]
        ]

    def fetch_closed_orders(self, _symbol, _since, _limit, params):
        return [
            order
            for order in self.orders.values()
            if order["status"] == "closed" and order["clientOrderId"] == params["orderLinkId"]
        ]

    def fetch_canceled_orders(self, _symbol, _since, _limit, params):
        return []


def engine(tmp_path, fake=None, **overrides):
    key = tmp_path / "key"
    secret = tmp_path / "secret"
    key.write_text("testnet-key-123", encoding="utf-8")
    secret.write_text("testnet-secret-123", encoding="utf-8")
    fake = fake or FakeBybit()
    instance = BybitTestnetExecutionEngine(
        api_key_path=key,
        api_secret_path=secret,
        state_path=tmp_path / "testnet-state.json",
        confirmation="I_UNDERSTAND_TESTNET_ONLY",
        exchange_factory=lambda _config: fake,
        **overrides,
    )
    return instance, fake


def buy_event():
    return {
        "timestamp": "2026-08-14T12:00:00+00:00",
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 100.0,
        "quantity": 0.02,
        "reason": "paper_entry",
    }


def test_sandbox_switch_is_first_call_and_endpoint_is_verified(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()
    assert fake.calls[0] == ("sandbox", True)
    assert fake.calls[1] == ("load_markets",)
    assert fake.calls[2] == ("fetch_balance",)
    assert instance.health()["sandbox_endpoint_verified"] is True
    assert instance.health()["authenticated"] is True
    assert instance.health()["live_authority"] is False
    assert instance.eligible_symbols("USDT") == {"BTC/USDT", "ETH/USDT"}


def test_market_unavailable_on_testnet_is_skipped_without_provider_call(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()
    event = {**buy_event(), "symbol": "DOGE/USDT"}
    result = instance.mirror_events([event])[0]
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "market_unavailable_on_bybit_testnet"
    assert fake.created == []


def test_production_endpoint_is_rejected(tmp_path):
    instance, _fake = engine(tmp_path, FakeBybit(unsafe_urls=True))
    with pytest.raises(SandboxSafetyError, match="not exclusively"):
        instance.start()


def test_runtime_endpoint_change_is_rejected_before_order(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()
    fake.urls = {"api": {"private": "https://api.bybit.com"}}
    with pytest.raises(SandboxSafetyError, match="not exclusively"):
        instance.mirror_events([buy_event()])
    assert fake.created == []


def test_invalid_private_credentials_cannot_report_ready(tmp_path):
    fake = FakeBybit()

    def reject_balance():
        raise RuntimeError("invalid api key")

    fake.fetch_balance = reject_balance
    instance, _ = engine(tmp_path, fake)
    with pytest.raises(RuntimeError, match="invalid api key"):
        instance.start()
    assert instance.authenticated is False


def test_order_is_minimum_aware_idempotent_and_persistent(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()
    first = instance.mirror_events([buy_event()])[0]
    duplicate = instance.mirror_events([buy_event()])[0]

    assert first["idempotent"] is False
    assert first["submitted_usd"] == 5.0
    assert duplicate["idempotent"] is True
    assert len(fake.created) == 1
    assert instance.health()["positions"]["BTC/USDT"] == pytest.approx(0.05)
    assert instance.health()["position_cost_usd"]["BTC/USDT"] == pytest.approx(5.01)
    assert instance.health()["account_balance"]["assets"]["USDT"] == 10_000.0
    assert instance.health()["performance"]["filled_orders"] == 1
    assert instance.health()["performance"]["average_adverse_slippage_bps"] == pytest.approx(0.0)

    saved = json.loads((tmp_path / "testnet-state.json").read_text(encoding="utf-8"))
    assert saved["daily_order_count"] == 1
    assert saved["daily_submitted_usd"] == 5.0


def test_partial_fill_reconciliation_applies_only_the_delta(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()
    client_id = instance._client_order_id(buy_event())
    fake_order = {
        "id": "partial-1",
        "clientOrderId": client_id,
        "status": "open",
        "filled": 0.02,
        "average": 100.0,
        "fee": {"cost": 0.0},
        "info": {"orderLinkId": client_id},
    }

    def create_partial(*_args, **_kwargs):
        fake.created.append(fake_order)
        fake.orders["partial-1"] = fake_order
        return dict(fake_order)

    fake.create_order = create_partial
    instance.mirror_events([buy_event()])
    assert instance.health()["positions"]["BTC/USDT"] == pytest.approx(0.02)

    fake.orders["partial-1"] = {**fake_order, "status": "closed", "filled": 0.05}
    instance.reconcile()
    instance.reconcile()
    assert instance.health()["positions"]["BTC/USDT"] == pytest.approx(0.05)


def test_unfilled_buy_reserves_position_capacity(tmp_path):
    instance, fake = engine(tmp_path, max_order_usd=5.0, max_position_usd=9.0)
    instance.start()

    def create_open(symbol, _order_type, side, amount, _price, params):
        observed = {
            "id": "open-1",
            "clientOrderId": params["orderLinkId"],
            "symbol": symbol,
            "side": side,
            "status": "open",
            "filled": 0.0,
            "info": {"orderLinkId": params["orderLinkId"]},
        }
        fake.created.append(observed)
        fake.orders["open-1"] = observed
        return observed

    fake.create_order = create_open
    instance.mirror_events([buy_event()])
    second = {**buy_event(), "timestamp": "2026-08-14T12:01:00+00:00"}
    blocked = instance.mirror_events([second])[0]
    assert blocked["status"] == "skipped"
    assert blocked["skip_reason"] == "position_notional_cap"
    assert len(fake.created) == 1


def test_restart_recovers_ambiguous_submission_by_client_id(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()
    event = buy_event()
    client_id = instance._client_order_id(event)
    instance.state["orders"][client_id] = {
        "client_order_id": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.05,
        "submitted_usd": 5.0,
        "status": "submitting",
        "order_id": None,
        "filled": 0.0,
        "applied_filled": 0.0,
    }
    instance._save_state()
    fake.orders["accepted-during-timeout"] = {
        "id": "accepted-during-timeout",
        "clientOrderId": client_id,
        "status": "closed",
        "filled": 0.05,
        "average": 100.0,
        "fee": {"cost": 0.0},
        "info": {"orderLinkId": client_id},
    }

    restarted, _ = engine(tmp_path, fake)
    restarted.start()
    result = restarted.mirror_events([event])[0]
    assert result["idempotent"] is True
    assert result["order_id"] == "accepted-during-timeout"
    assert restarted.health()["positions"]["BTC/USDT"] == pytest.approx(0.05)
    assert len(fake.created) == 0


def test_kill_switch_blocks_entries_but_does_not_block_exit(tmp_path):
    instance, fake = engine(tmp_path, max_orders_per_day=1, max_daily_submitted_usd=10.0)
    instance.start()
    (tmp_path / "TESTNET_HALT").touch()
    blocked = instance.mirror_events([buy_event()])[0]
    assert blocked["status"] == "skipped"
    assert blocked["skip_reason"] == "testnet_kill_switch"

    instance.state["positions"]["BTC/USDT"] = 0.05
    instance.state["daily_order_count"] = 1
    instance.state["daily_submitted_usd"] = 10.0
    sell = {
        **buy_event(),
        "timestamp": "2026-08-14T12:05:00+00:00",
        "side": "sell",
        "quantity": 0.05,
        "remaining_quantity": 0.0,
    }
    exited = instance.mirror_events([sell])[0]
    assert exited["status"] == "closed"
    assert exited["side"] == "sell"
    assert "BTC/USDT" not in instance.health()["positions"]
    assert len(fake.created) == 1


def test_testnet_round_trip_performance_is_observable(tmp_path):
    instance, _fake = engine(tmp_path)
    instance.start()
    instance.mirror_events([buy_event()])
    sell = {
        **buy_event(),
        "timestamp": "2026-08-14T12:10:00+00:00",
        "side": "sell",
        "quantity": 0.05,
        "remaining_quantity": 0.0,
        "reason": "paper_exit",
    }
    instance.mirror_events([sell])
    performance = instance.health()["performance"]
    assert performance["filled_orders"] == 2
    assert performance["closed_positions"] == 1
    assert performance["winning_positions"] == 0
    assert performance["realized_pnl_usd"] == pytest.approx(-0.02)
