from __future__ import annotations

import json

import pytest

from leantrader.production.testnet_execution import BybitTestnetExecutionEngine
from leantrader.production.testnet_execution import TestnetSafetyError as SandboxSafetyError


class FakeBybit:
    def __init__(self, _config=None, *, unsafe_urls: bool = False):
        self.id = "bybit"
        self.calls = []
        self.created = []
        self.orders = {}
        self.unsafe_urls = unsafe_urls
        self.has = {
            "fetchBalance": True,
            "createOrder": True,
            "createMarketOrder": True,
            "fetchOrder": True,
            "fetchOpenOrder": True,
            "fetchClosedOrder": True,
            "fetchOpenOrders": True,
            "fetchClosedOrders": True,
            "cancelOrder": True,
            "fetchMyTrades": True,
        }
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

    def private_get_v5_user_query_api(self):
        self.calls.append(("query_api_key",))
        return {
            "result": {
                "readOnly": 0,
                "permissions": {"Spot": ["SpotTrade"], "Wallet": []},
                "ips": ["169.58.175.192"],
                "type": 1,
            }
        }

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

    def fetch_order(self, order_id, _symbol, _params=None):
        return self.orders[order_id]

    def fetch_open_order(self, order_id, _symbol):
        order = self.orders.get(order_id)

        if (
            order is None
            or str(order.get("status") or "").lower()
            != "open"
        ):
            raise RuntimeError("order is not open")

        return order

    def fetch_closed_order(self, order_id, _symbol):
        order = self.orders.get(order_id)

        if (
            order is None
            or str(order.get("status") or "").lower()
            not in {"closed", "canceled"}
        ):
            raise RuntimeError("order is not closed")

        return order

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
    assert fake.calls[2] == ("query_api_key",)
    assert fake.calls[3] == ("fetch_balance",)
    assert instance.health()["sandbox_endpoint_verified"] is True
    assert instance.health()["authenticated"] is True
    assert instance.health()["live_authority"] is False
    assert instance.health()["api_attestation"]["verified"] is True
    assert instance.health()["api_attestation"]["withdrawal_permission"] is False
    assert instance.health()["exchange_capabilities"]["execution_market_type"] == "spot"
    assert instance.health()["exchange_capabilities"]["market_types_observed"]["spot"] == 3
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


def test_read_only_or_withdrawal_enabled_key_is_rejected(tmp_path):
    read_only = FakeBybit()
    read_only.private_get_v5_user_query_api = lambda: {
        "result": {"readOnly": 1, "permissions": {"Spot": [], "Wallet": []}}
    }
    instance, _ = engine(tmp_path, read_only)
    with pytest.raises(SandboxSafetyError, match="read-only"):
        instance.start()

    withdrawal = FakeBybit()
    withdrawal.private_get_v5_user_query_api = lambda: {
        "result": {
            "readOnly": 0,
            "permissions": {"Spot": ["SpotTrade"], "Wallet": ["Withdraw"]},
        }
    }
    instance, _ = engine(tmp_path, withdrawal)
    with pytest.raises(SandboxSafetyError, match="withdrawal"):
        instance.start()


def test_missing_required_exchange_capability_is_rejected(tmp_path):
    fake = FakeBybit()
    fake.has["createOrder"] = False
    instance, _ = engine(tmp_path, fake)
    with pytest.raises(SandboxSafetyError, match="createOrder"):
        instance.start()


def test_non_bybit_adapter_is_rejected(tmp_path):
    fake = FakeBybit()
    fake.id = "binance"
    instance, _ = engine(tmp_path, fake)
    with pytest.raises(SandboxSafetyError, match="unsupported testnet exchange"):
        instance.start()


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


def test_current_ccxt_bybit_template_urls_are_supported(tmp_path):
    fake = FakeBybit()
    fake.hostname = "bybit.com"

    def sandbox(enabled):
        fake.calls.append(("sandbox", enabled))
        fake.urls = {
            "api": {
                "spot": "https://api-testnet.{hostname}",
                "public": "https://api-testnet.{hostname}",
                "private": "https://api-testnet.{hostname}",
            }
        }

    fake.set_sandbox_mode = sandbox

    instance, _ = engine(tmp_path, fake)
    instance.start()

    assert instance.health()["sandbox_endpoint_verified"] is True
    assert instance.health()["authenticated"] is True


def test_ccxt_template_cannot_escape_approved_bybit_hosts(tmp_path):
    fake = FakeBybit()
    fake.hostname = "example.com"

    def sandbox(enabled):
        fake.calls.append(("sandbox", enabled))
        fake.urls = {
            "api": {
                "public": "https://api-testnet.{hostname}",
                "private": "https://api-testnet.{hostname}",
            }
        }

    fake.set_sandbox_mode = sandbox

    instance, _ = engine(tmp_path, fake)

    with pytest.raises(
        SandboxSafetyError,
        match="unexpected Bybit sandbox hostname",
    ):
        instance.start()


def test_required_reconciliation_fails_closed_and_recovers(tmp_path):
    instance, fake = engine(tmp_path)
    instance.start()

    healthy_fetch_balance = fake.fetch_balance

    def temporary_failure():
        raise RuntimeError("temporary balance failure")

    fake.fetch_balance = temporary_failure

    with pytest.raises(
        RuntimeError,
        match="testnet reconciliation is unresolved",
    ):
        instance.reconcile_required()

    assert (
        instance.health()["last_reconciliation_errors"]
    )

    fake.fetch_balance = healthy_fetch_balance

    result = instance.reconcile_required()

    assert result["reconciled"] is True
    assert (
        instance.health()["last_reconciliation_errors"]
        == []
    )


def test_bybit_spot_reconciliation_avoids_generic_fetch_order_limit(
    tmp_path,
):
    instance, fake = engine(tmp_path)
    instance.start()

    event = buy_event()
    client_id = instance._client_order_id(event)
    order_id = "spot-recovery-1"

    instance.state["orders"][client_id] = {
        "client_order_id": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.05,
        "submitted_usd": 5.0,
        "reference_price": 100.0,
        "reason": "paper_entry",
        "paper_event_timestamp": event["timestamp"],
        "status": "submitting",
        "order_id": order_id,
        "filled": 0.0,
        "applied_filled": 0.0,
        "filled_cost": 0.0,
        "applied_fill_cost": 0.0,
        "average": None,
        "fee": 0.0,
        "fee_currency": None,
        "applied_fee": 0.0,
        "fill_counted": False,
    }

    fake.orders[order_id] = {
        "id": order_id,
        "clientOrderId": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "status": "closed",
        "filled": 0.05,
        "average": 100.0,
        "cost": 5.0,
        "fee": {
            "cost": 0.0,
            "currency": "USDT",
        },
        "info": {
            "orderLinkId": client_id,
        },
    }

    def blocked_generic_fetch_order(*_args, **_kwargs):
        raise RuntimeError(
            "bybit fetchOrder() can only access "
            "an order if it is in last 500 orders"
        )

    fake.fetch_order = blocked_generic_fetch_order

    result = instance.reconcile_required()

    assert result["reconciled"] is True
    assert result["errors"] == []

    record = instance.state["orders"][client_id]

    assert record["status"] == "closed"
    assert record["filled"] == pytest.approx(0.05)
    assert (
        instance.health()["positions"]["BTC/USDT"]
        == pytest.approx(0.05)
    )



class NativeRecoveryBybit(FakeBybit):
    def __init__(
        self,
        *,
        native_order=None,
        native_empty=False,
    ):
        super().__init__()
        self.native_order = native_order
        self.native_empty = native_empty

    def market(self, symbol):
        row = super().market(symbol)
        return {
            **row,
            "id": symbol.replace("/", ""),
        }

    def _native_rows(self):
        if self.native_order is None:
            return []
        return [dict(self.native_order)]

    def private_get_v5_order_realtime(self, params):
        return {
            "retCode": 0,
            "result": {
                "list": self._native_rows(),
            },
        }

    def private_get_v5_order_history(self, params):
        return {
            "retCode": 0,
            "result": {
                "list": self._native_rows(),
            },
        }

    def private_get_v5_execution_list(self, params):
        return {
            "retCode": 0,
            "result": {
                "list": [],
            },
        }


def _ambiguous_without_exchange_id(
    instance,
    event,
):
    client_id = instance._client_order_id(event)

    instance.state["orders"][client_id] = {
        "client_order_id": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.05,
        "submitted_usd": 5.0,
        "reference_price": 100.0,
        "reason": "paper_entry",
        "paper_event_timestamp": (
            "2026-08-14T12:00:00+00:00"
        ),
        "status": "submitting",
        "order_id": None,
        "filled": 0.0,
        "applied_filled": 0.0,
        "filled_cost": 0.0,
        "applied_fill_cost": 0.0,
        "average": None,
        "fee": 0.0,
        "fee_currency": None,
        "applied_fee": 0.0,
        "fill_counted": False,
    }

    instance._save_state()

    return client_id


def test_native_bybit_order_link_id_recovers_without_order_id(
    tmp_path,
):
    event = buy_event()

    temporary, _ = engine(tmp_path)
    client_id = temporary._client_order_id(event)

    native = {
        "orderId": "native-order-1",
        "orderLinkId": client_id,
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderStatus": "Filled",
        "cumExecQty": "0.05",
        "cumExecValue": "5.0",
        "avgPrice": "100",
    }

    fake = NativeRecoveryBybit(
        native_order=native,
    )

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    client_id = _ambiguous_without_exchange_id(
        instance,
        event,
    )

    result = instance.reconcile_required()

    assert result["reconciled"] is True
    assert result["errors"] == []

    record = instance.state["orders"][client_id]

    assert record["order_id"] == "native-order-1"
    assert record["status"] == "closed"
    assert record["filled"] == pytest.approx(0.05)
    assert (
        record["reconciliation_resolution"]
        == "native_bybit_order_link_id"
    )


def test_native_bybit_authoritative_absence_resolves_old_ambiguity(
    tmp_path,
):
    fake = NativeRecoveryBybit(
        native_order=None,
        native_empty=True,
    )

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    client_id = _ambiguous_without_exchange_id(
        instance,
        buy_event(),
    )

    result = instance.reconcile_required()

    assert result["reconciled"] is True
    assert result["errors"] == []

    record = instance.state["orders"][client_id]

    assert record["status"] == "rejected"
    assert record["filled"] == 0.0
    assert (
        record["reconciliation_resolution"]
        == "native_bybit_authoritative_absence"
    )


class TransientNativeAbsenceBybit(
    NativeRecoveryBybit
):
    def __init__(self):
        super().__init__(
            native_order=None,
            native_empty=True,
        )
        self.realtime_calls = 0

    def private_get_v5_order_realtime(
        self,
        params,
    ):
        self.realtime_calls += 1

        if self.realtime_calls == 1:
            raise RuntimeError(
                "temporary testnet endpoint failure"
            )

        return super().private_get_v5_order_realtime(
            params
        )


def test_v1603_old_ambiguity_recovers_without_restart(
    tmp_path,
):
    fake = TransientNativeAbsenceBybit()

    instance, _ = engine(
        tmp_path,
        fake,
    )

    instance.start()

    client_id = _ambiguous_without_exchange_id(
        instance,
        buy_event(),
    )

    result = instance.reconcile_required()

    assert result["reconciled"] is True
    assert result["errors"] == []
    assert fake.realtime_calls >= 2

    record = instance.state["orders"][
        client_id
    ]

    assert record["status"] == "rejected"

    recovery = instance.health()[
        "automatic_reconciliation_recovery"
    ]

    assert recovery["enabled"] is True
    assert recovery["resubmission_allowed"] is False
    assert recovery["retry_successes"] >= 1


def test_spot_executor_loads_only_spot_market_metadata(tmp_path):
    key = tmp_path / "key-v1621"
    secret = tmp_path / "secret-v1621"

    key.write_text(
        "testnet-key-v1621",
        encoding="utf-8",
    )
    secret.write_text(
        "testnet-secret-v1621",
        encoding="utf-8",
    )

    fake = FakeBybit()
    captured = {}

    def factory(config):
        captured.update(config)
        return fake

    instance = BybitTestnetExecutionEngine(
        api_key_path=key,
        api_secret_path=secret,
        state_path=tmp_path / "state-v1621.json",
        confirmation="I_UNDERSTAND_TESTNET_ONLY",
        exchange_factory=factory,
    )

    instance.start()

    options = captured["options"]

    assert options["defaultType"] == "spot"

    assert options["fetchMarkets"] == {
        "types": ["spot"],
    }

    # Sandbox selection remains first.
    assert fake.calls[0] == (
        "sandbox",
        True,
    )

    assert fake.calls[1] == (
        "load_markets",
    )

    # Execution authority remains spot-only.
    assert instance.eligible_symbols(
        "USDT"
    ) == {
        "BTC/USDT",
        "ETH/USDT",
    }

    assert instance.health()[
        "live_authority"
    ] is False
