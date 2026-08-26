from __future__ import annotations

from tests.test_production_testnet_exit_recycle_v1608 import BalanceBybit
from tests.test_testnet_execution import engine


class PriceGuardBybit(BalanceBybit):
    def __init__(self):
        super().__init__()
        self.bid = 100.0
        self.ask = 101.0
        self.sell_limit = 90.0

    def market(self, symbol):
        row = super().market(symbol)
        row["id"] = symbol.replace("/", "")
        return row

    def fetch_ticker(self, symbol):
        self.calls.append(("fetch_ticker", symbol))
        return {"bid": self.bid, "ask": self.ask, "last": self.bid}

    def fetch_order_book(self, symbol, limit=5):
        self.calls.append(("fetch_order_book", symbol, limit))
        return {"bids": [[self.bid, 1000.0]], "asks": [[self.ask, 1000.0]]}

    def public_get_v5_market_price_limit(self, params):
        self.calls.append(("price_limit", dict(params)))
        return {
            "retCode": 0,
            "result": {
                "symbol": params["symbol"],
                "buyLmt": str(self.ask * 2.0),
                "sellLmt": str(self.sell_limit),
            },
        }


class RejectPriceLimitBybit(PriceGuardBybit):
    def create_order(self, symbol, order_type, side, amount, price, params):
        self.calls.append(("create_order", symbol, order_type, side, amount, price, params))
        client_id = params["orderLinkId"]
        observed = {
            "id": f"exchange-{len(self.created) + 1}",
            "clientOrderId": client_id,
            "symbol": symbol,
            "side": side,
            "status": "canceled",
            "filled": 0.0,
            "average": None,
            "fee": {"cost": 0.0},
            "info": {
                "orderLinkId": client_id,
                "rejectReason": "EC_ReachMarketPriceLimit",
                "cancelType": "UNKNOWN",
            },
        }
        self.created.append(observed)
        self.orders[observed["id"]] = observed
        return observed


def sell_event(event_id: str):
    return {
        "timestamp": "2026-08-25T20:45:00+00:00",
        "event_id": event_id,
        "symbol": "BTC/USDT",
        "side": "sell",
        "price": 100.0,
        "quantity": 0.1,
        "remaining_quantity": 0.0,
        "reason": "v1611_regression_exit",
    }


def seed(instance, fake, quantity: float, cost: float):
    instance.state["positions"]["BTC/USDT"] = quantity
    instance.state["position_cost_usd"]["BTC/USDT"] = cost
    fake.balance_total["BTC"] = quantity
    fake.balance_free["BTC"] = quantity
    instance._save_state()


def test_fresh_bid_converts_stale_reference_position_to_dust(tmp_path):
    fake = PriceGuardBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    seed(instance, fake, 0.05, 5.0)
    fake.bid = 80.0
    fake.ask = 81.0
    preparation = instance.prepare_sell("BTC/USDT", 0.05, 100.0)
    assert preparation["status"] == "dust"
    assert preparation["reason"] == "fresh_bid_below_exchange_executable_threshold"
    assert "BTC/USDT" not in instance.health()["positions"]
    assert fake.created == []


def test_price_limit_preflight_blocks_before_order_and_counter_use(tmp_path):
    fake = PriceGuardBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    seed(instance, fake, 0.1, 10.0)
    fake.sell_limit = 105.0
    result = instance.mirror_events([sell_event("blocked")])[0]
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "sell_preparation:bybit_market_price_limit_unexecutable"
    assert fake.created == []
    health = instance.health()
    assert health["daily_total_order_count"] == 0
    assert health["exit_price_guard"]["price_limit_preflight_blocks"] == 1
    assert health["live_authority"] is False


def test_clear_price_limit_preserves_existing_market_exit_path(tmp_path):
    fake = PriceGuardBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    seed(instance, fake, 0.1, 10.0)
    fake.sell_limit = 95.0
    result = instance.mirror_events([sell_event("allowed")])[0]
    assert result["status"] == "closed"
    assert len(fake.created) == 1
    assert fake.created[0]["side"] == "sell"
    assert "BTC/USDT" not in instance.health()["positions"]


def test_observed_price_limit_rejection_arms_cooldown_and_stops_second_order(tmp_path):
    fake = RejectPriceLimitBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()
    seed(instance, fake, 0.1, 10.0)
    fake.sell_limit = 95.0
    first = instance.mirror_events([sell_event("race-1")])[0]
    second = instance.mirror_events([sell_event("race-2")])[0]
    assert first["status"] == "canceled"
    assert second["status"] == "skipped"
    assert second["skip_reason"] == "sell_preparation:bybit_market_price_limit_cooldown"
    assert len(fake.created) == 1
    health = instance.health()
    assert health["exit_price_guard"]["price_limit_rejections"] == 1
    assert health["exit_price_guard"]["retry_storm_order_submission_allowed_while_blocked"] is False


def test_startup_sweep_recycles_persisted_fresh_bid_dust(tmp_path):
    fake = PriceGuardBybit()
    instance, _ = engine(tmp_path, fake)
    instance.state["positions"]["BTC/USDT"] = 0.05
    instance.state["position_cost_usd"]["BTC/USDT"] = 5.0
    fake.balance_total["BTC"] = 0.05
    fake.balance_free["BTC"] = 0.05
    fake.bid = 80.0
    fake.ask = 81.0
    instance._save_state()
    instance.start()
    health = instance.health()
    assert "BTC/USDT" not in health["positions"]
    assert "BTC/USDT" in health["non_tradeable_dust"]
    assert health["exit_price_guard"]["startup_fresh_bid_dust_sweep"]["recycled"] == ["BTC/USDT"]
    assert health["performance"]["closed_positions"] == 0
    assert health["live_authority"] is False


def test_startup_sweep_handles_raw_subminimum_before_precision(tmp_path):
    class StrictMinimumBybit(PriceGuardBybit):
        def amount_to_precision(self, symbol, amount):
            minimum = float(
                (
                    (
                        self.market(symbol).get("limits")
                        or {}
                    ).get("amount")
                    or {}
                ).get("min")
                or 0.0
            )
            if minimum > 0.0 and float(amount) < minimum:
                raise AssertionError(
                    "precision must not be called for raw subminimum dust"
                )
            return super().amount_to_precision(symbol, amount)

    fake = StrictMinimumBybit()
    instance, _ = engine(tmp_path, fake)

    instance.state["positions"]["BTC/USDT"] = 0.0005
    instance.state["position_cost_usd"]["BTC/USDT"] = 0.05
    fake.balance_total["BTC"] = 0.0005
    fake.balance_free["BTC"] = 0.0005
    fake.bid = 100.0
    fake.ask = 101.0

    instance._save_state()
    instance.start()

    health = instance.health()

    assert "BTC/USDT" not in health["positions"]
    assert "BTC/USDT" in health["non_tradeable_dust"]
    assert (
        health["exit_price_guard"]
        ["startup_fresh_bid_dust_sweep"]
        ["errors"]
        == []
    )
    assert health["performance"]["closed_positions"] == 0
    assert health["live_authority"] is False
