from __future__ import annotations

from typing import Any

from leantrader.production.exchange_protection import ExchangeProtectionOrchestrator


class IntelligenceStub:
    def __init__(self, market_type: str = "spot") -> None:
        self.market_type = market_type
        self.profile = {
            "exchange_id": "bybit",
            "environment": "public_market_data",
            "capabilities": {
                "fetchOrderBook": True,
                "fetchOHLCV": True,
                "fetchTrades": True,
                "fetchFundingRate": True,
                "fetchOpenInterest": True,
                "fetchTickers": True,
            },
        }

    def market_rules(self, symbol: str) -> dict[str, Any]:
        flags = {name: False for name in ("spot", "margin", "swap", "future", "option")}
        flags[self.market_type] = True
        if self.market_type == "spot":
            # CCXT may mark a spot market as margin-capable while its canonical
            # product type remains spot.
            flags["margin"] = True
        return {
            "available": True,
            "symbol": symbol,
            "type": self.market_type,
            "active": True,
            "precision": {"amount": 1e-6, "price": 0.01},
            "limits": {"cost": {"min": 1.0}},
            "taker_fee": 0.001,
            **flags,
        }


def engine_health() -> dict[str, dict[str, Any]]:
    return {
        name: {"healthy": True}
        for name in ExchangeProtectionOrchestrator.SPOT_EXECUTION_ENGINES
    }


def execution_health() -> dict[str, Any]:
    protections = {
        name: True
        for name in ExchangeProtectionOrchestrator.PRODUCT_PROTECTION_REQUIREMENTS["spot"]
    }
    return {
        "provider": "bybit",
        "environment": "testnet",
        "sandbox_endpoint_verified": True,
        "authenticated": True,
        "execution_authority": "testnet_only",
        "live_authority": False,
        "api_attestation": {
            "verified": True,
            "read_write": True,
            "spot_trade": True,
            "withdrawal_permission": False,
            "ip_bound": True,
        },
        "exchange_capabilities": {
            "methods": {
                "fetchBalance": True,
                "createOrder": True,
                "fetchOrder": True,
                "fetchOpenOrders": True,
                "fetchClosedOrders": True,
            }
        },
        "account_balance": {"timestamp": "2026-08-14T12:00:00+00:00"},
        "last_reconciliation": "2026-08-14T12:00:00+00:00",
        "last_reconciliation_errors": [],
        "protection_contract": protections,
        "kill_switch_active": False,
    }


def test_research_plan_selects_only_exchange_supported_engines():
    policy = ExchangeProtectionOrchestrator(IntelligenceStub())
    plan = policy.research_plan("BTC/USDT")
    assert {
        "fluid_liquidity",
        "smart_scalping",
        "moon_scout_dynamic_scanner",
    } <= set(plan["enabled_research_engines"])
    assert set(plan["available_unbound_observations"]) == {
        "public_trade_tape",
        "funding_rate",
        "open_interest",
    }
    assert plan["execution_authority"] is False


def test_verified_bybit_spot_testnet_receives_bounded_authority():
    policy = ExchangeProtectionOrchestrator(IntelligenceStub())
    result = policy.authorize_execution(
        symbol="BTC/USDT",
        side="buy",
        execution_health=execution_health(),
        engine_health=engine_health(),
    )
    assert result["allowed"] is True
    assert result["execution_authority"] == "testnet_only"
    assert result["live_authority"] is False
    assert result["missing_protections"] == []


def test_unbound_key_and_derivative_product_fail_closed():
    policy = ExchangeProtectionOrchestrator(IntelligenceStub("swap"))
    health = execution_health()
    health["api_attestation"]["ip_bound"] = False
    result = policy.authorize_execution(
        symbol="BTC/USDT:USDT",
        side="buy",
        execution_health=health,
        engine_health=engine_health(),
    )
    assert result["allowed"] is False
    assert "api_key_ip_bound" in result["missing_protections"]
    assert "spot_product_supported" in result["missing_protections"]
    assert any(
        item.startswith("missing_product_protection:")
        for item in result["missing_protections"]
    )
    assert result["live_authority"] is False


def test_kill_switch_blocks_entries_but_preserves_protected_exit_path():
    policy = ExchangeProtectionOrchestrator(IntelligenceStub())
    health = execution_health()
    health["kill_switch_active"] = True
    buy = policy.authorize_execution(
        symbol="BTC/USDT",
        side="buy",
        execution_health=health,
        engine_health=engine_health(),
    )
    sell = policy.authorize_execution(
        symbol="BTC/USDT",
        side="sell",
        execution_health=health,
        engine_health=engine_health(),
    )
    assert buy["allowed"] is False
    assert buy["reason"] == "kill_switch_allows_action"
    assert sell["allowed"] is True
