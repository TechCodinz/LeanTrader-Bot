from __future__ import annotations

from leantrader.production.exchange_protection import ExchangeProtectionOrchestrator
from tests.test_exchange_protection import IntelligenceStub, engine_health, execution_health


SELF_REFERENTIAL_EXECUTOR = "bybit_testnet_execution"


def fresh_execution_health():
    health = execution_health()
    health["account_balance"]["timestamp"] = "2026-08-24T23:35:00+00:00"
    health["last_reconciliation"] = "2026-08-24T23:35:00+00:00"
    health["last_reconciliation_errors"] = []
    return health


def test_stale_bybit_registry_health_does_not_block_fresh_verified_executor():
    policy = ExchangeProtectionOrchestrator(IntelligenceStub())
    runtime_health = engine_health()
    runtime_health[SELF_REFERENTIAL_EXECUTOR] = {
        "healthy": False,
        "degraded": True,
        "stale": True,
        "reason": "stale_registry_health",
    }
    execution = fresh_execution_health()

    result = policy.authorize_execution(
        symbol="BTC/USDT",
        side="buy",
        execution_health=execution,
        engine_health=runtime_health,
    )

    assert execution["authenticated"] is True
    assert execution["sandbox_endpoint_verified"] is True
    assert execution["last_reconciliation_errors"] == []
    assert result["allowed"] is True
    assert result["checks"]["required_runtime_engines_healthy"] is True
    assert SELF_REFERENTIAL_EXECUTOR in result["active_execution_engines"]
    assert result["execution_authority"] == "testnet_only"
    assert result["live_authority"] is False


def test_every_unrelated_required_engine_failure_still_fails_closed():
    policy = ExchangeProtectionOrchestrator(IntelligenceStub())
    unrelated_required_engines = [
        name
        for name in policy.SPOT_EXECUTION_ENGINES
        if name != SELF_REFERENTIAL_EXECUTOR
    ]
    assert unrelated_required_engines

    for failed_engine in unrelated_required_engines:
        runtime_health = engine_health()
        runtime_health[SELF_REFERENTIAL_EXECUTOR] = {
            "healthy": False,
            "degraded": True,
            "stale": True,
            "reason": "stale_registry_health",
        }
        runtime_health[failed_engine] = {
            "healthy": False,
            "reason": "required_engine_regression_failure",
        }

        result = policy.authorize_execution(
            symbol="BTC/USDT",
            side="buy",
            execution_health=fresh_execution_health(),
            engine_health=runtime_health,
        )

        assert result["allowed"] is False, failed_engine
        assert result["checks"]["required_runtime_engines_healthy"] is False, failed_engine
        assert result["reason"] == "required_runtime_engines_healthy", failed_engine
        assert result["active_execution_engines"] == [], failed_engine
        assert result["execution_authority"] is False, failed_engine
        assert result["live_authority"] is False, failed_engine
