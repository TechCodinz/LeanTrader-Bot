from __future__ import annotations

import time
from typing import Any


class ExchangeProtectionOrchestrator:
    """Translate exchange capabilities into fail-closed engine authority.

    Public APIs may activate compatible research engines. Authenticated order
    authority is a separate decision and is granted only when the adapter,
    environment, market rules, account permissions, reconciliation and runtime
    protection engines all attest successfully.
    """

    VERSION = "1.0"

    CORE_RESEARCH_ENGINES = (
        "adaptive_intelligence",
        "advanced_shadow_suite",
        "research_governor",
        "decision_router",
        "strategy_observatory",
        "market_temporal_guard",
        "news_awareness",
        "swarm_hivemind",
        "photographic_pattern_memory",
        "multi_timeframe_matrix",
        "technical_structure",
        "spectral_harmonics",
        "fundamental_market_context",
        "portfolio_risk",
    )
    SPOT_EXECUTION_ENGINES = (
        "market_data",
        "exchange_intelligence",
        "market_temporal_guard",
        "market_universe",
        "paper_ledger",
        "adaptive_intelligence",
        "advanced_shadow_suite",
        "research_governor",
        "decision_router",
        "operations_safety",
        "bybit_testnet_execution",
    )
    PRODUCT_PROTECTION_REQUIREMENTS = {
        "spot": (
            "market_precision_and_limits",
            "fee_and_slippage_model",
            "balance_reconciliation",
            "order_idempotency",
            "order_state_recovery",
            "position_and_daily_caps",
            "kill_switch",
        ),
        "margin": (
            "borrow_liability_reconciliation",
            "interest_accrual_model",
            "leverage_and_liquidation_guard",
            "collateral_health_monitor",
        ),
        "swap": (
            "contract_size_normalization",
            "funding_and_mark_price_monitor",
            "reduce_only_exit_semantics",
            "leverage_and_liquidation_guard",
            "position_mode_reconciliation",
        ),
        "future": (
            "contract_size_normalization",
            "expiry_and_settlement_guard",
            "reduce_only_exit_semantics",
            "leverage_and_liquidation_guard",
            "position_mode_reconciliation",
        ),
        "option": (
            "greeks_and_volatility_surface",
            "expiry_exercise_assignment_guard",
            "multi_leg_atomicity",
            "portfolio_margin_reconciliation",
        ),
        "forex": (
            "provider_contract_normalization",
            "session_and_rollover_guard",
            "leverage_and_margin_reconciliation",
            "broker_order_state_recovery",
        ),
        "arbitrage": (
            "per_venue_inventory_reconciliation",
            "dual_leg_failure_recovery",
            "transfer_latency_and_settlement_guard",
            "cross_venue_kill_switch",
        ),
    }

    def __init__(self, exchange_intelligence: Any) -> None:
        self.exchange_intelligence = exchange_intelligence
        self.research_profiles = 0
        self.authorization_checks = 0
        self.authorized = 0
        self.blocked = 0
        self.block_reasons: dict[str, int] = {}
        self.last_research_plan: dict[str, Any] = {}
        self.last_authorizations: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        self.research_plan()

    def research_plan(self, symbol: str | None = None) -> dict[str, Any]:
        profile = dict(getattr(self.exchange_intelligence, "profile", {}) or {})
        capabilities = dict(profile.get("capabilities") or {})
        enabled = list(self.CORE_RESEARCH_ENGINES)
        if capabilities.get("fetchOHLCV"):
            enabled.extend(("smart_scalping", "moon_scout_dynamic_scanner"))
        if capabilities.get("fetchOrderBook"):
            enabled.append("fluid_liquidity")
        available_observations = [
            name
            for name, capability in (
                ("public_trade_tape", "fetchTrades"),
                ("funding_rate", "fetchFundingRate"),
                ("open_interest", "fetchOpenInterest"),
            )
            if capabilities.get(capability)
        ]
        market_rules = self.exchange_intelligence.market_rules(symbol) if symbol else {}
        market_type = self._market_type(market_rules) if symbol else "portfolio"
        plan = {
            "exchange_id": profile.get("exchange_id"),
            "environment": profile.get("environment", "public_market_data"),
            "symbol": symbol,
            "market_type": market_type,
            "enabled_research_engines": list(dict.fromkeys(enabled)),
            "available_unbound_observations": available_observations,
            "disabled_research_reasons": self._disabled_research(capabilities),
            "execution_requested": False,
            "execution_authority": False,
            "provider_rules_dynamic": True,
            "planned_at": time.time(),
        }
        self.research_profiles += 1
        self.last_research_plan = plan
        return dict(plan)

    def authorize_execution(
        self,
        *,
        symbol: str,
        side: str,
        execution_health: dict[str, Any] | None,
        engine_health: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Authorize one paper-to-Testnet mirror; never authorizes live orders."""
        self.authorization_checks += 1
        execution = dict(execution_health or {})
        profile = dict(getattr(self.exchange_intelligence, "profile", {}) or {})
        rules = self.exchange_intelligence.market_rules(symbol)
        market_type = self._market_type(rules)
        required_protections = list(self.PRODUCT_PROTECTION_REQUIREMENTS.get(market_type, ()))
        checks: dict[str, bool] = {
            "known_exchange_identity": bool(profile.get("exchange_id")),
            "matching_exchange_identity": (
                str(execution.get("provider") or "").lower()
                == str(profile.get("exchange_id") or "").lower()
            ),
            "testnet_environment": execution.get("environment") == "testnet",
            "sandbox_endpoint_verified": execution.get("sandbox_endpoint_verified") is True,
            "authenticated": execution.get("authenticated") is True,
            "api_key_attested": (execution.get("api_attestation") or {}).get("verified") is True,
            "api_key_ip_bound": (execution.get("api_attestation") or {}).get("ip_bound") is True,
            "spot_trade_permission_only": (
                (execution.get("api_attestation") or {}).get("spot_trade") is True
                and (execution.get("api_attestation") or {}).get("read_write") is True
            ),
            "withdrawal_permission_absent": (
                execution.get("api_attestation") or {}
            ).get("withdrawal_permission")
            is False,
            "testnet_authority_only": execution.get("execution_authority") == "testnet_only",
            "live_authority_absent": execution.get("live_authority") is False,
            "market_loaded": rules.get("available") is True and rules.get("active") is True,
            "spot_product_supported": market_type == "spot",
            "market_precision_and_limits": bool(rules.get("precision")) and bool(rules.get("limits")),
            "fee_model_available": rules.get("taker_fee") is not None,
            "balance_reconciled": bool((execution.get("account_balance") or {}).get("timestamp")),
            "reconciliation_performed": bool(execution.get("last_reconciliation")),
            "order_reconciliation_clear": not execution.get("last_reconciliation_errors"),
            "kill_switch_allows_action": (
                side.lower() == "sell" or execution.get("kill_switch_active") is not True
            ),
            "required_runtime_engines_healthy": self._engines_healthy(
                engine_health, self.SPOT_EXECUTION_ENGINES
            ),
        }
        protection_contract = dict(execution.get("protection_contract") or {})
        for protection in self.PRODUCT_PROTECTION_REQUIREMENTS["spot"]:
            checks[f"product_protection:{protection}"] = protection_contract.get(protection) is True
        methods = ((execution.get("exchange_capabilities") or {}).get("methods") or {})
        checks["required_order_methods"] = bool(methods.get("fetchBalance")) and bool(
            methods.get("createOrder")
        )
        checks["order_state_recovery"] = (
            (
                bool(methods.get("fetchOpenOrder"))
                and bool(methods.get("fetchClosedOrder"))
            )
            or (
                bool(methods.get("fetchOpenOrders"))
                and bool(methods.get("fetchClosedOrders"))
            )
            or bool(methods.get("fetchOrder"))
        )

        missing = [name for name, passed in checks.items() if not passed]
        if market_type != "spot":
            missing.extend(
                f"missing_product_protection:{name}" for name in required_protections
            )
        # This verified runtime deliberately has one authenticated executor.
        if str(execution.get("provider") or "").lower() != "bybit":
            missing.append("authenticated_executor_not_implemented")
        allowed = not missing
        reason = "authorized_testnet_spot" if allowed else missing[0]
        plan = {
            "exchange_id": profile.get("exchange_id"),
            "environment": execution.get("environment"),
            "symbol": symbol,
            "side": side.lower(),
            "market_type": market_type,
            "allowed": allowed,
            "reason": reason,
            "checks": checks,
            "missing_protections": list(dict.fromkeys(missing)),
            "required_product_protections": required_protections,
            "active_execution_engines": list(self.SPOT_EXECUTION_ENGINES) if allowed else [],
            "execution_authority": "testnet_only" if allowed else False,
            "live_authority": False,
            "evaluated_at": time.time(),
        }
        key = f"{symbol}:{side.lower()}"
        self.last_authorizations[key] = plan
        self.last_authorizations = dict(list(self.last_authorizations.items())[-50:])
        if allowed:
            self.authorized += 1
        else:
            self.blocked += 1
            self.block_reasons[reason] = self.block_reasons.get(reason, 0) + 1
        return dict(plan)

    @staticmethod
    def _engines_healthy(
        engine_health: dict[str, dict[str, Any]], required: tuple[str, ...]
    ) -> bool:
        return all((engine_health.get(name) or {}).get("healthy") is True for name in required)

    @staticmethod
    def _market_type(rules: dict[str, Any]) -> str:
        declared = str(rules.get("type") or "").lower()
        if declared in {"spot", "margin", "swap", "future", "option", "forex", "arbitrage"}:
            return declared
        for market_type in ("option", "future", "swap", "margin", "spot"):
            if rules.get(market_type) is True:
                return market_type
        return "unknown"

    @staticmethod
    def _disabled_research(capabilities: dict[str, Any]) -> dict[str, str]:
        mapping = {
            "fluid_liquidity": "fetchOrderBook",
            "smart_scalping": "fetchOHLCV",
            "moon_scout_dynamic_scanner": "fetchOHLCV",
        }
        disabled: dict[str, str] = {}
        for engine, capability in mapping.items():
            available = capabilities.get(capability)
            if not available:
                disabled[engine] = f"exchange_does_not_advertise:{capability}"
        return disabled

    def health(self) -> dict[str, Any]:
        return {
            "capability_driven": True,
            "fail_closed": True,
            "public_api_research_only": True,
            "authenticated_executor": "bybit_testnet_spot_only",
            "supported_execution_products": ["spot"],
            "blocked_execution_products": ["margin", "swap", "future", "option", "forex", "arbitrage"],
            "product_protection_requirements": {
                name: list(requirements)
                for name, requirements in self.PRODUCT_PROTECTION_REQUIREMENTS.items()
            },
            "research_profiles": self.research_profiles,
            "authorization_checks": self.authorization_checks,
            "authorized": self.authorized,
            "blocked": self.blocked,
            "block_reasons": dict(self.block_reasons),
            "last_research_plan": dict(self.last_research_plan),
            "last_authorizations": dict(self.last_authorizations),
            "execution_authority": "testnet_policy_only",
            "live_authority": False,
        }
