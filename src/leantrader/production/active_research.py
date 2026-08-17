from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class ActiveResearchPlanner:
    """Turns uncertainty and rare market structure into bounded research tasks.

    The planner is intentionally not a free-form crawler.  It identifies which
    *class* of evidence is missing, formulates falsifiable questions, and exposes
    a safe adapter agenda.  New external sources must be implemented as explicit
    read-only adapters; the planner cannot invent URLs, credentials or execution
    authority by itself.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    MAX_TASKS = 2_000
    SOURCE_CATALOG: dict[str, dict[str, Any]] = {
        "spot_ohlcv": {"tier": "canonical", "description": "spot price/volume/candles"},
        "order_book": {"tier": "canonical", "description": "top-of-book/depth/liquidity"},
        "multi_timeframe": {"tier": "canonical", "description": "fast/tactical/strategic timeframe structure"},
        "cross_venue_quotes": {"tier": "canonical", "description": "public cross-venue price/basis comparison"},
        "public_fundamentals": {"tier": "canonical", "description": "market cap, volume, dominance, trending context"},
        "public_news": {"tier": "canonical", "description": "timestamped market news context"},
        "closed_trade_memory": {"tier": "canonical", "description": "real paper/testnet closed outcomes"},
        "costed_shadow_episodes": {"tier": "canonical", "description": "cost-adjusted shadow strategy episodes"},
        "derivatives_funding": {"tier": "adapter_needed", "description": "funding rates and perpetual basis"},
        "open_interest": {"tier": "adapter_needed", "description": "derivatives positioning/participation"},
        "liquidations": {"tier": "adapter_needed", "description": "forced-position unwind pressure"},
        "options_surface": {"tier": "adapter_needed", "description": "implied volatility, skew, term structure"},
        "onchain_flows": {"tier": "adapter_needed", "description": "exchange/stablecoin/whale flow context"},
        "exchange_onchain_flows": {"tier": "adapter_needed", "description": "labelled exchange inflow/outflow and whale-flow context"},
        "chain_liquidity_flows": {"tier": "adapter_needed", "description": "cross-chain stablecoin and TVL migration"},
        "bridge_flows": {"tier": "adapter_needed", "description": "cross-chain bridge deposit/withdrawal context"},
        "institutional_flows": {"tier": "adapter_needed", "description": "ETF/institutional asset flow context"},
        "whale_concentration": {"tier": "adapter_needed", "description": "large-holder supply concentration and distribution context"},
        "chain_congestion": {"tier": "adapter_needed", "description": "network fee, utilization and prioritization pressure"},
        "stablecoin_mint_burn": {"tier": "adapter_needed", "description": "direct stablecoin issuance/redemption event context"},
        "macro_calendar": {"tier": "adapter_needed", "description": "scheduled macro event risk"},
        "rates_fx_cross_asset": {"tier": "adapter_needed", "description": "rates, dollar, equities and risk-asset coupling"},
        "stablecoin_liquidity": {"tier": "adapter_needed", "description": "stablecoin supply/flow/liquidity state"},
    }

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.plans = int(self.state.get("plans") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.plans = int(self.state.get("plans") or 0)

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _task_id(symbol: str, question: str) -> str:
        return hashlib.sha256(f"{symbol}|{question}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _source_status(
        self,
        *,
        world: dict[str, Any],
        engine_health: dict[str, Any],
        public_context: dict[str, Any],
        arbitrage: dict[str, Any] | None,
        sensor_snapshot: dict[str, Any] | None = None,
        external_capabilities: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        status = {name: "adapter_needed" for name, row in self.SOURCE_CATALOG.items() if row["tier"] == "adapter_needed"}
        status.update(
            {
                "spot_ohlcv": "available" if (engine_health.get("market_data") or {}).get("healthy") else "degraded",
                "order_book": "available" if "order_book_liquidity" not in (world.get("unknowns") or []) else "intermittent_or_missing",
                "multi_timeframe": "available" if float(world.get("timeframe_coverage") or 0.0) >= 0.50 else "partial",
                "cross_venue_quotes": "available" if (arbitrage or {}).get("available") else "partial_or_unavailable",
                "public_fundamentals": "available" if public_context.get("market_data_fresh") else "stale_or_unavailable",
                "public_news": "available" if public_context.get("news_fresh") else "stale_or_unavailable",
                "closed_trade_memory": "available" if (engine_health.get("memory_retention") or {}).get("healthy") else "degraded",
                "costed_shadow_episodes": "available" if (engine_health.get("strategy_observatory") or {}).get("healthy") else "degraded",
            }
        )
        for name, value in ((sensor_snapshot or {}).get("source_status") or {}).items():
            if name in status and value:
                status[name] = str(value)
        for name, value in (external_capabilities or {}).items():
            if name not in status:
                continue
            row = value if isinstance(value, dict) else {}
            if row.get("status") == "available_external_shadow":
                status[name] = "available_external_shadow"
        return status

    def plan_symbol(
        self,
        *,
        symbol: str,
        world: dict[str, Any],
        self_model: dict[str, Any],
        council: dict[str, Any],
        critic: dict[str, Any],
        hypotheses: dict[str, Any],
        engine_health: dict[str, Any],
        public_context_health: dict[str, Any],
        arbitrage: dict[str, Any] | None = None,
        market_world: dict[str, Any] | None = None,
        sensor_snapshot: dict[str, Any] | None = None,
        external_capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        patterns = set(str(value) for value in (world.get("latent_patterns") or []))
        unknowns = set(str(value) for value in (self_model.get("unknowns") or []))
        questions: list[str] = []
        required_sources: list[str] = []
        priority = 0.10

        if "liquidity_price_divergence" in patterns:
            questions.append("Does the observed order-book imbalance persist across venues and precede price convergence after realistic costs?")
            required_sources += ["order_book", "cross_venue_quotes", "open_interest", "liquidations"]
            priority += 0.25
        if "volatility_liquidity_coupling" in patterns:
            questions.append("Is the volatility/liquidity shock driven by forced deleveraging, genuine spot demand, or venue-specific stress?")
            required_sources += ["order_book", "open_interest", "liquidations", "derivatives_funding", "cross_venue_quotes"]
            priority += 0.30
        if "compression_with_participation_anomaly" in patterns:
            questions.append("Does compressed price volatility with abnormal participation predict expansion, and which market side is absorbing flow?")
            required_sources += ["spot_ohlcv", "order_book", "open_interest", "options_surface"]
            priority += 0.22
        if "multi_timeframe_phase_fracture" in patterns:
            questions.append("Which timeframe historically leads resolution when fast, tactical and strategic structures diverge in this regime?")
            required_sources += ["multi_timeframe", "closed_trade_memory", "costed_shadow_episodes"]
            priority += 0.18
        if "cross_model_disagreement" in patterns:
            questions.append("Which specialist is historically best calibrated in this regime, and what missing evidence explains current model disagreement?")
            required_sources += ["closed_trade_memory", "costed_shadow_episodes", "public_news"]
            priority += 0.20
        if "narrative_price_divergence" in patterns:
            questions.append("Is public narrative leading price, lagging price, or unrelated after controlling for liquidity and broader market direction?")
            required_sources += ["public_news", "public_fundamentals", "rates_fx_cross_asset", "macro_calendar"]
            priority += 0.18
        if "crowded_derivatives_positioning" in patterns:
            questions.append("Is crowded funding/positioning being absorbed by spot demand or creating asymmetric squeeze risk?")
            required_sources += ["derivatives_funding", "open_interest", "liquidations", "order_book"]
            priority += 0.24
        if "leverage_build_without_price_confirmation" in patterns:
            questions.append("Is open-interest growth without price confirmation accumulation, hedging, or unstable leverage that is likely to unwind?")
            required_sources += ["open_interest", "derivatives_funding", "liquidations", "cross_venue_quotes"]
            priority += 0.25
        if "one_sided_liquidation_cascade" in patterns:
            questions.append("Are forced liquidations propagating across venues, and is spot liquidity absorbing or amplifying the cascade?")
            required_sources += ["liquidations", "order_book", "cross_venue_quotes", "open_interest"]
            priority += 0.32
        if "options_skew_stress" in patterns:
            questions.append("Does options skew/volatility stress confirm directional tail demand or merely temporary hedging demand?")
            required_sources += ["options_surface", "open_interest", "public_news"]
            priority += 0.22
        if "exchange_inflow_supply_pressure" in patterns:
            questions.append("Are labelled exchange inflows persistent, broadly distributed, and accompanied by spot selling pressure rather than internal exchange wallet movement?")
            required_sources += ["exchange_onchain_flows", "order_book", "cross_venue_quotes", "closed_trade_memory"]
            priority += 0.24
        if "exchange_withdrawal_accumulation_pressure" in patterns:
            questions.append("Do persistent exchange withdrawals coincide with reduced sell-side liquidity and historically precede durable accumulation rather than custody reshuffling?")
            required_sources += ["exchange_onchain_flows", "order_book", "closed_trade_memory"]
            priority += 0.22
        if "stablecoin_liquidity_expansion" in patterns or "stablecoin_liquidity_contraction" in patterns:
            questions.append("Is stablecoin liquidity expansion/contraction broad-based across chains, and does it lead risk-asset demand after controlling for bridge migration and depegs?")
            required_sources += ["stablecoin_liquidity", "chain_liquidity_flows", "bridge_flows", "closed_trade_memory"]
            priority += 0.20
        if "stablecoin_net_mint_impulse" in patterns or "stablecoin_net_burn_impulse" in patterns:
            questions.append("Does the observed stablecoin mint/burn impulse represent deployable liquidity or treasury/redemption mechanics, and does it persist across supply and bridge evidence?")
            required_sources += ["stablecoin_mint_burn", "stablecoin_liquidity", "bridge_flows", "closed_trade_memory"]
            priority += 0.20
        if "cross_chain_liquidity_rotation" in patterns:
            questions.append("Which chains are gaining liquidity, is the rotation persistent after bridge flows are reconciled, and which assets historically respond with a measurable lag?")
            required_sources += ["chain_liquidity_flows", "bridge_flows", "spot_ohlcv", "closed_trade_memory"]
            priority += 0.24
        if "institutional_price_divergence" in patterns:
            questions.append("Are institutional ETF flows diverging from spot price because of temporary hedging/basis activity or because price has not yet reflected persistent allocation pressure?")
            required_sources += ["institutional_flows", "derivatives_funding", "open_interest", "spot_ohlcv"]
            priority += 0.25
        if "bridge_liquidity_migration" in patterns:
            questions.append("Is bridge flow a genuine destination-chain liquidity migration or short-lived routing activity, and does it survive a seven-day persistence test?")
            required_sources += ["bridge_flows", "chain_liquidity_flows", "stablecoin_liquidity"]
            priority += 0.18
        if "onchain_derivatives_positioning_divergence" in patterns:
            questions.append("Why do on-chain liquidity flows and derivatives positioning disagree, and which side has historically led resolution in comparable regimes?")
            required_sources += ["onchain_flows", "open_interest", "derivatives_funding", "liquidations", "closed_trade_memory"]
            priority += 0.30
        if "cross_sensor_flow_contradiction" in patterns:
            questions.append("Which flow sensor is stale, structurally biased, or observing a different participant cohort, and what independent evidence can falsify the apparent contradiction?")
            required_sources += ["onchain_flows", "chain_liquidity_flows", "institutional_flows", "bridge_flows", "public_news"]
            priority += 0.28
        if "whale_supply_concentration_rising" in patterns or "whale_supply_concentration_falling" in patterns:
            questions.append("Is the observed large-holder concentration change persistent, entity-clean, and associated with exchange or custody flows rather than address reshuffling?")
            required_sources += ["whale_concentration", "exchange_onchain_flows", "closed_trade_memory"]
            priority += 0.18
        if "chain_congestion_stress" in patterns:
            questions.append("Is network congestion a genuine demand/stress regime, and does fee/utilization pressure transmit into liquidity, bridge routing, or spot execution quality?")
            required_sources += ["chain_congestion", "chain_liquidity_flows", "bridge_flows", "order_book"]
            priority += 0.20
        if "out_of_distribution_market_state" in patterns or world.get("knowledge_state") == "out_of_distribution":
            questions.append("Which measurable features make this state out-of-distribution, and do similar states exist in a broader cross-asset or on-chain history?")
            required_sources += ["closed_trade_memory", "rates_fx_cross_asset", "onchain_flows", "stablecoin_liquidity"]
            priority += 0.30
        relationships = list((market_world or {}).get("lead_lag_research_candidates") or [])
        related = [
            row for row in relationships
            if symbol in {str(row.get("leader") or "").upper(), str(row.get("follower") or "").upper()}
        ]
        if related:
            strongest = related[0]
            questions.append(
                "Is the observed cross-market lead/lag relationship stable out-of-sample, robust to multiple-comparison bias, "
                "and still present after realistic latency/cost assumptions?"
            )
            required_sources += [
                "spot_ohlcv", "cross_venue_quotes", "closed_trade_memory", "rates_fx_cross_asset"
            ]
            priority += min(0.22, 0.10 + max(0.0, float(strongest.get("incremental_strength") or 0.0)))

        if "similar_closed_outcomes" in unknowns:
            questions.append("What additional comparable closed episodes are required before the current thesis can be treated as calibrated rather than exploratory?")
            required_sources += ["closed_trade_memory", "costed_shadow_episodes"]
        if "fresh_public_context" in unknowns:
            questions.append("Is there fresh market-moving public information that could explain the current price/liquidity state?")
            required_sources += ["public_news", "macro_calendar"]
        if not questions and float((world.get("senses") or {}).get("rare_scope_score") or 0.0) >= 0.45:
            questions.append("What independent evidence could falsify or confirm this unusual sensor conjunction without assuming it is a profitable edge?")
            required_sources += ["spot_ohlcv", "order_book", "closed_trade_memory"]
            priority += 0.10

        questions.extend(str(value) for value in (critic.get("falsification_questions") or [])[:2])
        questions = self._dedupe(questions)
        required_sources = self._dedupe(required_sources)
        source_status = self._source_status(
            world=world,
            engine_health=engine_health,
            public_context=public_context_health,
            arbitrage=arbitrage,
            sensor_snapshot=sensor_snapshot,
            external_capabilities=external_capabilities,
        )
        missing_adapters = [source for source in required_sources if source_status.get(source) == "adapter_needed"]
        degraded_sources = [
            source
            for source in required_sources
            if source_status.get(source) not in {"available", "available_external_shadow", "adapter_needed"}
        ]
        priority = min(
            1.0,
            priority
            + 0.25 * float((world.get("senses") or {}).get("rare_scope_score") or 0.0)
            + 0.20 * float(self_model.get("uncertainty") or 0.0)
            + 0.10 * float(council.get("disagreement") or 0.0),
        )

        tasks: list[dict[str, Any]] = []
        for question in questions:
            task = {
                "task_id": self._task_id(symbol, question),
                "symbol": symbol,
                "question": question,
                "priority": priority,
                "required_sources": required_sources,
                "missing_adapters": missing_adapters,
                "degraded_sources": degraded_sources,
                "knowledge_state": world.get("knowledge_state", "unknown"),
                "hypothesis_ids": [
                    str(row.get("hypothesis_id"))
                    for row in (hypotheses.get("active_for_symbol") or [])
                    if row.get("hypothesis_id")
                ],
                "status": "needs_adapter" if missing_adapters else "ready_for_bounded_research",
                "created_at": time.time(),
                "read_only": True,
                "execution_authority": False,
            }
            tasks.append(task)
            self.state.setdefault("tasks", {})[task["task_id"]] = task

        if len(self.state.get("tasks") or {}) > self.MAX_TASKS:
            ordered = sorted(
                (self.state.get("tasks") or {}).items(),
                key=lambda item: float(item[1].get("created_at") or 0.0),
            )
            for key, _ in ordered[: len(ordered) - self.MAX_TASKS]:
                self.state["tasks"].pop(key, None)

        result = {
            "symbol": symbol,
            "priority": priority,
            "questions": questions,
            "required_sources": required_sources,
            "source_status": {source: source_status.get(source, "unknown") for source in required_sources},
            "missing_adapters": missing_adapters,
            "degraded_sources": degraded_sources,
            "tasks": tasks,
            "knows_what_it_does_not_have": True,
            "can_request_configured_read_only_research": True,
            "cannot_invent_external_authority": True,
            "cannot_add_credentials": True,
            "cannot_execute_trades": True,
            "execution_authority": False,
        }
        self.state.setdefault("latest", {})[symbol] = result
        self.plans += 1
        self.state["plans"] = self.plans
        self._save()
        return result

    def agenda(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = list((self.state.get("tasks") or {}).values())
        rows.sort(key=lambda row: (float(row.get("priority") or 0.0), float(row.get("created_at") or 0.0)), reverse=True)
        return [dict(row) for row in rows[: max(0, int(limit))]]

    def adapter_backlog(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        priorities: dict[str, float] = {}
        for task in (self.state.get("tasks") or {}).values():
            for source in task.get("missing_adapters") or []:
                counts[source] = counts.get(source, 0) + 1
                priorities[source] = max(priorities.get(source, 0.0), float(task.get("priority") or 0.0))
        rows = [
            {
                "source": source,
                "requests": count,
                "max_priority": priorities.get(source, 0.0),
                "description": (self.SOURCE_CATALOG.get(source) or {}).get("description"),
                "implementation_rule": "explicit_read_only_adapter_required",
            }
            for source, count in counts.items()
        ]
        rows.sort(key=lambda row: (float(row["max_priority"]), int(row["requests"])), reverse=True)
        return rows

    def model_research_context(self, limit: int = 20) -> dict[str, Any]:
        return {
            "research_agenda": self.agenda(limit),
            "adapter_backlog": self.adapter_backlog()[:20],
            "rules": {
                "external_data_must_be_cited_or_supplied": True,
                "no_orders": True,
                "no_credentials": True,
                "no_live_authority": True,
                "novelty_is_not_profit_evidence": True,
            },
        }

    def health(self) -> dict[str, Any]:
        tasks = list((self.state.get("tasks") or {}).values())
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "plans": self.plans,
            "tasks": len(tasks),
            "ready_tasks": sum(1 for row in tasks if row.get("status") == "ready_for_bounded_research"),
            "tasks_needing_adapters": sum(1 for row in tasks if row.get("status") == "needs_adapter"),
            "adapter_backlog": self.adapter_backlog()[:20],
            "source_catalog": self.SOURCE_CATALOG,
            "open_web_crawler": False,
            "explicit_adapter_policy": True,
            "execution_authority": False,
            "can_add_credentials": False,
            "can_enable_live": False,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {"schema_version": self.SCHEMA_VERSION, "plans": 0, "tasks": {}, "latest": {}}
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["plans"] = self.plans
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
