from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .active_research import ActiveResearchPlanner
from .adversarial_critic import AdversarialCritic
from .advanced_engines import UltraEngineSuite
from .arbitrage_monitor import CrossVenueQuoteCollector
from .brain import TradingBrain
from .hypothesis_lab import HypothesisLab
from .intelligence_council import IntelligenceCouncil
from .capital_growth import CapitalGrowthGovernor
from .cns import CentralNervousSystem
from .decision_router import BoundedDecisionRouter, MarketEvidenceGate
from .engine_control import EngineRegistry
from .error_attribution import ErrorAttributionTracker
from .exchange_intelligence import ExchangeIntelligence, timeframe_seconds
from .exchange_protection import ExchangeProtectionOrchestrator
from .intelligence import AdaptiveIntelligence, IntelligenceDecision, MINIMUM_INTELLIGENCE_CANDLES
from .ledger import PaperLedger
from .market_universe import MarketUniverse
from .market_world_model import MarketWorldModel
from .market_sensor_fabric import MarketSensorFabric
from .memory_retention import MarketFingerprint, MemoryRetentionEngine
from .meta_cognition import MetaCognitiveSelfModel
from .model_research import ModelResearchEngine, StructuredResearchProvider
from .operations_engines import OperationsEngineSuite
from .public_context import PublicMarketContextEngine
from .research_engines import ResearchEngineSuite
from .settings import Settings
from .strategy_observatory import StrategyObservatory
from .tail_risk_sentinel import TailRiskSentinel
from .temporal_guard import MarketTemporalGuard
from .testnet_execution import BybitTestnetExecutionEngine

LOGGER = logging.getLogger("leantrader.production")


def _is_symbol_history_unavailable(exc: Exception) -> bool:
    """Return True only for a genuine not-yet-mature market-history gap.

    A newly listed symbol can be active and liquid while still having fewer than
    the minimum 220 closed candles required by the adaptive intelligence layer.
    That is an availability state, not a required-engine failure. Other quality
    problems (invalid prices, gaps, malformed timestamps, etc.) remain failures.
    """
    return (
        isinstance(exc, ValueError)
        and str(exc).strip() == "market data rejected: fewer than 220 candles"
    )


def atr_sized_notional(
    *,
    equity: float,
    price: float,
    atr: float,
    stop_multiple: float,
    risk_fraction: float,
    position_cap_fraction: float,
    order_cap: float,
    existing_notional: float = 0.0,
) -> float:
    """Size notional so the ATR stop risks no more than the configured equity fraction."""
    stop_distance = atr * stop_multiple
    if equity <= 0 or price <= 0 or stop_distance <= 0:
        return 0.0
    risk_budget = equity * risk_fraction
    risk_based = risk_budget * price / stop_distance
    position_room = max(0.0, equity * position_cap_fraction - existing_notional)
    return max(0.0, min(order_cap, risk_based, position_room))


class MarketFeed:
    """Public, read-only CCXT market-data feed. No credentials are loaded."""

    def __init__(self, exchange_id: str) -> None:
        import ccxt  # type: ignore

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"unknown CCXT exchange: {exchange_id}")
        self.exchange = exchange_class({"enableRateLimit": True, "timeout": 20_000})
        self._markets_loaded = False
        self._last_discovery: dict[str, Any] = {}
        self._candle_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}
        self._candle_cache_hits = 0
        self._candle_cache_misses = 0

    def _load_markets(self) -> None:
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        self._load_markets()
        key = (symbol, timeframe, limit)
        cached = self._candle_cache.get(key)
        ttl = self._timeframe_cache_seconds(timeframe)
        if cached is not None and time.monotonic() - cached[0] < ttl:
            self._candle_cache_hits += 1
            return cached[1].copy()
        self._candle_cache_misses += 1
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            raise RuntimeError(f"no candles returned for {symbol}")
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._candle_cache[key] = (time.monotonic(), frame)
        return frame.copy()

    @staticmethod
    def _timeframe_cache_seconds(timeframe: str) -> float:
        try:
            seconds = timeframe_seconds(timeframe)
        except ValueError:
            seconds = 60
        return max(15.0, min(float(seconds) * 0.25, 3_600.0))

    def order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        self._load_markets()
        if not self.exchange.has.get("fetchOrderBook", False):
            raise RuntimeError("exchange does not support order-book inspection")
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def discover_markets(
        self,
        *,
        quote: str,
        min_quote_volume_usd: float,
        max_spread_bps: float,
    ) -> dict[str, Any]:
        """Return every active, liquid spot market ranked by quote volume."""
        self._load_markets()
        if not self.exchange.has.get("fetchTickers", False):
            raise RuntimeError("exchange does not support bulk ticker discovery")
        tickers = self.exchange.fetch_tickers()
        accepted: list[dict[str, Any]] = []
        rejected: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1

        for symbol, market in self.exchange.markets.items():
            if not market.get("spot") or market.get("active") is False:
                reject("not_active_spot")
                continue
            if str(market.get("quote", "")).upper() != quote.upper():
                reject("quote_mismatch")
                continue
            if market.get("leveraged") is True or self._looks_leveraged(str(market.get("base", ""))):
                reject("leveraged_token")
                continue
            ticker = tickers.get(symbol) or {}
            last = float(ticker.get("last") or ticker.get("close") or 0.0)
            quote_volume = float(ticker.get("quoteVolume") or 0.0)
            if quote_volume <= 0:
                quote_volume = float(ticker.get("baseVolume") or 0.0) * last
            if last <= 0 or quote_volume < min_quote_volume_usd:
                reject("insufficient_volume")
                continue
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            if bid <= 0 or ask <= 0 or ask < bid:
                reject("invalid_spread")
                continue
            spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
            if spread_bps > max_spread_bps:
                reject("spread_too_wide")
                continue
            accepted.append(
                {
                    "symbol": symbol.upper(),
                    "quote_volume_usd": quote_volume,
                    "spread_bps": spread_bps,
                    "last": last,
                }
            )
        accepted.sort(key=lambda item: (-float(item["quote_volume_usd"]), str(item["symbol"])))
        self._last_discovery = {
            "quote": quote.upper(),
            "eligible": len(accepted),
            "markets_seen": len(self.exchange.markets),
            "rejection_counts": rejected,
        }
        return {"candidates": accepted, **self._last_discovery}

    @staticmethod
    def _looks_leveraged(base: str) -> bool:
        upper = base.upper()
        return upper.endswith(("3L", "3S", "5L", "5S", "BULL", "BEAR"))

    def health(self) -> dict[str, Any]:
        capabilities = {
            name: bool(self.exchange.has.get(name, False))
            for name in ("fetchMarkets", "fetchTickers", "fetchOHLCV", "fetchOrderBook", "fetchTime")
        }
        return {
            "read_only": True,
            "environment": "live_public_market_data",
            "credentials_loaded": False,
            "trading_authority": False,
            "exchange_id": str(getattr(self.exchange, "id", "unknown")),
            "capabilities": capabilities,
            "markets_loaded": self._markets_loaded,
            "last_discovery": dict(self._last_discovery),
            "candle_cache": {
                "entries": len(getattr(self, "_candle_cache", {})),
                "hits": getattr(self, "_candle_cache_hits", 0),
                "misses": getattr(self, "_candle_cache_misses", 0),
                "timeframe_aware": True,
            },
        }


class PaperRunner:
    def __init__(self, settings: Settings, feed: MarketFeed | Any) -> None:
        self.settings = settings
        self.feed = feed
        self.ledger = PaperLedger(settings.state_path, settings.starting_cash)
        self.intelligence = AdaptiveIntelligence(
            settings.intelligence_state_path,
            learning_rate=settings.learning_rate,
            min_samples=settings.learning_min_samples,
        )
        self.advanced = UltraEngineSuite(
            settings.pattern_memory_path,
            settings.news_state_path,
            news_max_age_seconds=settings.news_max_age_seconds,
            news_max_future_skew_seconds=settings.news_max_future_skew_seconds,
        )
        self.research = ResearchEngineSuite(settings.research_state_path)
        model_provider = (
            StructuredResearchProvider(
                provider=settings.model_research_provider,
                model=settings.model_research_model,
                api_key_path=settings.model_research_api_key_path,
                endpoint=settings.model_research_endpoint,
            )
            if settings.model_research_enabled
            else None
        )
        self.model_research = ModelResearchEngine(
            settings.model_research_state_path,
            enabled=settings.model_research_enabled,
            interval_cycles=settings.model_research_interval_cycles,
            provider=model_provider,
        )
        self.public_context = PublicMarketContextEngine(
            settings.public_context_state_path,
            enabled=settings.public_context_enabled,
            refresh_seconds=settings.public_context_refresh_seconds,
        )
        self.exchange_intelligence = ExchangeIntelligence(
            state_path=settings.exchange_intelligence_state_path,
            exchange_id=settings.exchange,
            feed=feed,
            base_timeframe=settings.timeframe,
            requested_timeframes=settings.confirm_timeframes,
        )
        self.exchange_protection = ExchangeProtectionOrchestrator(self.exchange_intelligence)
        self.temporal_guard = MarketTemporalGuard(
            feed,
            max_clock_offset_ms=settings.max_clock_offset_ms,
            clock_sync_seconds=settings.clock_sync_seconds,
            candle_stale_multiplier=settings.candle_stale_multiplier,
        )
        self.arbitrage_monitor = CrossVenueQuoteCollector(
            primary_feed=feed,
            venues=settings.arbitrage_venues,
            enabled=settings.arbitrage_enabled and hasattr(feed, "exchange"),
            refresh_seconds=settings.arbitrage_refresh_seconds,
            assumed_slippage_bps=settings.arbitrage_slippage_bps,
        )
        self.strategy_observatory = StrategyObservatory(
            settings.strategy_observatory_state_path,
            round_trip_cost_bps=2 * (settings.fee_bps + settings.slippage_bps),
        )
        self.memory = MemoryRetentionEngine(
            settings.memory_retention_state_path,
            max_episodes=settings.memory_max_episodes,
            half_life_hours=settings.memory_half_life_hours,
            legacy_memory_path=settings.legacy_memory_path,
        )
        self.error_attribution = ErrorAttributionTracker(settings.error_attribution_state_path)
        self.cns = CentralNervousSystem(settings.cns_state_path)
        self.brain = TradingBrain(
            settings.brain_state_path,
            min_strategy_samples=settings.brain_min_strategy_samples,
            negative_expectancy_floor=settings.brain_negative_expectancy_floor,
            quarantine_min_samples=settings.brain_quarantine_min_samples,
            quarantine_expectancy_floor=settings.brain_quarantine_expectancy_floor,
            recovery_expectancy_floor=settings.brain_recovery_expectancy_floor,
        )
        self.market_world_model = MarketWorldModel(settings.market_world_model_state_path)
        self.market_sensor_fabric = MarketSensorFabric(
            settings.market_sensor_fabric_state_path,
            fred_api_key_file=settings.fred_api_key_path,
            glassnode_api_key_file=settings.glassnode_api_key_path,
            defillama_api_key_file=settings.defillama_api_key_path,
            ethereum_rpc_url_file=settings.ethereum_rpc_url_path,
            solana_rpc_url_file=settings.solana_rpc_url_path,
            derivatives_refresh_seconds=settings.sensor_derivatives_refresh_seconds,
            options_refresh_seconds=settings.sensor_options_refresh_seconds,
            macro_refresh_seconds=settings.sensor_macro_refresh_seconds,
            onchain_refresh_seconds=settings.sensor_onchain_refresh_seconds,
            enabled=hasattr(feed, "exchange"),
        )
        self.meta_cognition = MetaCognitiveSelfModel(settings.meta_cognition_state_path)
        self.intelligence_council = IntelligenceCouncil(settings.intelligence_council_state_path)
        self.adversarial_critic = AdversarialCritic(settings.adversarial_critic_state_path)
        self.hypothesis_lab = HypothesisLab(settings.hypothesis_lab_state_path)
        self.active_research = ActiveResearchPlanner(settings.active_research_state_path)
        self.tail_risk = TailRiskSentinel(settings.tail_risk_state_path)
        self.capital_growth = CapitalGrowthGovernor(
            settings.capital_growth_state_path,
            starting_equity=settings.starting_cash,
            principal_floor_fraction=settings.capital_principal_floor_fraction,
            profit_reinvest_fraction=settings.capital_profit_reinvest_fraction,
        )
        self.decision_router = BoundedDecisionRouter(
            MarketEvidenceGate(
                settings.decision_router_state_path,
                minimum_samples=settings.market_evidence_min_samples,
                rolling_window=settings.market_evidence_window,
            ),
            minimum_advanced_confidence=settings.router_min_advanced_confidence,
            minimum_combined_score=settings.router_min_combined_score,
            negative_consensus_veto=settings.router_negative_consensus_veto,
        )
        self.operations = OperationsEngineSuite(settings.provenance_path, settings.metrics_path)
        self.universe = MarketUniverse(
            state_path=settings.market_universe_state_path,
            mode=settings.market_universe_mode,
            configured_symbols=settings.symbols,
            quote=settings.market_quote,
            batch_size=settings.market_scan_batch_size,
            refresh_seconds=settings.market_refresh_seconds,
        )
        self._logged_event_ids = self._load_logged_event_ids(settings.log_path)
        self.testnet = (
            BybitTestnetExecutionEngine(
                api_key_path=settings.testnet_api_key_path,
                api_secret_path=settings.testnet_api_secret_path,
                state_path=settings.testnet_state_path,
                confirmation=settings.testnet_confirmation,
                max_order_usd=settings.testnet_max_order_usd,
                max_position_usd=settings.testnet_max_position_usd,
                max_daily_submitted_usd=settings.testnet_max_daily_submitted_usd,
                max_orders_per_day=settings.testnet_max_orders_per_day,
            )
            if settings.testnet_enabled
            else None
        )
        self.engines = EngineRegistry(
            failure_threshold=settings.engine_failure_threshold,
            recovery_seconds=settings.engine_recovery_seconds,
        )
        self.engines.register("market_data", self.feed, version="ccxt-public-v1")
        self.engines.register(
            "exchange_intelligence",
            self.exchange_intelligence,
            dependencies=("market_data",),
            version=self.exchange_intelligence.VERSION,
        )
        self.engines.register(
            "market_temporal_guard",
            self.temporal_guard,
            dependencies=("market_data", "exchange_intelligence"),
            version=self.temporal_guard.VERSION,
        )
        self.engines.register(
            "exchange_protection",
            self.exchange_protection,
            dependencies=("exchange_intelligence", "market_temporal_guard"),
            version=self.exchange_protection.VERSION,
        )
        self.engines.register(
            "cross_venue_arbitrage",
            self.arbitrage_monitor,
            required=False,
            dependencies=("market_data", "exchange_intelligence"),
            version=self.arbitrage_monitor.VERSION,
        )
        universe_dependencies = (
            ("market_data", "exchange_intelligence", "bybit_testnet_execution")
            if self.testnet is not None
            else ("market_data", "exchange_intelligence")
        )
        self.engines.register(
            "market_universe",
            self.universe,
            dependencies=universe_dependencies,
            version=self.universe.VERSION,
        )
        self.engines.register("paper_ledger", self.ledger, version="2")
        self.engines.register(
            "adaptive_intelligence",
            self.intelligence,
            dependencies=("market_data",),
            version=self.intelligence.VERSION,
        )
        self.engines.register(
            "advanced_shadow_suite",
            self.advanced,
            dependencies=("market_data",),
            version=self.advanced.VERSION,
        )
        self.engines.register(
            "public_market_context",
            self.public_context,
            required=False,
            dependencies=("market_data",),
            version=self.public_context.VERSION,
        )
        self.engines.register(
            "research_governor",
            self.research,
            dependencies=("adaptive_intelligence", "paper_ledger"),
            version=self.research.VERSION,
        )
        self.engines.register(
            "model_research",
            self.model_research,
            required=False,
            dependencies=("research_governor", "public_market_context", "strategy_observatory"),
            version=self.model_research.VERSION,
        )
        self.engines.register(
            "decision_router",
            self.decision_router,
            dependencies=("adaptive_intelligence", "advanced_shadow_suite", "research_governor", "market_universe"),
            version=self.decision_router.VERSION,
        )
        self.engines.register(
            "strategy_observatory",
            self.strategy_observatory,
            dependencies=("advanced_shadow_suite", "market_data"),
            version=self.strategy_observatory.VERSION,
        )
        self.engines.register(
            "memory_retention",
            self.memory,
            dependencies=("paper_ledger",),
            version=self.memory.VERSION,
        )
        self.engines.register(
            "error_attribution",
            self.error_attribution,
            required=False,
            version=self.error_attribution.VERSION,
        )
        self.engines.register(
            "central_nervous_system",
            self.cns,
            dependencies=("adaptive_intelligence", "advanced_shadow_suite", "decision_router", "memory_retention"),
            version=self.cns.VERSION,
        )
        self.engines.register(
            "trading_brain",
            self.brain,
            dependencies=("central_nervous_system", "memory_retention", "strategy_observatory", "decision_router"),
            version=self.brain.VERSION,
        )
        self.engines.register(
            "market_sensor_fabric",
            self.market_sensor_fabric,
            required=False,
            dependencies=("market_data",),
            version=self.market_sensor_fabric.VERSION,
        )
        self.engines.register(
            "market_world_model",
            self.market_world_model,
            required=False,
            dependencies=("adaptive_intelligence", "advanced_shadow_suite", "public_market_context", "market_sensor_fabric"),
            version=self.market_world_model.VERSION,
        )
        self.engines.register(
            "meta_cognitive_self_model",
            self.meta_cognition,
            required=False,
            dependencies=("market_world_model", "central_nervous_system", "trading_brain", "memory_retention", "strategy_observatory"),
            version=self.meta_cognition.VERSION,
        )
        self.engines.register(
            "intelligence_council",
            self.intelligence_council,
            required=False,
            dependencies=("market_world_model", "meta_cognitive_self_model", "adaptive_intelligence", "advanced_shadow_suite"),
            version=self.intelligence_council.VERSION,
        )
        self.engines.register(
            "adversarial_critic",
            self.adversarial_critic,
            required=False,
            dependencies=("intelligence_council", "meta_cognitive_self_model", "trading_brain"),
            version=self.adversarial_critic.VERSION,
        )
        self.engines.register(
            "hypothesis_lab",
            self.hypothesis_lab,
            required=False,
            dependencies=("market_world_model", "intelligence_council", "adversarial_critic"),
            version=self.hypothesis_lab.VERSION,
        )
        self.engines.register(
            "active_research_planner",
            self.active_research,
            required=False,
            dependencies=("hypothesis_lab", "market_world_model", "meta_cognitive_self_model"),
            version=self.active_research.VERSION,
        )
        self.engines.register(
            "tail_risk_sentinel",
            self.tail_risk,
            required=False,
            dependencies=("market_world_model", "advanced_shadow_suite"),
            version=self.tail_risk.VERSION,
        )
        self.engines.register(
            "capital_growth",
            self.capital_growth,
            dependencies=("paper_ledger", "research_governor"),
            version=self.capital_growth.VERSION,
        )
        self.engines.register(
            "operations_safety",
            self.operations,
            required=False,
            dependencies=("adaptive_intelligence", "paper_ledger"),
            version=self.operations.VERSION,
        )
        if self.testnet is not None:
            self.engines.register(
                "bybit_testnet_execution",
                self.testnet,
                dependencies=(
                    "market_data",
                    "paper_ledger",
                    "operations_safety",
                    "exchange_protection",
                ),
                version=self.testnet.VERSION,
            )
        self.engines.start_all()
        self._recover_open_position_memory()
        self.stop_requested = False

    def _shadow_call(
        self,
        cycle_errors: dict[str, str],
        key: str,
        engine_name: str,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call an optional intelligence engine without changing trading authority.

        v12.6 meta/world/research engines are deliberately shadow-only. Their
        failures must remain visible but can never turn an otherwise valid market
        decision into a required symbol-pipeline failure.
        """
        if not self.error_attribution.should_attempt(key):
            return {
                "available": False,
                "reason": "optional_error_cooldown",
                "engine": engine_name,
                "execution_authority": False,
            }
        try:
            result = self.engines.call(engine_name, method, *args, **kwargs)
            self.error_attribution.success(key)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as exc:  # noqa: BLE001 - optional intelligence cannot interrupt canonical trading
            error_text = f"{type(exc).__name__}: {exc}"
            cycle_errors[key] = error_text
            self.error_attribution.failure(
                key, error_text, optional=True, component=engine_name,
                symbol=str(kwargs.get("symbol") or "") or None,
            )
            LOGGER.warning("optional intelligence failure engine=%s key=%s error=%s", engine_name, key, error_text)
            return {
                "available": False,
                "error": error_text,
                "engine": engine_name,
                "execution_authority": False,
            }

    def cycle(self) -> dict[str, Any]:
        decisions: dict[str, IntelligenceDecision] = {}
        frames: dict[str, pd.DataFrame] = {}
        advanced_decisions: dict[str, dict[str, Any]] = {}
        routed_decisions: dict[str, dict[str, Any]] = {}
        research_observations: dict[str, dict[str, Any]] = {}
        memory_fingerprints: dict[str, MarketFingerprint] = {}
        memory_summaries: dict[str, dict[str, Any]] = {}
        cns_packets: dict[str, dict[str, Any]] = {}
        brain_decisions: dict[str, dict[str, Any]] = {}
        world_states: dict[str, dict[str, Any]] = {}
        self_awareness: dict[str, dict[str, Any]] = {}
        council_decisions: dict[str, dict[str, Any]] = {}
        critic_reviews: dict[str, dict[str, Any]] = {}
        hypothesis_updates: dict[str, dict[str, Any]] = {}
        research_plans: dict[str, dict[str, Any]] = {}
        tail_risk_states: dict[str, dict[str, Any]] = {}
        public_symbol_context: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        self.engines.call("market_temporal_guard", "sync_clock")
        allowed_testnet_symbols = (
            self.engines.call("bybit_testnet_execution", "eligible_symbols", self.settings.market_quote)
            if self.testnet is not None
            else None
        )
        if self.universe.needs_refresh():
            discovery = self.engines.call(
                "market_data",
                "discover_markets",
                quote=self.settings.market_quote,
                min_quote_volume_usd=self.settings.market_min_quote_volume_usd,
                max_spread_bps=self.settings.market_max_spread_bps,
            )
            self.engines.call(
                "market_universe",
                "refresh",
                discovery["candidates"],
                allowed_symbols=allowed_testnet_symbols,
                rejection_counts=discovery["rejection_counts"],
            )
        elif allowed_testnet_symbols is not None:
            self.engines.call(
                "market_universe",
                "apply_testnet_intersection",
                allowed_testnet_symbols,
            )
        cycle_symbols = self.engines.call(
            "market_universe",
            "next_batch",
            mandatory_symbols=set(self.ledger.positions),
        )
        sensor_snapshot: dict[str, Any] = self._shadow_call(
            errors,
            "market_sensor_fabric:collect",
            "market_sensor_fabric",
            "collect",
            cycle_symbols,
        )
        sensor_symbols = sensor_snapshot.get("symbols") or {} if isinstance(sensor_snapshot, dict) else {}
        context_refresh = self.engines.call(
            "public_market_context", "refresh", tuple(self.universe.symbols)
        )
        arbitrage_key = "cross_venue_arbitrage"
        if self.error_attribution.should_attempt(arbitrage_key):
            try:
                arbitrage_collection = self.engines.call(
                    "cross_venue_arbitrage", "collect", tuple(cycle_symbols)
                )
                self.error_attribution.success(arbitrage_key)
            except Exception as exc:  # noqa: BLE001 - cross-venue research cannot interrupt canonical trading
                errors[arbitrage_key] = f"{type(exc).__name__}: {exc}"
                self.error_attribution.failure(
                    arbitrage_key, errors[arbitrage_key], optional=True, component="cross_venue_arbitrage"
                )
                arbitrage_collection = {"available": False, "quotes": [], "error": errors[arbitrage_key]}
        else:
            arbitrage_collection = {"available": False, "quotes": [], "reason": "optional_error_cooldown"}
        resolved_timeframes = self.engines.call("exchange_intelligence", "resolve_timeframes")
        cycle_engine_health = self.engines.snapshot()
        news_items = list(context_refresh.get("news_items") or [])
        if news_items:
            self.advanced.news.ingest(news_items)
        for symbol in cycle_symbols:
            try:
                frame = self.engines.call(
                    "market_data", "candles", symbol, self.settings.timeframe, self.settings.candle_limit
                )
                frame = self.engines.call(
                    "market_temporal_guard",
                    "filter_closed_candles",
                    frame,
                    self.settings.timeframe,
                    source_requires_timestamp=hasattr(self.feed, "exchange"),
                )
                if len(frame) < MINIMUM_INTELLIGENCE_CANDLES:
                    # Market discovery can legitimately find a newly listed, liquid
                    # spot market before enough base-timeframe history exists for
                    # the adaptive intelligence contract.  Do not fetch the full
                    # multi-timeframe matrix or classify this as a required engine
                    # failure.  Preserve observability and automatically retry when
                    # the symbol rotates through the universe again.
                    reason = (
                        f"{self.settings.timeframe} history immature: "
                        f"{len(frame)} closed candles; requires "
                        f"{MINIMUM_INTELLIGENCE_CANDLES}"
                    )
                    self.error_attribution.unavailable(
                        symbol, reason, component="symbol_history", symbol=symbol
                    )
                    LOGGER.info(
                        "symbol history unavailable symbol=%s reason=%s", symbol, reason
                    )
                    continue
                frames[symbol] = frame
                research_observations[symbol] = self.engines.call(
                    "research_governor", "observe_symbol", symbol, frame
                )
                context_frames: dict[str, pd.DataFrame] = {}
                for timeframe in resolved_timeframes:
                    if timeframe == self.settings.timeframe:
                        context_frames[timeframe] = frame
                        continue
                    error_key = f"{symbol}:timeframe:{timeframe}"
                    if not self.error_attribution.should_attempt(error_key):
                        context_frames[timeframe] = pd.DataFrame()
                        continue
                    try:
                        context_frame = self.feed.candles(
                            symbol, timeframe, self.settings.candle_limit
                        )
                        context_frames[timeframe] = self.engines.call(
                            "market_temporal_guard",
                            "filter_closed_candles",
                            context_frame,
                            timeframe,
                            source_requires_timestamp=hasattr(self.feed, "exchange"),
                        )
                        self.error_attribution.success(error_key)
                    except Exception as exc:  # noqa: BLE001 - one interval must not hide all others
                        context_frames[timeframe] = pd.DataFrame()
                        error_text = f"{type(exc).__name__}: {exc}"
                        # A newly listed asset can legitimately have no completed
                        # weekly/monthly candle yet.  Treat that as unavailable
                        # evidence, not as a runtime failure.
                        if isinstance(exc, ValueError) and "has no closed candles" in str(exc):
                            self.error_attribution.unavailable(
                                error_key, error_text, component="multi_timeframe", symbol=symbol
                            )
                        else:
                            errors[error_key] = error_text
                            self.error_attribution.failure(
                                error_key, error_text, optional=True, component="multi_timeframe", symbol=symbol
                            )
                decisions[symbol] = self.engines.call(
                    "adaptive_intelligence",
                    "evaluate",
                    frame,
                    context_frames=context_frames,
                    symbol=symbol,
                )
                order_book = None
                if hasattr(self.feed, "order_book"):
                    try:
                        # Depth is an optional enhancement to an otherwise valid
                        # candle decision, so an unavailable book is observable
                        # without falsely degrading the required candle feed.
                        order_book = self.feed.order_book(symbol, 10)
                    except Exception as exc:  # noqa: BLE001 - candle decisions remain observable without depth
                        error_key = f"{symbol}:order_book"
                        errors[error_key] = f"{type(exc).__name__}: {exc}"
                        self.error_attribution.failure(
                            error_key, errors[error_key], optional=True, component="order_book", symbol=symbol
                        )
                public_symbol_context[symbol] = self.engines.call(
                    "public_market_context", "evaluate", symbol
                )
                advanced_decisions[symbol] = self.engines.call(
                    "advanced_shadow_suite",
                    "evaluate_symbol",
                    symbol,
                    frame,
                    order_book,
                    self.settings.order_usd / max(decisions[symbol].close, 1e-12),
                    context_frames,
                    public_symbol_context[symbol],
                )
                base_score = sum(
                    decisions[symbol].component_scores[name] * decisions[symbol].weights[name]
                    for name in decisions[symbol].component_scores
                )
                world_states[symbol] = self._shadow_call(
                    errors,
                    f"{symbol}:market_world_model",
                    "market_world_model",
                    "observe_symbol",
                    symbol,
                    frame,
                    adaptive={
                        "score": base_score,
                        "confidence": decisions[symbol].confidence,
                        "regime": decisions[symbol].regime,
                    },
                    advanced=advanced_decisions[symbol],
                    public_context=public_symbol_context[symbol],
                    timeframe_signals=decisions[symbol].timeframe_signals,
                    timeframe_coverage=decisions[symbol].multi_timeframe_coverage,
                    external_sensors=sensor_symbols.get(symbol, {}),
                )
                routed_decisions[symbol] = self.engines.call(
                    "decision_router",
                    "route",
                    symbol=symbol,
                    base_enter=decisions[symbol].enter_long,
                    base_score=base_score,
                    base_confidence=decisions[symbol].confidence,
                    advanced=advanced_decisions[symbol],
                )
                temporal_session = self.engines.call(
                    "market_temporal_guard",
                    "session_status",
                    symbol,
                    market_rules=self.engines.call("exchange_intelligence", "market_rules", symbol),
                )
                routed_decisions[symbol]["temporal_session"] = temporal_session
                routed_decisions[symbol]["exchange_research_plan"] = self.engines.call(
                    "exchange_protection", "research_plan", symbol
                )
                if not temporal_session["allowed"]:
                    routed_decisions[symbol]["allowed"] = False
                    routed_decisions[symbol]["reason"] = str(temporal_session["reason"])
                    routed_decisions[symbol]["size_multiplier"] = 0.0
                if self.testnet is not None and not self.temporal_guard.testnet_clock_safe():
                    routed_decisions[symbol]["allowed"] = False
                    routed_decisions[symbol]["reason"] = "exchange_clock_not_verified"
                    routed_decisions[symbol]["size_multiplier"] = 0.0
                if self.testnet is not None:
                    protection = self.engines.call(
                        "exchange_protection",
                        "authorize_execution",
                        symbol=symbol,
                        side="buy",
                        execution_health=self.testnet.health(),
                        engine_health=self.engines.snapshot(),
                    )
                    routed_decisions[symbol]["exchange_protection"] = protection
                    if not protection["allowed"]:
                        routed_decisions[symbol]["allowed"] = False
                        routed_decisions[symbol]["reason"] = (
                            f"exchange_protection:{protection['reason']}"
                        )
                        routed_decisions[symbol]["size_multiplier"] = 0.0

                fingerprint = self._market_fingerprint(decisions[symbol], advanced_decisions[symbol])
                memory_fingerprints[symbol] = fingerprint
                memory_summaries[symbol] = self.engines.call(
                    "memory_retention", "summarize", symbol=symbol, fingerprint=fingerprint
                )
                cns_packets[symbol] = self.engines.call(
                    "central_nervous_system",
                    "integrate",
                    symbol=symbol,
                    adaptive={
                        "score": base_score,
                        "confidence": decisions[symbol].confidence,
                        "regime": decisions[symbol].regime,
                    },
                    advanced=advanced_decisions[symbol],
                    routed=routed_decisions[symbol],
                    memory_summary=memory_summaries[symbol],
                    exchange_protection=routed_decisions[symbol].get("exchange_protection", {}),
                    runtime_errors=errors,
                )
                upstream_allowed = bool(routed_decisions[symbol].get("allowed"))
                routed_decisions[symbol]["router_allowed_pre_brain"] = upstream_allowed
                routed_decisions[symbol]["router_reason_pre_brain"] = str(
                    routed_decisions[symbol].get("reason") or "unknown"
                )
                strategy_evidence = self.engines.call(
                    "strategy_observatory",
                    "evidence",
                    "engine:bounded_decision_router",
                    symbol,
                )
                brain_decisions[symbol] = self.engines.call(
                    "trading_brain",
                    "evaluate",
                    symbol=symbol,
                    cns=cns_packets[symbol],
                    memory=memory_summaries[symbol],
                    strategy_evidence=strategy_evidence,
                    upstream_allowed=upstream_allowed,
                    strategy_name=f"bounded_decision_router:{symbol}",
                )
                routed_decisions[symbol]["brain"] = brain_decisions[symbol]
                routed_decisions[symbol]["size_multiplier"] = max(
                    0.0,
                    min(
                        1.0,
                        float(routed_decisions[symbol].get("size_multiplier", 1.0))
                        * float(brain_decisions[symbol]["risk_multiplier"]),
                    ),
                )
                if upstream_allowed and not brain_decisions[symbol]["allow_entry"]:
                    routed_decisions[symbol]["allowed"] = False
                    reason = next(
                        (value for value in brain_decisions[symbol]["reasons"] if value != "upstream_router_rejected"),
                        "brain_veto",
                    )
                    routed_decisions[symbol]["reason"] = f"brain:{reason}"
                    routed_decisions[symbol]["size_multiplier"] = 0.0

                self_awareness[symbol] = self._shadow_call(
                    errors,
                    f"{symbol}:meta_cognition",
                    "meta_cognitive_self_model",
                    "assess_symbol",
                    symbol=symbol,
                    world=world_states.get(symbol, {}),
                    cns=cns_packets[symbol],
                    memory=memory_summaries[symbol],
                    route=routed_decisions[symbol],
                    brain=brain_decisions[symbol],
                    strategy_evidence=strategy_evidence,
                    engine_health=cycle_engine_health,
                )
                council_decisions[symbol] = self._shadow_call(
                    errors,
                    f"{symbol}:intelligence_council",
                    "intelligence_council",
                    "deliberate",
                    symbol=symbol,
                    adaptive={
                        "score": base_score,
                        "confidence": decisions[symbol].confidence,
                    },
                    advanced=advanced_decisions[symbol],
                    world=world_states.get(symbol, {}),
                    self_model=self_awareness.get(symbol, {}),
                    memory=memory_summaries[symbol],
                    public_context=public_symbol_context.get(symbol, {}),
                )
                critic_reviews[symbol] = self._shadow_call(
                    errors,
                    f"{symbol}:adversarial_critic",
                    "adversarial_critic",
                    "review",
                    symbol=symbol,
                    council=council_decisions.get(symbol, {}),
                    world=world_states.get(symbol, {}),
                    self_model=self_awareness.get(symbol, {}),
                    memory=memory_summaries[symbol],
                    route=routed_decisions[symbol],
                    brain=brain_decisions[symbol],
                    public_context=public_symbol_context.get(symbol, {}),
                )
                hypothesis_updates[symbol] = self._shadow_call(
                    errors,
                    f"{symbol}:hypothesis_lab",
                    "hypothesis_lab",
                    "observe",
                    symbol=symbol,
                    world=world_states.get(symbol, {}),
                    council=council_decisions.get(symbol, {}),
                    critic=critic_reviews.get(symbol, {}),
                )

                observatory_signals = list(advanced_decisions[symbol].get("signals", []))
                observatory_signals.extend(
                    {
                        "engine": f"adaptive_component:{name}",
                        "score": score,
                        "confidence": decisions[symbol].confidence,
                    }
                    for name, score in decisions[symbol].component_scores.items()
                )
                observatory_signals.extend(
                    [
                        {
                            "engine": "adaptive_ensemble",
                            "score": base_score,
                            "confidence": decisions[symbol].confidence,
                        },
                        advanced_decisions[symbol].get(
                            "swarm",
                            {"engine": "swarm_hivemind", "score": 0.0, "confidence": 0.0},
                        ),
                        {
                            "engine": "bounded_decision_router",
                            "score": routed_decisions[symbol].get("combined_score", 0.0),
                            "confidence": routed_decisions[symbol].get("predicted_probability", 0.0),
                        },
                    ]
                )
                self.engines.call(
                    "strategy_observatory",
                    "observe",
                    symbol,
                    decisions[symbol].close,
                    observatory_signals,
                    decisions[symbol].timeframe_signals,
                )
                self.engines.call(
                    "operations_safety",
                    "record_decision",
                    symbol,
                    {
                        "close": decisions[symbol].close,
                        "regime": decisions[symbol].regime,
                        "confidence": decisions[symbol].confidence,
                        "quality_score": decisions[symbol].quality_score,
                        "component_scores": decisions[symbol].component_scores,
                        "weights": decisions[symbol].weights,
                        "advanced_shadow": advanced_decisions[symbol],
                        "decision_route": routed_decisions[symbol],
                        "cns": cns_packets[symbol],
                        "brain": brain_decisions[symbol],
                        "memory_summary": memory_summaries[symbol],
                        "market_world_model": world_states.get(symbol, {}),
                        "meta_cognitive_self_model": self_awareness.get(symbol, {}),
                        "intelligence_council": council_decisions.get(symbol, {}),
                        "adversarial_critic": critic_reviews.get(symbol, {}),
                        "hypothesis_lab": hypothesis_updates.get(symbol, {}),
                        "research_observation": research_observations[symbol],
                    },
                )
                # Clear any previous symbol-level failure/unavailable state only
                # after the full canonical symbol pipeline completes successfully.
                # This lets transient provider failures and young-market maturity
                # gaps recover automatically without hiding malformed-data faults.
                self.error_attribution.success(symbol)
            except Exception as exc:  # noqa: BLE001 - isolate individual symbol/feed failures
                error_text = f"{type(exc).__name__}: {exc}"
                if _is_symbol_history_unavailable(exc):
                    # Young/newly listed markets can be tradable before enough
                    # closed base-timeframe history exists for the intelligence
                    # quality contract. Keep them visible and retry later, but do
                    # not turn an expected maturity gap into a required failure.
                    self.error_attribution.unavailable(
                        symbol, error_text, component="symbol_history", symbol=symbol
                    )
                    LOGGER.info(
                        "symbol history unavailable symbol=%s reason=%s", symbol, error_text
                    )
                else:
                    errors[symbol] = error_text
                    self.error_attribution.failure(
                        symbol, error_text, optional=False, component="symbol_pipeline", symbol=symbol
                    )
                    LOGGER.warning(
                        "symbol pipeline failure symbol=%s error=%s", symbol, error_text
                    )

        world_market_snapshot = self._shadow_call(
            errors,
            "market_world_model:market_snapshot",
            "market_world_model",
            "observe_market",
            frames,
        )
        post_symbol_engine_health = self.engines.snapshot()
        for symbol in world_states:
            tail_risk_states[symbol] = self._shadow_call(
                errors,
                f"{symbol}:tail_risk_sentinel",
                "tail_risk_sentinel",
                "assess",
                symbol=symbol,
                world=world_states.get(symbol, {}),
                market_world=world_market_snapshot,
                advanced=advanced_decisions.get(symbol, {}),
                runtime_errors=errors,
            )
            research_plans[symbol] = self._shadow_call(
                errors,
                f"{symbol}:active_research_planner",
                "active_research_planner",
                "plan_symbol",
                symbol=symbol,
                world=world_states.get(symbol, {}),
                self_model=self_awareness.get(symbol, {}),
                council=council_decisions.get(symbol, {}),
                critic=critic_reviews.get(symbol, {}),
                hypotheses=hypothesis_updates.get(symbol, {}),
                engine_health=post_symbol_engine_health,
                public_context_health={
                    key: value for key, value in context_refresh.items() if key != "news_items"
                },
                arbitrage=arbitrage_collection,
                market_world=world_market_snapshot,
                sensor_snapshot=sensor_snapshot,
            )

        prices = {symbol: decision.close for symbol, decision in decisions.items()}
        events: list[dict[str, Any]] = []
        entry_blocks: dict[str, str] = {}
        halt = self.engines.call(
            "paper_ledger",
            "risk_check",
            prices,
            self.settings.max_daily_loss_pct,
            self.settings.max_drawdown_pct,
        )
        current_equity = self.engines.call("paper_ledger", "equity", prices)
        current_notionals = {
            symbol: position.quantity * prices.get(symbol, position.entry_price)
            for symbol, position in self.ledger.positions.items()
        }
        drawdown = (self.ledger.peak_equity - current_equity) / max(self.ledger.peak_equity, 1e-9)
        daily_loss = (self.ledger.day_start_equity - current_equity) / max(self.ledger.day_start_equity, 1e-9)
        research_state = self.engines.call(
            "research_governor",
            "runtime_snapshot",
            notionals=current_notionals,
            drawdown=drawdown,
            daily_loss=daily_loss,
            data_healthy=bool(decisions),
            required_engines_healthy=self.engines.required_healthy(),
        )
        capital_state = research_state["capital_preservation"]
        bounded_research_context = self._shadow_call(
            errors,
            "active_research_planner:model_context",
            "active_research_planner",
            "model_research_context",
            20,
        )
        try:
            model_research_observation = self.engines.call(
                "model_research",
                "observe",
                {
                    "timestamp": time.time(),
                    "exchange": self.settings.exchange,
                    "resolved_timeframes": list(resolved_timeframes),
                    "cycle_symbols": list(cycle_symbols),
                    "decisions": {
                        symbol: {
                            "regime": decision.regime,
                            "confidence": decision.confidence,
                            "quality_score": decision.quality_score,
                            "component_scores": decision.component_scores,
                            "timeframe_score": decision.multi_timeframe_score,
                            "timeframe_coverage": decision.multi_timeframe_coverage,
                            "route": routed_decisions.get(symbol),
                        }
                        for symbol, decision in decisions.items()
                    },
                    "distribution_drift": research_state["distribution_drift"],
                    "strategy_evidence": self.strategy_observatory.health()["strategies"],
                    "public_context": {
                        key: value for key, value in context_refresh.items() if key != "news_items"
                    },
                    "market_sensor_fabric": {
                "snapshot": sensor_snapshot,
                "health": self.market_sensor_fabric.health(),
                "read_only": True,
                "execution_authority": False,
            },
            "market_world_model": {
                        "market": world_market_snapshot,
                        "research_candidates": self.market_world_model.research_candidates(20),
                    },
                    "meta_cognitive_self_model": self.meta_cognition.health(),
                    "hypothesis_agenda": self.hypothesis_lab.agenda(20),
                    "active_research": bounded_research_context,
                    "tail_risk": self.tail_risk.health(),
                },
            )
        except Exception as exc:  # noqa: BLE001 - external research never interrupts trading/accounting
            errors["model_research"] = f"{type(exc).__name__}: {exc}"
            model_research_observation = {"requested": True, "accepted": False, "error": errors["model_research"]}
        if capital_state["state"] == "halt" and not halt:
            self.ledger.halt_reason = "capital_preservation"
            halt = self.ledger.halt_reason

        for symbol in list(self.ledger.positions):
            decision = decisions.get(symbol)
            if decision is None:
                continue
            self.engines.call("paper_ledger", "update_peak", symbol, decision.close, decision.atr)
            position = self.ledger.positions[symbol]
            fixed_stop = position.entry_price - self.settings.atr_stop_multiple * position.atr
            trailing_stop = position.peak_price - self.settings.atr_trail_multiple * position.atr
            stop_price = max(fixed_stop, trailing_stop)
            reason = None
            fraction = 1.0
            if halt:
                reason = f"risk_halt:{halt}"
            elif decision.close <= stop_price:
                reason = "atr_stop"
            elif decision.close >= position.entry_price + self.settings.final_take_profit_atr * position.atr:
                reason = "take_profit_final"
            elif (
                float((advanced_decisions.get(symbol, {}).get("swarm") or {}).get("score") or 0.0) <= -0.55
                and float((advanced_decisions.get(symbol, {}).get("swarm") or {}).get("confidence") or 0.0) >= 0.50
            ):
                reason = "ultra_consensus_exit"
            elif not decision.trend_up:
                reason = "trend_reversal"
            elif (
                not position.partial_taken
                and decision.close >= position.entry_price + self.settings.partial_take_profit_atr * position.atr
            ):
                reason = "take_profit_partial"
                fraction = self.settings.partial_take_profit_fraction
            if reason:
                event = self.engines.call(
                    "paper_ledger",
                    "sell",
                    symbol,
                    decision.close,
                    self.settings.fee_bps,
                    self.settings.slippage_bps,
                    reason,
                    fraction,
                )
                events.append(event)
                if event["remaining_quantity"] <= 0:
                    try:
                        promoted = self.engines.call(
                            "adaptive_intelligence",
                            "learn",
                            event.get("position_metadata", {}),
                            event.get("trade_realized_return_total", 0.0),
                        )
                        event["intelligence_promoted"] = promoted
                        self.engines.call(
                            "advanced_shadow_suite",
                            "learn",
                            event.get("position_metadata", {}),
                            event.get("trade_realized_return_total", 0.0),
                            symbol,
                        )
                        event["market_evidence"] = self.engines.call(
                            "decision_router",
                            "record_outcome",
                            symbol,
                            event.get("trade_realized_return_total", 0.0),
                            (
                                event.get("position_metadata", {})
                                .get("decision_route", {})
                                .get("predicted_probability")
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 - execution completed; expose learning failure
                        errors[f"{symbol}:learning"] = f"{type(exc).__name__}: {exc}"
                    try:
                        event["memory_retention"] = self._settle_trade_memory(event)
                    except Exception as exc:  # noqa: BLE001 - closed accounting is final; expose memory failure
                        errors[f"{symbol}:memory_retention"] = f"{type(exc).__name__}: {exc}"
                    try:
                        event["meta_cognition_outcome"] = self.engines.call(
                            "meta_cognitive_self_model",
                            "record_outcome",
                            dict(event.get("position_metadata") or {}),
                            float(event.get("trade_realized_return_total") or 0.0),
                        )
                    except Exception as exc:  # noqa: BLE001 - accounting is final; self-model learning is optional
                        error_key = f"{symbol}:meta_cognition_outcome"
                        errors[error_key] = f"{type(exc).__name__}: {exc}"
                        self.error_attribution.failure(
                            error_key, errors[error_key], optional=True, component="meta_cognitive_self_model", symbol=symbol
                        )

        # Re-evaluate protected capital after exits so a realized loss can
        # block same-cycle re-entry instead of waiting for the next poll.
        post_exit_equity = self.engines.call("paper_ledger", "equity", prices)
        post_exit_notional = sum(
            position.quantity * prices.get(symbol, position.entry_price)
            for symbol, position in self.ledger.positions.items()
        )
        growth_state = self.engines.call(
            "capital_growth",
            "evaluate",
            equity=post_exit_equity,
            realized_pnl=self.ledger.realized_pnl,
            open_notional=post_exit_notional,
        )
        available_deployable_notional = float(growth_state["remaining_deployable_notional"])

        entry_attempts = 0
        entry_failures = 0
        if not halt:
            for symbol, decision in decisions.items():
                route = routed_decisions.get(symbol)
                if route is None:
                    entry_blocks[symbol] = "decision_route_unavailable"
                    continue
                if not route["allowed"]:
                    entry_blocks[symbol] = str(route["reason"])
                    continue
                if not growth_state["new_entries_allowed"]:
                    entry_blocks[symbol] = "capital_growth:principal_floor"
                    continue
                position = self.ledger.positions.get(symbol)
                if position and (
                    position.entry_count >= self.settings.max_entries_per_symbol
                    or decision.confidence < self.settings.scale_in_min_confidence
                ):
                    continue
                if position is None and len(self.ledger.positions) >= self.settings.max_open_positions:
                    break
                equity = self.engines.call("paper_ledger", "equity", prices)
                existing_notional = position.quantity * decision.close if position else 0.0
                notional = atr_sized_notional(
                    equity=equity,
                    price=decision.close,
                    atr=decision.atr,
                    stop_multiple=self.settings.atr_stop_multiple,
                    risk_fraction=self.settings.risk_per_trade_pct,
                    position_cap_fraction=self.settings.max_position_pct,
                    order_cap=self.settings.order_usd,
                    existing_notional=existing_notional,
                )
                notional *= (
                    float(capital_state["size_multiplier"])
                    * float(route["size_multiplier"])
                    * float(growth_state["risk_multiplier"])
                )
                notional = min(notional, available_deployable_notional)
                if notional <= 0:
                    entry_blocks[symbol] = "capital_growth:no_remaining_deployable_notional"
                    continue
                memory_decision_id = None
                entry_metadata = {
                    "regime": decision.regime,
                    "confidence": decision.confidence,
                    "quality_score": decision.quality_score,
                    "component_scores": decision.component_scores,
                    "weights": decision.weights,
                    "rationale": list(decision.rationale),
                    "multi_timeframe_confirmed": decision.multi_timeframe_confirmed,
                    "multi_timeframe_score": decision.multi_timeframe_score,
                    "multi_timeframe_coverage": decision.multi_timeframe_coverage,
                    "timeframe_signals": decision.timeframe_signals,
                    "session_allowed": decision.session_allowed,
                    "symbol": symbol,
                    "advanced_feature_vector": advanced_decisions.get(symbol, {}).get("feature_vector", {}),
                    "advanced_shadow": advanced_decisions.get(symbol, {}),
                    "decision_route": route,
                    "cns": cns_packets.get(symbol, {}),
                    "brain": brain_decisions.get(symbol, {}),
                    "memory_summary": memory_summaries.get(symbol, {}),
                    "market_world_model": world_states.get(symbol, {}),
                    "meta_cognitive_self_model": self_awareness.get(symbol, {}),
                    "intelligence_council": council_decisions.get(symbol, {}),
                    "adversarial_critic": critic_reviews.get(symbol, {}),
                    "hypothesis_lab": hypothesis_updates.get(symbol, {}),
                    "tail_risk_sentinel": tail_risk_states.get(symbol, {}),
                    "active_research_plan": research_plans.get(symbol, {}),
                    "capital_growth": growth_state,
                }
                if position is None:
                    memory_decision_id = f"memory-{symbol.replace('/', '_')}-{time.time_ns()}"
                    fingerprint = memory_fingerprints[symbol]
                    entry_metadata.update(
                        {
                            "memory_decision_id": memory_decision_id,
                            "memory_strategy": "bounded_decision_router",
                            "memory_fingerprint": dict(fingerprint.__dict__),
                            "memory_observed_at": time.time(),
                        }
                    )
                try:
                    entry_attempts += 1
                    event = self.engines.call(
                        "paper_ledger",
                        "buy",
                        symbol,
                        decision.close,
                        notional,
                        decision.atr,
                        self.settings.fee_bps,
                        self.settings.slippage_bps,
                        metadata=entry_metadata,
                        allow_add=position is not None,
                    )
                    events.append(event)
                    available_deployable_notional = max(0.0, available_deployable_notional - notional)
                    if position is None and memory_decision_id is not None:
                        try:
                            self.engines.call(
                                "memory_retention",
                                "remember_decision",
                                memory_decision_id,
                                symbol=symbol,
                                strategy="bounded_decision_router",
                                fingerprint=memory_fingerprints[symbol],
                                confidence=decision.confidence,
                                metadata={
                                    "route_reason": route.get("reason"),
                                    "brain_reasons": brain_decisions.get(symbol, {}).get("reasons", []),
                                },
                                observed_at=entry_metadata["memory_observed_at"],
                            )
                        except Exception as exc:  # noqa: BLE001 - paper fill is final; expose memory capture failure
                            errors[f"{symbol}:memory_capture"] = f"{type(exc).__name__}: {exc}"
                except ValueError as exc:
                    entry_failures += 1
                    errors[symbol] = str(exc)

        equity = self.engines.call("paper_ledger", "equity", prices)
        final_open_notional = sum(
            position.quantity * prices.get(symbol, position.entry_price)
            for symbol, position in self.ledger.positions.items()
        )
        growth_state = self.engines.call(
            "capital_growth",
            "evaluate",
            equity=equity,
            realized_pnl=self.ledger.realized_pnl,
            open_notional=final_open_notional,
        )
        self.engines.call("paper_ledger", "save")
        delivery_events = list(self.ledger.pending_events)
        for event in delivery_events:
            self._append_event(event)
        testnet_events: list[dict[str, Any]] = []
        if self.testnet is not None:
            try:
                protected_events: list[dict[str, Any]] = []
                for event in delivery_events:
                    protection = self.engines.call(
                        "exchange_protection",
                        "authorize_execution",
                        symbol=str(event["symbol"]),
                        side=str(event["side"]),
                        execution_health=self.testnet.health(),
                        engine_health=self.engines.snapshot(),
                    )
                    if protection["allowed"]:
                        protected_events.append(event)
                    else:
                        testnet_events.append(
                            {
                                "symbol": event["symbol"],
                                "side": event["side"],
                                "status": "blocked",
                                "skip_reason": f"exchange_protection:{protection['reason']}",
                                "protection": protection,
                            }
                        )
                if protected_events:
                    testnet_events.extend(
                        self.engines.call(
                            "bybit_testnet_execution", "mirror_events", protected_events
                        )
                    )
                self.engines.call(
                    "paper_ledger",
                    "acknowledge_events",
                    [str(event["event_id"]) for event in delivery_events],
                )
            except Exception as exc:  # noqa: BLE001 - expose and fail health; never fall through to live
                errors["bybit_testnet_execution"] = f"{type(exc).__name__}: {exc}"
        else:
            self.engines.call(
                "paper_ledger",
                "acknowledge_events",
                [str(event["event_id"]) for event in delivery_events],
            )
        try:
            operation_alerts = self.engines.call(
                "operations_safety", "alert_events", events, self.ledger.halt_reason
            )
        except Exception as exc:  # noqa: BLE001 - alerts cannot interrupt accounting/status persistence
            errors["operations_safety:alerts"] = f"{type(exc).__name__}: {exc}"
            operation_alerts = [{"sent": False, "reason": "alert_engine_unavailable"}]
        notionals = {
            symbol: position.quantity * prices.get(symbol, position.entry_price)
            for symbol, position in self.ledger.positions.items()
        }
        try:
            advanced_market = self.engines.call(
                "advanced_shadow_suite",
                "market_snapshot",
                frames,
                notionals,
                list(arbitrage_collection.get("quotes") or []),
            )
            advanced_market["business_performance"] = self.engines.call(
                "advanced_shadow_suite", "business_snapshot", events
            )
        except Exception as exc:  # noqa: BLE001 - advanced suite is shadow-only
            advanced_market = {"error": f"{type(exc).__name__}: {exc}"}
        block_reason_counts: dict[str, int] = {}
        for reason in entry_blocks.values():
            block_reason_counts[str(reason)] = block_reason_counts.get(str(reason), 0) + 1
        execution_funnel = {
            "symbols_evaluated": len(decisions),
            "base_enter_candidates": sum(1 for decision in decisions.values() if decision.enter_long),
            "router_approved_pre_brain": sum(
                1 for route in routed_decisions.values() if route.get("router_allowed_pre_brain") is True
            ),
            "brain_approved": sum(
                1
                for symbol, brain in brain_decisions.items()
                if brain.get("allow_entry") is True
                and (routed_decisions.get(symbol) or {}).get("router_allowed_pre_brain") is True
            ),
            "final_route_allowed": sum(
                1 for route in routed_decisions.values() if route.get("allowed") is True
            ),
            "entry_attempts": entry_attempts,
            "entry_failures": entry_failures,
            "buy_events": sum(1 for event in events if event.get("side") == "buy"),
            "sell_events": sum(1 for event in events if event.get("side") == "sell"),
            "entry_blocks": len(entry_blocks),
            "entry_block_reasons": block_reason_counts,
            "open_positions": len(self.ledger.positions),
            "paper_trade_events_total": self.ledger.trade_count,
            "halted": bool(halt),
        }
        self_system_awareness = self._shadow_call(
            errors,
            "meta_cognitive_self_model:system",
            "meta_cognitive_self_model",
            "observe_system",
            engines=self.engines.snapshot(),
            execution_funnel=execution_funnel,
            errors=errors,
            capital=growth_state,
            world_health=self.market_world_model.health(),
        )
        engine_status = self.engines.snapshot()
        error_attribution = self.error_attribution.cycle_summary(errors)
        if errors:
            LOGGER.warning(
                "cycle error attribution count=%s optional=%s required=%s keys=%s",
                error_attribution["count"],
                error_attribution["optional_count"],
                error_attribution["required_count"],
                ",".join(error_attribution["keys"]),
            )
        observatory_health = self.strategy_observatory.health()
        memory_health = self.memory.health()
        availability_total = self.error_attribution.health().get("total_unavailable")

        status = {
            "timestamp": time.time(),
            "healthy": bool(decisions) and self.engines.required_healthy(),
            "mode": "paper",
            "runtime": "verified-multi-engine-v12.8.2-deep-flow-self-model-hygiene",
            "exchange": self.settings.exchange,
            "resolved_timeframes": list(resolved_timeframes),
            "cycle_symbols": cycle_symbols,
            "equity": equity,
            "cash": self.ledger.cash,
            "open_positions": sorted(self.ledger.positions),
            "realized_pnl": self.ledger.realized_pnl,
            "halt_reason": self.ledger.halt_reason,
            "events": events,
            "errors": errors,
            "error_attribution": error_attribution,
            "availability_total": availability_total,
            "entry_blocks": entry_blocks,
            "execution_funnel": execution_funnel,
            "strategy_observatory_health": observatory_health,
            "memory_health": memory_health,
            "decisions": {
                symbol: {
                    "regime": decision.regime,
                    "confidence": decision.confidence,
                    "quality_score": decision.quality_score,
                    "enter_long": decision.enter_long,
                    "component_scores": decision.component_scores,
                    "weights": decision.weights,
                    "rationale": list(decision.rationale),
                    "multi_timeframe_confirmed": decision.multi_timeframe_confirmed,
                    "multi_timeframe_score": decision.multi_timeframe_score,
                    "multi_timeframe_coverage": decision.multi_timeframe_coverage,
                    "timeframe_signals": decision.timeframe_signals,
                    "session_allowed": decision.session_allowed,
                    "route": routed_decisions.get(symbol),
                }
                for symbol, decision in decisions.items()
            },
            "engines": engine_status,
            "advanced_shadow": {
                "symbols": advanced_decisions,
                "market": advanced_market,
                "execution_authority": False,
            },
            "research_governor": research_state,
            "capital_growth": growth_state,
            "cns": {
                "symbols": cns_packets,
                "execution_authority": False,
            },
            "brain": {
                "symbols": brain_decisions,
                "execution_authority": False,
                "can_increase_upstream_risk": False,
            },
            "memory_retention": {
                "summaries": memory_summaries,
                "health": memory_health,
                "execution_authority": False,
            },
            "market_world_model": {
                "symbols": world_states,
                "market": world_market_snapshot,
                "health": self.market_world_model.health(),
                "execution_authority": False,
            },
            "meta_cognitive_self_model": {
                "symbols": self_awareness,
                "system": self_system_awareness,
                "health": self.meta_cognition.health(),
                "execution_authority": False,
                "consciousness_claim": False,
            },
            "intelligence_council": {
                "symbols": council_decisions,
                "health": self.intelligence_council.health(),
                "execution_authority": False,
            },
            "adversarial_critic": {
                "symbols": critic_reviews,
                "health": self.adversarial_critic.health(),
                "shadow_only": True,
                "execution_authority": False,
            },
            "hypothesis_lab": {
                "symbols": hypothesis_updates,
                "agenda": self.hypothesis_lab.agenda(25),
                "health": self.hypothesis_lab.health(),
                "research_only": True,
                "execution_authority": False,
            },
            "active_research": {
                "symbols": research_plans,
                "agenda": self.active_research.agenda(50),
                "adapter_backlog": self.active_research.adapter_backlog(),
                "health": self.active_research.health(),
                "execution_authority": False,
            },
            "tail_risk_sentinel": {
                "symbols": tail_risk_states,
                "health": self.tail_risk.health(),
                "shadow_only": True,
                "execution_authority": False,
            },
            "error_attribution_health": self.error_attribution.health(),
            "research_observations": research_observations,
            "model_research_observation": model_research_observation,
            "public_context_refresh": {
                key: value for key, value in context_refresh.items() if key != "news_items"
            },
            "arbitrage_collection": {
                key: value for key, value in arbitrage_collection.items() if key != "quotes"
            },
            "operation_alerts": operation_alerts,
            "market_sensor_fabric": {
                "enabled_on_public_exchange_feed": True,
                "version": self.market_sensor_fabric.VERSION,
                "snapshot": sensor_snapshot,
                "health": self.market_sensor_fabric.health(),
                "bybit_derivatives_positioning": True,
                "bybit_liquidation_websocket": True,
                "deribit_options_surface": True,
                "defillama_stablecoin_liquidity": True,
                "defillama_chain_liquidity_flows": True,
                "glassnode_exchange_whale_flows_optional": True,
                "glassnode_whale_concentration_optional": True,
                "defillama_pro_institutional_bridge_flows_optional": True,
                "ethereum_rpc_congestion_optional": True,
                "solana_rpc_congestion_optional": True,
                "ethereum_stablecoin_mint_burn_optional": True,
                "fred_macro_rates_calendar_optional": True,
                "flow_intelligence_synthesizer": True,
                "read_only": True,
                "credentials_loaded_by_default": False,
                "execution_authority": False,
                "can_increase_upstream_risk": False,
            },
        "testnet_execution": {
                "enabled": self.testnet is not None,
                "events": testnet_events,
                "live_authority": False,
            },
        }
        try:
            status["telegram_notifications"] = self.engines.call(
                "operations_safety", "notify_cycle", status
            )
        except Exception as exc:  # noqa: BLE001 - notification failure cannot interrupt accounting
            errors["operations_safety:telegram"] = f"{type(exc).__name__}: {exc}"
            status["telegram_notifications"] = [
                {"sent": False, "reason": "notification_engine_unavailable"}
            ]
        status["engines"] = self.engines.snapshot()
        try:
            status["operation_metrics"] = self.engines.call("operations_safety", "record_metrics", status)
        except Exception as exc:  # noqa: BLE001 - metrics cannot interrupt heartbeat persistence
            errors["operations_safety:metrics"] = f"{type(exc).__name__}: {exc}"
            status["operation_metrics"] = {"written": False, "reason": "metrics_engine_unavailable"}
        status["engines"] = self.engines.snapshot()
        self._write_json_atomic(self.settings.heartbeat_path, status)
        return status

    def run(self, once: bool = False) -> None:
        try:
            while not self.stop_requested:
                started = time.monotonic()
                try:
                    status = self.cycle()
                    LOGGER.info(
                        "cycle equity=%.2f cash=%.2f positions=%s events=%d errors=%d",
                        status["equity"],
                        status["cash"],
                        status["open_positions"],
                        len(status["events"]),
                        len(status["errors"]),
                    )
                except Exception:
                    LOGGER.exception("paper cycle failed")
                if once:
                    return
                remaining = max(1.0, self.settings.poll_seconds - (time.monotonic() - started))
                time.sleep(remaining)
        finally:
            self.engines.stop_all()

    @staticmethod
    def _market_fingerprint(decision: IntelligenceDecision, advanced: dict[str, Any]) -> MarketFingerprint:
        swarm = advanced.get("swarm") or {}
        liquidity = advanced.get("liquidity") or {}
        scores = decision.component_scores
        return MarketFingerprint(
            regime=str(decision.regime or "unknown"),
            volatility=float(decision.atr / max(decision.close, 1e-12)),
            trend=float(scores.get("trend") or 0.0),
            momentum=float(scores.get("momentum") or 0.0),
            spread_bps=float(liquidity.get("spread_bps") or 0.0),
            liquidity_imbalance=float(liquidity.get("imbalance") or 0.0),
            ultra_score=float(swarm.get("score") or 0.0),
            ultra_confidence=float(swarm.get("confidence") or 0.0),
        )

    def _legacy_fingerprint_from_metadata(self, metadata: dict[str, Any]) -> MarketFingerprint | None:
        scores = metadata.get("component_scores") or {}
        advanced = metadata.get("advanced_shadow") or {}
        route = metadata.get("decision_route") or {}
        raw_liquidity = (advanced.get("liquidity") or {}) if isinstance(advanced, dict) else {}
        swarm = (advanced.get("swarm") or {}) if isinstance(advanced, dict) else {}
        if not scores and not route and not advanced:
            return None
        return MarketFingerprint(
            regime=str(metadata.get("regime") or "legacy_unknown"),
            volatility=float(metadata.get("atr_ratio") or 0.0),
            trend=float(scores.get("trend") or 0.0),
            momentum=float(scores.get("momentum") or 0.0),
            spread_bps=float(raw_liquidity.get("spread_bps") or 0.0),
            liquidity_imbalance=float(raw_liquidity.get("imbalance") or 0.0),
            ultra_score=float(swarm.get("score") or route.get("ultra_score") or 0.0),
            ultra_confidence=float(swarm.get("confidence") or route.get("ultra_confidence") or 0.0),
        )

    def _settle_trade_memory(self, event: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(event.get("position_metadata") or {})
        decision_id = str(metadata.get("memory_decision_id") or "")
        net_return = float(event.get("trade_realized_return_total") or 0.0)
        if not decision_id:
            legacy_fp = self._legacy_fingerprint_from_metadata(metadata)
            if legacy_fp is None:
                return {"settled": False, "reason": "legacy_position_without_recoverable_context"}
            decision_id = f"legacy-close-{str(event.get('event_id') or time.time_ns())}"
            episode = self.engines.call(
                "memory_retention",
                "record_closed_observation",
                observation_id=decision_id,
                symbol=str(event.get("symbol") or ""),
                strategy="bounded_decision_router",
                fingerprint=legacy_fp,
                confidence=float(metadata.get("confidence") or 0.0),
                net_return=net_return,
                metadata={"source": "recovered_v11_position_close"},
                observed_at=float(metadata.get("memory_observed_at") or time.time()),
            )
        else:
            try:
                episode = self.engines.call(
                    "memory_retention", "close_decision", decision_id, net_return=net_return
                )
            except KeyError:
                raw_fingerprint = metadata.get("memory_fingerprint")
                fingerprint = (
                    MarketFingerprint(**raw_fingerprint)
                    if isinstance(raw_fingerprint, dict)
                    else self._legacy_fingerprint_from_metadata(metadata)
                )
                if fingerprint is None:
                    return {"settled": False, "reason": "missing_recovery_fingerprint"}
                episode = self.engines.call(
                    "memory_retention",
                    "record_closed_observation",
                    observation_id=decision_id,
                    symbol=str(event.get("symbol") or ""),
                    strategy=str(metadata.get("memory_strategy") or "bounded_decision_router"),
                    fingerprint=fingerprint,
                    confidence=float(metadata.get("confidence") or 0.0),
                    net_return=net_return,
                    metadata={"source": "recovered_pending_or_legacy_position"},
                    observed_at=float(metadata.get("memory_observed_at") or time.time()),
                )
        return {
            "settled": True,
            "decision_id": decision_id,
            "net_return": float(episode.get("net_return") or 0.0),
            "retention_score": float(episode.get("retention_score") or 0.0),
        }

    def _recover_open_position_memory(self) -> None:
        changed = False
        for symbol, position in self.ledger.positions.items():
            metadata = dict(position.metadata or {})
            decision_id = str(metadata.get("memory_decision_id") or "")
            raw_fingerprint = metadata.get("memory_fingerprint")
            fingerprint = (
                MarketFingerprint(**raw_fingerprint)
                if isinstance(raw_fingerprint, dict)
                else self._legacy_fingerprint_from_metadata(metadata)
            )
            if fingerprint is None:
                continue
            if not decision_id:
                decision_id = f"recovered-{symbol.replace('/', '_')}-{int(position.entry_price * 1_000_000)}"
                position.metadata["memory_decision_id"] = decision_id
                position.metadata["memory_strategy"] = "bounded_decision_router"
                position.metadata["memory_fingerprint"] = dict(fingerprint.__dict__)
                position.metadata["memory_observed_at"] = time.time()
                position.metadata["memory_recovered_from_legacy_position"] = True
                changed = True
            self.engines.call(
                "memory_retention",
                "remember_decision",
                decision_id,
                symbol=symbol,
                strategy=str(position.metadata.get("memory_strategy") or "bounded_decision_router"),
                fingerprint=fingerprint,
                confidence=float(position.metadata.get("confidence") or 0.0),
                metadata={"recovered_after_restart": True},
                observed_at=float(position.metadata.get("memory_observed_at") or time.time()),
            )
        if changed:
            self.engines.call("paper_ledger", "save")

    def _append_event(self, event: dict[str, Any]) -> None:
        event_id = str(event.get("event_id", ""))
        if event_id and event_id in self._logged_event_ids:
            return
        self.settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if event_id:
            self._logged_event_ids.add(event_id)

    @staticmethod
    def _load_logged_event_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        output: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event_id = str(json.loads(line).get("event_id", ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if event_id:
                output.add(event_id)
        return output

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def preflight(settings: Settings) -> dict[str, Any]:
    settings.state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.intelligence_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pattern_memory_path.parent.mkdir(parents=True, exist_ok=True)
    settings.news_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.public_context_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.exchange_intelligence_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.strategy_observatory_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.market_universe_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.research_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_research_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.decision_router_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.cns_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.brain_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.memory_retention_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.error_attribution_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.capital_growth_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.market_world_model_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.meta_cognition_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.intelligence_council_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.adversarial_critic_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.hypothesis_lab_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.active_research_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.tail_risk_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.market_sensor_fabric_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.provenance_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.testnet_enabled:
        settings.testnet_state_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "mode": "paper",
        "exchange": settings.exchange,
        "symbols": settings.symbols,
        "market_universe": {
            "mode": settings.market_universe_mode,
            "quote": settings.market_quote,
            "scan_batch_size": settings.market_scan_batch_size,
            "refresh_seconds": settings.market_refresh_seconds,
            "minimum_quote_volume_usd": settings.market_min_quote_volume_usd,
            "maximum_spread_bps": settings.market_max_spread_bps,
        },
        "starting_cash": settings.starting_cash,
        "order_usd": settings.order_usd,
        "runtime": "verified-multi-engine-v12.8.2-deep-flow-self-model-hygiene",
        "cns_brain_memory": {
            "cns_execution_authority": False,
            "brain_can_only_reduce_or_veto": True,
            "memory_closed_outcomes_only": True,
            "memory_max_episodes": settings.memory_max_episodes,
            "memory_half_life_hours": settings.memory_half_life_hours,
            "brain_min_strategy_samples": settings.brain_min_strategy_samples,
            "brain_negative_expectancy_floor": settings.brain_negative_expectancy_floor,
            "brain_quarantine_min_samples": settings.brain_quarantine_min_samples,
            "brain_quarantine_expectancy_floor": settings.brain_quarantine_expectancy_floor,
        },
        "capital_growth": {
            "principal_floor_fraction": settings.capital_principal_floor_fraction,
            "profit_reinvest_fraction": settings.capital_profit_reinvest_fraction,
            "martingale": False,
            "can_increase_upstream_risk": False,
        },
        "world_model_self_awareness": {
            "market_world_model": True,
            "meta_cognitive_self_model": True,
            "intelligence_council": True,
            "adversarial_critic": True,
            "hypothesis_lab": True,
            "active_research_planner": True,
            "tail_risk_sentinel": True,
            "cross_market_lead_lag_discovery": True,
            "consciousness_claim": False,
            "novelty_is_not_trade_authority": True,
            "relationship_discovery_is_not_trade_authority": True,
            "execution_authority": False,
            "can_increase_upstream_risk": False,
            "can_enable_live": False,
            "can_modify_code": False,
            "can_deploy": False,
        },
        "market_sensor_fabric": {
            "version": "2.1",
            "bybit_derivatives_positioning": True,
            "bybit_liquidation_websocket": True,
            "deribit_options_surface": True,
            "defillama_stablecoin_liquidity": True,
            "defillama_chain_liquidity_flows": True,
            "glassnode_exchange_whale_flows_optional": True,
            "glassnode_whale_concentration_optional": True,
            "defillama_pro_institutional_bridge_flows_optional": True,
            "ethereum_rpc_congestion_optional": True,
            "solana_rpc_congestion_optional": True,
            "ethereum_stablecoin_mint_burn_optional": True,
            "fred_macro_rates_calendar_optional": True,
            "flow_intelligence_synthesizer": True,
            "read_only": True,
            "execution_authority": False,
            "can_increase_upstream_risk": False,
            "can_add_credentials": False,
            "can_enable_live": False,
        },
        "testnet_execution": {
            "enabled": settings.testnet_enabled,
            "provider": "bybit" if settings.testnet_enabled else None,
            "environment": "testnet" if settings.testnet_enabled else None,
            "live_authority": False,
        },
        "adaptive_learning": {
            "learning_rate": settings.learning_rate,
            "minimum_closed_trades_per_promotion": settings.learning_min_samples,
            "weight_bounds": [0.10, 0.70],
        },
        "strategy_controls": {
            "confirm_timeframes": settings.confirm_timeframes or ("AUTO_FROM_EXCHANGE",),
            "risk_per_trade_pct": settings.risk_per_trade_pct,
            "max_entries_per_symbol": settings.max_entries_per_symbol,
            "partial_take_profit_atr": settings.partial_take_profit_atr,
            "final_take_profit_atr": settings.final_take_profit_atr,
        },
        "time_integrity": {
            "utc_internal": True,
            "exchange_clock_max_offset_ms": settings.max_clock_offset_ms,
            "clock_sync_seconds": settings.clock_sync_seconds,
            "closed_candles_only": True,
            "candle_stale_multiplier": settings.candle_stale_multiplier,
            "dst_aware_forex": True,
        },
        "news_integrity": {
            "max_age_seconds": settings.news_max_age_seconds,
            "max_future_skew_seconds": settings.news_max_future_skew_seconds,
            "collector": "coingecko_plus_rss",
        },
        "arbitrage_monitor": {
            "enabled": settings.arbitrage_enabled,
            "venues": settings.arbitrage_venues,
            "refresh_seconds": settings.arbitrage_refresh_seconds,
            "execution_authority": False,
        },
        "exchange_protection": {
            "capability_driven": True,
            "fail_closed": True,
            "authenticated_executor": "bybit_testnet_spot_only",
            "live_authority": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LeanTrader paper runner with optional bounded Bybit testnet mirror")
    parser.add_argument("--once", action="store_true", help="run one market cycle and exit")
    parser.add_argument("--preflight", action="store_true", help="validate safe configuration without network access")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    if args.preflight:
        print(json.dumps(preflight(settings), indent=2))
        return

    runner = PaperRunner(settings, MarketFeed(settings.exchange))

    def request_stop(_signum: int, _frame: Any) -> None:
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    runner.run(once=args.once)


if __name__ == "__main__":
    main()
