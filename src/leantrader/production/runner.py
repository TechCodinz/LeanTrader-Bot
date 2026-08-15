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

from .advanced_engines import UltraEngineSuite
from .arbitrage_monitor import CrossVenueQuoteCollector
from .decision_router import BoundedDecisionRouter, MarketEvidenceGate
from .engine_control import EngineRegistry
from .exchange_intelligence import ExchangeIntelligence, timeframe_seconds
from .exchange_protection import ExchangeProtectionOrchestrator
from .intelligence import AdaptiveIntelligence, IntelligenceDecision
from .ledger import PaperLedger
from .market_universe import MarketUniverse
from .model_research import ModelResearchEngine, StructuredResearchProvider
from .operations_engines import OperationsEngineSuite
from .public_context import PublicMarketContextEngine
from .research_engines import ResearchEngineSuite
from .settings import Settings
from .strategy_observatory import StrategyObservatory
from .temporal_guard import MarketTemporalGuard
from .testnet_execution import BybitTestnetExecutionEngine

LOGGER = logging.getLogger("leantrader.production")


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
        self.stop_requested = False

    def cycle(self) -> dict[str, Any]:
        decisions: dict[str, IntelligenceDecision] = {}
        frames: dict[str, pd.DataFrame] = {}
        advanced_decisions: dict[str, dict[str, Any]] = {}
        routed_decisions: dict[str, dict[str, Any]] = {}
        research_observations: dict[str, dict[str, Any]] = {}
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
        context_refresh = self.engines.call(
            "public_market_context", "refresh", tuple(self.universe.symbols)
        )
        try:
            arbitrage_collection = self.engines.call(
                "cross_venue_arbitrage", "collect", tuple(cycle_symbols)
            )
        except Exception as exc:  # noqa: BLE001 - cross-venue research cannot interrupt canonical trading
            errors["cross_venue_arbitrage"] = f"{type(exc).__name__}: {exc}"
            arbitrage_collection = {"available": False, "quotes": [], "error": errors["cross_venue_arbitrage"]}
        resolved_timeframes = self.engines.call("exchange_intelligence", "resolve_timeframes")
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
                frames[symbol] = frame
                research_observations[symbol] = self.engines.call(
                    "research_governor", "observe_symbol", symbol, frame
                )
                context_frames: dict[str, pd.DataFrame] = {}
                for timeframe in resolved_timeframes:
                    if timeframe == self.settings.timeframe:
                        context_frames[timeframe] = frame
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
                    except Exception as exc:  # noqa: BLE001 - one interval must not hide all others
                        context_frames[timeframe] = pd.DataFrame()
                        errors[f"{symbol}:timeframe:{timeframe}"] = f"{type(exc).__name__}: {exc}"
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
                        errors[f"{symbol}:order_book"] = f"{type(exc).__name__}: {exc}"
                advanced_decisions[symbol] = self.engines.call(
                    "advanced_shadow_suite",
                    "evaluate_symbol",
                    symbol,
                    frame,
                    order_book,
                    self.settings.order_usd / max(decisions[symbol].close, 1e-12),
                    context_frames,
                    self.engines.call("public_market_context", "evaluate", symbol),
                )
                base_score = sum(
                    decisions[symbol].component_scores[name] * decisions[symbol].weights[name]
                    for name in decisions[symbol].component_scores
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
                        "research_observation": research_observations[symbol],
                    },
                )
            except Exception as exc:  # noqa: BLE001 - isolate individual symbol/feed failures
                errors[symbol] = f"{type(exc).__name__}: {exc}"

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

        if not halt:
            for symbol, decision in decisions.items():
                route = routed_decisions.get(symbol)
                if route is None:
                    entry_blocks[symbol] = "decision_route_unavailable"
                    continue
                if not route["allowed"]:
                    entry_blocks[symbol] = str(route["reason"])
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
                notional *= float(capital_state["size_multiplier"]) * float(route["size_multiplier"])
                if notional <= 0:
                    continue
                try:
                    events.append(
                        self.engines.call(
                            "paper_ledger",
                            "buy",
                            symbol,
                            decision.close,
                            notional,
                            decision.atr,
                            self.settings.fee_bps,
                            self.settings.slippage_bps,
                            metadata={
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
                                "advanced_feature_vector": advanced_decisions.get(symbol, {}).get("feature_vector", {}),
                                "decision_route": route,
                            },
                            allow_add=position is not None,
                        )
                    )
                except ValueError as exc:
                    errors[symbol] = str(exc)

        equity = self.engines.call("paper_ledger", "equity", prices)
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
        engine_status = self.engines.snapshot()
        status = {
            "timestamp": time.time(),
            "healthy": bool(decisions) and self.engines.required_healthy(),
            "mode": "paper",
            "runtime": "verified-multi-engine-v11-exchange-protection",
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
            "entry_blocks": entry_blocks,
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
            "research_observations": research_observations,
            "model_research_observation": model_research_observation,
            "public_context_refresh": {
                key: value for key, value in context_refresh.items() if key != "news_items"
            },
            "arbitrage_collection": {
                key: value for key, value in arbitrage_collection.items() if key != "quotes"
            },
            "operation_alerts": operation_alerts,
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
        "runtime": "verified-multi-engine-v11-exchange-protection",
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
