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
from .engine_control import EngineRegistry
from .intelligence import AdaptiveIntelligence, IntelligenceDecision
from .ledger import PaperLedger
from .operations_engines import OperationsEngineSuite
from .research_engines import ResearchEngineSuite
from .settings import Settings
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

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            raise RuntimeError(f"no candles returned for {symbol}")
        return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    def health(self) -> dict[str, Any]:
        return {"read_only": True, "markets_loaded": self._markets_loaded}


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
        self.advanced = UltraEngineSuite(settings.pattern_memory_path, settings.news_state_path)
        self.research = ResearchEngineSuite(settings.research_state_path)
        self.operations = OperationsEngineSuite(settings.provenance_path, settings.metrics_path)
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
            required=False,
            dependencies=("market_data",),
            version=self.advanced.VERSION,
        )
        self.engines.register(
            "research_governor",
            self.research,
            dependencies=("adaptive_intelligence", "paper_ledger"),
            version=self.research.VERSION,
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
                dependencies=("market_data", "paper_ledger", "operations_safety"),
                version=self.testnet.VERSION,
            )
        self.engines.start_all()
        self.stop_requested = False

    def cycle(self) -> dict[str, Any]:
        decisions: dict[str, IntelligenceDecision] = {}
        frames: dict[str, pd.DataFrame] = {}
        advanced_decisions: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for symbol in self.settings.symbols:
            try:
                frame = self.engines.call(
                    "market_data", "candles", symbol, self.settings.timeframe, self.settings.candle_limit
                )
                frames[symbol] = frame
                context_frames = {
                    timeframe: self.engines.call(
                        "market_data", "candles", symbol, timeframe, self.settings.candle_limit
                    )
                    for timeframe in self.settings.confirm_timeframes
                }
                decisions[symbol] = self.engines.call(
                    "adaptive_intelligence",
                    "evaluate",
                    frame,
                    context_frames=context_frames,
                    symbol=symbol,
                )
                advanced_decisions[symbol] = self.engines.call(
                    "advanced_shadow_suite", "evaluate_symbol", symbol, frame
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
                    except Exception as exc:  # noqa: BLE001 - execution completed; expose learning failure
                        errors[f"{symbol}:learning"] = f"{type(exc).__name__}: {exc}"

        if not halt:
            for symbol, decision in decisions.items():
                if not decision.enter_long:
                    continue
                if symbol not in advanced_decisions:
                    entry_blocks[symbol] = "advanced_context_unavailable"
                    continue
                if advanced_decisions.get(symbol, {}).get("news_blackout") is True:
                    entry_blocks[symbol] = "high_impact_news_blackout"
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
                notional *= float(capital_state["size_multiplier"])
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
                                "session_allowed": decision.session_allowed,
                                "advanced_feature_vector": advanced_decisions.get(symbol, {}).get("feature_vector", {}),
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
                testnet_events = self.engines.call("bybit_testnet_execution", "mirror_events", delivery_events)
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
            advanced_market = self.engines.call("advanced_shadow_suite", "market_snapshot", frames, notionals)
            advanced_market["business_performance"] = self.advanced.business.summarize(events)
        except Exception as exc:  # noqa: BLE001 - advanced suite is shadow-only
            advanced_market = {"error": f"{type(exc).__name__}: {exc}"}
        engine_status = self.engines.snapshot()
        status = {
            "timestamp": time.time(),
            "healthy": bool(decisions) and self.engines.required_healthy(),
            "mode": "paper",
            "runtime": "verified-multi-engine-v4-testnet",
            "exchange": self.settings.exchange,
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
                    "session_allowed": decision.session_allowed,
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
            "operation_alerts": operation_alerts,
            "testnet_execution": {
                "enabled": self.testnet is not None,
                "events": testnet_events,
                "live_authority": False,
            },
        }
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
    settings.research_state_path.parent.mkdir(parents=True, exist_ok=True)
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
        "starting_cash": settings.starting_cash,
        "order_usd": settings.order_usd,
        "runtime": "verified-multi-engine-v4-testnet",
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
            "confirm_timeframes": settings.confirm_timeframes,
            "risk_per_trade_pct": settings.risk_per_trade_pct,
            "max_entries_per_symbol": settings.max_entries_per_symbol,
            "partial_take_profit_atr": settings.partial_take_profit_atr,
            "final_take_profit_atr": settings.final_take_profit_atr,
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
