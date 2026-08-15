from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
BYBIT_KLINE_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M")


class SafetyError(ValueError):
    """Raised when a configuration could place real orders."""


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false")
    return raw.strip().lower() in TRUE_VALUES


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    exchange: str
    symbols: tuple[str, ...]
    market_universe_mode: str
    market_quote: str
    market_universe_state_path: Path
    exchange_intelligence_state_path: Path
    market_scan_batch_size: int
    market_refresh_seconds: int
    market_min_quote_volume_usd: float
    market_max_spread_bps: float
    timeframe: str
    confirm_timeframes: tuple[str, ...]
    candle_limit: int
    poll_seconds: int
    starting_cash: float
    order_usd: float
    max_open_positions: int
    max_position_pct: float
    risk_per_trade_pct: float
    max_entries_per_symbol: int
    scale_in_min_confidence: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    fee_bps: float
    slippage_bps: float
    atr_stop_multiple: float
    atr_trail_multiple: float
    partial_take_profit_atr: float
    partial_take_profit_fraction: float
    final_take_profit_atr: float
    state_path: Path
    intelligence_state_path: Path
    pattern_memory_path: Path
    news_state_path: Path
    public_context_state_path: Path
    public_context_enabled: bool
    public_context_refresh_seconds: int
    news_max_age_seconds: int
    news_max_future_skew_seconds: int
    max_clock_offset_ms: float
    clock_sync_seconds: int
    candle_stale_multiplier: float
    arbitrage_enabled: bool
    arbitrage_venues: tuple[str, ...]
    arbitrage_refresh_seconds: int
    arbitrage_slippage_bps: float
    strategy_observatory_state_path: Path
    research_state_path: Path
    model_research_state_path: Path
    model_research_enabled: bool
    model_research_provider: str
    model_research_model: str
    model_research_api_key_path: Path
    model_research_endpoint: str
    model_research_interval_cycles: int
    decision_router_state_path: Path
    provenance_path: Path
    metrics_path: Path
    heartbeat_path: Path
    log_path: Path
    learning_rate: float
    learning_min_samples: int
    market_evidence_min_samples: int
    market_evidence_window: int
    router_min_advanced_confidence: float
    router_min_combined_score: float
    router_negative_consensus_veto: float
    engine_failure_threshold: int
    engine_recovery_seconds: float
    testnet_enabled: bool
    testnet_confirmation: str
    testnet_api_key_path: Path
    testnet_api_secret_path: Path
    testnet_state_path: Path
    testnet_max_order_usd: float
    testnet_max_position_usd: float
    testnet_max_daily_submitted_usd: float
    testnet_max_orders_per_day: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(override=False)

        trading_mode = os.getenv("TRADING_MODE", "paper").strip().lower()
        live_confirm = os.getenv("LIVE_CONFIRM", "NO").strip().upper()
        if trading_mode != "paper" or _bool("ENABLE_LIVE") or _bool("ALLOW_LIVE") or live_confirm == "YES":
            raise SafetyError(
                "The supported VPS runner is paper-only. Set TRADING_MODE=paper, "
                "ENABLE_LIVE=false, ALLOW_LIVE=false, and LIVE_CONFIRM=NO."
            )

        exchange = os.getenv("DATA_EXCHANGE", "bybit").strip().lower()
        raw_symbols = os.getenv("PAPER_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").strip()
        market_universe_mode = "dynamic" if raw_symbols.upper() in {"AUTO", "ALL", "DYNAMIC", "*"} else "configured"
        symbols = tuple(
            value.strip().upper()
            for value in raw_symbols.split(",")
            if value.strip()
        ) if market_universe_mode == "configured" else ()
        if market_universe_mode == "configured" and not symbols:
            raise ValueError("PAPER_SYMBOLS must contain at least one symbol")
        raw_timeframes = os.getenv("CONFIRM_TIMEFRAMES", "AUTO").strip()
        confirm_timeframes = () if raw_timeframes.upper() == "AUTO" else tuple(
            dict.fromkeys(value.strip() for value in raw_timeframes.split(",") if value.strip())
        )
        arbitrage_venues = tuple(
            dict.fromkeys(
                value.strip().lower()
                for value in os.getenv("ARBITRAGE_VENUES", "bybit,okx").split(",")
                if value.strip()
            )
        )

        settings = cls(
            exchange=exchange,
            symbols=symbols,
            market_universe_mode=market_universe_mode,
            market_quote=os.getenv("MARKET_QUOTE", "USDT").strip().upper(),
            market_universe_state_path=Path(
                os.getenv("MARKET_UNIVERSE_STATE_PATH", "runtime/vps_market_universe.json")
            ),
            exchange_intelligence_state_path=Path(
                os.getenv("EXCHANGE_INTELLIGENCE_STATE_PATH", "runtime/vps_exchange_intelligence.json")
            ),
            market_scan_batch_size=_int("MARKET_SCAN_BATCH_SIZE", 18),
            market_refresh_seconds=_int("MARKET_REFRESH_SECONDS", 3600),
            market_min_quote_volume_usd=_float("MARKET_MIN_QUOTE_VOLUME_USD", 250_000.0),
            market_max_spread_bps=_float("MARKET_MAX_SPREAD_BPS", 75.0),
            timeframe=os.getenv("PAPER_TIMEFRAME", "15m").strip(),
            confirm_timeframes=confirm_timeframes,
            candle_limit=_int("PAPER_CANDLE_LIMIT", 320),
            poll_seconds=_int("POLL_INTERVAL", 60),
            starting_cash=_float("PAPER_START_CASH", 50.0),
            order_usd=_float("PAPER_ORDER_USD", 2.0),
            max_open_positions=_int("MAX_OPEN_POSITIONS", 2),
            max_position_pct=_float("MAX_POSITION_PCT", 0.10),
            risk_per_trade_pct=_float("RISK_PER_TRADE_PCT", 0.005),
            max_entries_per_symbol=_int("MAX_ENTRIES_PER_SYMBOL", 2),
            scale_in_min_confidence=_float("SCALE_IN_MIN_CONFIDENCE", 0.75),
            max_daily_loss_pct=_float("MAX_DAILY_LOSS_PCT", 0.02),
            max_drawdown_pct=_float("MAX_DRAWDOWN_PCT", 0.10),
            fee_bps=_float("PAPER_FEE_BPS", 10.0),
            slippage_bps=_float("PAPER_SLIPPAGE_BPS", 5.0),
            atr_stop_multiple=_float("ATR_STOP_MULT", 1.5),
            atr_trail_multiple=_float("ATR_TRAIL_MULT", 2.0),
            partial_take_profit_atr=_float("PARTIAL_TAKE_PROFIT_ATR", 1.5),
            partial_take_profit_fraction=_float("PARTIAL_TAKE_PROFIT_FRACTION", 0.50),
            final_take_profit_atr=_float("FINAL_TAKE_PROFIT_ATR", 3.0),
            state_path=Path(os.getenv("PAPER_STATE_PATH", "runtime/vps_paper_state.json")),
            intelligence_state_path=Path(os.getenv("INTELLIGENCE_STATE_PATH", "runtime/vps_intelligence_state.json")),
            pattern_memory_path=Path(os.getenv("PATTERN_MEMORY_PATH", "runtime/vps_pattern_memory.json")),
            news_state_path=Path(os.getenv("NEWS_STATE_PATH", "runtime/vps_news_state.json")),
            public_context_state_path=Path(
                os.getenv("PUBLIC_CONTEXT_STATE_PATH", "runtime/vps_public_context.json")
            ),
            public_context_enabled=_bool("PUBLIC_CONTEXT_ENABLED", True),
            public_context_refresh_seconds=_int("PUBLIC_CONTEXT_REFRESH_SECONDS", 900),
            news_max_age_seconds=_int("NEWS_MAX_AGE_SECONDS", 86_400),
            news_max_future_skew_seconds=_int("NEWS_MAX_FUTURE_SKEW_SECONDS", 300),
            max_clock_offset_ms=_float("MAX_CLOCK_OFFSET_MS", 5_000.0),
            clock_sync_seconds=_int("CLOCK_SYNC_SECONDS", 300),
            candle_stale_multiplier=_float("CANDLE_STALE_MULTIPLIER", 2.5),
            arbitrage_enabled=_bool("ARBITRAGE_MONITOR_ENABLED", True),
            arbitrage_venues=arbitrage_venues,
            arbitrage_refresh_seconds=_int("ARBITRAGE_REFRESH_SECONDS", 60),
            arbitrage_slippage_bps=_float("ARBITRAGE_SLIPPAGE_BPS", 3.0),
            strategy_observatory_state_path=Path(
                os.getenv("STRATEGY_OBSERVATORY_STATE_PATH", "runtime/vps_strategy_observatory.json")
            ),
            research_state_path=Path(os.getenv("RESEARCH_STATE_PATH", "runtime/vps_research_governor.json")),
            model_research_state_path=Path(
                os.getenv("MODEL_RESEARCH_STATE_PATH", "runtime/vps_model_research.json")
            ),
            model_research_enabled=_bool("MODEL_RESEARCH_ENABLED"),
            model_research_provider=os.getenv("MODEL_RESEARCH_PROVIDER", "").strip().lower(),
            model_research_model=os.getenv("MODEL_RESEARCH_MODEL", "").strip(),
            model_research_api_key_path=Path(
                os.getenv("MODEL_RESEARCH_API_KEY_FILE", "/run/secrets/model_research_api_key")
            ),
            model_research_endpoint=os.getenv("MODEL_RESEARCH_ENDPOINT", "").strip(),
            model_research_interval_cycles=_int("MODEL_RESEARCH_INTERVAL_CYCLES", 60),
            decision_router_state_path=Path(
                os.getenv("DECISION_ROUTER_STATE_PATH", "runtime/vps_decision_router.json")
            ),
            provenance_path=Path(os.getenv("PROVENANCE_PATH", "runtime/vps_decision_provenance.jsonl")),
            metrics_path=Path(os.getenv("PROMETHEUS_METRICS_PATH", "runtime/vps_metrics.prom")),
            heartbeat_path=Path(os.getenv("HEARTBEAT_PATH", "runtime/vps_heartbeat.json")),
            log_path=Path(os.getenv("TRADES_LOG_PATH", "logs/vps_trades.jsonl")),
            learning_rate=_float("ADAPTIVE_LEARNING_RATE", 0.08),
            learning_min_samples=_int("ADAPTIVE_MIN_SAMPLES", 5),
            market_evidence_min_samples=_int("MARKET_EVIDENCE_MIN_SAMPLES", 8),
            market_evidence_window=_int("MARKET_EVIDENCE_WINDOW", 50),
            router_min_advanced_confidence=_float("ROUTER_MIN_ADVANCED_CONFIDENCE", 0.20),
            router_min_combined_score=_float("ROUTER_MIN_COMBINED_SCORE", 0.20),
            router_negative_consensus_veto=_float("ROUTER_NEGATIVE_CONSENSUS_VETO", -0.25),
            engine_failure_threshold=_int("ENGINE_FAILURE_THRESHOLD", 3),
            engine_recovery_seconds=_float("ENGINE_RECOVERY_SECONDS", 60.0),
            testnet_enabled=_bool("BYBIT_TESTNET_ENABLED"),
            testnet_confirmation=os.getenv("BYBIT_TESTNET_CONFIRM", "").strip(),
            testnet_api_key_path=Path(
                os.getenv("BYBIT_TESTNET_API_KEY_FILE", "/run/secrets/bybit_testnet_api_key")
            ),
            testnet_api_secret_path=Path(
                os.getenv("BYBIT_TESTNET_API_SECRET_FILE", "/run/secrets/bybit_testnet_api_secret")
            ),
            testnet_state_path=Path(
                os.getenv("BYBIT_TESTNET_STATE_PATH", "runtime/vps_testnet_execution.json")
            ),
            testnet_max_order_usd=_float("BYBIT_TESTNET_MAX_ORDER_USD", 10.0),
            testnet_max_position_usd=_float("BYBIT_TESTNET_MAX_POSITION_USD", 20.0),
            testnet_max_daily_submitted_usd=_float("BYBIT_TESTNET_MAX_DAILY_SUBMITTED_USD", 50.0),
            testnet_max_orders_per_day=_int("BYBIT_TESTNET_MAX_ORDERS_PER_DAY", 20),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.exchange:
            raise ValueError("DATA_EXCHANGE must name a CCXT exchange adapter")
        if not self.market_quote or "/" in self.market_quote:
            raise ValueError("MARKET_QUOTE must be a quote asset such as USDT")
        if not 1 <= self.market_scan_batch_size <= 100:
            raise ValueError("MARKET_SCAN_BATCH_SIZE must be in [1, 100]")
        if self.market_refresh_seconds < 60:
            raise ValueError("MARKET_REFRESH_SECONDS must be at least 60")
        if self.market_min_quote_volume_usd < 0:
            raise ValueError("MARKET_MIN_QUOTE_VOLUME_USD cannot be negative")
        if not 0 < self.market_max_spread_bps <= 1_000:
            raise ValueError("MARKET_MAX_SPREAD_BPS must be in (0, 1000]")
        if self.candle_limit < 220:
            raise ValueError("PAPER_CANDLE_LIMIT must be at least 220")
        if self.exchange == "bybit" and self.confirm_timeframes:
            unsupported_timeframes = sorted(set(self.confirm_timeframes) - set(BYBIT_KLINE_TIMEFRAMES))
            if unsupported_timeframes:
                raise ValueError(
                    "CONFIRM_TIMEFRAMES contains unsupported Bybit intervals: "
                    + ",".join(unsupported_timeframes)
                )
            if set(self.confirm_timeframes) != set(BYBIT_KLINE_TIMEFRAMES):
                raise ValueError("Bybit CONFIRM_TIMEFRAMES must include the complete verified matrix or AUTO")
        if self.public_context_refresh_seconds < 300:
            raise ValueError("PUBLIC_CONTEXT_REFRESH_SECONDS must be at least 300")
        if self.news_max_age_seconds < 3_600:
            raise ValueError("NEWS_MAX_AGE_SECONDS must be at least 3600")
        if not 0 <= self.news_max_future_skew_seconds <= 3_600:
            raise ValueError("NEWS_MAX_FUTURE_SKEW_SECONDS must be in [0, 3600]")
        if not 100 <= self.max_clock_offset_ms <= 60_000:
            raise ValueError("MAX_CLOCK_OFFSET_MS must be in [100, 60000]")
        if self.clock_sync_seconds < 60:
            raise ValueError("CLOCK_SYNC_SECONDS must be at least 60")
        if not 1.0 <= self.candle_stale_multiplier <= 10.0:
            raise ValueError("CANDLE_STALE_MULTIPLIER must be in [1, 10]")
        if self.arbitrage_enabled and len(self.arbitrage_venues) < 2:
            raise ValueError("ARBITRAGE_VENUES must contain at least two venues when enabled")
        if len(self.arbitrage_venues) > 5:
            raise ValueError("ARBITRAGE_VENUES supports at most five public venues")
        if self.arbitrage_refresh_seconds < 30:
            raise ValueError("ARBITRAGE_REFRESH_SECONDS must be at least 30")
        if not 0 <= self.arbitrage_slippage_bps <= 100:
            raise ValueError("ARBITRAGE_SLIPPAGE_BPS must be in [0, 100]")
        if self.model_research_interval_cycles < 10:
            raise ValueError("MODEL_RESEARCH_INTERVAL_CYCLES must be at least 10")
        if self.model_research_enabled:
            if self.model_research_provider not in {"openai", "anthropic", "gemini"}:
                raise ValueError("MODEL_RESEARCH_PROVIDER must be openai, anthropic, or gemini")
            if not self.model_research_model:
                raise ValueError("MODEL_RESEARCH_MODEL is required when model research is enabled")
            if self.model_research_endpoint and not self.model_research_endpoint.startswith("https://"):
                raise ValueError("MODEL_RESEARCH_ENDPOINT must use HTTPS")
        if self.poll_seconds < 10:
            raise ValueError("POLL_INTERVAL must be at least 10 seconds")
        if self.starting_cash <= 0 or self.order_usd <= 0:
            raise ValueError("PAPER_START_CASH and PAPER_ORDER_USD must be positive")
        if self.max_open_positions < 1:
            raise ValueError("MAX_OPEN_POSITIONS must be at least 1")
        for name, value in (
            ("MAX_POSITION_PCT", self.max_position_pct),
            ("RISK_PER_TRADE_PCT", self.risk_per_trade_pct),
            ("MAX_DAILY_LOSS_PCT", self.max_daily_loss_pct),
            ("MAX_DRAWDOWN_PCT", self.max_drawdown_pct),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee and slippage cannot be negative")
        if self.max_entries_per_symbol < 1:
            raise ValueError("MAX_ENTRIES_PER_SYMBOL must be at least 1")
        if not 0 <= self.scale_in_min_confidence <= 1:
            raise ValueError("SCALE_IN_MIN_CONFIDENCE must be between 0 and 1")
        if not 0 < self.partial_take_profit_fraction < 1:
            raise ValueError("PARTIAL_TAKE_PROFIT_FRACTION must be between 0 and 1")
        if self.partial_take_profit_atr <= 0 or self.final_take_profit_atr <= self.partial_take_profit_atr:
            raise ValueError("take-profit ATR levels must be positive and increasing")
        if not 0 < self.learning_rate <= 0.25:
            raise ValueError("ADAPTIVE_LEARNING_RATE must be in (0, 0.25]")
        if self.learning_min_samples < 3:
            raise ValueError("ADAPTIVE_MIN_SAMPLES must be at least 3")
        if self.market_evidence_min_samples < 3:
            raise ValueError("MARKET_EVIDENCE_MIN_SAMPLES must be at least 3")
        if self.market_evidence_window < self.market_evidence_min_samples:
            raise ValueError("MARKET_EVIDENCE_WINDOW cannot be smaller than MARKET_EVIDENCE_MIN_SAMPLES")
        if not 0 <= self.router_min_advanced_confidence <= 1:
            raise ValueError("ROUTER_MIN_ADVANCED_CONFIDENCE must be between 0 and 1")
        if not -1 <= self.router_min_combined_score <= 1:
            raise ValueError("ROUTER_MIN_COMBINED_SCORE must be between -1 and 1")
        if not -1 <= self.router_negative_consensus_veto <= 0:
            raise ValueError("ROUTER_NEGATIVE_CONSENSUS_VETO must be between -1 and 0")
        if self.engine_failure_threshold < 1 or self.engine_recovery_seconds < 0:
            raise ValueError("invalid engine circuit-breaker configuration")
        if self.testnet_enabled:
            if self.exchange != "bybit":
                raise SafetyError("authenticated testnet execution is supported only for Bybit")
            if self.testnet_confirmation != "I_UNDERSTAND_TESTNET_ONLY":
                raise SafetyError("BYBIT_TESTNET_CONFIRM must explicitly confirm testnet-only execution")
            if not 0 < self.testnet_max_order_usd <= 100:
                raise SafetyError("BYBIT_TESTNET_MAX_ORDER_USD must be in (0, 100]")
            if not self.testnet_max_order_usd <= self.testnet_max_position_usd <= 500:
                raise SafetyError("BYBIT_TESTNET_MAX_POSITION_USD exceeds the safe testnet boundary")
            if not self.testnet_max_order_usd <= self.testnet_max_daily_submitted_usd <= 1_000:
                raise SafetyError("BYBIT_TESTNET_MAX_DAILY_SUBMITTED_USD exceeds the safe boundary")
            if not 1 <= self.testnet_max_orders_per_day <= 100:
                raise SafetyError("BYBIT_TESTNET_MAX_ORDERS_PER_DAY must be in [1, 100]")
