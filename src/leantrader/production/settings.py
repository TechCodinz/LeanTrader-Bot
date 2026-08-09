from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

TRUE_VALUES = {"1", "true", "yes", "y", "on"}


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
    research_state_path: Path
    provenance_path: Path
    heartbeat_path: Path
    log_path: Path
    learning_rate: float
    learning_min_samples: int
    engine_failure_threshold: int
    engine_recovery_seconds: float

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

        symbols = tuple(
            value.strip().upper()
            for value in os.getenv("PAPER_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
            if value.strip()
        )
        if not symbols:
            raise ValueError("PAPER_SYMBOLS must contain at least one symbol")
        confirm_timeframes = tuple(
            value.strip() for value in os.getenv("CONFIRM_TIMEFRAMES", "1h,4h").split(",") if value.strip()
        )

        settings = cls(
            exchange=os.getenv("DATA_EXCHANGE", "bybit").strip().lower(),
            symbols=symbols,
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
            research_state_path=Path(os.getenv("RESEARCH_STATE_PATH", "runtime/vps_research_governor.json")),
            provenance_path=Path(os.getenv("PROVENANCE_PATH", "runtime/vps_decision_provenance.jsonl")),
            heartbeat_path=Path(os.getenv("HEARTBEAT_PATH", "runtime/vps_heartbeat.json")),
            log_path=Path(os.getenv("TRADES_LOG_PATH", "logs/vps_trades.jsonl")),
            learning_rate=_float("ADAPTIVE_LEARNING_RATE", 0.08),
            learning_min_samples=_int("ADAPTIVE_MIN_SAMPLES", 5),
            engine_failure_threshold=_int("ENGINE_FAILURE_THRESHOLD", 3),
            engine_recovery_seconds=_float("ENGINE_RECOVERY_SECONDS", 60.0),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.candle_limit < 220:
            raise ValueError("PAPER_CANDLE_LIMIT must be at least 220")
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
        if self.engine_failure_threshold < 1 or self.engine_recovery_seconds < 0:
            raise ValueError("invalid engine circuit-breaker configuration")
