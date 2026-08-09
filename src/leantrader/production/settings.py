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
    candle_limit: int
    poll_seconds: int
    starting_cash: float
    order_usd: float
    max_open_positions: int
    max_position_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    fee_bps: float
    slippage_bps: float
    atr_stop_multiple: float
    atr_trail_multiple: float
    state_path: Path
    heartbeat_path: Path
    log_path: Path

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

        settings = cls(
            exchange=os.getenv("DATA_EXCHANGE", "bybit").strip().lower(),
            symbols=symbols,
            timeframe=os.getenv("PAPER_TIMEFRAME", "15m").strip(),
            candle_limit=_int("PAPER_CANDLE_LIMIT", 320),
            poll_seconds=_int("POLL_INTERVAL", 60),
            starting_cash=_float("PAPER_START_CASH", 50.0),
            order_usd=_float("PAPER_ORDER_USD", 2.0),
            max_open_positions=_int("MAX_OPEN_POSITIONS", 2),
            max_position_pct=_float("MAX_POSITION_PCT", 0.10),
            max_daily_loss_pct=_float("MAX_DAILY_LOSS_PCT", 0.02),
            max_drawdown_pct=_float("MAX_DRAWDOWN_PCT", 0.10),
            fee_bps=_float("PAPER_FEE_BPS", 10.0),
            slippage_bps=_float("PAPER_SLIPPAGE_BPS", 5.0),
            atr_stop_multiple=_float("ATR_STOP_MULT", 1.5),
            atr_trail_multiple=_float("ATR_TRAIL_MULT", 2.0),
            state_path=Path(os.getenv("PAPER_STATE_PATH", "runtime/vps_paper_state.json")),
            heartbeat_path=Path(os.getenv("HEARTBEAT_PATH", "runtime/vps_heartbeat.json")),
            log_path=Path(os.getenv("TRADES_LOG_PATH", "logs/vps_trades.jsonl")),
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
            ("MAX_DAILY_LOSS_PCT", self.max_daily_loss_pct),
            ("MAX_DRAWDOWN_PCT", self.max_drawdown_pct),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee and slippage cannot be negative")
