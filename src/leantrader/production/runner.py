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

from .ledger import PaperLedger
from .settings import Settings
from .strategy import Decision, decide

LOGGER = logging.getLogger("leantrader.production")


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


class PaperRunner:
    def __init__(self, settings: Settings, feed: MarketFeed | Any) -> None:
        self.settings = settings
        self.feed = feed
        self.ledger = PaperLedger(settings.state_path, settings.starting_cash)
        self.stop_requested = False

    def cycle(self) -> dict[str, Any]:
        decisions: dict[str, Decision] = {}
        errors: dict[str, str] = {}
        for symbol in self.settings.symbols:
            try:
                frame = self.feed.candles(symbol, self.settings.timeframe, self.settings.candle_limit)
                decisions[symbol] = decide(frame)
            except Exception as exc:  # noqa: BLE001 - isolate individual symbol/feed failures
                errors[symbol] = f"{type(exc).__name__}: {exc}"

        prices = {symbol: decision.close for symbol, decision in decisions.items()}
        events: list[dict[str, Any]] = []
        halt = self.ledger.risk_check(prices, self.settings.max_daily_loss_pct, self.settings.max_drawdown_pct)

        for symbol in list(self.ledger.positions):
            decision = decisions.get(symbol)
            if decision is None:
                continue
            self.ledger.update_peak(symbol, decision.close, decision.atr)
            position = self.ledger.positions[symbol]
            fixed_stop = position.entry_price - self.settings.atr_stop_multiple * position.atr
            trailing_stop = position.peak_price - self.settings.atr_trail_multiple * position.atr
            stop_price = max(fixed_stop, trailing_stop)
            reason = None
            if halt:
                reason = f"risk_halt:{halt}"
            elif decision.close <= stop_price:
                reason = "atr_stop"
            elif not decision.trend_up:
                reason = "trend_reversal"
            if reason:
                events.append(
                    self.ledger.sell(
                        symbol,
                        decision.close,
                        self.settings.fee_bps,
                        self.settings.slippage_bps,
                        reason,
                    )
                )

        if not halt:
            for symbol, decision in decisions.items():
                if symbol in self.ledger.positions or not decision.enter_long:
                    continue
                if len(self.ledger.positions) >= self.settings.max_open_positions:
                    break
                equity = self.ledger.equity(prices)
                notional = min(self.settings.order_usd, equity * self.settings.max_position_pct)
                if notional <= 0:
                    continue
                try:
                    events.append(
                        self.ledger.buy(
                            symbol,
                            decision.close,
                            notional,
                            decision.atr,
                            self.settings.fee_bps,
                            self.settings.slippage_bps,
                        )
                    )
                except ValueError as exc:
                    errors[symbol] = str(exc)

        equity = self.ledger.equity(prices)
        self.ledger.save()
        for event in events:
            self._append_event(event)
        status = {
            "timestamp": time.time(),
            "healthy": bool(decisions),
            "mode": "paper",
            "exchange": self.settings.exchange,
            "equity": equity,
            "cash": self.ledger.cash,
            "open_positions": sorted(self.ledger.positions),
            "realized_pnl": self.ledger.realized_pnl,
            "halt_reason": self.ledger.halt_reason,
            "events": events,
            "errors": errors,
        }
        self._write_json_atomic(self.settings.heartbeat_path, status)
        return status

    def run(self, once: bool = False) -> None:
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

    def _append_event(self, event: dict[str, Any]) -> None:
        self.settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

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
    settings.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "mode": "paper",
        "exchange": settings.exchange,
        "symbols": settings.symbols,
        "starting_cash": settings.starting_cash,
        "order_usd": settings.order_usd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LeanTrader supported paper-only VPS runner")
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
