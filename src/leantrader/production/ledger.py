from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Position:
    quantity: float
    entry_price: float
    peak_price: float
    atr: float
    entry_fee: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PaperLedger:
    """Small persistent ledger that fills against observed prices, including costs."""

    def __init__(self, path: Path, starting_cash: float) -> None:
        self.path = path
        self.cash = float(starting_cash)
        self.initial_cash = float(starting_cash)
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.peak_equity = float(starting_cash)
        self.day = dt.datetime.now(dt.UTC).date().isoformat()
        self.day_start_equity = float(starting_cash)
        self.halt_reason: str | None = None
        self.trade_count = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.cash = float(data["cash"])
        self.initial_cash = float(data["initial_cash"])
        self.positions = {symbol: Position(**position) for symbol, position in data.get("positions", {}).items()}
        self.realized_pnl = float(data.get("realized_pnl", 0.0))
        self.peak_equity = float(data.get("peak_equity", self.initial_cash))
        self.day = str(data.get("day", self.day))
        self.day_start_equity = float(data.get("day_start_equity", self.initial_cash))
        self.halt_reason = data.get("halt_reason")
        self.trade_count = int(data.get("trade_count", 0))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "positions": {symbol: asdict(position) for symbol, position in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "peak_equity": self.peak_equity,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "halt_reason": self.halt_reason,
            "trade_count": self.trade_count,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, position in self.positions.items():
            value += position.quantity * prices.get(symbol, position.entry_price)
        self.peak_equity = max(self.peak_equity, value)
        return value

    def refresh_day(self, equity: float) -> None:
        today = dt.datetime.now(dt.UTC).date().isoformat()
        if today != self.day:
            self.day = today
            self.day_start_equity = equity
            self.halt_reason = None

    def risk_check(self, prices: dict[str, float], daily_limit: float, drawdown_limit: float) -> str | None:
        equity = self.equity(prices)
        self.refresh_day(equity)
        daily_loss = (self.day_start_equity - equity) / max(self.day_start_equity, 1e-9)
        drawdown = (self.peak_equity - equity) / max(self.peak_equity, 1e-9)
        if daily_loss >= daily_limit:
            self.halt_reason = f"daily_loss:{daily_loss:.4f}"
        elif drawdown >= drawdown_limit:
            self.halt_reason = f"drawdown:{drawdown:.4f}"
        return self.halt_reason

    def buy(
        self,
        symbol: str,
        market_price: float,
        notional: float,
        atr: float,
        fee_bps: float,
        slippage_bps: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if symbol in self.positions:
            raise ValueError(f"position already open for {symbol}")
        fill = market_price * (1 + slippage_bps / 10_000)
        fee = notional * fee_bps / 10_000
        total = notional + fee
        if total > self.cash:
            raise ValueError("insufficient paper cash")
        quantity = notional / fill
        self.cash -= total
        self.positions[symbol] = Position(
            quantity=quantity,
            entry_price=fill,
            peak_price=fill,
            atr=atr,
            entry_fee=fee,
            metadata=dict(metadata or {}),
        )
        self.trade_count += 1
        self.save()
        event = self._event("buy", symbol, quantity, fill, fee, "signal")
        event["position_metadata"] = self.positions[symbol].metadata
        return event

    def sell(
        self, symbol: str, market_price: float, fee_bps: float, slippage_bps: float, reason: str
    ) -> dict[str, Any]:
        position = self.positions.pop(symbol)
        fill = market_price * (1 - slippage_bps / 10_000)
        gross = position.quantity * fill
        fee = gross * fee_bps / 10_000
        cost_basis = position.quantity * position.entry_price + position.entry_fee
        pnl = gross - fee - cost_basis
        self.cash += gross - fee
        self.realized_pnl += pnl
        self.trade_count += 1
        self.save()
        event = self._event("sell", symbol, position.quantity, fill, fee, reason)
        event["realized_pnl"] = pnl
        event["realized_return"] = pnl / max(cost_basis, 1e-9)
        event["position_metadata"] = position.metadata
        return event

    def update_peak(self, symbol: str, price: float, atr: float) -> None:
        position = self.positions[symbol]
        position.peak_price = max(position.peak_price, price)
        if atr > 0:
            position.atr = atr

    def health(self) -> dict[str, Any]:
        return {
            "persistent": True,
            "state_path": str(self.path),
            "open_positions": len(self.positions),
            "trade_count": self.trade_count,
            "halt_reason": self.halt_reason,
        }

    @staticmethod
    def _event(side: str, symbol: str, quantity: float, price: float, fee: float, reason: str) -> dict[str, Any]:
        return {
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "side": side,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "reason": reason,
        }
