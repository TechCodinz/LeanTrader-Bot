from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import time
from typing import Any


class AgentRole(str, Enum):
    SCALP = "scalp"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    ARBITRAGE = "arbitrage"
    TREND = "trend"


class TrancheState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class AgentTranche:
    tranche_id: str
    agent_id: str
    role: AgentRole
    timeframe: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    capital: float
    confidence: float
    expected_edge_bps: float
    opened_at: float = field(default_factory=time.time)
    state: TrancheState = TrancheState.OPEN
    exit_price: float | None = None
    realized_pnl: float = 0.0
    closed_at: float | None = None

    @property
    def open_notional(self) -> float:
        return self.capital if self.state == TrancheState.OPEN else 0.0


@dataclass
class SharedPosition:
    symbol: str
    side: str
    tranches: dict[str, AgentTranche] = field(default_factory=dict)

    @property
    def open_tranches(self) -> list[AgentTranche]:
        return [row for row in self.tranches.values() if row.state == TrancheState.OPEN]

    @property
    def open_notional(self) -> float:
        return sum(row.open_notional for row in self.tranches.values())

    @property
    def realized_pnl(self) -> float:
        return sum(row.realized_pnl for row in self.tranches.values())


class PositionCoordinator:
    """Coordinate independently-owned agent tranches on shared symbol positions.

    The coordinator never grants execution authority. It is a paper-side state and
    exposure primitive that prevents multiple agents from independently believing
    they each own the full symbol risk budget.
    """

    VERSION = "1.0"

    def __init__(
        self,
        *,
        max_symbol_exposure_fraction: float = 0.20,
        max_portfolio_exposure_fraction: float = 0.70,
    ) -> None:
        if not 0.0 < max_symbol_exposure_fraction <= 1.0:
            raise ValueError("max_symbol_exposure_fraction must be in (0, 1]")
        if not 0.0 < max_portfolio_exposure_fraction <= 1.0:
            raise ValueError("max_portfolio_exposure_fraction must be in (0, 1]")
        if max_symbol_exposure_fraction > max_portfolio_exposure_fraction:
            raise ValueError("symbol exposure cap cannot exceed portfolio cap")
        self.max_symbol_exposure_fraction = float(max_symbol_exposure_fraction)
        self.max_portfolio_exposure_fraction = float(max_portfolio_exposure_fraction)
        self.positions: dict[tuple[str, str], SharedPosition] = {}
        self._tranche_sequence = 0
        self.blocked_exposure = 0

    @staticmethod
    def _validate_side(side: str) -> str:
        side = str(side).lower()
        if side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        return side

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def symbol_open_notional(self, symbol: str) -> float:
        symbol = str(symbol).upper()
        return sum(
            position.open_notional
            for (candidate, _), position in self.positions.items()
            if candidate == symbol
        )

    @property
    def total_open_notional(self) -> float:
        return sum(position.open_notional for position in self.positions.values())

    def remaining_capacity(self, *, symbol: str, equity: float) -> dict[str, float]:
        equity = self._positive_finite(equity, "equity")
        symbol = str(symbol).upper()
        symbol_cap = equity * self.max_symbol_exposure_fraction
        portfolio_cap = equity * self.max_portfolio_exposure_fraction
        symbol_remaining = max(0.0, symbol_cap - self.symbol_open_notional(symbol))
        portfolio_remaining = max(0.0, portfolio_cap - self.total_open_notional)
        return {
            "symbol_remaining": symbol_remaining,
            "portfolio_remaining": portfolio_remaining,
            "available": min(symbol_remaining, portfolio_remaining),
        }

    def attach_tranche(
        self,
        *,
        agent_id: str,
        role: AgentRole,
        timeframe: str,
        symbol: str,
        side: str,
        entry_price: float,
        capital: float,
        confidence: float,
        expected_edge_bps: float,
        equity: float,
    ) -> AgentTranche:
        symbol = str(symbol).upper()
        side = self._validate_side(side)
        entry_price = self._positive_finite(entry_price, "entry_price")
        capital = self._positive_finite(capital, "capital")
        equity = self._positive_finite(equity, "equity")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        expected_edge_bps = float(expected_edge_bps)
        if not math.isfinite(expected_edge_bps):
            raise ValueError("expected_edge_bps must be finite")
        capacity = self.remaining_capacity(symbol=symbol, equity=equity)
        if capital > capacity["available"] + 1e-12:
            self.blocked_exposure += 1
            raise ValueError("requested tranche exceeds coordinated exposure capacity")

        self._tranche_sequence += 1
        tranche_id = f"{symbol}:{side}:{agent_id}:{self._tranche_sequence}"
        tranche = AgentTranche(
            tranche_id=tranche_id,
            agent_id=str(agent_id),
            role=AgentRole(role),
            timeframe=str(timeframe),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=capital / entry_price,
            capital=capital,
            confidence=confidence,
            expected_edge_bps=expected_edge_bps,
        )
        position = self.positions.setdefault((symbol, side), SharedPosition(symbol=symbol, side=side))
        position.tranches[tranche_id] = tranche
        return tranche

    def close_tranche(self, tranche_id: str, *, exit_price: float) -> AgentTranche:
        exit_price = self._positive_finite(exit_price, "exit_price")
        for position in self.positions.values():
            tranche = position.tranches.get(tranche_id)
            if tranche is None:
                continue
            if tranche.state != TrancheState.OPEN:
                raise ValueError("tranche is already closed")
            direction = 1.0 if tranche.side == "long" else -1.0
            tranche.realized_pnl = direction * (exit_price - tranche.entry_price) * tranche.quantity
            tranche.exit_price = exit_price
            tranche.closed_at = time.time()
            tranche.state = TrancheState.CLOSED
            return tranche
        raise KeyError(f"unknown tranche_id: {tranche_id}")

    def snapshot(self) -> dict[str, Any]:
        positions: list[dict[str, Any]] = []
        for (_, _), position in sorted(self.positions.items()):
            positions.append(
                {
                    "symbol": position.symbol,
                    "side": position.side,
                    "open_notional": position.open_notional,
                    "realized_pnl": position.realized_pnl,
                    "tranches": [asdict(row) for row in position.tranches.values()],
                }
            )
        return {
            "version": self.VERSION,
            "positions": positions,
            "open_notional": self.total_open_notional,
            "blocked_exposure": self.blocked_exposure,
            "max_symbol_exposure_fraction": self.max_symbol_exposure_fraction,
            "max_portfolio_exposure_fraction": self.max_portfolio_exposure_fraction,
            "shared_position_intelligence": True,
            "independent_tranche_exit": True,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
