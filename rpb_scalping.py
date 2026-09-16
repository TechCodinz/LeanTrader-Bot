"""Scalping confluence and micro-wallet growth for REAL_PROFIT_BOT.

Two existing engines are connected here, both of which were built and never
reached the bot that executes:

* SMART_SCALPING_ENGINE -- multi-timeframe confluence, market-session
  detection and per-symbol/per-session win-rate tracking. Its indicator
  boundary was a hardcoded dict that ignored the candles it was handed and
  therefore returned SELL at full strength for everything, always. That
  boundary is now repaired in place; the confluence and session logic around
  it is the original and is untouched.

* november_growth_strategy -- the balance-tiered position-sizing ladder. Its
  tiers are real; what it lacked was a connection to an authenticated wallet,
  so it scaled against an internal number. Here the tier is selected from the
  balance the bot actually reads from the exchange.

Neither engine can block a trade. Confluence adjusts confidence and abstains
when it has nothing to say; the growth ladder only sizes. REAL_PROFIT_BOT
remains the execution owner and nothing here places an order.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Timeframes the confluence check runs across. The engine needs at least three
# cached before it will report confluence at all.
CONFLUENCE_TIMEFRAMES = tuple(
    tf.strip()
    for tf in os.getenv("RPB_SCALP_TIMEFRAMES", "5m,15m,1h").split(",")
    if tf.strip()
)
CANDLE_LIMIT = int(os.getenv("RPB_SCALP_CANDLE_LIMIT", "60") or 60)
CACHE_SECONDS = float(os.getenv("RPB_SCALP_CACHE_SECONDS", "45") or 45)

CONFLUENCE_AGREE_BONUS = float(os.getenv("RPB_SCALP_AGREE_BONUS", "8"))
CONFLUENCE_CONFLICT_PENALTY = float(os.getenv("RPB_SCALP_CONFLICT_PENALTY", "6"))

# The growth ladder, taken from november_growth_strategy's phases: as the
# authenticated wallet grows, a larger fraction of it is committed per trade.
# (balance_at_least, fraction_of_free_quote, phase_name)
GROWTH_LADDER: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 0.25, "Micro"),
    (100.0, 0.20, "Foundation"),
    (300.0, 0.15, "Acceleration"),
    (800.0, 0.12, "Expansion"),
    (2000.0, 0.10, "Maturity"),
)

MIN_GROWTH_FRACTION = float(os.getenv("RPB_GROWTH_MIN_FRACTION", "0.05"))
MAX_GROWTH_FRACTION = float(os.getenv("RPB_GROWTH_MAX_FRACTION", "0.35"))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def growth_phase(balance: float) -> Tuple[str, float]:
    """The phase and position fraction for an authenticated balance.

    Below $100 the wallet is in the micro phase and commits the largest
    fraction, because a smaller fraction of a tiny balance cannot clear venue
    minimums at all. The fraction tapers as the wallet grows, which is what
    the original ladder did.
    """
    balance = max(0.0, _f(balance))
    phase_name, fraction = "Micro", GROWTH_LADDER[0][1]
    for floor, frac, name in GROWTH_LADDER:
        if balance >= floor:
            phase_name, fraction = name, frac
    return phase_name, max(MIN_GROWTH_FRACTION, min(MAX_GROWTH_FRACTION, fraction))


def target_position_usd(balance: float, minimum: float = 3.50) -> Tuple[float, str]:
    """What to commit on the next entry, from the real wallet.

    No ceiling: the historical $12 cap stopped the wallet compounding past
    roughly $48. The floor stays because an order below it cannot execute.
    """
    phase_name, fraction = growth_phase(balance)
    return max(minimum, _f(balance) * fraction), phase_name


class ScalpingConfluence:
    """Multi-timeframe agreement from the existing scalping engine."""

    def __init__(self) -> None:
        self.available = False
        self.reason = "not_initialised"
        self._analyzer = None
        self._tracker = None
        self._session = None
        self._last: Dict[str, float] = {}
        try:
            from SMART_SCALPING_ENGINE import (
                MarketSession,
                MultiTimeframeAnalyzer,
                SessionPerformanceTracker,
            )

            self._analyzer = MultiTimeframeAnalyzer()
            self._tracker = SessionPerformanceTracker()
            self._session = MarketSession
            self.available = True
            self.reason = "ready"
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    def current_session(self) -> str:
        try:
            return self._session.get_current_session()
        except Exception:
            return "unknown"

    def refresh(self, exchange: Any, symbol: str) -> bool:
        """Load real candles for each timeframe into the analyzer."""
        if not self.available:
            return False
        now = time.time()
        if now - self._last.get(symbol, 0.0) < CACHE_SECONDS:
            return True

        loaded = 0
        for timeframe in CONFLUENCE_TIMEFRAMES:
            try:
                rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
            except Exception:
                continue
            if not rows or len(rows) < 30:
                continue
            try:
                self._analyzer.analyze_timeframe(symbol, timeframe, rows)
                loaded += 1
            except Exception:
                continue

        if loaded:
            self._last[symbol] = now
        return loaded >= 3

    def evaluate(
        self,
        exchange: Any,
        symbol: str,
        signal: str,
        confidence: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Adjust confidence by multi-timeframe agreement.

        Never returns a different signal and never vetoes. When the engine has
        no opinion -- which is common, because its vote is mean-reversion
        flavoured and cancels itself on strong trends -- confidence is returned
        untouched.
        """
        detail: Dict[str, Any] = {"applied": False, "reason": "", "session": self.current_session()}
        if not self.available or signal not in {"BUY", "SELL"}:
            detail["reason"] = self.reason if not self.available else "no_directional_signal"
            return confidence, detail

        if not self.refresh(exchange, symbol):
            detail["reason"] = "insufficient_timeframes"
            return confidence, detail

        try:
            has_confluence, direction, consensus = self._analyzer.check_confluence(symbol)
        except Exception as exc:
            detail["reason"] = f"confluence_error:{type(exc).__name__}"
            return confidence, detail

        detail.update(
            {
                "has_confluence": bool(has_confluence),
                "direction": direction,
                "consensus": round(_f(consensus), 4),
            }
        )

        if not has_confluence or direction == "NEUTRAL":
            detail["reason"] = "no_confluence"
            return confidence, detail

        adjusted = float(confidence)
        if direction == signal:
            adjusted += CONFLUENCE_AGREE_BONUS
            detail["reason"] = f"{len(CONFLUENCE_TIMEFRAMES)}tf_agree_{direction}"
        else:
            adjusted -= CONFLUENCE_CONFLICT_PENALTY
            detail["reason"] = f"confluence_says_{direction}"

        detail["applied"] = True
        detail["confidence_before"] = round(float(confidence), 2)
        detail["confidence_after"] = round(max(0.0, min(99.0, adjusted)), 2)
        return max(0.0, min(99.0, adjusted)), detail

    def record_result(self, symbol: str, realized_net_pnl: float) -> None:
        """Feed an authenticated close back into session win-rate tracking."""
        if not self.available:
            return
        try:
            self._tracker.record_trade(
                symbol,
                self.current_session(),
                float(realized_net_pnl),
                bool(realized_net_pnl > 0),
            )
        except Exception:
            pass

    def session_stats(self, symbol: str) -> Dict[str, Any]:
        if not self.available:
            return {}
        try:
            return dict(self._tracker.performance.get(symbol, {}).get(self.current_session(), {}))
        except Exception:
            return {}

    def health(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "session": self.current_session(),
            "timeframes": list(CONFLUENCE_TIMEFRAMES),
        }
