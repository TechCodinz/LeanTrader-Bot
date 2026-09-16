"""Alpha ensemble and risk management for REAL_PROFIT_BOT.

Two more engines that were built and never reachable.

alpha_engines.AlphaRouter fuses nine real strategies -- oscillator
confluence, trend squeeze, naked price action, Donchian and Keltner
breakouts, VWAP bounce, volume spike, volatility regime and session bias --
into a calibrated probability with a size multiplier, and keeps per
(symbol, timeframe) reliability memory. It imports cleanly; nothing was
consuming it.

core.risk_manager.RiskManager implements Kelly-criterion sizing with a
safety factor, plus exposure, drawdown, VaR, correlation and concentration
checks. It could not be imported at all: like core/strategy_engine, its
import block had been stripped (Enum, dataclass, Optional, Dict, Any, List).
Those imports are restored; none of its logic is touched.

Neither can stop the bot trading. Alpha only raises or leaves confidence and
abstains when it has no opinion. Risk caps position size and reports
drawdown, but never sizes below the executable floor, and its hard halt is
opt-in and off by default -- a risk engine that silently stops a working bot
is the failure this integration exists to avoid.

Nothing here places an order.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

ALPHA_AGREE_BONUS = float(os.getenv("RPB_ALPHA_AGREE_BONUS", "7"))
ALPHA_MIN_PROB = float(os.getenv("RPB_ALPHA_MIN_PROB", "0.58"))
ALPHA_TIMEFRAME = os.getenv("RPB_ALPHA_TIMEFRAME", "5m")
ALPHA_CANDLES = int(os.getenv("RPB_ALPHA_CANDLES", "120") or 120)
ALPHA_CACHE_SECONDS = float(os.getenv("RPB_ALPHA_CACHE_SECONDS", "45") or 45)

# Risk. The drawdown halt is deliberately opt-in.
RISK_ENABLED = os.getenv("RPB_RISK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
RISK_HALT_ON_DRAWDOWN = os.getenv("RPB_RISK_HALT_ON_DRAWDOWN", "0").strip().lower() in {"1", "true", "yes", "on"}
RISK_PER_TRADE = float(os.getenv("RPB_RISK_PER_TRADE", "0.02"))
RISK_STOP_LOSS_PCT = float(os.getenv("RPB_RISK_STOP_LOSS_PCT", "0.003"))
EXECUTABLE_FLOOR_USD = float(os.getenv("RPB_EXECUTABLE_FLOOR_USD", "3.50"))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class AlphaEnsemble:
    """The nine-strategy alpha router, scoring the bot's own signal."""

    def __init__(self) -> None:
        self.available = False
        self.reason = "not_initialised"
        self._router = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        try:
            from alpha_engines import AlphaRouter

            self._router = AlphaRouter()
            self.available = True
            self.reason = "ready"
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    def strategy_names(self) -> List[str]:
        if not self.available:
            return []
        try:
            return [s.name for s in self._router.strats]
        except Exception:
            return []

    def _frame(self, exchange: Any, symbol: str):
        """Real candles as the DataFrame AlphaRouter expects."""
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached[0] < ALPHA_CACHE_SECONDS:
            return cached[1]
        try:
            import pandas as pd

            rows = exchange.fetch_ohlcv(symbol, timeframe=ALPHA_TIMEFRAME, limit=ALPHA_CANDLES)
        except Exception:
            return None
        if not rows or len(rows) < 60:
            return None
        try:
            frame = pd.DataFrame(
                rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        except Exception:
            return None
        self._cache[symbol] = (now, frame)
        return frame

    def evaluate(
        self,
        exchange: Any,
        symbol: str,
        signal: str,
        confidence: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Return (confidence, size_multiplier, detail).

        The alpha router is long-only -- it emits 'buy' or nothing -- which
        matches spot. It never changes the bot's direction and never vetoes;
        an absent opinion returns the confidence unchanged with a neutral
        multiplier of 1.0.
        """
        detail: Dict[str, Any] = {"applied": False, "reason": ""}
        if not self.available or signal != "BUY":
            detail["reason"] = self.reason if not self.available else "alpha_is_long_only"
            return confidence, 1.0, detail

        frame = self._frame(exchange, symbol)
        if frame is None:
            detail["reason"] = "insufficient_real_candles"
            return confidence, 1.0, detail

        try:
            decision = self._router.pick(frame, symbol, ALPHA_TIMEFRAME)
        except Exception as exc:
            detail["reason"] = f"alpha_error:{type(exc).__name__}"
            return confidence, 1.0, detail

        prob = _f(getattr(decision, "prob", 0.0))
        side = getattr(decision, "side", None)
        size_mult = _f(getattr(decision, "size_mult", 1.0), 1.0)
        votes = getattr(decision, "votes", {}) or {}

        detail.update(
            {
                "prob": round(prob, 4),
                "side": side,
                "size_mult": round(size_mult, 3),
                "top_votes": dict(
                    sorted(votes.items(), key=lambda kv: -abs(_f(kv[1])))[:3]
                ),
                "notes": str(getattr(decision, "notes", ""))[:160],
            }
        )

        if side != "buy" or prob < ALPHA_MIN_PROB:
            # No conviction. The bot's own signal stands, unmodified.
            detail["reason"] = f"alpha_no_conviction_p{prob:.2f}"
            return confidence, 1.0, detail

        adjusted = min(99.0, float(confidence) + ALPHA_AGREE_BONUS)
        detail["applied"] = True
        detail["reason"] = f"alpha_agrees_p{prob:.2f}"
        detail["confidence_before"] = round(float(confidence), 2)
        detail["confidence_after"] = round(adjusted, 2)
        return adjusted, max(0.5, min(2.0, size_mult)), detail

    def health(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "strategies": self.strategy_names(),
            "timeframe": ALPHA_TIMEFRAME,
        }


class RiskGovernor:
    """Kelly sizing and exposure limits over the authenticated wallet."""

    def __init__(self) -> None:
        self.available = False
        self.reason = "not_initialised"
        self._manager = None
        if not RISK_ENABLED:
            self.reason = "disabled_by_env"
            return
        try:
            from core.risk_manager import RiskManager

            self._manager = RiskManager()
            self.available = True
            self.reason = "ready"
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    def sync_wallet(self, portfolio_value: float) -> None:
        """Risk is measured against the real wallet, never a remembered one."""
        if not self.available:
            return
        try:
            self._manager.update_portfolio_value(max(0.0, _f(portfolio_value)))
        except Exception:
            pass

    def sync_position(self, symbol: str, amount: float, price: float) -> None:
        if not self.available:
            return
        try:
            self._manager.update_position(symbol, _f(amount), _f(price))
        except Exception:
            pass

    def cap_position(
        self,
        symbol: str,
        proposed_usd: float,
        confidence: float,
        size_multiplier: float = 1.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """Scale the proposed position, never to zero.

        The growth ladder proposes; this caps. The executable floor is always
        respected -- sizing below it would not place a smaller order, it would
        place no order at all, and a risk engine that silently stops a working
        bot is the exact failure this integration avoids. Whether the floor is
        actually affordable stays preflight's decision, as it already is.
        """
        detail: Dict[str, Any] = {"applied": False, "reason": ""}
        proposed = max(0.0, _f(proposed_usd))

        # The alpha multiplier applies whether or not risk is available.
        multiplied = proposed * max(0.5, min(2.0, _f(size_multiplier, 1.0)))

        if not self.available:
            detail["reason"] = self.reason
            return max(EXECUTABLE_FLOOR_USD, multiplied), detail

        try:
            from core.order_manager import OrderSide

            kelly = _f(
                self._manager.calculate_position_size(
                    symbol,
                    OrderSide.BUY,
                    max(0.0, min(1.0, _f(confidence) / 100.0)),
                    RISK_STOP_LOSS_PCT,
                    RISK_PER_TRADE,
                )
            )
            metrics = self._manager.get_risk_metrics()
        except Exception as exc:
            detail["reason"] = f"risk_error:{type(exc).__name__}"
            return max(EXECUTABLE_FLOOR_USD, multiplied), detail

        capped = min(multiplied, kelly) if kelly > 0 else multiplied
        final = max(EXECUTABLE_FLOOR_USD, capped)

        detail.update(
            {
                "applied": True,
                "proposed_usd": round(proposed, 6),
                "after_alpha_usd": round(multiplied, 6),
                "kelly_usd": round(kelly, 6),
                "final_usd": round(final, 6),
                "floored": final > capped,
                "drawdown": round(_f(metrics.get("current_drawdown")), 4),
                "exposure_ratio": round(_f(metrics.get("exposure_ratio")), 4),
                "risk_level": metrics.get("risk_level"),
                "reason": "kelly_cap" if capped < multiplied else "ladder_within_risk",
            }
        )
        return final, detail

    def should_halt(self) -> Tuple[bool, str]:
        """Only ever True when an operator has explicitly opted in."""
        if not self.available or not RISK_HALT_ON_DRAWDOWN:
            return False, ""
        try:
            metrics = self._manager.get_risk_metrics()
            drawdown = _f(metrics.get("current_drawdown"))
            limit = _f(getattr(self._manager, "max_drawdown_limit", 0.20), 0.20)
            if drawdown >= limit:
                return True, f"drawdown_{drawdown:.2%}_exceeds_{limit:.2%}"
        except Exception:
            return False, ""
        return False, ""

    def metrics(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": self.reason}
        try:
            return {"available": True, **self._manager.get_risk_metrics()}
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
