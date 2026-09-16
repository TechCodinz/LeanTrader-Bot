"""Market intelligence for REAL_PROFIT_BOT.

REAL_PROFIT_BOT decided entries from 24h percentage change and 24h quote
volume alone. Every indicator engine this project already contains sat
unconnected beside it -- and core/strategy_engine.py, which holds the real
RSI, MACD, Bollinger and ATR implementations, could not even be imported
because nine names had been stripped from its import block. That is the
assembly failure: the intelligence existed and was never reachable.

This connector fetches real candles and a real order book through the bot's
own exchange client, runs the existing indicator math from
core/strategy_engine.py, and returns a confidence adjustment.

Two rules govern it, in this order:

1. It must never starve execution. The bot's own signal stands on its own.
   Intelligence adjusts confidence and can veto only a genuinely bad or
   non-executable setup -- never "all engines must agree". Any failure to
   compute returns the base signal completely unchanged, so a broken feed,
   a rate limit or a thin market can slow the bot down but can never stop it
   trading.

2. It computes from real data or it abstains. No synthetic candles, no
   default indicator values standing in for absent data.

Nothing here places an order. REAL_PROFIT_BOT remains the execution owner.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

# Confidence movement, in points on the bot's existing 0-100 scale.
CONFIRM_BONUS = float(os.getenv("RPB_INTEL_CONFIRM_BONUS", "6"))
CONTRADICT_PENALTY = float(os.getenv("RPB_INTEL_CONTRADICT_PENALTY", "8"))

# Vetoes. Deliberately few, and each one describes a setup that is either
# economically bad or literally not executable.
RSI_BLOWOFF = float(os.getenv("RPB_INTEL_RSI_BLOWOFF", "85"))
RSI_CAPITULATION = float(os.getenv("RPB_INTEL_RSI_CAPITULATION", "15"))
MAX_ENTRY_SPREAD_BPS = float(os.getenv("RPB_INTEL_MAX_SPREAD_BPS", "80"))

CANDLE_TIMEFRAME = os.getenv("RPB_INTEL_TIMEFRAME", "1m")
CANDLE_LIMIT = int(os.getenv("RPB_INTEL_CANDLE_LIMIT", "60") or 60)
CACHE_SECONDS = float(os.getenv("RPB_INTEL_CACHE_SECONDS", "20") or 20)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class MarketIntelligence:
    """Reads real market structure and scores the bot's own signal."""

    def __init__(self) -> None:
        self._candles: Dict[str, Tuple[float, Any]] = {}
        self._strategy = None
        self.available = False
        self.reason = "not_initialised"
        try:
            from core.strategy_engine import RSIStrategy

            # RSIStrategy is a concrete TechnicalStrategy, so it carries the
            # whole indicator toolkit: RSI, MACD, Bollinger and ATR. Using it
            # rather than reimplementing keeps one copy of the math.
            self._strategy = RSIStrategy()
            self.available = True
            self.reason = "ready"
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------- data

    def _fetch_candles(self, exchange: Any, symbol: str):
        """Real candles, briefly cached so a fast loop does not hammer the API."""
        now = time.time()
        cached = self._candles.get(symbol)
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1]
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=CANDLE_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception:
            return None
        if not rows or len(rows) < 30:
            return None
        self._candles[symbol] = (now, rows)
        return rows

    @staticmethod
    def _order_book_state(exchange: Any, symbol: str) -> Dict[str, float]:
        """Spread, depth imbalance and microprice shift from a real book."""
        state = {"spread_bps": 0.0, "imbalance": 0.0, "microprice_shift_bps": 0.0, "usable": False}
        try:
            book = exchange.fetch_order_book(symbol, limit=20) or {}
        except Exception:
            return state

        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return state

        best_bid, best_ask = _f(bids[0][0]), _f(asks[0][0])
        if best_bid <= 0 or best_ask <= 0:
            return state

        mid = (best_bid + best_ask) / 2.0
        bid_size = sum(_f(level[1]) for level in bids[:10])
        ask_size = sum(_f(level[1]) for level in asks[:10])
        total = bid_size + ask_size

        state["spread_bps"] = ((best_ask - best_bid) / mid) * 10_000.0
        if total > 0:
            state["imbalance"] = (bid_size - ask_size) / total
            microprice = (best_bid * ask_size + best_ask * bid_size) / total
            state["microprice_shift_bps"] = ((microprice - mid) / mid) * 10_000.0
        state["usable"] = True
        return state

    # ------------------------------------------------------- evaluation

    def evaluate(
        self,
        exchange: Any,
        symbol: str,
        signal: str,
        confidence: float,
        price: float,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Score the bot's signal against real structure.

        Returns (signal, adjusted_confidence, detail). On any failure the
        signal and confidence come back exactly as they went in -- intelligence
        that cannot compute must not be able to block a working bot.
        """
        detail: Dict[str, Any] = {"applied": False, "reason": "", "agreements": [], "conflicts": []}

        if not self.available or signal not in {"BUY", "SELL"}:
            detail["reason"] = self.reason if not self.available else "no_directional_signal"
            return signal, confidence, detail

        rows = self._fetch_candles(exchange, symbol)
        if rows is None:
            # Abstain rather than guess. The bot's own signal stands.
            detail["reason"] = "insufficient_real_candles"
            return signal, confidence, detail

        try:
            import pandas as pd

            closes = pd.Series([_f(r[4]) for r in rows])
            highs = pd.Series([_f(r[2]) for r in rows])
            lows = pd.Series([_f(r[3]) for r in rows])

            rsi = float(self._strategy.calculate_rsi(closes).iloc[-1])
            macd = self._strategy.calculate_macd(closes)
            histogram = float(macd["histogram"].iloc[-1])
            bands = self._strategy.calculate_bollinger_bands(closes)
            upper = float(bands["upper"].iloc[-1])
            lower = float(bands["lower"].iloc[-1])
            middle = float(bands["middle"].iloc[-1])
            atr = float(self._strategy.calculate_atr(highs, lows, closes).iloc[-1])
            last = float(closes.iloc[-1])
        except Exception as exc:
            detail["reason"] = f"indicator_error:{type(exc).__name__}"
            return signal, confidence, detail

        if any(v != v for v in (rsi, histogram, upper, lower, middle)):  # NaN guard
            detail["reason"] = "indicators_not_warm"
            return signal, confidence, detail

        book = self._order_book_state(exchange, symbol)

        detail.update(
            {
                "applied": True,
                "rsi": round(rsi, 2),
                "macd_histogram": round(histogram, 8),
                "bb_position": round((last - lower) / (upper - lower), 4)
                if upper > lower
                else 0.5,
                "atr_pct": round((atr / last) * 100.0, 4) if last > 0 else 0.0,
                "spread_bps": round(book["spread_bps"], 2),
                "imbalance": round(book["imbalance"], 4),
                "microprice_shift_bps": round(book["microprice_shift_bps"], 3),
            }
        )

        # --- vetoes: genuinely bad or non-executable, nothing else ---

        if book["usable"] and book["spread_bps"] > MAX_ENTRY_SPREAD_BPS:
            # A spread this wide eats the move before it happens.
            detail["reason"] = f"spread_{book['spread_bps']:.0f}bps_exceeds_{MAX_ENTRY_SPREAD_BPS:.0f}"
            return "HOLD", 0.0, detail

        # A stretched RSI on its own is NOT a veto. This bot enters on strong
        # 24h momentum, and strong momentum produces a high RSI by definition
        # -- vetoing on that alone would reject precisely the setups it exists
        # to catch. A blow-off is an extreme reading where momentum has already
        # begun rolling over, so both conditions are required.
        if signal == "BUY" and rsi >= RSI_BLOWOFF and histogram < 0:
            detail["reason"] = f"rsi_{rsi:.1f}_blowoff_with_macd_rollover"
            return "HOLD", 0.0, detail

        if signal == "SELL" and rsi <= RSI_CAPITULATION and histogram > 0:
            detail["reason"] = f"rsi_{rsi:.1f}_capitulation_with_macd_turn"
            return "HOLD", 0.0, detail

        # --- confirmation scoring: adjusts, never gates ---

        adjusted = float(confidence)

        def agree(name: str) -> None:
            detail["agreements"].append(name)

        def conflict(name: str) -> None:
            detail["conflicts"].append(name)

        if signal == "BUY":
            (agree if histogram > 0 else conflict)("macd")
            (agree if rsi < 70 else conflict)("rsi")
            (agree if last > middle else conflict)("bollinger_mid")
            if book["usable"]:
                (agree if book["imbalance"] > 0.05 else conflict)("book_imbalance")
                (agree if book["microprice_shift_bps"] > 0 else conflict)("microprice")
        else:
            (agree if histogram < 0 else conflict)("macd")
            (agree if rsi > 30 else conflict)("rsi")
            (agree if last < middle else conflict)("bollinger_mid")
            if book["usable"]:
                (agree if book["imbalance"] < -0.05 else conflict)("book_imbalance")
                (agree if book["microprice_shift_bps"] < 0 else conflict)("microprice")

        adjusted += CONFIRM_BONUS * len(detail["agreements"])
        adjusted -= CONTRADICT_PENALTY * len(detail["conflicts"])
        adjusted = max(0.0, min(99.0, adjusted))

        detail["confidence_before"] = round(float(confidence), 2)
        detail["confidence_after"] = round(adjusted, 2)
        detail["reason"] = (
            f"{len(detail['agreements'])} agree / {len(detail['conflicts'])} conflict"
        )
        return signal, adjusted, detail

    def health(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "cached_symbols": len(self._candles),
            "timeframe": CANDLE_TIMEFRAME,
        }
