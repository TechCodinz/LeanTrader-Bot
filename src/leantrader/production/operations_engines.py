from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SimulatedFill:
    requested_quantity: float
    filled_quantity: float
    average_price: float
    fee: float
    impact_bps: float
    partial: bool


class ExecutionRealityEngine:
    """Order-book paper fills with depth, fees, latency drift, and partial fills."""

    VERSION = "1.0"

    def market_fill(
        self,
        order_book: dict[str, Any],
        *,
        side: str,
        quantity: float,
        fee_bps: float,
        latency_bps: float = 0.0,
    ) -> SimulatedFill:
        if quantity <= 0 or side.lower() not in {"buy", "sell"}:
            raise ValueError("positive quantity and buy/sell side required")
        key = "asks" if side.lower() == "buy" else "bids"
        reverse = side.lower() == "sell"
        levels = sorted(
            [(float(price), float(available)) for price, available, *_ in order_book.get(key, [])],
            key=lambda row: row[0],
            reverse=reverse,
        )
        opposite = "bids" if key == "asks" else "asks"
        other = [(float(price), float(available)) for price, available, *_ in order_book.get(opposite, [])]
        if not levels or not other:
            raise ValueError("two-sided order book required")
        best_bid = max(float(row[0]) for row in order_book["bids"])
        best_ask = min(float(row[0]) for row in order_book["asks"])
        mid = (best_bid + best_ask) / 2.0
        remaining, cost, filled = quantity, 0.0, 0.0
        latency_multiplier = 1 + (latency_bps / 10_000) * (1 if side.lower() == "buy" else -1)
        for price, available in levels:
            take = min(remaining, max(0.0, available))
            cost += take * price * latency_multiplier
            filled += take
            remaining -= take
            if remaining <= 1e-12:
                break
        average = cost / filled if filled else 0.0
        direction = 1 if side.lower() == "buy" else -1
        impact = direction * (average - mid) / mid * 10_000 if filled else math.inf
        return SimulatedFill(quantity, filled, average, cost * fee_bps / 10_000, impact, filled + 1e-12 < quantity)

    def health(self) -> dict[str, Any]:
        return {"depth_aware": True, "partial_fills": True, "latency_model": True}


class ForexEngine:
    """Broker-neutral FX/gold symbol, pip, session, and risk-unit calculations."""

    VERSION = "1.0"

    @staticmethod
    def normalize(symbol: str, provider: str) -> str:
        compact = symbol.upper().replace("/", "").replace("_", "")
        if len(compact) != 6:
            raise ValueError("FX symbol must contain six characters, for example XAUUSD")
        if provider.lower() == "oanda":
            return f"{compact[:3]}_{compact[3:]}"
        if provider.lower() == "mt5":
            return compact
        raise ValueError(f"unsupported FX provider: {provider}")

    @staticmethod
    def pip_size(symbol: str) -> float:
        compact = symbol.upper().replace("/", "").replace("_", "")
        if compact.startswith("XAU"):
            return 0.01
        return 0.01 if compact.endswith("JPY") else 0.0001

    def risk_units(
        self,
        symbol: str,
        *,
        equity: float,
        risk_fraction: float,
        stop_distance: float,
        quote_to_account_rate: float = 1.0,
    ) -> int:
        if min(equity, risk_fraction, stop_distance, quote_to_account_rate) <= 0:
            return 0
        units = equity * risk_fraction / (stop_distance * quote_to_account_rate)
        return max(0, int(units))

    @staticmethod
    def session_allowed(when: dt.datetime) -> bool:
        utc = when.astimezone(dt.UTC)
        return utc.weekday() < 5 and 6 <= utc.hour < 21

    def health(self) -> dict[str, Any]:
        return {"providers": ["oanda", "mt5"], "execution_authority": False, "xauusd": True}


class ReconciliationEngine:
    """Detects ledger-versus-broker position mismatches before live promotion."""

    VERSION = "1.0"

    def compare(
        self,
        expected: dict[str, float],
        observed: dict[str, float],
        tolerance: float = 1e-8,
    ) -> dict[str, Any]:
        mismatches = []
        for symbol in sorted(set(expected) | set(observed)):
            delta = float(observed.get(symbol, 0.0) - expected.get(symbol, 0.0))
            if abs(delta) > tolerance:
                mismatches.append(
                    {
                        "symbol": symbol,
                        "expected": float(expected.get(symbol, 0.0)),
                        "observed": float(observed.get(symbol, 0.0)),
                        "delta": delta,
                    }
                )
        return {"reconciled": not mismatches, "mismatches": mismatches}

    def health(self) -> dict[str, Any]:
        return {"live_promotion_gate": True, "automatic_correction": False}


class MarketManipulationEngine:
    """Flags abrupt book cancellation and volume/price divergence patterns."""

    VERSION = "1.0"

    def evaluate(
        self, previous_book: dict[str, Any], current_book: dict[str, Any], frame: pd.DataFrame
    ) -> dict[str, Any]:
        def depth(book: dict[str, Any], side: str) -> float:
            return float(sum(float(row[1]) for row in book.get(side, [])[:10]))

        previous_bid, current_bid = depth(previous_book, "bids"), depth(current_book, "bids")
        previous_ask, current_ask = depth(previous_book, "asks"), depth(current_book, "asks")
        bid_cancel = max(0.0, 1 - current_bid / max(previous_bid, 1e-12))
        ask_cancel = max(0.0, 1 - current_ask / max(previous_ask, 1e-12))
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        volume_ratio = float(volume.iloc[-1] / max(float(volume.rolling(30).median().iloc[-1]), 1e-12))
        price_move = abs(float(close.iloc[-1] / close.iloc[-2] - 1.0))
        wash_like = volume_ratio >= 5.0 and price_move <= 0.0005
        spoof_like = max(bid_cancel, ask_cancel) >= 0.80
        return {
            "spoof_like": spoof_like,
            "wash_volume_like": wash_like,
            "bid_cancel_fraction": bid_cancel,
            "ask_cancel_fraction": ask_cancel,
            "volume_ratio": volume_ratio,
        }

    def health(self) -> dict[str, Any]:
        return {"heuristic_alert_only": True, "trade_authority": False}


class StrategyCapacityEngine:
    """Maximum executable quantity under an observed impact budget."""

    VERSION = "1.0"

    def estimate(self, order_book: dict[str, Any], side: str, impact_cap_bps: float = 20.0) -> dict[str, float]:
        bids = [(float(row[0]), float(row[1])) for row in order_book.get("bids", [])]
        asks = [(float(row[0]), float(row[1])) for row in order_book.get("asks", [])]
        if not bids or not asks:
            raise ValueError("two-sided order book required")
        mid = (max(price for price, _ in bids) + min(price for price, _ in asks)) / 2.0
        levels = sorted(asks) if side.lower() == "buy" else sorted(bids, reverse=True)
        direction = 1 if side.lower() == "buy" else -1
        quantity = notional = 0.0
        for price, available in levels:
            impact = direction * (price - mid) / mid * 10_000
            if impact > impact_cap_bps:
                break
            quantity += available
            notional += available * price
        return {"max_quantity": quantity, "max_notional": notional, "impact_cap_bps": impact_cap_bps}

    def health(self) -> dict[str, Any]:
        return {"order_book_capacity": True, "impact_budgeted": True}


class DataProvenanceEngine:
    """Append-only decision fingerprints for exact source/model traceability."""

    VERSION = "1.0"

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, symbol: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        event = {
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "symbol": symbol,
            "fingerprint": fingerprint,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return fingerprint

    def health(self) -> dict[str, Any]:
        return {"append_only": True, "sha256": True, "path": str(self.path)}


class TelegramAlertEngine:
    """Optional outbound paper alerts; no inbound commands or execution capability."""

    VERSION = "1.0"

    def __init__(self, token: str = "", chat_id: str = "", timeout: float = 5.0) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout

    def send(self, message: str) -> dict[str, Any]:
        if not self.token or not self.chat_id:
            return {"sent": False, "reason": "telegram not configured"}
        body = urllib.parse.urlencode({"chat_id": self.chat_id, "text": message[:4000]}).encode()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data=body,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return {"sent": 200 <= response.status < 300, "status": response.status}
        except OSError as exc:
            return {"sent": False, "reason": type(exc).__name__}

    def health(self) -> dict[str, Any]:
        return {"configured": bool(self.token and self.chat_id), "outbound_only": True, "execution_authority": False}


class OperationsEngineSuite:
    VERSION = "2.0"

    def __init__(self, provenance_path: Path) -> None:
        self.execution_reality = ExecutionRealityEngine()
        self.forex = ForexEngine()
        self.reconciliation = ReconciliationEngine()
        self.manipulation = MarketManipulationEngine()
        self.capacity = StrategyCapacityEngine()
        self.provenance = DataProvenanceEngine(provenance_path)
        self.telegram = TelegramAlertEngine()

    def record_decision(self, symbol: str, payload: dict[str, Any]) -> str:
        return self.provenance.record(symbol, payload)

    def alert_events(self, events: list[dict[str, Any]], halt_reason: str | None) -> list[dict[str, Any]]:
        results = []
        for event in events:
            results.append(
                self.telegram.send(f"LeanTrader paper {event['side']} {event['symbol']} reason={event['reason']}")
            )
        if halt_reason:
            results.append(self.telegram.send(f"LeanTrader paper halt: {halt_reason}"))
        return results

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "execution_reality": self.execution_reality.health(),
            "forex_xauusd": self.forex.health(),
            "reconciliation": self.reconciliation.health(),
            "market_manipulation": self.manipulation.health(),
            "strategy_capacity": self.capacity.health(),
            "data_provenance": self.provenance.health(),
            "telegram": self.telegram.health(),
        }
