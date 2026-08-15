from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .temporal_guard import MarketTemporalGuard


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
        return bool(MarketTemporalGuard.forex_session_status(when)["allowed"])

    def health(self) -> dict[str, Any]:
        return {
            "providers": ["oanda", "mt5"],
            "execution_authority": False,
            "xauusd": True,
            "session_timezone": "America/New_York",
            "dst_aware": True,
            "rollover_buffer": True,
        }


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


class PrometheusMetricsEngine:
    """Atomic Prometheus textfile snapshot derived only from canonical heartbeat fields."""

    VERSION = "1.0"

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _number(value: Any) -> float:
        number = float(value)
        return number if math.isfinite(number) else 0.0

    def write(self, status: dict[str, Any]) -> dict[str, Any]:
        engines = status.get("engines", {})
        universe = engines.get("market_universe", {})
        advanced = engines.get("advanced_shadow_suite", {})
        capabilities = advanced.get("capabilities", {})
        moon = capabilities.get("moon_scout_dynamic_scanner", {})
        arbitrage = capabilities.get("arbitrage", {})
        telegram = (engines.get("operations_safety", {}).get("telegram", {}))
        exchange_protection = engines.get("exchange_protection", {})
        lines = [
            "# HELP leantrader_healthy Whether the canonical paper runtime is healthy.",
            "# TYPE leantrader_healthy gauge",
            f"leantrader_healthy {1 if status.get('healthy') else 0}",
            "# TYPE leantrader_equity_usd gauge",
            f"leantrader_equity_usd {self._number(status.get('equity', 0.0)):.12g}",
            "# TYPE leantrader_cash_usd gauge",
            f"leantrader_cash_usd {self._number(status.get('cash', 0.0)):.12g}",
            "# TYPE leantrader_realized_pnl_usd gauge",
            f"leantrader_realized_pnl_usd {self._number(status.get('realized_pnl', 0.0)):.12g}",
            "# TYPE leantrader_open_positions gauge",
            f"leantrader_open_positions {len(status.get('open_positions', []))}",
            "# TYPE leantrader_cycle_errors gauge",
            f"leantrader_cycle_errors {len(status.get('errors', {}))}",
            "# TYPE leantrader_risk_halted gauge",
            f"leantrader_risk_halted {1 if status.get('halt_reason') else 0}",
            "# TYPE leantrader_market_universe_eligible gauge",
            f"leantrader_market_universe_eligible {int(universe.get('eligible_symbols', 0))}",
            "# TYPE leantrader_market_universe_scanned gauge",
            f"leantrader_market_universe_scanned {int(universe.get('last_scan_count', 0))}",
            "# TYPE leantrader_market_universe_full_sweeps counter",
            f"leantrader_market_universe_full_sweeps {int(universe.get('full_sweeps', 0))}",
            "# TYPE leantrader_moon_scout_scans counter",
            f"leantrader_moon_scout_scans {int(moon.get('scans', 0))}",
            "# TYPE leantrader_arbitrage_scans counter",
            f"leantrader_arbitrage_scans {int(arbitrage.get('scans', 0))}",
            "# TYPE leantrader_arbitrage_opportunities counter",
            f"leantrader_arbitrage_opportunities {int(arbitrage.get('opportunities_seen', 0))}",
            "# TYPE leantrader_telegram_messages_sent counter",
            f"leantrader_telegram_messages_sent {int(telegram.get('sent', 0))}",
            "# TYPE leantrader_telegram_delivery_failures counter",
            f"leantrader_telegram_delivery_failures {int(telegram.get('failed', 0))}",
            "# TYPE leantrader_exchange_authorization_checks counter",
            f"leantrader_exchange_authorization_checks {int(exchange_protection.get('authorization_checks', 0))}",
            "# TYPE leantrader_exchange_authorizations counter",
            f"leantrader_exchange_authorizations {int(exchange_protection.get('authorized', 0))}",
            "# TYPE leantrader_exchange_authority_blocks counter",
            f"leantrader_exchange_authority_blocks {int(exchange_protection.get('blocked', 0))}",
            "# TYPE leantrader_engine_healthy gauge",
        ]
        for name, state in sorted(engines.items()):
            safe_name = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
            lines.append(f'leantrader_engine_healthy{{engine="{safe_name}"}} {1 if state.get("healthy") else 0}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        return {"written": True, "path": str(self.path), "series": len(lines)}

    def health(self) -> dict[str, Any]:
        return {"format": "prometheus_textfile", "atomic": True, "path": str(self.path)}


class TelegramAlertEngine:
    """Tiered outbound paper/Testnet intelligence with no order authority."""

    VERSION = "2.0"

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        timeout: float = 5.0,
        cooldown_seconds: int | None = None,
        monitor_interval_cycles: int | None = None,
    ) -> None:
        token_path = Path(os.getenv("TELEGRAM_BOT_TOKEN_FILE", "/run/secrets/telegram_bot_token"))
        self.token = token or self._read_optional_secret(token_path) or os.getenv("TELEGRAM_BOT_TOKEN", "")
        admin = chat_id or os.getenv("TELEGRAM_ADMIN_CHAT_ID", "") or os.getenv("TELEGRAM_CHAT_ID", "")
        self.audiences = {
            "admin": self._chat_ids(admin, os.getenv("TELEGRAM_ADMIN_CHAT_IDS", "")),
            "free": self._chat_ids(os.getenv("TELEGRAM_FREE_CHAT_ID", ""), os.getenv("TELEGRAM_FREE_CHAT_IDS", "")),
            "paid": self._chat_ids(os.getenv("TELEGRAM_PAID_CHAT_ID", ""), os.getenv("TELEGRAM_PAID_CHAT_IDS", "")),
        }
        self.chat_id = admin
        self.timeout = timeout
        self.cooldown_seconds = (
            int(os.getenv("TELEGRAM_SIGNAL_COOLDOWN_SECONDS", "900"))
            if cooldown_seconds is None
            else cooldown_seconds
        )
        self.monitor_interval_cycles = (
            int(os.getenv("TELEGRAM_MONITOR_INTERVAL_CYCLES", "60"))
            if monitor_interval_cycles is None
            else monitor_interval_cycles
        )
        self.free_min_confidence = float(os.getenv("TELEGRAM_FREE_MIN_CONFIDENCE", "0.85"))
        self.paid_min_confidence = float(os.getenv("TELEGRAM_PAID_MIN_CONFIDENCE", "0.70"))
        self.moon_min_score = float(os.getenv("TELEGRAM_MOON_MIN_SCORE", "1.0"))
        self.testnet_trade_url = os.getenv("TELEGRAM_TESTNET_TRADE_URL", "https://testnet.bybit.com/").strip()
        if not 0 <= self.paid_min_confidence <= self.free_min_confidence <= 1:
            raise ValueError("Telegram confidence thresholds must satisfy 0 <= paid <= free <= 1")
        if self.cooldown_seconds < 60:
            raise ValueError("TELEGRAM_SIGNAL_COOLDOWN_SECONDS must be at least 60")
        if self.monitor_interval_cycles < 1:
            raise ValueError("TELEGRAM_MONITOR_INTERVAL_CYCLES must be positive")
        if self.moon_min_score < 0:
            raise ValueError("TELEGRAM_MOON_MIN_SCORE cannot be negative")
        self.sent = 0
        self.failed = 0
        self.sent_by_audience = {"admin": 0, "free": 0, "paid": 0}
        self.failed_by_audience = {"admin": 0, "free": 0, "paid": 0}
        self.skipped_cooldown = 0
        self.publish_cycles = 0
        self.last_error: str | None = None
        self.last_sent_at: str | None = None
        self.last_keys: dict[str, float] = {}

    @staticmethod
    def _read_optional_secret(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @staticmethod
    def _chat_ids(*values: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.strip()
                for value in values
                for item in value.split(",")
                if item.strip()
            )
        )

    def send(
        self,
        message: str,
        *,
        audience: str = "admin",
        reply_markup: dict[str, Any] | None = None,
        dedupe_key: str = "",
    ) -> dict[str, Any]:
        chats = self.audiences.get(audience, ())
        if not self.token or not chats:
            return {"sent": False, "reason": "telegram not configured"}
        now = time.time()
        if dedupe_key and now - float(self.last_keys.get(dedupe_key) or 0.0) < self.cooldown_seconds:
            self.skipped_cooldown += 1
            return {"sent": False, "reason": "cooldown", "audience": audience}
        deliveries: list[dict[str, Any]] = []
        for target in chats:
            payload: dict[str, Any] = {
                "chat_id": target,
                "text": message[:4000],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup, separators=(",", ":"))
            body = urllib.parse.urlencode(payload).encode()
            request = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data=body,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8") or "{}")
                    delivered = 200 <= response.status < 300 and response_payload.get("ok") is True
                    deliveries.append({"chat_id": target, "sent": delivered, "status": response.status})
                    if delivered:
                        self.sent += 1
                        self.sent_by_audience[audience] = self.sent_by_audience.get(audience, 0) + 1
                    else:
                        self.failed += 1
                        self.failed_by_audience[audience] = self.failed_by_audience.get(audience, 0) + 1
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.failed += 1
                self.failed_by_audience[audience] = self.failed_by_audience.get(audience, 0) + 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                deliveries.append({"chat_id": target, "sent": False, "reason": type(exc).__name__})
        delivered = any(item["sent"] for item in deliveries)
        if delivered:
            if dedupe_key:
                self.last_keys[dedupe_key] = now
            self.last_sent_at = dt.datetime.now(dt.UTC).isoformat()
            self.last_error = None
        return {"sent": delivered, "audience": audience, "deliveries": deliveries}

    def publish_cycle(self, status: dict[str, Any]) -> list[dict[str, Any]]:
        """Publish gated signals, Moon Scout, arbitrage and periodic admin health."""
        self.publish_cycles += 1
        if not self.token:
            return [{"sent": False, "reason": "telegram not configured"}]
        results: list[dict[str, Any]] = []
        testnet_enabled = bool((status.get("testnet_execution") or {}).get("enabled"))
        keyboard = (
            {"inline_keyboard": [[{"text": "Open Bybit Testnet", "url": self.testnet_trade_url}]]}
            if testnet_enabled and self._safe_testnet_url()
            else None
        )
        for symbol, decision in (status.get("decisions") or {}).items():
            route = decision.get("route") or {}
            if route.get("allowed") is not True:
                continue
            confidence = float(decision.get("confidence") or 0.0)
            safe_symbol = html.escape(str(symbol))
            side = "BUY" if decision.get("enter_long") else "OBSERVE"
            if confidence >= self.paid_min_confidence:
                paid_message = (
                    f"<b>LeanTrader verified signal</b>\n"
                    f"Market: <code>{safe_symbol}</code>\nAction: <b>{side}</b>\n"
                    f"Confidence: {confidence:.1%}\nRegime: {html.escape(str(decision.get('regime')))}\n"
                    f"Timeframe score: {float(decision.get('multi_timeframe_score') or 0.0):.3f}\n"
                    f"Route: {html.escape(str(route.get('reason')))}\n"
                    "Authority: paper/Testnet only"
                )
                results.append(
                    self.send(
                        paid_message,
                        audience="paid",
                        reply_markup=keyboard,
                        dedupe_key=f"paid:signal:{symbol}:{side}",
                    )
                )
            if confidence >= self.free_min_confidence:
                free_message = (
                    f"<b>LeanTrader market signal</b>\nMarket: <code>{safe_symbol}</code>\n"
                    f"Direction: <b>{side}</b>\nConfidence band: high\nPaper research—not financial advice"
                )
                results.append(
                    self.send(free_message, audience="free", dedupe_key=f"free:signal:{symbol}:{side}")
                )

        advanced_market = ((status.get("advanced_shadow") or {}).get("market") or {})
        moon = list(advanced_market.get("moon_scout_ranking") or [])
        if moon and float(moon[0].get("score") or 0.0) >= self.moon_min_score:
            top = moon[0]
            message = (
                f"<b>Moon Scout anomaly</b>\nMarket: <code>{html.escape(str(top.get('symbol')))}</code>\n"
                f"Cross-sectional score: {float(top.get('score') or 0.0):.3f}\n"
                f"Momentum: {float(top.get('momentum') or 0.0):.2%}\n"
                f"Volume spike: {float(top.get('volume_spike') or 0.0):.2f}x\n"
                "Scanner observation only"
            )
            results.append(
                self.send(message, audience="paid", dedupe_key=f"paid:moon:{top.get('symbol')}")
            )
            results.append(
                self.send(message, audience="admin", dedupe_key=f"admin:moon:{top.get('symbol')}")
            )

        opportunities = list(advanced_market.get("arbitrage_opportunities") or [])
        if opportunities:
            top = opportunities[0]
            message = (
                f"<b>Cross-venue spread observed</b>\nMarket: <code>{html.escape(str(top.get('symbol')))}</code>\n"
                f"Buy: {html.escape(str(top.get('buy_venue')))} @ {float(top.get('buy_price') or 0.0):.8g}\n"
                f"Sell: {html.escape(str(top.get('sell_venue')))} @ {float(top.get('sell_price') or 0.0):.8g}\n"
                f"Net after modeled costs: {float(top.get('net_bps') or 0.0):.2f} bps\n"
                f"Liquidity verified: {bool(top.get('liquidity_verified'))}\nNo execution authority"
            )
            results.append(
                self.send(message, audience="paid", dedupe_key=f"paid:arbitrage:{top.get('symbol')}")
            )
            results.append(
                self.send(message, audience="admin", dedupe_key=f"admin:arbitrage:{top.get('symbol')}")
            )

        protection = ((status.get("engines") or {}).get("exchange_protection") or {})
        block_reasons = dict(protection.get("block_reasons") or {})
        if block_reasons:
            reason_text = ", ".join(
                f"{html.escape(str(reason))}={int(count)}"
                for reason, count in sorted(block_reasons.items())
            )
            message = (
                "<b>Exchange protection blocked authority</b>\n"
                f"Reasons: {reason_text}\n"
                f"Checks: {int(protection.get('authorization_checks') or 0)}\n"
                "No blocked order was submitted"
            )
            results.append(
                self.send(
                    message,
                    audience="admin",
                    dedupe_key=f"admin:exchange-protection:{','.join(sorted(block_reasons))}",
                )
            )

        if self.publish_cycles % self.monitor_interval_cycles == 0:
            message = (
                f"<b>LeanTrader monitor</b>\nHealthy: {bool(status.get('healthy'))}\n"
                f"Equity: ${float(status.get('equity') or 0.0):.2f}\n"
                f"Open positions: {len(status.get('open_positions') or [])}\n"
                f"Cycle errors: {len(status.get('errors') or {})}\n"
                f"Runtime: <code>{html.escape(str(status.get('runtime')))}</code>"
            )
            results.append(self.send(message, audience="admin", dedupe_key=f"admin:monitor:{self.publish_cycles}"))
        return results

    def _safe_testnet_url(self) -> bool:
        parsed = urllib.parse.urlparse(self.testnet_trade_url)
        return parsed.scheme == "https" and "testnet" in parsed.netloc.lower()

    def health(self) -> dict[str, Any]:
        return {
            "configured": bool(self.token and any(self.audiences.values())),
            "audiences": {name: len(chats) for name, chats in self.audiences.items()},
            "free_min_confidence": self.free_min_confidence,
            "paid_min_confidence": self.paid_min_confidence,
            "cooldown_seconds": self.cooldown_seconds,
            "monitor_interval_cycles": self.monitor_interval_cycles,
            "testnet_trade_link_safe": self._safe_testnet_url(),
            "sent": self.sent,
            "failed": self.failed,
            "sent_by_audience": dict(self.sent_by_audience),
            "failed_by_audience": dict(self.failed_by_audience),
            "skipped_cooldown": self.skipped_cooldown,
            "publish_cycles": self.publish_cycles,
            "last_sent_at": self.last_sent_at,
            "last_error": self.last_error,
            "outbound_only": True,
            "inbound_commands": False,
            "paid_access_source": "configured_channel_or_chat",
            "payment_verification": False,
            "execution_authority": False,
        }


class OperationsEngineSuite:
    VERSION = "3.0"

    def __init__(self, provenance_path: Path, metrics_path: Path | None = None) -> None:
        self.execution_reality = ExecutionRealityEngine()
        self.forex = ForexEngine()
        self.reconciliation = ReconciliationEngine()
        self.manipulation = MarketManipulationEngine()
        self.capacity = StrategyCapacityEngine()
        self.provenance = DataProvenanceEngine(provenance_path)
        self.metrics = PrometheusMetricsEngine(metrics_path or provenance_path.with_name("vps_metrics.prom"))
        self.telegram = TelegramAlertEngine()

    def record_decision(self, symbol: str, payload: dict[str, Any]) -> str:
        return self.provenance.record(symbol, payload)

    def record_metrics(self, status: dict[str, Any]) -> dict[str, Any]:
        return self.metrics.write(status)

    def notify_cycle(self, status: dict[str, Any]) -> list[dict[str, Any]]:
        return self.telegram.publish_cycle(status)

    def alert_events(self, events: list[dict[str, Any]], halt_reason: str | None) -> list[dict[str, Any]]:
        results = []
        for event in events:
            results.append(
                self.telegram.send(
                    f"LeanTrader paper {event['side']} {event['symbol']} reason={event['reason']}",
                    audience="admin",
                )
            )
        if halt_reason:
            results.append(self.telegram.send(f"LeanTrader paper halt: {halt_reason}", audience="admin"))
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
            "prometheus_metrics": self.metrics.health(),
            "telegram": self.telegram.health(),
        }
