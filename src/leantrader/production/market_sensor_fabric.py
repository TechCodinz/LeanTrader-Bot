from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests

from .onchain_flow_sensors import (
    DefiLlamaChainLiquiditySensor,
    DefiLlamaProFlowSensor,
    EvmChainCongestionSensor,
    EthereumStablecoinIssuanceSensor,
    FlowIntelligenceSynthesizer,
    GlassnodeExchangeFlowSensor,
    SolanaNetworkCongestionSensor,
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _base(symbol: str) -> str:
    return symbol.upper().split("/", 1)[0]


def _contract(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace(":", "")


@dataclass
class SensorReading:
    source: str
    sensor: str
    symbol: str
    observed_at: float
    source_timestamp: float | None
    values: dict[str, Any]
    freshness_seconds: float | None
    confidence: float
    status: str = "available"
    provenance: str = "public_read_only"
    execution_authority: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _JsonHttp:
    def __init__(self, timeout: float = 8.0) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    def get(self, url: str, *, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("JSON response is not an object")
        return payload


class BybitDerivativesSensor:
    """Read-only public perpetual positioning sensor.

    Uses Bybit V5 public market endpoints only. It never loads credentials and
    has no order methods. Unsupported spot symbols degrade to `unsupported`
    rather than becoming runtime failures.
    """

    VERSION = "1.0"
    BASE_URL = "https://api.bybit.com"

    def __init__(self, refresh_seconds: int = 300, http: _JsonHttp | None = None) -> None:
        self.refresh_seconds = max(30, int(refresh_seconds))
        self.http = http or _JsonHttp()
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.calls = self.successes = self.failures = self.unsupported = 0
        self.last_error: str | None = None

    def _result(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = self.http.get(f"{self.BASE_URL}{path}", params=params)
        if int(payload.get("retCode") or 0) != 0:
            raise LookupError(str(payload.get("retMsg") or "Bybit market endpoint rejected request"))
        result = payload.get("result") or {}
        return result if isinstance(result, dict) else {}

    def collect(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        cached = self.cache.get(symbol)
        if cached and time.monotonic() - cached[0] < self.refresh_seconds:
            return dict(cached[1])
        self.calls += 1
        contract = _contract(symbol)
        now = time.time()
        try:
            ticker_r = self._result("/v5/market/tickers", {"category": "linear", "symbol": contract})
            ticker_rows = ticker_r.get("list") or []
            if not ticker_rows:
                raise LookupError("no linear perpetual ticker")
            ticker = ticker_rows[0]

            funding_r = self._result(
                "/v5/market/funding/history",
                {"category": "linear", "symbol": contract, "limit": 8},
            )
            funding_rows = funding_r.get("list") or []
            funding = [_finite(row.get("fundingRate")) for row in funding_rows if isinstance(row, dict)]

            oi_r = self._result(
                "/v5/market/open-interest",
                {"category": "linear", "symbol": contract, "intervalTime": "15min", "limit": 12},
            )
            oi_rows = oi_r.get("list") or []
            oi = [_finite(row.get("openInterest")) for row in oi_rows if isinstance(row, dict)]

            ratio_r = self._result(
                "/v5/market/account-ratio",
                {"category": "linear", "symbol": contract, "period": "15min", "limit": 12},
            )
            ratio_rows = ratio_r.get("list") or []
            ratios = [
                (_finite(row.get("buyRatio")), _finite(row.get("sellRatio")))
                for row in ratio_rows if isinstance(row, dict)
            ]

            latest_funding = funding[0] if funding else _finite(ticker.get("fundingRate"))
            funding_mean = sum(funding) / len(funding) if funding else latest_funding
            funding_change = latest_funding - (funding[-1] if len(funding) > 1 else latest_funding)
            latest_oi = oi[0] if oi else _finite(ticker.get("openInterest"))
            oldest_oi = oi[-1] if oi else latest_oi
            oi_change = (latest_oi / oldest_oi - 1.0) if latest_oi > 0 and oldest_oi > 0 else 0.0
            buy_ratio, sell_ratio = ratios[0] if ratios else (0.5, 0.5)
            positioning_skew = buy_ratio - sell_ratio
            mark = _finite(ticker.get("markPrice"))
            index = _finite(ticker.get("indexPrice"))
            basis = (mark / index - 1.0) if mark > 0 and index > 0 else _finite(ticker.get("basisRate"))
            source_ms = max(
                [_finite(row.get("timestamp")) for row in ratio_rows if isinstance(row, dict)]
                + [_finite(row.get("timestamp")) for row in oi_rows if isinstance(row, dict)]
                + [0.0]
            )
            source_ts = source_ms / 1000.0 if source_ms > 0 else now
            reading = SensorReading(
                source="Bybit V5 public market data",
                sensor="derivatives_positioning",
                symbol=symbol,
                observed_at=now,
                source_timestamp=source_ts,
                freshness_seconds=max(0.0, now - source_ts),
                confidence=0.90,
                values={
                    "funding_rate": latest_funding,
                    "funding_mean_8": funding_mean,
                    "funding_change": funding_change,
                    "open_interest": latest_oi,
                    "open_interest_change_15m_window": oi_change,
                    "long_ratio": buy_ratio,
                    "short_ratio": sell_ratio,
                    "positioning_skew": positioning_skew,
                    "perpetual_basis": basis,
                    "mark_price": mark,
                    "index_price": index,
                    "next_funding_time_ms": int(_finite(ticker.get("nextFundingTime"))),
                },
            ).as_dict()
            self.successes += 1
            self.last_error = None
        except LookupError as exc:
            self.unsupported += 1
            reading = SensorReading(
                source="Bybit V5 public market data", sensor="derivatives_positioning", symbol=symbol,
                observed_at=now, source_timestamp=None, freshness_seconds=None, confidence=0.0,
                values={}, status="unsupported", provenance=f"public_read_only:{type(exc).__name__}"
            ).as_dict()
        except Exception as exc:  # noqa: BLE001 - optional sensor isolation
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            reading = SensorReading(
                source="Bybit V5 public market data", sensor="derivatives_positioning", symbol=symbol,
                observed_at=now, source_timestamp=None, freshness_seconds=None, confidence=0.0,
                values={}, status="degraded", provenance="public_read_only"
            ).as_dict()
        self.cache[symbol] = (time.monotonic(), reading)
        return dict(reading)

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION, "calls": self.calls, "successes": self.successes,
            "failures": self.failures, "unsupported": self.unsupported, "last_error": self.last_error,
            "credentials_loaded": False, "read_only": True, "execution_authority": False,
        }


class BybitLiquidationTape:
    """Read-only background tape of Bybit public liquidation events."""

    VERSION = "1.0"
    WS_URL = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, window_seconds: int = 900, max_symbols: int = 180) -> None:
        self.window_seconds = max(60, int(window_seconds))
        self.max_symbols = max(1, int(max_symbols))
        self.events: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=5_000))
        self.desired: set[str] = set()
        self.subscribed: set[str] = set()
        self.ws: Any = None
        self.thread: threading.Thread | None = None
        self.lock = threading.RLock()
        self.connected = False
        self.messages = self.failures = 0
        self.last_error: str | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        try:
            import websocket  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return

        def on_open(ws: Any) -> None:
            self.connected = True
            self._subscribe(ws, sorted(self.desired))

        def on_message(_ws: Any, raw: str) -> None:
            try:
                msg = json.loads(raw)
                topic = str(msg.get("topic") or "")
                if not topic.startswith("allLiquidation."):
                    return
                now = time.time()
                for row in msg.get("data") or []:
                    contract = str(row.get("s") or "").upper()
                    if not contract.endswith("USDT"):
                        continue
                    symbol = f"{contract[:-4]}/USDT"
                    event = {
                        "timestamp": _finite(row.get("T")) / 1000.0,
                        "side": str(row.get("S") or ""),
                        "quantity": _finite(row.get("v")),
                        "price": _finite(row.get("p")),
                    }
                    with self.lock:
                        self.events[symbol].append(event)
                    self.messages += 1
                    self._prune(symbol, now)
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

        def on_error(_ws: Any, err: Any) -> None:
            self.failures += 1
            self.last_error = str(err)

        def on_close(_ws: Any, *_args: Any) -> None:
            self.connected = False
            self.subscribed.clear()

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self.ws = websocket.WebSocketApp(
                        self.WS_URL, on_open=on_open, on_message=on_message,
                        on_error=on_error, on_close=on_close,
                    )
                    self.ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as exc:  # noqa: BLE001
                    self.failures += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                if not self._stop.wait(5.0):
                    continue

        self.thread = threading.Thread(target=run, name="bybit-liquidation-tape", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def ensure_symbols(self, symbols: list[str]) -> None:
        contracts = [_contract(symbol) for symbol in symbols if symbol.upper().endswith("/USDT")]
        with self.lock:
            for contract in contracts:
                if len(self.desired) >= self.max_symbols:
                    break
                self.desired.add(contract)
            pending = sorted(self.desired - self.subscribed)
            ws = self.ws
        if self.connected and ws is not None and pending:
            self._subscribe(ws, pending)

    def _subscribe(self, ws: Any, contracts: list[str]) -> None:
        if not contracts:
            return
        # Keep command packets modest even though linear futures permit much larger args payloads.
        for i in range(0, len(contracts), 50):
            batch = contracts[i:i + 50]
            try:
                ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{c}" for c in batch]}))
                with self.lock:
                    self.subscribed.update(batch)
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

    def _prune(self, symbol: str, now: float) -> None:
        with self.lock:
            rows = self.events.get(symbol)
            if not rows:
                return
            cutoff = now - self.window_seconds
            while rows and _finite(rows[0].get("timestamp")) < cutoff:
                rows.popleft()

    def collect(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        now = time.time()
        self._prune(symbol, now)
        with self.lock:
            rows = list(self.events.get(symbol) or [])
        long_liq = sum(row["quantity"] * row["price"] for row in rows if row.get("side") == "Buy")
        short_liq = sum(row["quantity"] * row["price"] for row in rows if row.get("side") == "Sell")
        total = long_liq + short_liq
        imbalance = (short_liq - long_liq) / total if total > 0 else 0.0
        newest = max((_finite(row.get("timestamp")) for row in rows), default=0.0)
        return SensorReading(
            source="Bybit V5 public all-liquidation websocket",
            sensor="liquidation_tape", symbol=symbol, observed_at=now,
            source_timestamp=newest or None,
            freshness_seconds=(max(0.0, now - newest) if newest else None),
            confidence=0.90 if self.connected else 0.35,
            status="available" if self.connected else "warming_or_degraded",
            values={
                "window_seconds": self.window_seconds,
                "events": len(rows),
                "long_liquidated_notional": long_liq,
                "short_liquidated_notional": short_liq,
                "liquidation_notional": total,
                "liquidation_imbalance": imbalance,
            },
        ).as_dict()

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION, "connected": self.connected, "messages": self.messages,
            "failures": self.failures, "last_error": self.last_error,
            "desired_symbols": len(self.desired), "subscribed_symbols": len(self.subscribed),
            "credentials_loaded": False, "read_only": True, "execution_authority": False,
        }


class DeribitOptionsSensor:
    """Public options-volatility context for BTC/ETH without credentials."""

    VERSION = "1.0"
    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(self, refresh_seconds: int = 600, http: _JsonHttp | None = None) -> None:
        self.refresh_seconds = max(60, int(refresh_seconds))
        self.http = http or _JsonHttp(timeout=10.0)
        self.cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _call(self, method: str, params: dict[str, Any]) -> Any:
        payload = self.http.get(f"{self.BASE_URL}/{method}", params=params)
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload.get("result")

    @staticmethod
    def _kind(name: str) -> str:
        if name.endswith("-C"):
            return "call"
        if name.endswith("-P"):
            return "put"
        return "unknown"

    def collect(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        base = _base(symbol)
        if base not in {"BTC", "ETH"}:
            return SensorReading(
                source="Deribit public API", sensor="options_surface", symbol=symbol,
                observed_at=time.time(), source_timestamp=None, freshness_seconds=None,
                confidence=0.0, values={}, status="not_applicable"
            ).as_dict()
        cached = self.cache.get(base)
        if cached and time.monotonic() - cached[0] < self.refresh_seconds:
            return dict(cached[1])
        self.calls += 1
        now = time.time()
        try:
            rows = self._call("get_book_summary_by_currency", {"currency": base, "kind": "option"}) or []
            rows = [row for row in rows if isinstance(row, dict)]
            call_oi = sum(_finite(row.get("open_interest")) for row in rows if self._kind(str(row.get("instrument_name") or "")) == "call")
            put_oi = sum(_finite(row.get("open_interest")) for row in rows if self._kind(str(row.get("instrument_name") or "")) == "put")
            iv_weighted_num = 0.0
            iv_weighted_den = 0.0
            call_iv_num = put_iv_num = call_iv_den = put_iv_den = 0.0
            source_ms = 0.0
            for row in rows:
                iv = _finite(row.get("mark_iv"))
                weight = max(_finite(row.get("open_interest")), 0.0)
                source_ms = max(source_ms, _finite(row.get("creation_timestamp")))
                if iv > 0 and weight > 0:
                    iv_weighted_num += iv * weight
                    iv_weighted_den += weight
                    if self._kind(str(row.get("instrument_name") or "")) == "call":
                        call_iv_num += iv * weight; call_iv_den += weight
                    elif self._kind(str(row.get("instrument_name") or "")) == "put":
                        put_iv_num += iv * weight; put_iv_den += weight
            mean_iv = iv_weighted_num / iv_weighted_den if iv_weighted_den else 0.0
            call_iv = call_iv_num / call_iv_den if call_iv_den else 0.0
            put_iv = put_iv_num / put_iv_den if put_iv_den else 0.0
            now_ms = int(now * 1000)
            dvol = self._call(
                "get_volatility_index_data",
                {"currency": base, "start_timestamp": now_ms - 6 * 3600 * 1000, "end_timestamp": now_ms, "resolution": "3600"},
            ) or {}
            dvol_rows = dvol.get("data") or [] if isinstance(dvol, dict) else []
            dvol_close = _finite(dvol_rows[-1][4]) if dvol_rows else 0.0
            source_ts = source_ms / 1000.0 if source_ms else now
            reading = SensorReading(
                source="Deribit public API", sensor="options_surface", symbol=symbol,
                observed_at=now, source_timestamp=source_ts,
                freshness_seconds=max(0.0, now - source_ts), confidence=0.85,
                values={
                    "option_instruments": len(rows),
                    "call_open_interest": call_oi,
                    "put_open_interest": put_oi,
                    "put_call_open_interest_ratio": put_oi / call_oi if call_oi > 0 else 0.0,
                    "open_interest_skew": (put_oi - call_oi) / (put_oi + call_oi) if put_oi + call_oi > 0 else 0.0,
                    "open_interest_weighted_mark_iv": mean_iv,
                    "put_minus_call_iv_proxy": put_iv - call_iv,
                    "dvol_close": dvol_close,
                },
            ).as_dict()
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            reading = SensorReading(
                source="Deribit public API", sensor="options_surface", symbol=symbol,
                observed_at=now, source_timestamp=None, freshness_seconds=None,
                confidence=0.0, values={}, status="degraded"
            ).as_dict()
        self.cache[base] = (time.monotonic(), reading)
        return dict(reading)

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION, "calls": self.calls, "successes": self.successes,
            "failures": self.failures, "last_error": self.last_error,
            "credentials_loaded": False, "read_only": True, "execution_authority": False,
        }


class FredMacroSensor:
    """Optional FRED macro/rates sensor; stays unconfigured without a key."""

    VERSION = "1.0"
    URL = "https://api.stlouisfed.org/fred/series/observations"
    RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/releases/dates"
    DEFAULT_SERIES = ("DFF", "DGS2", "DGS10", "DTWEXBGS", "VIXCLS")

    def __init__(self, api_key_file: Path, refresh_seconds: int = 1800, http: _JsonHttp | None = None) -> None:
        self.api_key_file = api_key_file
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.http = http or _JsonHttp(timeout=10.0)
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    def _key(self) -> str:
        env = os.getenv("FRED_API_KEY", "").strip()
        if env:
            return env
        try:
            return self.api_key_file.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        key = self._key()
        now = time.time()
        if not key:
            result = {"status": "unconfigured", "values": {}, "read_only": True, "execution_authority": False}
            self.cache = (time.monotonic(), result)
            return dict(result)
        self.calls += 1
        values: dict[str, Any] = {}
        try:
            for series in self.DEFAULT_SERIES:
                payload = self.http.get(self.URL, params={
                    "series_id": series, "api_key": key, "file_type": "json",
                    "sort_order": "desc", "limit": 3,
                })
                observations = [row for row in payload.get("observations") or [] if row.get("value") not in {None, "."}]
                numeric = [_finite(row.get("value"), math.nan) for row in observations]
                numeric = [v for v in numeric if math.isfinite(v)]
                if numeric:
                    values[series] = {"latest": numeric[0], "change": numeric[0] - numeric[1] if len(numeric) > 1 else 0.0}
            calendar = self.http.get(self.RELEASE_DATES_URL, params={
                "api_key": key, "file_type": "json", "sort_order": "asc",
                "include_release_dates_with_no_data": "true", "limit": 100,
            })
            values["release_calendar"] = [
                {"date": row.get("date"), "release_id": row.get("release_id"), "release_name": row.get("release_name")}
                for row in (calendar.get("release_dates") or [])[:100]
            ]
            result = SensorReading(
                source="Federal Reserve Bank of St. Louis FRED API", sensor="macro_rates_cross_asset",
                symbol="GLOBAL", observed_at=now, source_timestamp=now, freshness_seconds=0.0,
                confidence=0.80, values=values, provenance="configured_read_only_api"
            ).as_dict()
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            result = {"status": "degraded", "values": values, "read_only": True, "execution_authority": False}
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION, "configured": bool(self._key()), "calls": self.calls,
            "successes": self.successes, "failures": self.failures, "last_error": self.last_error,
            "read_only": True, "execution_authority": False,
        }


class DefiLlamaStablecoinSensor:
    """Free read-only aggregate stablecoin liquidity/supply sensor."""

    VERSION = "1.0"
    BASE_URL = "https://stablecoins.llama.fi"

    def __init__(self, refresh_seconds: int = 1800, http: _JsonHttp | None = None) -> None:
        self.refresh_seconds = max(300, int(refresh_seconds))
        self.http = http or _JsonHttp(timeout=10.0)
        self.cache: tuple[float, dict[str, Any]] | None = None
        self.calls = self.successes = self.failures = 0
        self.last_error: str | None = None

    @staticmethod
    def _usd(value: Any) -> float:
        if isinstance(value, dict):
            return _finite(value.get("peggedUSD"))
        return _finite(value)

    def collect(self) -> dict[str, Any]:
        if self.cache and time.monotonic() - self.cache[0] < self.refresh_seconds:
            return dict(self.cache[1])
        self.calls += 1
        now = time.time()
        try:
            directory = self.http.get(f"{self.BASE_URL}/stablecoins", params={"includePrices": "true"})
            assets = directory.get("peggedAssets") or directory.get("stablecoins") or []
            total = 0.0
            depeg = 0
            top: list[dict[str, Any]] = []
            for row in assets:
                if not isinstance(row, dict):
                    continue
                circulating = self._usd(row.get("circulating"))
                total += max(0.0, circulating)
                price = _finite(row.get("price"), 1.0)
                if price > 0 and abs(price - 1.0) >= 0.01:
                    depeg += 1
                top.append({"symbol": row.get("symbol"), "circulating_usd": circulating, "price": price})
            top.sort(key=lambda row: float(row.get("circulating_usd") or 0.0), reverse=True)
            chart = self.http.get(f"{self.BASE_URL}/stablecoincharts/all", params={})
            chart_rows = chart if isinstance(chart, list) else chart.get("data") or chart.get("chart") or []
            hist: list[float] = []
            for row in chart_rows[-8:]:
                if isinstance(row, dict):
                    hist.append(self._usd(row.get("totalCirculatingUSD") or row.get("totalCirculating")))
            change = hist[-1] / hist[-2] - 1.0 if len(hist) >= 2 and hist[-2] > 0 else 0.0
            result = SensorReading(
                source="DefiLlama free stablecoin API", sensor="stablecoin_liquidity", symbol="GLOBAL",
                observed_at=now, source_timestamp=now, freshness_seconds=0.0, confidence=0.80,
                values={"total_circulating_usd": total, "recent_supply_change": change, "depeg_count_1pct": depeg, "largest": top[:10]},
            ).as_dict()
            self.successes += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            result = {"status": "degraded", "values": {}, "read_only": True, "execution_authority": False}
        self.cache = (time.monotonic(), result)
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {"version": self.VERSION, "calls": self.calls, "successes": self.successes, "failures": self.failures, "last_error": self.last_error, "read_only": True, "execution_authority": False}


class MarketSensorFabric:
    """Governed read-only market sensor bus for v12.8.

    Sensors may provide evidence to world/self/research models but never directly
    alter routes, sizing, risk ceilings or execution authority.
    """

    VERSION = "2.1"
    SCHEMA_VERSION = 3

    def __init__(
        self,
        state_path: Path,
        *,
        fred_api_key_file: Path,
        glassnode_api_key_file: Path | None = None,
        defillama_api_key_file: Path | None = None,
        ethereum_rpc_url_file: Path | None = None,
        solana_rpc_url_file: Path | None = None,
        derivatives_refresh_seconds: int = 300,
        options_refresh_seconds: int = 600,
        macro_refresh_seconds: int = 1800,
        onchain_refresh_seconds: int = 900,
        enabled: bool = True,
    ) -> None:
        self.state_path = state_path
        self.enabled = bool(enabled)
        self.derivatives = BybitDerivativesSensor(derivatives_refresh_seconds)
        self.liquidations = BybitLiquidationTape()
        self.options = DeribitOptionsSensor(options_refresh_seconds)
        self.macro = FredMacroSensor(fred_api_key_file, macro_refresh_seconds)
        self.stablecoins = DefiLlamaStablecoinSensor(macro_refresh_seconds)
        self.exchange_onchain = GlassnodeExchangeFlowSensor(
            glassnode_api_key_file or Path('/run/secrets/glassnode_api_key'),
            onchain_refresh_seconds,
        )
        self.chain_liquidity = DefiLlamaChainLiquiditySensor(macro_refresh_seconds)
        self.pro_flows = DefiLlamaProFlowSensor(
            defillama_api_key_file or Path('/run/secrets/defillama_api_key'),
            macro_refresh_seconds,
        )
        self.evm_congestion = EvmChainCongestionSensor(
            ethereum_rpc_url_file or Path('/run/secrets/ethereum_rpc_url'),
            derivatives_refresh_seconds,
        )
        self.solana_congestion = SolanaNetworkCongestionSensor(
            solana_rpc_url_file or Path('/run/secrets/solana_rpc_url'),
            derivatives_refresh_seconds,
        )
        self.stablecoin_issuance = EthereumStablecoinIssuanceSensor(
            ethereum_rpc_url_file or Path('/run/secrets/ethereum_rpc_url'),
            derivatives_refresh_seconds,
        )
        self.flow_synthesizer = FlowIntelligenceSynthesizer()
        self.state = self._load()
        self.last_error: str | None = None
        self.cycles = int(self.state.get("cycles") or 0)

    def start(self) -> None:
        if self.enabled:
            self.liquidations.start()

    def stop(self) -> None:
        self.liquidations.stop()
        self._save()

    def collect(self, symbols: list[str]) -> dict[str, Any]:
        symbols = [str(symbol).upper() for symbol in symbols]
        if not self.enabled:
            return {"timestamp": time.time(), "symbols": {}, "macro": {"status": "disabled"}, "source_status": {}, "read_only": True, "execution_authority": False, "can_increase_upstream_risk": False}
        self.liquidations.ensure_symbols(symbols)
        # Global sensors are cached and independent of individual symbols.
        # Collect them concurrently so research-only network I/O cannot serialize
        # the core market loop.
        with ThreadPoolExecutor(max_workers=7, thread_name_prefix="global-market-sensor") as global_pool:
            macro_future = global_pool.submit(self.macro.collect)
            stable_future = global_pool.submit(self.stablecoins.collect)
            chain_future = global_pool.submit(self.chain_liquidity.collect)
            pro_future = global_pool.submit(self.pro_flows.collect)
            evm_future = global_pool.submit(self.evm_congestion.collect)
            solana_future = global_pool.submit(self.solana_congestion.collect)
            issuance_future = global_pool.submit(self.stablecoin_issuance.collect)
            macro = macro_future.result()
            stablecoins = stable_future.result()
            chain_liquidity = chain_future.result()
            pro_flows = pro_future.result()
            evm_congestion = evm_future.result()
            solana_congestion = solana_future.result()
            stablecoin_issuance = issuance_future.result()
        per_symbol: dict[str, dict[str, Any]] = {}

        def collect_symbol(symbol: str) -> tuple[str, dict[str, Any]]:
            exchange_onchain = self.exchange_onchain.collect(symbol)
            row = {
                "derivatives": self.derivatives.collect(symbol),
                "liquidations": self.liquidations.collect(symbol),
                "options": self.options.collect(symbol),
                "exchange_onchain": exchange_onchain,
                "chain_liquidity": chain_liquidity,
                "stablecoins": stablecoins,
                "institutional_bridge_flows": pro_flows,
                "evm_network_congestion": evm_congestion,
                "solana_network_congestion": solana_congestion,
                "stablecoin_issuance": stablecoin_issuance,
            }
            row["flow_intelligence"] = self.flow_synthesizer.synthesize(
                symbol,
                exchange_onchain=exchange_onchain,
                chain_liquidity=chain_liquidity,
                stablecoins=stablecoins,
                pro_flows=pro_flows,
                evm_congestion=evm_congestion,
                solana_congestion=solana_congestion,
                stablecoin_issuance=stablecoin_issuance,
            )
            return symbol, row

        # Network-bound public sensors are isolated and bounded so a rotating
        # 20-symbol universe does not serialize dozens of HTTP requests into the
        # core trading cycle. Provider-side rate limiting still applies inside
        # each adapter/cache.
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(symbols))), thread_name_prefix="market-sensor") as pool:
            futures = [pool.submit(collect_symbol, symbol) for symbol in symbols]
            for future in as_completed(futures):
                symbol, row = future.result()
                per_symbol[symbol] = row
        self.cycles += 1
        snapshot = {
            "timestamp": time.time(),
            "symbols": per_symbol,
            "macro": macro,
            "stablecoins": stablecoins,
            "chain_liquidity": chain_liquidity,
            "institutional_bridge_flows": pro_flows,
            "evm_network_congestion": evm_congestion,
            "solana_network_congestion": solana_congestion,
            "stablecoin_issuance": stablecoin_issuance,
            "source_status": self.source_status(
                per_symbol, macro, stablecoins, chain_liquidity, pro_flows, evm_congestion, solana_congestion, stablecoin_issuance
            ),
            "read_only": True,
            "execution_authority": False,
            "can_increase_upstream_risk": False,
        }
        self.state["cycles"] = self.cycles
        self.state["latest"] = snapshot
        self._save()
        return snapshot

    @staticmethod
    def source_status(
        per_symbol: dict[str, Any],
        macro: dict[str, Any],
        stablecoins: dict[str, Any] | None = None,
        chain_liquidity: dict[str, Any] | None = None,
        pro_flows: dict[str, Any] | None = None,
        evm_congestion: dict[str, Any] | None = None,
        solana_congestion: dict[str, Any] | None = None,
        stablecoin_issuance: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        derivative_available = any((row.get("derivatives") or {}).get("status") == "available" for row in per_symbol.values())
        liquidation_available = any((row.get("liquidations") or {}).get("status") == "available" for row in per_symbol.values())
        options_available = any((row.get("options") or {}).get("status") == "available" for row in per_symbol.values())
        onchain_available = any((row.get("exchange_onchain") or {}).get("status") == "available" for row in per_symbol.values())
        flow_available = any((row.get("flow_intelligence") or {}).get("status") == "available" for row in per_symbol.values())
        macro_status = str(macro.get("status") or "available")
        stablecoin_status = str((stablecoins or {}).get("status") or "available")
        chain_status = str((chain_liquidity or {}).get("status") or "unavailable")
        pro_status = str((pro_flows or {}).get("status") or "unconfigured")
        evm_status = str((evm_congestion or {}).get("status") or "unconfigured")
        solana_status = str((solana_congestion or {}).get("status") or "unconfigured")
        issuance_status = str((stablecoin_issuance or {}).get("status") or "unconfigured")
        concentration_available = any(
            str((((row.get("exchange_onchain") or {}).get("values") or {}).get("whale_concentration_status") or "")) == "available"
            for row in per_symbol.values()
        )
        return {
            "derivatives_funding": "available" if derivative_available else "partial_or_unavailable",
            "open_interest": "available" if derivative_available else "partial_or_unavailable",
            "derivatives_positioning": "available" if derivative_available else "partial_or_unavailable",
            "liquidations": "available" if liquidation_available else "warming_or_unavailable",
            "options_surface": "available" if options_available else "partial_or_unavailable",
            "macro_calendar": "available" if macro_status == "available" else macro_status,
            "rates_fx_cross_asset": "available" if macro_status == "available" else macro_status,
            "onchain_flows": "available" if (onchain_available or flow_available or chain_status in {"available", "partial"}) else "unconfigured_or_unavailable",
            "exchange_onchain_flows": "available" if onchain_available else "unconfigured_or_partial",
            "chain_liquidity_flows": "available" if chain_status in {"available", "partial"} else chain_status,
            "institutional_flows": "available" if pro_status in {"available", "partial"} else pro_status,
            "bridge_flows": "available" if pro_status in {"available", "partial"} else pro_status,
            "whale_concentration": "available" if concentration_available else "unconfigured_or_partial",
            "ethereum_network_congestion": "available" if evm_status == "available" else evm_status,
            "solana_network_congestion": "available" if solana_status == "available" else solana_status,
            "chain_congestion": "available" if "available" in {evm_status, solana_status} else "unconfigured_or_unavailable",
            "stablecoin_mint_burn": "available" if issuance_status == "available" else issuance_status,
            "stablecoin_liquidity": "available" if stablecoin_status == "available" else stablecoin_status,
        }

    def symbol_context(self, symbol: str) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        return dict((latest.get("symbols") or {}).get(symbol.upper()) or {})

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        return {
            "healthy": True,
            "enabled": self.enabled,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "cycles": self.cycles,
            "source_status": latest.get("source_status") or {},
            "derivatives": self.derivatives.health(),
            "liquidations": self.liquidations.health(),
            "options": self.options.health(),
            "macro": self.macro.health(),
            "stablecoins": self.stablecoins.health(),
            "exchange_onchain": self.exchange_onchain.health(),
            "chain_liquidity": self.chain_liquidity.health(),
            "institutional_bridge_flows": self.pro_flows.health(),
            "evm_network_congestion": self.evm_congestion.health(),
            "solana_network_congestion": self.solana_congestion.health(),
            "stablecoin_issuance": self.stablecoin_issuance.health(),
            "flow_synthesizer": {
                "version": self.flow_synthesizer.VERSION,
                "read_only": True,
                "execution_authority": False,
            },
            "read_only": True,
            "execution_authority": False,
            "can_add_credentials": False,
            "can_enable_live": False,
            "state_path": str(self.state_path),
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": self.SCHEMA_VERSION, "cycles": 0, "latest": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(data.get("schema_version") or 0) == self.SCHEMA_VERSION:
                return data
        except Exception:
            pass
        return {"schema_version": self.SCHEMA_VERSION, "cycles": 0, "latest": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["cycles"] = self.cycles
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
