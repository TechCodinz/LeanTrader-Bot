from __future__ import annotations

import copy
import time
from typing import Any

from .testnet_exit_recycle import _record_non_tradeable_dust

PRICE_LIMIT_COOLDOWN_SECONDS = 300.0


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fresh_bid(self: Any, symbol: str) -> tuple[float, float]:
    ticker = self.exchange.fetch_ticker(symbol) or {}
    bid = max(0.0, _n(ticker.get("bid")))
    ask = max(0.0, _n(ticker.get("ask")))
    if bid > 0.0:
        return bid, ask
    book = self.exchange.fetch_order_book(symbol, limit=5) or {}
    bids = book.get("bids") or []
    return (max(0.0, _n(bids[0][0])) if bids else 0.0), ask


def _price_limit(self: Any, symbol: str) -> dict[str, Any]:
    method = None
    for name in ("public_get_v5_market_price_limit", "publicGetV5MarketPriceLimit"):
        candidate = getattr(self.exchange, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None:
        return {"supported": False, "ok": False}

    market = self.exchange.market(symbol)
    native = str(market.get("id") or symbol.replace("/", "")).upper()
    try:
        response = method({"category": "spot", "symbol": native})
    except Exception as exc:
        return {"supported": True, "ok": False, "error": type(exc).__name__}

    result = (response or {}).get("result") or {}
    try:
        code = int((response or {}).get("retCode", 0))
    except (TypeError, ValueError):
        code = -1
    sell_limit = max(0.0, _n(result.get("sellLmt")))
    return {
        "supported": True,
        "ok": code == 0 and sell_limit > 0.0,
        "sell_limit": sell_limit,
        "buy_limit": max(0.0, _n(result.get("buyLmt"))),
        "native_symbol": native,
    }


def _dust_or_guard(self: Any, symbol: str, preparation: dict[str, Any]) -> dict[str, Any]:
    if not callable(getattr(self.exchange, "fetch_ticker", None)):
        return preparation

    bid, ask = _fresh_bid(self, symbol)
    if bid <= 0.0:
        return {**preparation, "status": "blocked", "reason": "fresh_executable_bid_unavailable", "live_authority": False}

    current = max(0.0, _n((self.state.get("positions") or {}).get(symbol)))
    free_qty = max(0.0, _n(preparation.get("free_quantity")))
    market = self.exchange.market(symbol)
    limits = market.get("limits") or {}
    min_amount = max(0.0, _n((limits.get("amount") or {}).get("min")))
    min_cost = max(0.0, _n((limits.get("cost") or {}).get("min")))
    qty = max(0.0, _n(self.exchange.amount_to_precision(symbol, min(current, free_qty))))
    value = qty * bid

    if qty <= 0.0 or (min_amount > 0.0 and qty < min_amount) or (min_cost > 0.0 and value < min_cost):
        return _record_non_tradeable_dust(
            self,
            symbol=symbol,
            quantity=current,
            reference_price=bid,
            minimum_amount=min_amount,
            minimum_cost=min_cost,
            free_quantity=free_qty,
            reason="fresh_bid_below_exchange_executable_threshold",
        )

    now = time.time()
    blocked_until = _n((self.state.get("v1611_price_limit_blocked_until") or {}).get(symbol))
    if blocked_until > now:
        return {
            **preparation,
            "status": "blocked",
            "reason": "bybit_market_price_limit_cooldown",
            "fresh_bid": bid,
            "fresh_ask": ask,
            "blocked_until": blocked_until,
            "live_authority": False,
        }

    limit = _price_limit(self, symbol)
    if limit.get("supported") is True:
        if limit.get("ok") is not True:
            return {
                **preparation,
                "status": "blocked",
                "reason": "bybit_price_limit_unavailable",
                "fresh_bid": bid,
                "price_limit": limit,
                "live_authority": False,
            }
        sell_limit = max(0.0, _n(limit.get("sell_limit")))
        if sell_limit > 0.0 and bid + 1e-12 < sell_limit:
            until = now + PRICE_LIMIT_COOLDOWN_SECONDS
            with self._io_lock:
                self.state.setdefault("v1611_price_limit_blocked_until", {})[symbol] = until
                self.state["v1611_price_limit_preflight_blocks"] = int(
                    self.state.get("v1611_price_limit_preflight_blocks") or 0
                ) + 1
                self.state["v1611_last_price_limit_block"] = {
                    "symbol": symbol,
                    "fresh_bid": bid,
                    "sell_limit": sell_limit,
                    "blocked_at": now,
                    "blocked_until": until,
                    "live_authority": False,
                }
                self._save_state()
            return {
                **preparation,
                "status": "blocked",
                "reason": "bybit_market_price_limit_unexecutable",
                "fresh_bid": bid,
                "fresh_ask": ask,
                "bybit_sell_limit": sell_limit,
                "blocked_until": until,
                "price_limit": limit,
                "live_authority": False,
            }

    return {
        **preparation,
        "fresh_bid": bid,
        "fresh_ask": ask,
        "fresh_executable_value_usd": value,
        "price_limit": limit,
        "live_authority": False,
    }


def _startup_dust_sweep(self: Any) -> dict[str, Any]:
    if not callable(getattr(self.exchange, "fetch_ticker", None)):
        return {"enabled": True, "supported": False, "recycled": [], "live_authority": False}
    recycled: list[str] = []
    errors: list[dict[str, str]] = []
    for symbol in list((self.state.get("positions") or {}).keys()):
        try:
            current = max(0.0, _n((self.state.get("positions") or {}).get(symbol)))
            base = str(symbol).split("/", 1)[0]
            free_qty = max(0.0, _n((((self.state.get("account_balance") or {}).get("free") or {}).get(base))))
            market = self.exchange.market(symbol)
            limits = market.get("limits") or {}
            min_amount = max(0.0, _n((limits.get("amount") or {}).get("min")))
            min_cost = max(0.0, _n((limits.get("cost") or {}).get("min")))
            available_raw = min(current, free_qty)
            bid, _ = _fresh_bid(self, symbol)
            if bid <= 0.0:
                continue
            if (
                available_raw <= 0.0
                or (
                    min_amount > 0.0
                    and available_raw < min_amount
                )
            ):
                _record_non_tradeable_dust(
                    self,
                    symbol=symbol,
                    quantity=current,
                    reference_price=bid,
                    minimum_amount=min_amount,
                    minimum_cost=min_cost,
                    free_quantity=free_qty,
                    reason="startup_fresh_bid_below_exchange_executable_threshold",
                )
                recycled.append(symbol)
                continue
            qty = max(
                0.0,
                _n(
                    self.exchange.amount_to_precision(
                        symbol,
                        available_raw,
                    )
                ),
            )
            if qty <= 0.0 or (min_amount > 0.0 and qty < min_amount) or (min_cost > 0.0 and qty * bid < min_cost):
                _record_non_tradeable_dust(
                    self,
                    symbol=symbol,
                    quantity=current,
                    reference_price=bid,
                    minimum_amount=min_amount,
                    minimum_cost=min_cost,
                    free_quantity=free_qty,
                    reason="startup_fresh_bid_below_exchange_executable_threshold",
                )
                recycled.append(symbol)
        except Exception as exc:
            errors.append({"symbol": str(symbol), "error": type(exc).__name__})
    return {"enabled": True, "supported": True, "recycled": recycled, "errors": errors, "live_authority": False}


def install_testnet_exit_price_guard_v1611() -> None:
    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane
    from .testnet_execution import BybitTestnetExecutionEngine
    from .velocity_sniper_testnet import VelocitySniperTestnetLane

    if getattr(BybitTestnetExecutionEngine, "_v1611_exit_price_guard_installed", False):
        return

    original_start = BybitTestnetExecutionEngine.start
    original_prepare = BybitTestnetExecutionEngine.prepare_sell
    original_merge = BybitTestnetExecutionEngine._merge_observed
    original_health = BybitTestnetExecutionEngine.health
    original_hyper_health = HyperSpeedCollectiveTestnetLane.health

    def start(self: Any) -> None:
        original_start(self)
        with self._io_lock:
            self.state["v1611_startup_dust_sweep"] = _startup_dust_sweep(self)
            self._save_state()

    def prepare_sell(self: Any, symbol: str, requested_quantity: float, reference_price: float) -> dict[str, Any]:
        prepared = original_prepare(self, symbol, requested_quantity, reference_price)
        if str(prepared.get("status") or "") != "executable":
            return prepared
        return _dust_or_guard(self, str(symbol).upper(), prepared)

    def merge_observed(self: Any, record: dict[str, Any], observed: dict[str, Any]) -> None:
        original_merge(self, record, observed)
        info = observed.get("info") or {}
        reject = str(info.get("rejectReason") or info.get("reject_reason") or "")
        cancel_type = str(info.get("cancelType") or info.get("cancel_type") or "")
        if reject:
            record["exchange_reject_reason"] = reject
        if cancel_type:
            record["exchange_cancel_type"] = cancel_type
        if reject == "EC_ReachMarketPriceLimit" and _n(record.get("filled")) <= 0.0:
            symbol = str(record.get("symbol") or "").upper()
            now = time.time()
            if record.get("v1611_price_limit_rejection_counted") is not True:
                self.state["v1611_price_limit_rejections"] = int(self.state.get("v1611_price_limit_rejections") or 0) + 1
                record["v1611_price_limit_rejection_counted"] = True
            if symbol:
                self.state.setdefault("v1611_price_limit_blocked_until", {})[symbol] = now + PRICE_LIMIT_COOLDOWN_SECONDS
            self.state["v1611_last_price_limit_rejection"] = {
                "symbol": symbol,
                "reject_reason": reject,
                "observed_at": now,
                "blocked_until": now + PRICE_LIMIT_COOLDOWN_SECONDS,
                "live_authority": False,
            }

    def health(self: Any) -> dict[str, Any]:
        payload = original_health(self)
        payload["exit_price_guard"] = {
            "version": "1.60.11",
            "enabled": True,
            "fresh_executable_bid_minimum_check": True,
            "bybit_price_limit_preflight": True,
            "startup_fresh_bid_dust_sweep": copy.deepcopy(self.state.get("v1611_startup_dust_sweep") or {}),
            "price_limit_preflight_blocks": int(self.state.get("v1611_price_limit_preflight_blocks") or 0),
            "price_limit_rejections": int(self.state.get("v1611_price_limit_rejections") or 0),
            "last_price_limit_block": copy.deepcopy(self.state.get("v1611_last_price_limit_block") or {}),
            "last_price_limit_rejection": copy.deepcopy(self.state.get("v1611_last_price_limit_rejection") or {}),
            "retry_storm_order_submission_allowed_while_blocked": False,
            "ambiguous_order_resubmission_allowed": False,
            "live_authority": False,
        }
        payload["live_authority"] = False
        return payload

    def hyper_health(self: Any) -> dict[str, Any]:
        payload = original_hyper_health(self)
        try:
            snapshot = self.testnet.safe_snapshot()
        except Exception:
            snapshot = {}
        payload["version"] = "1.60.11"
        payload["exit_price_guard"] = copy.deepcopy(snapshot.get("exit_price_guard") or {})
        payload["live_authority"] = False
        return payload

    BybitTestnetExecutionEngine.start = start
    BybitTestnetExecutionEngine.prepare_sell = prepare_sell
    BybitTestnetExecutionEngine._merge_observed = merge_observed
    BybitTestnetExecutionEngine.health = health
    BybitTestnetExecutionEngine.VERSION = "2.8"
    BybitTestnetExecutionEngine._v1611_exit_price_guard_installed = True
    HyperSpeedCollectiveTestnetLane.health = hyper_health
    HyperSpeedCollectiveTestnetLane.VERSION = "1.60.11"
    VelocitySniperTestnetLane.VERSION = "1.60.11"
