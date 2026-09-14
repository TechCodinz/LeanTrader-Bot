"""The fast lane's executor seam, backed by the central order authority.

The mature fast lane talks to its executor through exactly two methods:
``safe_snapshot()`` for account truth and ``mirror_events()`` to submit. The
historical implementation behind that seam was ``BybitTestnetExecutionEngine``,
which built its own ccxt client, called ``create_order`` directly, and refused
to run anywhere but Bybit Testnet -- it verified the testnet URLs on every
submission and raised otherwise.

Restoring that engine would give the system a second order path, invisible to
preflight, route_order, lineage and inventory, and would permanently weld the
fast lane to Testnet. The fast lane is not a Testnet feature; Testnet is where
it is being proven. So the seam is filled here instead, and every order it
submits goes through the same authority every other order goes through:

    lane -> ExecutionIntent -> preflight.prepare_order -> route_order
         -> BrokerCCXT -> the selected environment

which is mode-neutral: paper, testnet and live are a configuration of the
authority below, not of the lane above.

Two things are deliberately preserved from the historical engine because they
were hard-won and are not reproducible from the modern path alone:

* idempotency keyed on a deterministic client order id, so a timeout after the
  exchange accepted an order is reconciled rather than resubmitted;
* the rule that an acknowledgement is not a fill. A Bybit ack can carry an
  order id, status ``open`` and ``filled`` 0, and the lane's position
  accounting must not treat that as inventory it now owns.

Nothing here fabricates a fill, a position or a price.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ..execution import intent as execution_intent
from ..execution import inventory as inventory_reconciler
from ..execution import preflight

# Statuses that mean the venue is finished with an order, one way or another.
TERMINAL_STATUSES = frozenset({"closed", "canceled", "cancelled", "rejected", "expired"})


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class CentralAuthorityExecutor:
    """Account truth and order submission for the fast lane.

    ``environment`` is read from the broker rather than assumed, so this object
    reports the authority it actually has instead of the one it was hoping for.
    """

    def __init__(
        self,
        *,
        broker_provider: Optional[Callable[[], Any]] = None,
        route: Optional[Callable[..., Dict[str, Any]]] = None,
        quote: str = "USDT",
        snapshot_ttl_seconds: float = 2.0,
        max_order_usd: float = 5.0,
    ) -> None:
        self._broker_provider = broker_provider or preflight.shared_broker
        # Injected only so the seam can be exercised without an exchange.
        # Production leaves it None, which resolves to route_order.
        self._route = route
        self.quote = str(quote or "USDT").upper()
        self.snapshot_ttl_seconds = max(0.0, float(snapshot_ttl_seconds))
        self.max_order_usd = max(0.0, float(max_order_usd))

        self._lock = threading.RLock()
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._snapshot_cache: Optional[Dict[str, Any]] = None
        self._snapshot_at = 0.0

    # ------------------------------------------------------------- accounting

    def _router(self) -> Callable[..., Dict[str, Any]]:
        if self._route is not None:
            return self._route
        from ..execution.router import route_order

        return route_order

    def safe_snapshot(self) -> Dict[str, Any]:
        """What the account actually holds, or an empty, honest snapshot.

        The lane polls this at sub-second cadence, so reads are cached for a
        short TTL. A failed read returns ``fresh: False`` with no positions,
        which the lane treats as "do not act" -- never as "flat".
        """
        now = time.time()
        with self._lock:
            cached = self._snapshot_cache
            if cached is not None and now - self._snapshot_at < self.snapshot_ttl_seconds:
                return dict(cached)

        snapshot: Dict[str, Any] = {
            "fresh": False,
            "positions": {},
            "open_orders": 0,
            "kill_switch_active": False,
            "free_quote": 0.0,
            "portfolio_equity": 0.0,
            "environment": "",
            "risk_limits": {"max_order_usd": self.max_order_usd},
        }

        try:
            broker = self._broker_provider()
            environment = broker.resolve_mode()
            snapshot["environment"] = environment

            report = inventory_reconciler.reconcile_from_broker(
                broker, quote=self.quote
            )
            if report.get("available"):
                # Only inventory the venue would accept a sell for counts as a
                # position the lane can manage. Dust and non-executable rows
                # are real holdings but they are not exitable, and offering
                # them as positions is how a lane ends up trying to sell
                # something no venue will take.
                positions = {
                    str(item["asset"]).upper() + "/" + self.quote: _number(item["total"])
                    for item in report.get("items") or []
                    if item.get("sellable")
                }
                snapshot["positions"] = positions
                capital = report.get("capital") or {}
                snapshot["free_quote"] = _number(
                    capital.get("cash_spendable_now")
                )
                snapshot["portfolio_equity"] = _number(
                    capital.get("portfolio_value")
                )
                snapshot["fresh"] = True
        except Exception:
            # Telemetry, not an exception path: the lane must degrade to
            # inaction rather than crash the orchestrator's task.
            snapshot["fresh"] = False

        with self._lock:
            snapshot["open_orders"] = sum(
                1
                for record in self._orders.values()
                if str(record.get("status") or "").lower() not in TERMINAL_STATUSES
                and str(record.get("status") or "").lower() != "skipped"
            )
            self._snapshot_cache = dict(snapshot)
            self._snapshot_at = now

        return snapshot

    def invalidate(self) -> None:
        with self._lock:
            self._snapshot_cache = None
            self._snapshot_at = 0.0

    # ------------------------------------------------------------- submission

    @staticmethod
    def client_order_id(event: Dict[str, Any]) -> str:
        """A deterministic id for one lane decision.

        The same decision resubmitted after a timeout produces the same id, so
        it is recognised rather than duplicated. Built from the fields that
        identify the decision, not from the clock.
        """
        explicit = str(event.get("client_order_id") or "").strip()
        if explicit:
            return explicit
        material = "|".join(
            str(event.get(key) or "")
            for key in ("symbol", "side", "reason", "timestamp", "quantity")
        )
        return "lt" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:20]

    def mirror_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Submit the lane's decisions through the central authority."""
        return [self._mirror_event(dict(event or {})) for event in events or []]

    def _skip(self, client_id: str, symbol: str, side: str, reason: str) -> Dict[str, Any]:
        record = {
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "status": "skipped",
            "reason": reason,
            "filled": 0.0,
            "average": None,
            "order_id": None,
        }
        with self._lock:
            self._orders[client_id] = record
        # Every result carries the same shape, so a caller never has to know
        # which branch produced it to read whether this was a resubmission.
        return {"idempotent": False, **record}

    def _mirror_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        symbol = preflight.normalize_symbol(event.get("symbol")) or str(
            event.get("symbol") or ""
        ).upper()
        side = str(event.get("side") or "").lower()
        price = _number(event.get("price"))
        quantity = _number(event.get("quantity"))

        client_id = self.client_order_id(event)

        with self._lock:
            existing = self._orders.get(client_id)
            if existing is not None:
                # Already submitted. Return what is known rather than sending
                # a second order for the same decision.
                return {"idempotent": True, **existing}

        if side not in {"buy", "sell"} or price <= 0.0 or quantity <= 0.0:
            return self._skip(client_id, symbol, side, "invalid_lane_event")

        lane_intent = execution_intent.ExecutionIntent(
            symbol=symbol,
            side=side,
            source_engine="velocity_sniper_lane",
            source_strategy=str(event.get("reason") or "fast_collective"),
            candidate_id=client_id,
            confidence=_number(event.get("confidence"), 1.0),
        )
        lane_intent.advance(
            execution_intent.CANDIDATE, f"{side} {quantity} {symbol} @ {price}"
        )
        lane_intent.advance(execution_intent.PREFLIGHT)

        order_intent: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "confidence": _number(event.get("confidence"), 1.0),
            "order_type": "market",
            "price": price,
            "source": "velocity_sniper_lane",
        }
        if side == "sell":
            # An exit closes what is held. Sizing it from a fraction of a
            # balance would leave most of the position behind.
            order_intent["amount"] = quantity
        elif self.max_order_usd > 0.0:
            order_intent["risk_budget"] = self.max_order_usd

        prepared, blocked = preflight.prepare_order(order_intent)

        if prepared is None:
            lane_intent.stop(
                execution_intent.PREFLIGHT, blocked.blocker, blocked.detail
            )
            return self._skip(client_id, symbol, side, blocked.blocker.lower())

        preflight.record_event("submitted")
        lane_intent.advance(execution_intent.ROUTE_ORDER)

        payload = prepared.to_payload()
        params = payload.setdefault("params", {})
        params.update(lane_intent.to_payload())
        # Carried to the venue so a timeout can be reconciled against the
        # order the exchange already has.
        params["clientOrderId"] = client_id
        params["orderLinkId"] = client_id

        receipt = self._router()(payload)
        blocker = preflight.classify_receipt(receipt)

        if blocker is not None:
            detail = str((receipt or {}).get("error", ""))[:200]
            preflight.record_blocker(blocker, f"{symbol}:{detail}")
            lane_intent.stop(execution_intent.EXCHANGE_ORDER, blocker, detail)
            return self._skip(client_id, symbol, side, blocker.lower())

        order = (receipt or {}).get("order") or {}
        filled = _number(order.get("filled"))
        status = str(order.get("status") or "").lower()

        preflight.record_event("acknowledged")
        lane_intent.advance(execution_intent.EXCHANGE_ORDER, str(order.get("id") or ""))

        if filled > 0.0:
            preflight.record_event("fills")
            lane_intent.succeed(execution_intent.FILL, str(filled))
            if side == "sell":
                preflight.record_event("closes")
                lane_intent.succeed(execution_intent.CLOSE, symbol)
        elif status and status not in TERMINAL_STATUSES:
            # Acknowledged, not filled. The lane reads `status` to decide
            # whether it now holds something; reporting this as closed would
            # invent a position or a close that does not exist.
            status = "open"

        record = {
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "status": status or "open",
            "order_id": str(order.get("id") or "") or None,
            "filled": filled,
            "average": order.get("average") or order.get("price"),
            "quantity": _number(prepared.amount),
            "submitted_usd": _number(prepared.notional),
            "fee": _number((order.get("fee") or {}).get("cost"))
            if isinstance(order.get("fee"), dict)
            else 0.0,
            "reason": str(event.get("reason") or ""),
        }

        with self._lock:
            self._orders[client_id] = record

        preflight.invalidate_balance_cache()
        self.invalidate()

        return {"idempotent": False, **record}

    # ---------------------------------------------------------- observability

    def orders(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._orders.values()]

    def health(self) -> Dict[str, Any]:
        snapshot = self.safe_snapshot()
        with self._lock:
            records = list(self._orders.values())
        return {
            "environment": snapshot.get("environment", ""),
            "account_readable": bool(snapshot.get("fresh")),
            "positions": len(snapshot.get("positions") or {}),
            "free_quote": snapshot.get("free_quote", 0.0),
            "orders_submitted": sum(1 for r in records if r.get("order_id")),
            "orders_skipped": sum(
                1 for r in records if str(r.get("status")) == "skipped"
            ),
            "orders_filled": sum(1 for r in records if _number(r.get("filled")) > 0),
        }
