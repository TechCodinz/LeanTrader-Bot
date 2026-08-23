from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class TestnetSafetyError(RuntimeError):
    """Raised when the testnet boundary cannot be proven safe."""


class BybitTestnetExecutionEngine:
    """Mirror approved paper events to Bybit sandbox with bounded authority.

    The paper ledger remains the strategy and learning authority. This engine
    exercises authenticated order placement, fills and reconciliation against
    exchange test funds only. It never accepts production credentials or URLs.
    """

    VERSION = "2.1"
    TESTNET_CONFIRMATION = "I_UNDERSTAND_TESTNET_ONLY"

    def __init__(
        self,
        *,
        api_key_path: Path,
        api_secret_path: Path,
        state_path: Path,
        confirmation: str,
        max_order_usd: float = 10.0,
        max_position_usd: float = 20.0,
        max_daily_submitted_usd: float = 50.0,
        max_orders_per_day: int = 20,
        exchange_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if confirmation != self.TESTNET_CONFIRMATION:
            raise TestnetSafetyError("explicit Bybit testnet confirmation is required")
        if not 0 < max_order_usd <= 100:
            raise TestnetSafetyError("testnet max order must be in (0, 100] USD")
        if not max_order_usd <= max_position_usd <= 500:
            raise TestnetSafetyError("testnet max position must be between the order cap and 500 USD")
        if not max_order_usd <= max_daily_submitted_usd <= 1_000:
            raise TestnetSafetyError("testnet daily submitted cap must be between the order cap and 1000 USD")
        if not 1 <= max_orders_per_day <= 100:
            raise TestnetSafetyError("testnet daily order count must be in [1, 100]")

        self.api_key_path = api_key_path
        self.api_secret_path = api_secret_path
        self.state_path = state_path
        self.max_order_usd = float(max_order_usd)
        self.max_position_usd = float(max_position_usd)
        self.max_daily_submitted_usd = float(max_daily_submitted_usd)
        self.max_orders_per_day = int(max_orders_per_day)
        self.exchange_factory = exchange_factory
        self.exchange: Any | None = None
        self.endpoint_verified = False
        self.authenticated = False
        self.credential_fingerprint: str | None = None
        self._eligible_symbols: set[str] = set()
        self.api_attestation: dict[str, Any] = {"verified": False}
        self.exchange_capabilities: dict[str, Any] = {}
        self.state: dict[str, Any] = self._load_state()

    def start(self) -> None:
        api_key = self._read_secret(self.api_key_path, "API key")
        api_secret = self._read_secret(self.api_secret_path, "API secret")
        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 20_000,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            },
        }
        if self.exchange_factory is None:
            import ccxt  # type: ignore

            exchange = ccxt.bybit(config)
        else:
            exchange = self.exchange_factory(config)

        try:
            # CCXT requires sandbox selection to be the first call after creation.
            exchange.set_sandbox_mode(True)
            self.exchange = exchange
            self._verify_testnet_urls()
            self.exchange.load_markets()
            self._eligible_symbols = {
                str(symbol).upper()
                for symbol, market in self.exchange.markets.items()
                if market.get("spot") and market.get("active") is not False
            }
            if not self._eligible_symbols:
                raise TestnetSafetyError("Bybit Testnet returned no active spot markets")
            self._verify_testnet_urls()
            self._attest_exchange_capabilities()
            self._attest_api_key()
            self._update_balance_snapshot(self.exchange.fetch_balance())
            self.authenticated = True
            self.credential_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
            self.reconcile()
        except Exception as exc:
            self.authenticated = False
            raise TestnetSafetyError(f"Bybit testnet startup failed: {self._redact(str(exc))}") from exc

    def stop(self) -> None:
        close = getattr(self.exchange, "close", None)
        if callable(close):
            close()

    def mirror_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._require_started()
        reconciliation = self.reconcile()
        if not reconciliation["reconciled"]:
            raise RuntimeError("testnet order reconciliation is unresolved; new mirrors are paused")
        results: list[dict[str, Any]] = []
        for event in events:
            if event.get("side") not in {"buy", "sell"}:
                continue
            try:
                results.append(self._mirror_event(event))
            except Exception as exc:
                raise RuntimeError(self._redact(str(exc))) from exc
        return results

    def eligible_symbols(self, quote: str = "USDT") -> set[str]:
        self._require_started()
        suffix = f"/{quote.upper()}"
        return {symbol for symbol in self._eligible_symbols if symbol.endswith(suffix)}

    def reconcile(self) -> dict[str, Any]:
        self._require_started()
        self._verify_testnet_urls()
        checked = 0
        errors: list[dict[str, str]] = []
        for client_id, record in list(self.state["orders"].items()):
            if record.get("status") in {"closed", "canceled", "rejected", "skipped"}:
                continue
            order_id = record.get("order_id")
            try:
                observed = (
                    self.exchange.fetch_order(order_id, record["symbol"])
                    if order_id
                    else self._find_order_by_client_id(record["symbol"], client_id)
                )
                if observed is None:
                    errors.append({"client_order_id": client_id, "reason": "order_state_ambiguous"})
                    continue
                self._merge_observed(record, observed)
                checked += 1
            except Exception as exc:  # noqa: BLE001 - normalize any provider reconciliation failure
                errors.append({"client_order_id": client_id, "reason": self._redact(str(exc))})
        try:
            self._update_balance_snapshot(self.exchange.fetch_balance())
        except Exception as exc:  # noqa: BLE001 - a stale balance pauses new mirrors safely
            errors.append({"client_order_id": "account_balance", "reason": self._redact(str(exc))})
        self.state["last_reconciliation"] = dt.datetime.now(dt.UTC).isoformat()
        self.state["last_reconciliation_errors"] = errors
        self._save_state()
        return {"reconciled": not errors, "checked": checked, "errors": errors}

    def reconcile_required(self) -> dict[str, Any]:
        """Require a clean Testnet reconciliation before new mirrors."""
        result = self.reconcile()

        if result.get("reconciled") is not True:
            issues = len(result.get("errors") or [])
            raise RuntimeError(
                "testnet reconciliation is unresolved "
                f"({issues} issue(s))"
            )

        return result

    def health(self) -> dict[str, Any]:
        orders = list(self.state["orders"].values())
        execution_notional = float(self.state["execution_notional_usd"])
        average_slippage = (
            float(self.state["weighted_slippage_bps_usd"]) / execution_notional
            if execution_notional > 0
            else 0.0
        )
        methods = dict(self.exchange_capabilities.get("methods") or {})
        return {
            "provider": "bybit",
            "environment": "testnet",
            "execution_authority": "testnet_only",
            "live_authority": False,
            "sandbox_endpoint_verified": self.endpoint_verified,
            "authenticated": self.authenticated,
            "credential_fingerprint": self.credential_fingerprint,
            "eligible_spot_markets": len(self._eligible_symbols),
            "exchange_capabilities": dict(self.exchange_capabilities),
            "api_attestation": dict(self.api_attestation),
            "persistent": True,
            "state_path": str(self.state_path),
            "orders": len(orders),
            "open_orders": sum(record.get("status") in {"open", "submitting"} for record in orders),
            "positions": dict(self.state["positions"]),
            "position_cost_usd": dict(self.state["position_cost_usd"]),
            "account_balance": dict(self.state["account_balance"]),
            "performance": {
                "realized_pnl_usd": float(self.state["realized_pnl_usd"]),
                "closed_positions": int(self.state["closed_positions"]),
                "winning_positions": int(self.state["winning_positions"]),
                "filled_orders": int(self.state["filled_orders"]),
                "execution_notional_usd": execution_notional,
                "average_adverse_slippage_bps": average_slippage,
                "last_fill": self.state.get("last_fill"),
            },
            "daily_submitted_usd": float(self.state["daily_submitted_usd"]),
            "daily_order_count": int(self.state["daily_order_count"]),
            "last_reconciliation": self.state.get("last_reconciliation"),
            "last_reconciliation_errors": list(self.state.get("last_reconciliation_errors", [])),
            "protection_contract": {
                "market_precision_and_limits": True,
                "fee_and_slippage_model": True,
                "balance_reconciliation": True,
                "order_idempotency": True,
                "order_state_recovery": bool(methods.get("fetchOrder"))
                or (bool(methods.get("fetchOpenOrders")) and bool(methods.get("fetchClosedOrders"))),
                "position_and_daily_caps": True,
                "kill_switch": True,
            },
            "risk_limits": {
                "max_order_usd": self.max_order_usd,
                "max_position_usd": self.max_position_usd,
                "max_daily_submitted_usd": self.max_daily_submitted_usd,
                "max_orders_per_day": self.max_orders_per_day,
            },
            "kill_switch_active": (self.state_path.parent / "TESTNET_HALT").exists(),
        }

    def _mirror_event(self, event: dict[str, Any]) -> dict[str, Any]:
        symbol = str(event["symbol"]).upper()
        side = str(event["side"]).lower()
        price = float(event["price"])
        requested_quantity = float(event["quantity"])
        if price <= 0 or requested_quantity <= 0:
            raise ValueError("positive testnet price and quantity are required")

        client_id = self._client_order_id(event)
        existing = self.state["orders"].get(client_id)
        if existing is not None:
            return {"client_order_id": client_id, "idempotent": True, **self._public_record(existing)}

        if symbol not in self._eligible_symbols:
            return self._skip(client_id, symbol, side, "market_unavailable_on_bybit_testnet")

        self._refresh_day()
        if side == "buy" and (self.state_path.parent / "TESTNET_HALT").exists():
            return self._skip(client_id, symbol, side, "testnet_kill_switch")
        if side == "buy" and self.state["daily_order_count"] >= self.max_orders_per_day:
            return self._skip(client_id, symbol, side, "daily_order_count_cap")
        if side == "buy" and float(self.state["daily_submitted_usd"]) >= self.max_daily_submitted_usd:
            return self._skip(client_id, symbol, side, "daily_submitted_notional_cap")

        market = self.exchange.market(symbol)
        minimum_cost = float(((market.get("limits") or {}).get("cost") or {}).get("min") or 0.0)
        minimum_amount = float(((market.get("limits") or {}).get("amount") or {}).get("min") or 0.0)
        current_quantity = float(self.state["positions"].get(symbol, 0.0))
        if side == "buy":
            submitted_usd = max(requested_quantity * price, minimum_cost, minimum_amount * price)
            if submitted_usd > self.max_order_usd:
                return self._skip(client_id, symbol, side, "exchange_minimum_exceeds_order_cap")
            reserved_notional = self._pending_buy_notional(symbol, price)
            if current_quantity * price + reserved_notional + submitted_usd > self.max_position_usd:
                return self._skip(client_id, symbol, side, "position_notional_cap")
            quantity = submitted_usd / price
        else:
            if current_quantity <= 0:
                return self._skip(client_id, symbol, side, "no_testnet_position")
            remaining = max(0.0, float(event.get("remaining_quantity", 0.0)))
            fraction = 1.0 if remaining <= 0 else requested_quantity / (requested_quantity + remaining)
            quantity = current_quantity * min(1.0, max(0.0, fraction))
            submitted_usd = quantity * price

        quantity = float(self.exchange.amount_to_precision(symbol, quantity))
        if side == "sell":
            quantity = min(quantity, current_quantity)
        if quantity <= 0:
            return self._skip(client_id, symbol, side, "quantity_below_exchange_precision")
        submitted_usd = quantity * price
        if side == "buy" and submitted_usd > self.max_order_usd:
            return self._skip(client_id, symbol, side, "exchange_precision_exceeds_order_cap")
        if side == "buy" and float(self.state["daily_submitted_usd"]) + submitted_usd > self.max_daily_submitted_usd:
            return self._skip(client_id, symbol, side, "daily_submitted_notional_cap")

        record = {
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "submitted_usd": submitted_usd,
            "reference_price": price,
            "reason": str(event.get("reason", "paper_event")),
            "paper_event_timestamp": str(event.get("timestamp", "")),
            "status": "submitting",
            "order_id": None,
            "filled": 0.0,
            "applied_filled": 0.0,
            "filled_cost": 0.0,
            "applied_fill_cost": 0.0,
            "average": None,
            "fee": 0.0,
            "fee_currency": None,
            "applied_fee": 0.0,
            "fill_counted": False,
        }
        self.state["orders"][client_id] = record
        # Count before the network call. A timeout after exchange acceptance is
        # ambiguous, so the same client id is reconciled and never resubmitted.
        self.state["daily_submitted_usd"] += submitted_usd
        self.state["daily_order_count"] += 1
        self._save_state()

        self._verify_testnet_urls()
        observed = self.exchange.create_order(
            symbol,
            "market",
            side,
            quantity,
            None,
            {"orderLinkId": client_id},
        )
        self._merge_observed(record, observed)
        self._save_state()
        return {"client_order_id": client_id, "idempotent": False, **self._public_record(record)}

    def _find_order_by_client_id(self, symbol: str, client_id: str) -> dict[str, Any] | None:
        """Recover an order accepted during an ambiguous network failure."""
        params = {"orderLinkId": client_id}
        for method_name in ("fetch_open_orders", "fetch_closed_orders", "fetch_canceled_orders"):
            method = getattr(self.exchange, method_name, None)
            if not callable(method):
                continue
            try:
                candidates = method(symbol, None, None, params)
            except Exception:  # noqa: BLE001,S112 - query the next provider order collection
                continue
            for candidate in candidates or []:
                observed_client_id = (
                    candidate.get("clientOrderId")
                    or candidate.get("client_order_id")
                    or (candidate.get("info") or {}).get("orderLinkId")
                )
                if observed_client_id == client_id:
                    return candidate
        return None

    def _pending_buy_notional(self, symbol: str, price: float) -> float:
        reserved = 0.0
        for record in self.state["orders"].values():
            if record.get("symbol") != symbol or record.get("side") != "buy":
                continue
            if record.get("status") not in {"open", "submitting"}:
                continue
            remaining = max(0.0, float(record.get("quantity", 0.0)) - float(record.get("filled", 0.0)))
            reserved += remaining * price
        return reserved

    def _merge_observed(self, record: dict[str, Any], observed: dict[str, Any]) -> None:
        record["order_id"] = observed.get("id") or record.get("order_id")
        status = str(observed.get("status") or record.get("status") or "open").lower()
        record["status"] = status
        filled = max(float(record.get("filled", 0.0)), float(observed.get("filled") or 0.0))
        record["filled"] = filled
        if observed.get("average") is not None:
            record["average"] = float(observed["average"])
        elif observed.get("price") is not None:
            record["average"] = float(observed["price"])
        if observed.get("cost") is not None:
            record["filled_cost"] = max(float(record.get("filled_cost", 0.0)), float(observed["cost"]))
        elif record.get("average") is not None:
            record["filled_cost"] = max(
                float(record.get("filled_cost", 0.0)),
                filled * float(record["average"]),
            )
        fee = observed.get("fee") or {}
        if fee.get("cost") is not None:
            record["fee"] = float(fee["cost"])
            record["fee_currency"] = fee.get("currency")
        elif observed.get("fees"):
            record["fee"] = sum(float(item.get("cost") or 0.0) for item in observed["fees"])
            currencies = {item.get("currency") for item in observed["fees"] if item.get("currency")}
            record["fee_currency"] = next(iter(currencies)) if len(currencies) == 1 else None
        self._apply_new_fill(record)

    def _apply_new_fill(self, record: dict[str, Any]) -> None:
        filled = float(record.get("filled", 0.0))
        applied = float(record.get("applied_filled", 0.0))
        delta = max(0.0, filled - applied)
        if delta <= 0:
            return
        symbol = record["symbol"]
        base_asset, quote_asset = symbol.split("/", 1)
        current = float(self.state["positions"].get(symbol, 0.0))
        current_cost = float(self.state["position_cost_usd"].get(symbol, 0.0))
        total_fill_cost = float(record.get("filled_cost", 0.0))
        applied_fill_cost = float(record.get("applied_fill_cost", 0.0))
        fill_cost_delta = max(0.0, total_fill_cost - applied_fill_cost)
        fill_price = (
            fill_cost_delta / delta
            if delta > 0 and fill_cost_delta > 0
            else float(record.get("average") or record.get("reference_price") or 0.0)
        )
        total_fee = float(record.get("fee", 0.0))
        applied_fee = float(record.get("applied_fee", 0.0))
        fee_delta = max(0.0, total_fee - applied_fee)
        fee_currency = str(record.get("fee_currency") or quote_asset).upper()
        effective_delta = delta
        if record["side"] == "buy":
            if fee_currency == base_asset:
                effective_delta = max(0.0, delta - fee_delta)
            current += effective_delta
            current_cost += fill_cost_delta or delta * fill_price
            if fee_currency == quote_asset:
                current_cost += fee_delta
        else:
            sold = min(delta, current)
            allocated_cost = current_cost * (sold / current) if current > 0 else 0.0
            gross_fill_value = fill_cost_delta or delta * fill_price
            proceeds = gross_fill_value * (sold / delta) if delta > 0 else 0.0
            if fee_currency == quote_asset:
                proceeds -= fee_delta
            realized = proceeds - allocated_cost
            self.state["realized_pnl_usd"] += realized
            self.state["position_cycle_pnl_usd"][symbol] = (
                float(self.state["position_cycle_pnl_usd"].get(symbol, 0.0)) + realized
            )
            current = max(0.0, current - sold)
            current_cost = max(0.0, current_cost - allocated_cost)
        if current <= 1e-12:
            self.state["positions"].pop(symbol, None)
            self.state["position_cost_usd"].pop(symbol, None)
            if record["side"] == "sell":
                cycle_pnl = float(self.state["position_cycle_pnl_usd"].pop(symbol, 0.0))
                self.state["closed_positions"] += 1
                if cycle_pnl > 0:
                    self.state["winning_positions"] += 1
        else:
            self.state["positions"][symbol] = current
            self.state["position_cost_usd"][symbol] = current_cost
        if not record.get("fill_counted"):
            self.state["filled_orders"] += 1
            record["fill_counted"] = True
        reference_price = float(record.get("reference_price") or fill_price)
        if reference_price > 0 and fill_price > 0:
            adverse_slippage = (
                (fill_price - reference_price) / reference_price * 10_000
                if record["side"] == "buy"
                else (reference_price - fill_price) / reference_price * 10_000
            )
            fill_notional = delta * fill_price
            self.state["execution_notional_usd"] += fill_notional
            self.state["weighted_slippage_bps_usd"] += adverse_slippage * fill_notional
        self.state["last_fill"] = dt.datetime.now(dt.UTC).isoformat()
        record["applied_filled"] = filled
        record["applied_fill_cost"] = total_fill_cost
        record["applied_fee"] = total_fee

    def _skip(self, client_id: str, symbol: str, side: str, reason: str) -> dict[str, Any]:
        record = {
            "client_order_id": client_id,
            "symbol": symbol,
            "side": side,
            "status": "skipped",
            "skip_reason": reason,
            "order_id": None,
            "filled": 0.0,
            "average": None,
            "fee": 0.0,
            "fee_currency": None,
        }
        self.state["orders"][client_id] = record
        self._save_state()
        return {"client_order_id": client_id, "idempotent": False, **self._public_record(record)}

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: record.get(key)
            for key in (
                "symbol",
                "side",
                "status",
                "skip_reason",
                "order_id",
                "filled",
                "average",
                "fee",
                "submitted_usd",
            )
            if key in record
        }

    @staticmethod
    def _client_order_id(event: dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                "timestamp": event.get("timestamp"),
                "symbol": event.get("symbol"),
                "side": event.get("side"),
                "quantity": event.get("quantity"),
                "reason": event.get("reason"),
                "event_id": event.get("event_id"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"lt-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"

    def _verify_testnet_urls(self) -> None:
        self.endpoint_verified = False

        exchange_id = str(
            getattr(self.exchange, "id", "")
        ).strip().lower()

        if exchange_id != "bybit":
            raise TestnetSafetyError(
                f"unsupported testnet exchange adapter: {exchange_id}"
            )

        urls = getattr(
            self.exchange,
            "urls",
            {},
        ).get("api", {})

        flattened = self._flatten_urls(urls)

        if not flattened:
            raise TestnetSafetyError(
                "exchange endpoints are not exclusively Bybit testnet URLs"
            )

        # Modern CCXT represents Bybit endpoints as:
        # https://api-testnet.{hostname}
        #
        # Resolve that template only against CCXT's known Bybit domains.
        allowed_domains = {
            "bybit.com",
            "bytick.com",
            "bybit.nl",
            "bybit.com.hk",
        }

        configured_hostname = str(
            getattr(self.exchange, "hostname", "")
            or ""
        ).strip().lower().rstrip(".")

        template_present = any(
            "{hostname}" in str(url)
            for url in flattened
        )

        if template_present:
            if configured_hostname not in allowed_domains:
                raise TestnetSafetyError(
                    "unexpected Bybit sandbox hostname"
                )

        allowed_testnet_hosts = {
            f"api-testnet.{domain}"
            for domain in allowed_domains
        }

        for raw_url in flattened:
            url = str(raw_url)

            if "{hostname}" in url:
                url = url.replace(
                    "{hostname}",
                    configured_hostname,
                )

            parsed = urlparse(url)
            host = str(
                parsed.hostname or ""
            ).lower().rstrip(".")

            if (
                parsed.scheme.lower() != "https"
                or "testnet" not in host
            ):
                raise TestnetSafetyError(
                    "exchange endpoints are not exclusively "
                    "Bybit testnet URLs"
                )

            if host not in allowed_testnet_hosts:
                raise TestnetSafetyError(
                    "unexpected non-Bybit sandbox endpoint"
                )

        self.endpoint_verified = True

    def _attest_exchange_capabilities(self) -> None:
        exchange_id = str(getattr(self.exchange, "id", "bybit")).lower()
        if exchange_id != "bybit":
            raise TestnetSafetyError(f"unsupported testnet exchange adapter: {exchange_id}")
        advertised = getattr(self.exchange, "has", {}) or {}
        methods = {
            name: bool(advertised.get(name, False))
            for name in (
                "fetchBalance",
                "createOrder",
                "createMarketOrder",
                "fetchOrder",
                "fetchOpenOrders",
                "fetchClosedOrders",
                "cancelOrder",
                "fetchMyTrades",
            )
        }
        market_types = {
            market_type: sum(bool(market.get(market_type)) for market in self.exchange.markets.values())
            for market_type in ("spot", "swap", "future", "option")
        }
        quotes = sorted(
            {
                str(market.get("quote")).upper()
                for market in self.exchange.markets.values()
                if market.get("spot") and market.get("quote")
            }
        )
        self.exchange_capabilities = {
            "exchange_id": exchange_id,
            "environment": "testnet",
            "execution_market_type": "spot",
            "market_types_observed": market_types,
            "spot_quote_assets": quotes,
            "methods": methods,
        }
        required = ("fetchBalance", "createOrder")
        missing = [name for name in required if not methods[name]]
        recoverable = methods["fetchOrder"] or (
            methods["fetchOpenOrders"] and methods["fetchClosedOrders"]
        )
        if missing or not recoverable:
            detail = missing + ([] if recoverable else ["order_recovery"])
            raise TestnetSafetyError(f"Bybit Testnet lacks required execution capabilities: {', '.join(detail)}")

    def _attest_api_key(self) -> None:
        query = None
        for method_name in ("private_get_v5_user_query_api", "privateGetV5UserQueryApi"):
            candidate = getattr(self.exchange, method_name, None)
            if callable(candidate):
                query = candidate
                break
        if query is None:
            raise TestnetSafetyError("CCXT cannot inspect Bybit API-key permissions")
        payload = query()
        result = payload.get("result", payload) if isinstance(payload, dict) else {}
        permissions = result.get("permissions") or {}
        spot_permissions = {str(value) for value in permissions.get("Spot", [])}
        wallet_permissions = {str(value) for value in permissions.get("Wallet", [])}
        read_only = int(result.get("readOnly", 1))
        if read_only != 0:
            raise TestnetSafetyError("Bybit Testnet API key is read-only; SpotTrade permission is required")
        if "SpotTrade" not in spot_permissions:
            raise TestnetSafetyError("Bybit Testnet API key does not grant SpotTrade permission")
        if "Withdraw" in wallet_permissions:
            raise TestnetSafetyError("API keys with withdrawal permission are rejected")
        ips = [str(value) for value in result.get("ips", []) if str(value).strip()]
        self.api_attestation = {
            "verified": True,
            "provider": "bybit",
            "environment": "testnet",
            "read_write": True,
            "spot_trade": True,
            "withdrawal_permission": False,
            "ip_bound": bool(ips),
            "bound_ip_count": len(ips),
            "key_type": int(result.get("type", 0) or 0),
            "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        }

    @classmethod
    def _flatten_urls(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            output: list[str] = []
            for nested in value.values():
                output.extend(cls._flatten_urls(nested))
            return output
        if isinstance(value, (list, tuple)):
            output = []
            for nested in value:
                output.extend(cls._flatten_urls(nested))
            return output
        return []

    def _load_state(self) -> dict[str, Any]:
        today = dt.datetime.now(dt.UTC).date().isoformat()
        default = {
            "version": 1,
            "day": today,
            "daily_submitted_usd": 0.0,
            "daily_order_count": 0,
            "orders": {},
            "positions": {},
            "last_reconciliation": None,
            "last_reconciliation_errors": [],
            "position_cost_usd": {},
            "position_cycle_pnl_usd": {},
            "realized_pnl_usd": 0.0,
            "closed_positions": 0,
            "winning_positions": 0,
            "filled_orders": 0,
            "execution_notional_usd": 0.0,
            "weighted_slippage_bps_usd": 0.0,
            "last_fill": None,
            "account_balance": {},
        }
        if not self.state_path.exists():
            return default
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if loaded.get("version") != 1:
                raise ValueError("unsupported state version")
            return {**default, **loaded}
        except Exception as exc:
            raise TestnetSafetyError(f"testnet execution state is invalid: {exc}") from exc

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _refresh_day(self) -> None:
        today = dt.datetime.now(dt.UTC).date().isoformat()
        if self.state.get("day") != today:
            self.state["day"] = today
            self.state["daily_submitted_usd"] = 0.0
            self.state["daily_order_count"] = 0

    def _update_balance_snapshot(self, balance: dict[str, Any]) -> None:
        totals = balance.get("total") or {}
        watched_assets = {"USDT"}
        for symbol in self.state.get("positions", {}):
            watched_assets.update(symbol.split("/", 1))
        assets: dict[str, float] = {}
        for asset in sorted(watched_assets):
            value = totals.get(asset)
            if value is None and isinstance(balance.get(asset), dict):
                value = balance[asset].get("total")
            if value is not None:
                assets[asset] = float(value)
        self.state["account_balance"] = {
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "assets": assets,
        }

    def _require_started(self) -> None:
        if self.exchange is None or not self.endpoint_verified:
            raise TestnetSafetyError("Bybit testnet engine is not safely started")

    @staticmethod
    def _read_secret(path: Path, label: str) -> str:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TestnetSafetyError(f"{label} file is unavailable") from exc
        if len(value) < 8:
            raise TestnetSafetyError(f"{label} file is empty or invalid")
        return value

    def _redact(self, message: str) -> str:
        for path in (self.api_key_path, self.api_secret_path):
            try:
                secret = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message
