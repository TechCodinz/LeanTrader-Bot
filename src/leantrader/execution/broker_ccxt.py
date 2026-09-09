from __future__ import annotations

import os
import time
from typing import Any, Dict


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(
        name,
        "true" if default else "false",
    )

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class BrokerCCXT:
    """
    Native LeanTrader CCXT execution backend.

    Supports:
      - authenticated exchange Testnet execution
      - public market reads
      - explicitly configured live execution
      - no fabricated fill when execution authority is absent
    """

    def __init__(self) -> None:
        self.exchange_id = (
            os.getenv("CCXT_EXCHANGE")
            or os.getenv("EXCHANGE_ID")
            or "bybit"
        ).lower()

        self.mode = (
            os.getenv("EXCHANGE_MODE")
            or "spot"
        ).lower()

        self.testnet = (
            _env_bool("CCXT_TESTNET", False)
            or _env_bool("BYBIT_TESTNET", False)
        )

        self.enable_live = _env_bool(
            "ENABLE_LIVE",
            False,
        )

        self.allow_live = _env_bool(
            "ALLOW_LIVE",
            False,
        )

        self.live_confirm = (
            os.getenv(
                "LIVE_CONFIRM",
                "",
            ).strip().upper()
            == "YES"
        )

        prefix = self.exchange_id.upper()

        self.api_key = (
            os.getenv("API_KEY")
            or os.getenv(f"{prefix}_API_KEY")
            or ""
        )

        self.api_secret = (
            os.getenv("API_SECRET")
            or os.getenv(f"{prefix}_API_SECRET")
            or os.getenv(f"{prefix}_SECRET_KEY")
            or ""
        )

        self.has_credentials = bool(
            self.api_key
            and self.api_secret
        )

        self.live = bool(
            not self.testnet
            and self.enable_live
            and self.allow_live
            and self.live_confirm
            and self.has_credentials
        )

        self.testnet_authority = bool(
            self.testnet
            and self.has_credentials
        )

        if self.testnet_authority:
            self.authority = "testnet"
        elif self.live:
            self.authority = "live"
        else:
            self.authority = "none"

        self._ex = None

    def _ensure_ex(self):
        if self._ex is not None:
            return self._ex

        try:
            import ccxt
        except Exception as exc:
            raise RuntimeError(
                f"ccxt import failed: {exc}"
            ) from exc

        klass = getattr(
            ccxt,
            self.exchange_id,
            None,
        )

        if klass is None:
            raise RuntimeError(
                f"Unsupported CCXT exchange: "
                f"{self.exchange_id}"
            )

        opts: Dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": int(
                os.getenv(
                    "CCXT_TIMEOUT_MS",
                    "15000",
                )
            ),
            "options": {},
        }

        if self.exchange_id == "bybit":
            opts["options"]["defaultType"] = (
                "swap"
                if self.mode == "linear"
                else "spot"
            )

            if self.mode == "linear":
                opts["options"][
                    "defaultSubType"
                ] = "linear"

        elif (
            self.exchange_id == "binance"
            and self.mode == "linear"
        ):
            opts["options"][
                "defaultType"
            ] = "future"

        if self.has_credentials:
            opts["apiKey"] = self.api_key
            opts["secret"] = self.api_secret

        exchange = klass(opts)

        if self.testnet:
            sandbox = getattr(
                exchange,
                "set_sandbox_mode",
                None,
            )

            if not callable(sandbox):
                raise RuntimeError(
                    f"{self.exchange_id} does not "
                    "expose CCXT sandbox mode; "
                    "refusing to fall through to "
                    "production endpoints"
                )

            sandbox(True)

        self._ex = exchange
        return exchange

    def fetch_ticker(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        exchange = self._ensure_ex()

        ticker = exchange.fetch_ticker(
            symbol
        )

        return ticker or {}

    def fetch_balance(
        self,
    ) -> Dict[str, Any]:
        if not self.has_credentials:
            return {}

        exchange = self._ensure_ex()

        balance = exchange.fetch_balance()

        return balance or {}

    def market(
        self,
        symbol: str,
        side: str,
        qty: float,
        ref_price: float = 0.0,
    ) -> Dict[str, Any]:

        symbol = str(symbol or "")
        side = str(side or "").lower()

        try:
            qty = float(qty or 0.0)
        except Exception:
            qty = 0.0

        if (
            not symbol
            or side not in {"buy", "sell"}
            or qty <= 0.0
        ):
            return {
                "ok": False,
                "error": "invalid_order_request",
                "symbol": symbol,
                "side": side,
                "qty": qty,
            }

        if self.authority == "none":
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "none",
                "exchange": self.exchange_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "reference_price": float(
                    ref_price or 0.0
                ),
                "error": (
                    "no_authenticated_execution_"
                    "authority"
                ),
            }

        exchange = self._ensure_ex()

        try:
            order = exchange.create_order(
                symbol,
                "market",
                side,
                qty,
                None,
                {},
            )

            order = order or {}

            order_id = order.get("id")

            if (
                order_id
                and hasattr(
                    exchange,
                    "fetch_order",
                )
            ):
                for _ in range(3):
                    try:
                        time.sleep(0.25)

                        refreshed = (
                            exchange.fetch_order(
                                order_id,
                                symbol,
                            )
                        )

                        if refreshed:
                            order = refreshed

                        status = str(
                            order.get(
                                "status",
                                "",
                            )
                        ).lower()

                        if status in {
                            "closed",
                            "filled",
                        }:
                            break

                    except Exception:
                        break

            return {
                "ok": True,
                "executed": True,
                "simulated": False,
                "authority": self.authority,
                "exchange": self.exchange_id,
                "order": order,
            }

        except Exception as exc:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": self.authority,
                "exchange": self.exchange_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "error": str(exc),
            }
