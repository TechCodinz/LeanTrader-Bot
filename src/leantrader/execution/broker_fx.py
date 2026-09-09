from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _normalize_mode(
    value: Optional[str],
) -> str:
    value = str(
        value or ""
    ).strip().lower()

    aliases = {
        "sandbox": "testnet",
        "practice": "testnet",
        "demo": "testnet",
        "production": "live",
        "prod": "live",
        "real": "live",
        "sim": "paper",
        "simulation": "paper",
    }

    value = aliases.get(
        value,
        value,
    )

    if value in {
        "auto",
        "paper",
        "testnet",
        "live",
    }:
        return value

    return "auto"


class BrokerFX:
    """
    Historical OANDA/MT5 adapter using
    LeanTrader's global execution mode.
    """

    def __init__(
        self,
        execution_mode: Optional[
            str
        ] = None,
    ) -> None:
        self.backend = (
            os.getenv(
                "FX_BACKEND",
                "",
            )
            or ""
        ).strip().lower()

        self.requested_mode = (
            _normalize_mode(
                execution_mode
                or os.getenv(
                    "EXECUTION_MODE",
                    "auto",
                )
            )
        )

        self.mode = (
            self._resolve_mode()
        )

    def _resolve_mode(
        self,
    ) -> str:
        if self.requested_mode != "auto":
            return (
                self.requested_mode
            )

        hint = (
            os.getenv(
                "FX_ENVIRONMENT"
            )
            or os.getenv(
                "OANDA_ENV"
            )
            or os.getenv(
                "MT5_ENVIRONMENT"
            )
            or ""
        )

        hint = _normalize_mode(
            hint
        )

        if hint in {
            "testnet",
            "live",
        }:
            return hint

        # No selected authenticated
        # environment: paper.
        return "paper"

    def market(
        self,
        symbol: str,
        side: str,
        qty: float,
        ref_price: float,
    ) -> Dict[str, Any]:
        if self.mode == "paper":
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "paper",
                "execution_mode": (
                    "paper"
                ),
                "backend": (
                    self.backend
                    or "none"
                ),
                "error": (
                    "paper_orders_are_"
                    "handled_by_router"
                ),
            }

        if self.backend == "oanda":
            return self._oanda_market(
                symbol,
                side,
                qty,
            )

        if self.backend == "mt5":
            return self._mt5_market(
                symbol,
                side,
                qty,
            )

        return {
            "ok": False,
            "executed": False,
            "simulated": False,
            "authority": "none",
            "execution_mode": (
                self.mode
            ),
            "error": (
                "FX_BACKEND not "
                "configured"
            ),
        }

    def _oanda_market(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Dict[str, Any]:
        try:
            import oandapyV20.endpoints.orders as orders
            from oandapyV20 import API
        except Exception:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "none",
                "execution_mode": (
                    self.mode
                ),
                "error": (
                    "oandapyV20 "
                    "not installed"
                ),
            }

        account = os.getenv(
            "OANDA_ACCOUNT",
            "",
        )

        token = os.getenv(
            "OANDA_TOKEN",
            "",
        )

        if not (
            account
            and token
        ):
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "none",
                "execution_mode": (
                    self.mode
                ),
                "error": (
                    "OANDA credentials "
                    "missing"
                ),
            }

        environment = (
            "practice"
            if self.mode
            == "testnet"
            else "live"
        )

        try:
            api = API(
                access_token=token,
                environment=(
                    environment
                ),
            )

            data = {
                "order": {
                    "instrument": (
                        symbol.replace(
                            "/",
                            "_",
                        )
                    ),
                    "units": str(
                        int(
                            qty
                            if side
                            == "buy"
                            else -qty
                        )
                    ),
                    "type": "MARKET",
                    "positionFill": (
                        "DEFAULT"
                    ),
                }
            }

            request = (
                orders.OrderCreate(
                    account,
                    data=data,
                )
            )

            response = (
                api.request(
                    request
                )
            )

            return {
                "ok": True,
                "executed": True,
                "simulated": False,
                "authority": (
                    self.mode
                ),
                "execution_mode": (
                    self.mode
                ),
                "backend": "oanda",
                "order": response,
            }

        except Exception as exc:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": (
                    self.mode
                ),
                "execution_mode": (
                    self.mode
                ),
                "backend": "oanda",
                "error": str(exc),
            }

    def _mt5_market(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Dict[str, Any]:
        try:
            import MetaTrader5 as mt5
        except Exception:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "none",
                "execution_mode": (
                    self.mode
                ),
                "error": (
                    "MetaTrader5 "
                    "not installed"
                ),
            }

        path = os.getenv(
            "MT5_PATH"
        ) or os.getenv(
            "MTS_PATH"
        )

        try:
            kwargs = {}

            if path:
                kwargs["path"] = path

            if not mt5.initialize(
                **kwargs
            ):
                return {
                    "ok": False,
                    "executed": False,
                    "simulated": False,
                    "authority": "none",
                    "execution_mode": (
                        self.mode
                    ),
                    "error": (
                        "mt5.initialize "
                        "failed"
                    ),
                }

            order_type = (
                mt5.ORDER_TYPE_BUY
                if side == "buy"
                else mt5.ORDER_TYPE_SELL
            )

            request = {
                "action": (
                    mt5.TRADE_ACTION_DEAL
                ),
                "symbol": symbol,
                "volume": float(qty),
                "type": order_type,
                "deviation": 20,
                "magic": 123456,
                "comment": "leantrader",
            }

            result = (
                mt5.order_send(
                    request
                )
            )

            return {
                "ok": result
                is not None,
                "executed": result
                is not None,
                "simulated": False,
                "authority": (
                    self.mode
                ),
                "execution_mode": (
                    self.mode
                ),
                "backend": "mt5",
                "order": {
                    "retcode": (
                        getattr(
                            result,
                            "retcode",
                            None,
                        )
                    )
                },
            }

        except Exception as exc:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": (
                    self.mode
                ),
                "execution_mode": (
                    self.mode
                ),
                "backend": "mt5",
                "error": str(exc),
            }
