from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .broker_ccxt import (
    BrokerCCXT,
)
from .broker_emulator import (
    BrokerEmulator,
)
from .broker_fx import (
    BrokerFX,
)


EXECUTION_MODES = {
    "auto",
    "paper",
    "testnet",
    "live",
}

BACKENDS = {
    "ccxt",
    "fx",
    "emu",
}


def _normalize_execution_mode(
    value: Optional[str],
) -> str:
    value = str(
        value or ""
    ).strip().lower()

    aliases = {
        "sandbox": "testnet",
        "demo": "testnet",
        "practice": "testnet",
        "prod": "live",
        "production": "live",
        "real": "live",
        "sim": "paper",
        "simulation": "paper",
        "emu": "paper",
    }

    value = aliases.get(
        value,
        value,
    )

    if value in EXECUTION_MODES:
        return value

    return "auto"


def _requested_execution_mode(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    requested: Optional[str] = None,
) -> str:
    payload = payload or {}

    if requested:
        candidate = str(
            requested
        ).strip().lower()

        if candidate in EXECUTION_MODES:
            return (
                _normalize_execution_mode(
                    candidate
                )
            )

    candidate = payload.get(
        "execution_mode"
    )

    if candidate:
        return (
            _normalize_execution_mode(
                candidate
            )
        )

    configured = os.getenv(
        "EXECUTION_MODE",
        "",
    ).strip()

    if configured:
        return (
            _normalize_execution_mode(
                configured
            )
        )

    # Legacy compatibility only.
    if (
        os.getenv(
            "CCXT_TESTNET",
            "",
        ).strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
        or os.getenv(
            "BYBIT_TESTNET",
            "",
        ).strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    ):
        return "testnet"

    return "auto"


def _requested_backend(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    requested: Optional[str] = None,
) -> str:
    payload = payload or {}

    if requested:
        candidate = str(
            requested
        ).strip().lower()

        if candidate in BACKENDS:
            return candidate

    candidate = str(
        payload.get(
            "backend",
            "",
        )
        or payload.get(
            "broker",
            "",
        )
    ).strip().lower()

    if candidate in BACKENDS:
        return candidate

    configured = os.getenv(
        "BROKER_BACKEND",
        "ccxt",
    ).strip().lower()

    if configured in BACKENDS:
        return configured

    return "ccxt"


def _requested_exchange(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    exchange_id: Optional[
        str
    ] = None,
) -> str:
    payload = payload or {}

    value = (
        exchange_id
        or payload.get(
            "exchange_id"
        )
        or payload.get(
            "exchange"
        )
        or payload.get(
            "venue"
        )
        or os.getenv(
            "CCXT_EXCHANGE"
        )
        or os.getenv(
            "EXCHANGE_ID"
        )
        or "bybit"
    )

    return str(
        value
    ).strip().lower()


def resolve_execution_context(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    payload = payload or {}

    requested_mode = (
        _requested_execution_mode(
            payload,
            mode,
        )
    )

    backend = (
        _requested_backend(
            payload,
            mode,
        )
    )

    exchange = (
        _requested_exchange(
            payload,
            exchange_id,
        )
    )

    if backend == "emu":
        return {
            "requested_mode": (
                requested_mode
            ),
            "execution_mode": (
                "paper"
            ),
            "authority": "paper",
            "backend": "emu",
            "exchange": exchange,
        }

    if backend == "ccxt":
        broker = BrokerCCXT(
            execution_mode=(
                requested_mode
            ),
            exchange_id=exchange,
        )

        resolved = (
            broker.resolve_mode()
        )

        return {
            "requested_mode": (
                requested_mode
            ),
            "execution_mode": (
                resolved
            ),
            "authority": (
                broker.authority
            ),
            "backend": "ccxt",
            "exchange": (
                broker.exchange_id
            ),
            "market_mode": (
                broker.market_mode
            ),
            "authenticated": (
                broker.has_credentials
            ),
        }

    if backend == "fx":
        if requested_mode == "auto":
            hint = (
                os.getenv(
                    "FX_ENVIRONMENT"
                )
                or os.getenv(
                    "OANDA_ENV"
                )
                or ""
            )

            hint = (
                _normalize_execution_mode(
                    hint
                )
            )

            resolved = (
                hint
                if hint
                in {
                    "testnet",
                    "live",
                }
                else "paper"
            )
        else:
            resolved = (
                requested_mode
            )

        return {
            "requested_mode": (
                requested_mode
            ),
            "execution_mode": (
                resolved
            ),
            "authority": (
                resolved
                if resolved
                in {
                    "paper",
                    "testnet",
                    "live",
                }
                else "none"
            ),
            "backend": "fx",
            "exchange": exchange,
        }

    return {
        "requested_mode": (
            requested_mode
        ),
        "execution_mode": (
            "invalid"
        ),
        "authority": "none",
        "backend": backend,
        "exchange": exchange,
    }


def route_order(
    payload: Dict[str, Any],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One universal order intent.

    Strategy engines submit:
      symbol
      side
      quantity
      order type
      target exchange

    Runtime decides paper/testnet/live.
    """
    context = (
        resolve_execution_context(
            payload,
            mode,
        )
    )

    execution_mode = (
        context[
            "execution_mode"
        ]
    )

    backend = (
        context[
            "backend"
        ]
    )

    exchange = (
        context.get(
            "exchange"
        )
        or "bybit"
    )

    symbol = str(
        payload.get("symbol")
        or payload.get("pair")
        or ""
    )

    side = str(
        payload.get(
            "side",
            "buy",
        )
    ).strip().lower()

    qty = float(
        payload.get(
            "qty",
            0.0,
        )
        or payload.get(
            "quantity",
            0.0,
        )
        or payload.get(
            "amount",
            0.0,
        )
        or 0.0
    )

    price_value = (
        payload.get(
            "price"
        )
        if payload.get(
            "price"
        )
        is not None
        else payload.get(
            "reference_price"
        )
    )

    price = None

    if price_value is not None:
        try:
            price = float(
                price_value
            )
        except Exception:
            price = None

    order_type = str(
        payload.get(
            "order_type"
        )
        or payload.get(
            "type"
        )
        or (
            "market"
            if price is None
            else "limit"
        )
    ).strip().lower()

    params = dict(
        payload.get(
            "params"
        )
        or {}
    )

    if execution_mode == "paper":
        ref_price = float(
            price or 0.0
        )

        result = (
            BrokerEmulator()
            .market(
                symbol,
                side,
                qty,
                ref_price,
            )
        )

        return {
            "ok": True,
            "executed": True,
            "simulated": True,
            "authority": "paper",
            "execution_mode": (
                "paper"
            ),
            "backend": "emu",
            "exchange": exchange,
            "order_type": (
                order_type
            ),
            "order": result,
        }

    if backend == "ccxt":
        broker = BrokerCCXT(
            execution_mode=(
                execution_mode
            ),
            exchange_id=(
                exchange
            ),
        )

        return broker.order(
            symbol=symbol,
            order_type=(
                order_type
            ),
            side=side,
            qty=qty,
            price=price,
            params=params,
        )

    if backend == "fx":
        if order_type != "market":
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": (
                    execution_mode
                ),
                "execution_mode": (
                    execution_mode
                ),
                "backend": "fx",
                "error": (
                    "fx_adapter_currently_"
                    "supports_market_orders"
                ),
            }

        broker = BrokerFX(
            execution_mode=(
                execution_mode
            )
        )

        return broker.market(
            symbol,
            side,
            qty,
            float(
                price or 0.0
            ),
        )

    return {
        "ok": False,
        "executed": False,
        "simulated": False,
        "authority": "none",
        "execution_mode": (
            execution_mode
        ),
        "backend": backend,
        "exchange": exchange,
        "error": (
            "execution_route_"
            "unavailable"
        ),
    }


def route_balance(
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
        )
    )

    execution_mode = (
        context[
            "execution_mode"
        ]
    )

    if execution_mode == "paper":
        starting = float(
            os.getenv(
                "PAPER_STARTING_BALANCE",
                "1000.0",
            )
        )

        return {
            "free": {
                "USDT": starting,
            },
            "used": {
                "USDT": 0.0,
            },
            "total": {
                "USDT": starting,
            },
            "USDT": {
                "free": starting,
                "used": 0.0,
                "total": starting,
            },
            "authority": "paper",
            "execution_mode": (
                "paper"
            ),
        }

    if context[
        "backend"
    ] == "ccxt":
        return (
            BrokerCCXT(
                execution_mode=(
                    execution_mode
                ),
                exchange_id=(
                    context[
                        "exchange"
                    ]
                ),
            )
            .fetch_balance()
        )

    return {}


def route_ticker(
    symbol: str,
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
        )
    )

    return (
        BrokerCCXT(
            execution_mode=(
                context[
                    "execution_mode"
                ]
            ),
            exchange_id=(
                context[
                    "exchange"
                ]
            ),
        )
        .fetch_ticker(
            symbol
        )
    )


def route_ohlcv(
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100,
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
):
    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
        )
    )

    return (
        BrokerCCXT(
            execution_mode=(
                context[
                    "execution_mode"
                ]
            ),
            exchange_id=(
                context[
                    "exchange"
                ]
            ),
        )
        .fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )
    )


def route_order_book(
    symbol: str,
    limit: int = 20,
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
        )
    )

    return (
        BrokerCCXT(
            execution_mode=(
                context[
                    "execution_mode"
                ]
            ),
            exchange_id=(
                context[
                    "exchange"
                ]
            ),
        )
        .fetch_order_book(
            symbol,
            limit=limit,
        )
    )


def execution_status(
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
) -> Dict[str, Any]:
    return resolve_execution_context(
        {},
        mode,
        exchange_id,
    )
