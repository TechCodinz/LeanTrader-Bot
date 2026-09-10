from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .broker_ccxt import BrokerCCXT
from .broker_emulator import BrokerEmulator
from .broker_fx import BrokerFX


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
    requested: Optional[
        str
    ] = None,
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
    requested: Optional[
        str
    ] = None,
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


def _client_exchange_id(
    exchange_client: Any,
) -> str:
    if exchange_client is None:
        return ""

    for attr in (
        "id",
        "exchange_id",
    ):
        value = getattr(
            exchange_client,
            attr,
            None,
        )

        if isinstance(
            value,
            str,
        ) and value.strip():
            return (
                value.strip()
                .lower()
            )

    return ""


def _requested_exchange(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    exchange_id: Optional[
        str
    ] = None,
    exchange_client: Any = None,
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
        or _client_exchange_id(
            exchange_client
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


def _auth_profile(
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
) -> Dict[str, Any]:
    """
    Build a private execution profile.

    Secrets remain local to this call and are
    never returned by route/status functions.
    """
    source = (
        dict(auth_profile)
        if isinstance(
            auth_profile,
            dict,
        )
        else {}
    )

    output: Dict[
        str,
        Any,
    ] = {}

    aliases = {
        "api_key": (
            "api_key",
            "apiKey",
            "key",
        ),
        "secret": (
            "secret",
            "api_secret",
            "secret_key",
        ),
        "password": (
            "password",
            "api_password",
            "passphrase",
        ),
        "uid": (
            "uid",
            "account_id",
        ),
        "market_mode": (
            "market_mode",
            "exchange_mode",
        ),
    }

    for canonical, names in (
        aliases.items()
    ):
        for name in names:
            value = source.get(
                name
            )

            if value is not None:
                output[
                    canonical
                ] = value
                break

    if exchange_client is not None:
        client_aliases = {
            "api_key": (
                "apiKey",
                "api_key",
            ),
            "secret": (
                "secret",
                "api_secret",
            ),
            "password": (
                "password",
                "api_password",
            ),
            "uid": (
                "uid",
            ),
        }

        for canonical, names in (
            client_aliases.items()
        ):
            if output.get(
                canonical
            ):
                continue

            for name in names:
                value = getattr(
                    exchange_client,
                    name,
                    None,
                )

                if value:
                    output[
                        canonical
                    ] = value
                    break

    return output


def resolve_execution_context(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    mode: Optional[str] = None,
    exchange_id: Optional[
        str
    ] = None,
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
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
            exchange_client,
        )
    )

    profile = _auth_profile(
        auth_profile,
        exchange_client,
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
            credentials=profile,
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
                if hint in {
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
                if resolved in {
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
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
) -> Dict[str, Any]:
    profile = _auth_profile(
        auth_profile,
        exchange_client,
    )

    context = (
        resolve_execution_context(
            payload,
            mode,
            auth_profile=profile,
            exchange_client=(
                exchange_client
            ),
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

    try:
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
    except Exception:
        qty = 0.0

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

    if (
        not symbol
        or side not in {
            "buy",
            "sell",
        }
        or qty <= 0.0
    ):
        return {
            "ok": False,
            "executed": False,
            "simulated": False,
            "authority": (
                context.get(
                    "authority",
                    "none",
                )
            ),
            "execution_mode": (
                execution_mode
            ),
            "exchange": exchange,
            "error": (
                "invalid_order_request"
            ),
        }

    if execution_mode == "paper":
        emulator = (
            BrokerEmulator()
        )

        if order_type == "market":
            ref_price = float(
                price or 0.0
            )

            if ref_price <= 0.0:
                try:
                    ticker = (
                        BrokerCCXT(
                            execution_mode=(
                                "paper"
                            ),
                            exchange_id=(
                                exchange
                            ),
                        )
                        .fetch_ticker(
                            symbol
                        )
                        or {}
                    )

                    ref_price = float(
                        ticker.get(
                            "last"
                        )
                        or ticker.get(
                            "close"
                        )
                        or 0.0
                    )

                except Exception:
                    ref_price = 0.0

            if ref_price <= 0.0:
                return {
                    "ok": False,
                    "executed": False,
                    "simulated": True,
                    "authority": (
                        "paper"
                    ),
                    "execution_mode": (
                        "paper"
                    ),
                    "exchange": (
                        exchange
                    ),
                    "error": (
                        "paper_reference_"
                        "price_unavailable"
                    ),
                }

            result = (
                emulator.market(
                    symbol,
                    side,
                    qty,
                    ref_price,
                )
            )

            ok = (
                result.get(
                    "status"
                )
                == "filled"
            )

        else:
            result = (
                emulator
                .submit_pending(
                    symbol,
                    side,
                    qty,
                    order_type,
                    price,
                    params,
                )
            )

            ok = True

        return {
            "ok": ok,
            "executed": bool(
                ok
                and order_type == "market"
            ),
            "submitted": bool(ok),
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
            credentials=profile,
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
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
) -> Dict[str, Any]:
    profile = _auth_profile(
        auth_profile,
        exchange_client,
    )

    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
            auth_profile=profile,
            exchange_client=(
                exchange_client
            ),
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

    if (
        context[
            "backend"
        ]
        == "ccxt"
    ):
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
                credentials=profile,
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
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
) -> Dict[str, Any]:
    profile = _auth_profile(
        auth_profile,
        exchange_client,
    )

    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
            auth_profile=profile,
            exchange_client=(
                exchange_client
            ),
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
            credentials=profile,
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
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
):
    profile = _auth_profile(
        auth_profile,
        exchange_client,
    )

    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
            auth_profile=profile,
            exchange_client=(
                exchange_client
            ),
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
            credentials=profile,
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
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
) -> Dict[str, Any]:
    profile = _auth_profile(
        auth_profile,
        exchange_client,
    )

    context = (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
            auth_profile=profile,
            exchange_client=(
                exchange_client
            ),
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
            credentials=profile,
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
    *,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
    exchange_client: Any = None,
) -> Dict[str, Any]:
    return (
        resolve_execution_context(
            {},
            mode,
            exchange_id,
            auth_profile=(
                auth_profile
            ),
            exchange_client=(
                exchange_client
            ),
        )
    )


def route_legacy_order(
    *,
    exchange_client: Any = None,
    symbol: str,
    order_type: str,
    side: str,
    amount: float,
    price: Optional[float] = None,
    params: Optional[
        Dict[str, Any]
    ] = None,
    exchange_id: Optional[
        str
    ] = None,
    execution_mode: Optional[
        str
    ] = None,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Compatibility entrypoint for historical
    engines.

    Existing engine code can retain its strategy
    and exchange/account object while actual order
    authority remains inside route_order().

    Credentials extracted from exchange_client or
    auth_profile remain in memory and are never
    added to the returned receipt.
    """
    target_exchange = (
        exchange_id
        or _client_exchange_id(
            exchange_client
        )
        or "bybit"
    )

    result = route_order(
        {
            "symbol": symbol,
            "order_type": (
                order_type
                or "market"
            ),
            "side": side,
            "qty": amount,
            "price": price,
            "params": dict(
                params or {}
            ),
            "exchange_id": (
                target_exchange
            ),
            "backend": "ccxt",
        },
        execution_mode,
        auth_profile=auth_profile,
        exchange_client=exchange_client,
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "invalid_execution_receipt"
        )

    if not result.get(
        "ok"
    ):
        raise RuntimeError(
            str(
                result.get(
                    "error",
                    "order_rejected",
                )
            )
        )

    order = (
        result.get("order")
        or result
    )

    if not isinstance(
        order,
        dict,
    ):
        raise RuntimeError(
            "invalid_order_payload"
        )

    receipt = dict(
        order
    )

    # Safe execution metadata only.
    for field in (
        "authority",
        "execution_mode",
        "exchange",
        "simulated",
        "submitted",
        "executed",
        "order_type",
    ):
        if (
            field in result
            and field
            not in receipt
        ):
            receipt[field] = (
                result[field]
            )

    return receipt


async def route_legacy_order_async(
    *,
    exchange_client: Any = None,
    symbol: str,
    order_type: str,
    side: str,
    amount: float,
    price: Optional[float] = None,
    params: Optional[
        Dict[str, Any]
    ] = None,
    exchange_id: Optional[
        str
    ] = None,
    execution_mode: Optional[
        str
    ] = None,
    auth_profile: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Non-blocking compatibility facade for
    historical async engines.
    """
    import asyncio

    return await asyncio.to_thread(
        route_legacy_order,
        exchange_client=exchange_client,
        symbol=symbol,
        order_type=order_type,
        side=side,
        amount=amount,
        price=price,
        params=params,
        exchange_id=exchange_id,
        execution_mode=execution_mode,
        auth_profile=auth_profile,
    )
