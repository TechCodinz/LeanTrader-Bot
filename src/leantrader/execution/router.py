from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

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
    requested: Optional[
        str
    ] = None,
) -> str:
    payload = payload or {}

    if requested:
        candidate = str(
            requested
        ).lower()

        if candidate in (
            EXECUTION_MODES
        ):
            return (
                _normalize_execution_mode(
                    candidate
                )
            )

    candidate = payload.get(
        "execution_mode"
    )

    if candidate:
        return _normalize_execution_mode(
            candidate
        )

    configured = os.getenv(
        "EXECUTION_MODE",
        "",
    ).strip()

    if configured:
        return _normalize_execution_mode(
            configured
        )

    # Backwards compatibility only.
    if (
        os.getenv(
            "CCXT_TESTNET",
            "",
        ).lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
        or os.getenv(
            "BYBIT_TESTNET",
            "",
        ).lower()
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

    # Existing historical calls like
    # route_order(payload, "ccxt")
    # remain valid. "ccxt" now means
    # backend, not Testnet.
    if requested:
        candidate = str(
            requested
        ).lower()

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


def resolve_execution_context(
    payload: Optional[
        Dict[str, Any]
    ] = None,
    mode: Optional[str] = None,
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
        }

    if backend == "ccxt":
        broker = BrokerCCXT(
            execution_mode=(
                requested_mode
            )
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
            fx_hint = (
                os.getenv(
                    "FX_ENVIRONMENT"
                )
                or os.getenv(
                    "OANDA_ENV"
                )
                or ""
            ).strip().lower()

            if fx_hint in {
                "practice",
                "demo",
                "testnet",
                "sandbox",
            }:
                resolved = "testnet"
            elif fx_hint in {
                "live",
                "real",
                "production",
            }:
                resolved = "live"
            else:
                resolved = "paper"
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
    }


def route_order(
    payload: Dict[str, Any],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Universal historical LeanTrader order route.

    Engines submit one order intent.
    This router decides paper/testnet/live.
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
        context["backend"]
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
    ).lower()

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

    price = float(
        payload.get(
            "price",
            0.0,
        )
        or payload.get(
            "reference_price",
            0.0,
        )
        or 0.0
    )

    if execution_mode == "paper":
        result = (
            BrokerEmulator()
            .market(
                symbol,
                side,
                qty,
                price,
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
            "exchange": (
                context.get(
                    "exchange"
                )
            ),
            "order": result,
        }

    if (
        backend == "ccxt"
        and execution_mode
        in {
            "testnet",
            "live",
            "auto",
            "invalid",
        }
    ):
        broker = BrokerCCXT(
            execution_mode=(
                execution_mode
            )
        )

        return broker.market(
            symbol,
            side,
            qty,
            price,
        )

    if backend == "fx":
        # BrokerFX remains the historical
        # OANDA/MT5 adapter. Global mode
        # is passed through environment.
        previous = os.environ.get(
            "EXECUTION_MODE"
        )

        os.environ[
            "EXECUTION_MODE"
        ] = execution_mode

        try:
            result = (
                BrokerFX()
                .market(
                    symbol,
                    side,
                    qty,
                    price,
                )
            )
        finally:
            if previous is None:
                os.environ.pop(
                    "EXECUTION_MODE",
                    None,
                )
            else:
                os.environ[
                    "EXECUTION_MODE"
                ] = previous

        result = (
            result
            if isinstance(
                result,
                dict,
            )
            else {
                "ok": False,
                "error": (
                    "invalid_fx_result"
                ),
            }
        )

        result.setdefault(
            "execution_mode",
            execution_mode,
        )

        result.setdefault(
            "authority",
            execution_mode,
        )

        return result

    return {
        "ok": False,
        "executed": False,
        "simulated": False,
        "authority": "none",
        "execution_mode": (
            execution_mode
        ),
        "backend": backend,
        "error": (
            "execution_route_"
            "unavailable"
        ),
    }


def route_balance(
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    context = (
        resolve_execution_context(
            {},
            mode,
        )
    )

    execution_mode = (
        context[
            "execution_mode"
        ]
    )

    backend = (
        context["backend"]
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

    if backend == "ccxt":
        broker = BrokerCCXT(
            execution_mode=(
                execution_mode
            )
        )

        return (
            broker.fetch_balance()
        )

    return {}


def route_ticker(
    symbol: str,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    context = (
        resolve_execution_context(
            {},
            mode,
        )
    )

    # Market observation remains real
    # even when order execution is paper.
    if context[
        "backend"
    ] in {
        "ccxt",
        "emu",
    }:
        broker = BrokerCCXT(
            execution_mode=(
                context[
                    "execution_mode"
                ]
            )
        )

        return broker.fetch_ticker(
            symbol
        )

    return {}


def route_ohlcv(
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100,
    mode: Optional[str] = None,
):
    context = (
        resolve_execution_context(
            {},
            mode,
        )
    )

    if context[
        "backend"
    ] in {
        "ccxt",
        "emu",
    }:
        broker = BrokerCCXT(
            execution_mode=(
                context[
                    "execution_mode"
                ]
            )
        )

        return broker.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

    return []


def execution_status(
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    return resolve_execution_context(
        {},
        mode,
    )
