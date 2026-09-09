from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .broker_ccxt import BrokerCCXT
from .broker_emulator import BrokerEmulator
from .broker_fx import BrokerFX


def _env_bool(
    name: str,
    default: bool = False,
) -> bool:
    return os.getenv(
        name,
        "true" if default else "false",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _mode(
    requested: Optional[str] = None,
) -> str:
    if requested:
        return requested.lower()

    configured = os.getenv(
        "BROKER_MODE",
        "",
    ).strip().lower()

    if configured:
        return configured

    if (
        _env_bool("CCXT_TESTNET", False)
        or _env_bool(
            "BYBIT_TESTNET",
            False,
        )
    ):
        return "ccxt"

    return "emu"


def route_order(
    payload: Dict[str, Any],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Route one order through LeanTrader's
    historical execution package.
    """

    selected = _mode(mode)

    symbol = (
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
        payload.get("qty", 0.0)
        or payload.get(
            "quantity",
            0.0,
        )
        or 0.0
    )

    price = float(
        payload.get(
            "price",
            0.0,
        )
        or 0.0
    )

    if selected == "emu":
        return BrokerEmulator().market(
            symbol,
            side,
            qty,
            price,
        )

    if selected == "fx":
        return BrokerFX().market(
            symbol,
            side,
            qty,
            price,
        )

    if selected == "ccxt":
        return BrokerCCXT().market(
            symbol,
            side,
            qty,
            price,
        )

    return {
        "ok": False,
        "error": (
            f"unknown_broker_mode:"
            f"{selected}"
        ),
    }


def route_balance(
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    selected = _mode(mode)

    if selected != "ccxt":
        return {}

    return BrokerCCXT().fetch_balance()


def route_ticker(
    symbol: str,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    selected = _mode(mode)

    if selected != "ccxt":
        return {}

    return BrokerCCXT().fetch_ticker(
        symbol
    )
