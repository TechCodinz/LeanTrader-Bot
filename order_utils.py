"""
Historical order helper compatibility facade.

Old engines may still pass an ExchangeRouter or raw CCXT object.
All NEW order authority is delegated to LeanTrader's universal
execution router.
"""

from typing import Any, Dict, Optional


def _exchange_id(
    ex: Any,
) -> str:
    for attr in (
        "exchange_id",
        "id",
    ):
        value = getattr(
            ex,
            attr,
            None,
        )

        if isinstance(
            value,
            str,
        ) and value:
            return value.lower()

    inner = getattr(
        ex,
        "ex",
        None,
    )

    value = getattr(
        inner,
        "id",
        None,
    )

    if isinstance(
        value,
        str,
    ) and value:
        return value.lower()

    return "bybit"


def _is_paper(
    ex: Any,
) -> bool:
    name = (
        ex.__class__.__name__
        .lower()
    )

    return (
        "paper" in name
        or _exchange_id(
            ex
        )
        == "paper"
    )


def _flatten(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(
        result,
        dict,
    ):
        return {
            "ok": False,
            "error": (
                "invalid_router_result"
            ),
        }

    order = result.get(
        "order"
    )

    if (
        result.get("ok")
        and isinstance(
            order,
            dict,
        )
    ):
        out = dict(
            order
        )

        out.setdefault(
            "ok",
            True,
        )

        out.setdefault(
            "executed",
            result.get(
                "executed"
            ),
        )

        out.setdefault(
            "simulated",
            result.get(
                "simulated"
            ),
        )

        out.setdefault(
            "authority",
            result.get(
                "authority"
            ),
        )

        out.setdefault(
            "execution_mode",
            result.get(
                "execution_mode"
            ),
        )

        out.setdefault(
            "exchange",
            result.get(
                "exchange"
            ),
        )

        return out

    return result


def safe_create_order(
    ex: Any,
    typ: str,
    symbol: str,
    side: str,
    amount: float,
    price: Optional[
        float
    ] = None,
    params: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    from src.leantrader.execution.router import (
        route_order,
    )

    payload = {
        "symbol": symbol,
        "side": side,
        "qty": amount,
        "price": price,
        "order_type": (
            typ or "market"
        ),
        "params": dict(
            params or {}
        ),
        "exchange_id": (
            _exchange_id(
                ex
            )
        ),
        "backend": (
            "emu"
            if _is_paper(
                ex
            )
            else "ccxt"
        ),
    }

    if _is_paper(
        ex
    ):
        payload[
            "execution_mode"
        ] = "paper"

    try:
        return _flatten(
            route_order(
                payload
            )
        )

    except Exception as exc:
        return {
            "ok": False,
            "executed": False,
            "error": str(exc),
        }


def place_market(
    ex: Any,
    symbol: str,
    side: str,
    amount: float,
) -> Dict[str, Any]:
    return safe_create_order(
        ex,
        "market",
        symbol,
        side,
        amount,
        None,
        {},
    )


def place_oco_ccxt(
    ex: Any,
    symbol: str,
    side: str,
    amount: float,
    entry_px: float,
    stop_px: Optional[
        float
    ] = None,
    take_px: Optional[
        float
    ] = None,
) -> Dict[str, Any]:
    """
    Historical OCO facade.

    Every leg now goes through the same
    universal execution authority.
    """
    result: Dict[
        str,
        Any,
    ] = {
        "entry": None,
        "tp": None,
        "sl": None,
    }

    result[
        "entry"
    ] = safe_create_order(
        ex,
        "market",
        symbol,
        side,
        amount,
    )

    opposite = (
        "sell"
        if side.lower()
        == "buy"
        else "buy"
    )

    if take_px is not None:
        result[
            "tp"
        ] = safe_create_order(
            ex,
            "limit",
            symbol,
            opposite,
            amount,
            float(
                take_px
            ),
            {},
        )

    if stop_px is not None:
        result[
            "sl"
        ] = safe_create_order(
            ex,
            "stop",
            symbol,
            opposite,
            amount,
            float(
                stop_px
            ),
            {
                "stopPrice": float(
                    stop_px
                )
            },
        )

    return result
