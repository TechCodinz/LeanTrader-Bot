from __future__ import annotations

from typing import Any


_WRAPPER_MARKER = "_leantrader_public_spot_wrapper"


def install_public_spot_defaults() -> None:
    """Keep LeanTrader's public Bybit adapter on spot unless a caller opts out.

    CCXT's Bybit adapter defaults to derivatives, while the supported LeanTrader
    market universe and Testnet execution contract are spot-only.  Installing a
    narrow constructor wrapper prevents a default derivatives ticker request
    from being compared against spot markets during dynamic discovery.

    Explicit caller configuration always wins, so future research adapters can
    still request another market type deliberately without weakening the
    current production safety boundary.
    """
    import ccxt  # type: ignore

    exchange_class = getattr(ccxt, "bybit", None)
    if exchange_class is None or getattr(exchange_class, _WRAPPER_MARKER, False):
        return

    original_class = exchange_class

    class LeanTraderBybit(original_class):  # type: ignore[misc, valid-type]
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            normalized = dict(config or {})
            options = dict(normalized.get("options") or {})
            options.setdefault("defaultType", "spot")
            normalized["options"] = options
            super().__init__(normalized)

    setattr(LeanTraderBybit, _WRAPPER_MARKER, True)
    LeanTraderBybit.__name__ = original_class.__name__
    LeanTraderBybit.__qualname__ = original_class.__qualname__
    LeanTraderBybit.__module__ = original_class.__module__
    setattr(ccxt, "bybit", LeanTraderBybit)
