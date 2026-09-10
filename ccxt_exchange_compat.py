"""Resolve historical ccxt exchange ids against the installed ccxt version.

ccxt has renamed several exchange ids since this codebase was written: gateio
became gate, huobi became htx, coinbasepro became coinbaseexchange, and
ascendex was removed. Call sites that hardcode the old id raise AttributeError
on a current ccxt, which takes down orchestrator initialization.

This resolves an id to whatever the installed ccxt actually provides, trying the
historical name first and then its known successors, so the same source works
across ccxt versions and the operator's exchange selection is preserved. It does
not change execution behaviour or pick an exchange on the operator's behalf.
"""

from typing import Any, List, Optional

# Historical id -> successor ids, newest last. Both directions are tried, so a
# config naming either the old or the new id resolves on either ccxt version.
ALIASES = {
    "gateio": ["gate"],
    "gate": ["gateio"],
    "huobi": ["htx"],
    "htx": ["huobi"],
    "coinbasepro": ["coinbaseexchange", "coinbase"],
    "coinbaseexchange": ["coinbasepro"],
    "okex": ["okx"],
    "okx": ["okex"],
    "ftx": [],          # defunct; no successor
    "ascendex": [],     # removed from ccxt
    "bitfinex2": ["bitfinex"],
}


def candidate_ids(name: str) -> List[str]:
    """Historical id first, then known successors."""
    key = str(name or "").strip().lower()
    return [key] + [a for a in ALIASES.get(key, []) if a != key]


def resolve_exchange_id(ccxt_module: Any, name: str) -> Optional[str]:
    """Return the id this ccxt build actually exposes, or None."""
    for candidate in candidate_ids(name):
        if hasattr(ccxt_module, candidate):
            return candidate
    return None


def resolve_exchange_class(ccxt_module: Any, name: str) -> Any:
    """Return the ccxt exchange class for ``name``.

    Raises AttributeError naming the id and the ccxt version when neither the
    historical id nor any known successor exists, so the failure says which
    exchange is unavailable rather than surfacing as a bare attribute error.
    """
    resolved = resolve_exchange_id(ccxt_module, name)
    if resolved is None:
        version = getattr(ccxt_module, "__version__", "unknown")
        tried = ", ".join(candidate_ids(name))
        raise AttributeError(
            f"ccxt {version} provides no exchange for '{name}' (tried: {tried})"
        )
    return getattr(ccxt_module, resolved)
