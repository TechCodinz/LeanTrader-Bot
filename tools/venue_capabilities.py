#!/usr/bin/env python3
"""Classify what each exchange adapter can actually do, by testing it.

A capability written in a document goes stale the moment ccxt is upgraded.
This probes the installed build instead: it constructs each client, tries the
sandbox switch, and checks whether the switch reaches the wire.

It sends nothing. Every check is local -- class construction, URL and header
inspection, and the `has` capability map ccxt publishes. No request is made
and no order is placed.

Classifications:

  LIVE_CAPABLE      the adapter exists and advertises createOrder, so the
                    live path is reachable once an operator selects live and
                    supplies credentials
  TESTNET_CAPABLE   set_sandbox_mode(True) changes where requests go
  PUBLIC_DATA_ONLY  usable for market data, but not for orders
  CONFIG_REQUIRED   capable, but no credentials are configured for it
  INCOMPLETE        the installed ccxt has no adapter for this id

LIVE_CAPABLE describes the adapter, not permission. Execution mode decides
the destination; see docs/EXECUTION_MODES.md.

    python -m tools.venue_capabilities
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

VENUES = (
    "bybit",
    "binance",
    "okx",
    "kucoin",
    "gateio",
    "mexc",
    "bitget",
)


def _routing_fingerprint(exchange) -> str:
    return json.dumps(
        {
            "urls": exchange.urls,
            "hostname": getattr(exchange, "hostname", None),
            "headers": getattr(exchange, "headers", None),
        },
        sort_keys=True,
        default=str,
    )


def _credentials_present(venue: str) -> bool:
    prefix = venue.replace("-", "_").upper()
    for name in (
        f"{prefix}_API_KEY",
        f"{prefix}_TESTNET_API_KEY",
        f"{prefix}_API_KEY_FILE",
        f"{prefix}_TESTNET_API_KEY_FILE",
    ):
        if os.getenv(name):
            return True
    return False


def classify(venue: str) -> Dict[str, object]:
    """Everything decided by inspection; nothing by assertion."""
    import ccxt

    from ccxt_exchange_compat import resolve_exchange_class

    result: Dict[str, object] = {
        "venue": venue,
        "classifications": [],
        "detail": "",
    }

    try:
        exchange_class = resolve_exchange_class(ccxt, venue)
    except Exception as exc:
        result["classifications"] = ["INCOMPLETE"]
        result["detail"] = f"no adapter in installed ccxt ({type(exc).__name__})"
        return result

    exchange = exchange_class({"enableRateLimit": True})
    result["ccxt_id"] = exchange.id

    capabilities = exchange.has or {}
    can_order = bool(capabilities.get("createOrder"))

    classifications: List[str] = []
    if can_order:
        classifications.append("LIVE_CAPABLE")
    else:
        classifications.append("PUBLIC_DATA_ONLY")

    sandbox = getattr(exchange, "set_sandbox_mode", None)
    if callable(sandbox):
        before = _routing_fingerprint(exchange)
        try:
            sandbox(True)
        except Exception as exc:
            result["detail"] = f"sandbox unavailable ({type(exc).__name__})"
        else:
            if _routing_fingerprint(exchange) != before:
                classifications.append("TESTNET_CAPABLE")
                headers = getattr(exchange, "headers", {}) or {}
                result["detail"] = (
                    "sandbox via header"
                    if "x-simulated-trading" in headers
                    else "sandbox via endpoint"
                )
            else:
                result["detail"] = (
                    "set_sandbox_mode(True) changes nothing that reaches "
                    "the wire; Testnet not available"
                )
    else:
        result["detail"] = "no set_sandbox_mode"

    if not _credentials_present(venue):
        classifications.append("CONFIG_REQUIRED")

    result["classifications"] = classifications
    return result


def main() -> int:
    rows = [classify(venue) for venue in VENUES]

    width = max(len(str(r["venue"])) for r in rows)
    print("Exchange adapter capabilities (installed ccxt, probed locally)\n")
    for row in rows:
        labels = ", ".join(row["classifications"])
        print(f"  {str(row['venue']):<{width}}  {labels}")
        if row["detail"]:
            print(f"  {'':<{width}}  └─ {row['detail']}")

    print(
        "\nLIVE_CAPABLE describes the adapter, not permission to trade. "
        "Execution\nmode decides the destination; credentials only "
        "authenticate. See\ndocs/EXECUTION_MODES.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
