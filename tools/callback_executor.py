"""Callback executor: find a stored signal by id and route it for execution.

Reached from the Telegram callback handler, so it is the one place a chat
message can ask for an order. Telegram has no execution authority of its own:
this builds an intent and hands it to the universal router, which owns the
decision about which environment and which account the order goes to.

It used to answer a non-live request with {"ok": True, "simulated": True} and
an order dict carrying no id -- a success for an order nobody placed -- and on
the live path it went straight to PaperBroker.create_order, bypassing the
router entirely.
"""
from typing import Any, Dict
from pathlib import Path

import json
import os

ROOT = Path(__file__).resolve().parent.parent

def _find_signal(signal_id: str) -> Dict[str, Any] | None:
    qdir = ROOT / "runtime"
    for f in qdir.glob("signals-*.ndjson"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if obj.get("id") == signal_id:
                            return obj
                    except Exception:
                        continue
        except Exception:
            continue
    return None

def execute_signal_by_id(
    signal_id: str, user_id: str | int, live: bool = False
) -> Dict[str, Any]:
    """Route one stored signal through the universal execution router.

    ``live=False`` requests the paper environment; it does not mean "pretend".
    The paper broker produces a real simulated fill with an id, and the result
    reports what actually happened either way.

    ``live=True`` requests the operator-configured authenticated environment.
    It still requires ENABLE_LIVE, and the router refuses when there is no
    authenticated authority rather than quietly downgrading to paper.
    """
    from src.leantrader.execution.router import route_order

    sig = _find_signal(signal_id)
    if not sig:
        return {"ok": False, "error": "signal not found"}

    side = str(sig.get("side") or "").strip().lower()
    if side not in {"buy", "sell"}:
        return {"ok": False, "error": "signal carries no tradable side"}

    try:
        qty = float(sig.get("qty") or sig.get("amount") or 0.0)
    except (TypeError, ValueError):
        qty = 0.0
    if qty <= 0.0:
        return {"ok": False, "error": "signal carries no quantity"}

    intent: Dict[str, Any] = {
        "symbol": sig.get("symbol"),
        "side": side,
        "qty": qty,
        "order_type": "market",
    }

    entry = sig.get("entry")
    if entry is not None:
        intent["reference_price"] = entry

    if not live:
        return route_order(intent, "paper")

    if os.getenv("ENABLE_LIVE", "false").strip().lower() not in ("1", "true", "yes"):
        return {"ok": False, "error": "live execution disabled by config"}

    # Per-user PIN. The webhook verifies it before calling here; this is a
    # second check, and a failure to verify refuses rather than continuing.
    try:
        from tools.user_pins import verify_pin
    except Exception:
        return {"ok": False, "error": "pin verification unavailable"}

    if not verify_pin(user_id, os.getenv("LIVE_EXECUTION_PIN", "")):
        return {"ok": False, "error": "pin verification failed"}

    # No mode is forced here. The router resolves the authenticated
    # environment from the runtime configuration and refuses if there is none.
    return route_order(intent)
