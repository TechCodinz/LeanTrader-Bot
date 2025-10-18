from typing import Any, Dict

from tools.exchange_intel import get_profile
from tools.order_guard import enforce_rules, Throttler


def _fetch_last_price_safe(ex, symbol: str) -> float:
    try:
        t = ex.fetch_ticker(symbol)
        for k in ("last", "close", "bid", "ask"):
            v = t.get(k)
            if v:
                return float(v)
    except Exception:
        pass
    return 0.0


def enable_guard(ex) -> None:
    """Wrap ex.create_order so all orders pass through guardrails."""
    if getattr(ex, "_lt_guard_wrapped", False):
        return

    intel = get_profile(getattr(ex, "id", "unknown"), lambda: ex, force=False)
    throttler = Throttler(
        rate_limit_ms=intel.get("rate_limit_ms", 50),
        burst_limit=intel.get("burst_limit", 8),
        cooldown_secs=intel.get("cooldown_secs_on_throttle", 2),
    )

    _orig_create_order = ex.create_order

    def _guarded_create_order(
        symbol: str,
        type_: str,
        side: str,
        amount: Any,
        price: Any = None,
        params: Dict[str, Any] | None = None,
    ):
        throttler.before_call()
        _price = float(price) if price is not None else _fetch_last_price_safe(ex, symbol)
        _amount = float(amount)
        adj_price, adj_amount, warnings = enforce_rules(intel, symbol, _price, _amount)
        if warnings:
            print("[ORDER_GUARD] ", "; ".join(warnings))
        return _orig_create_order(
            symbol,
            type_,
            side,
            adj_amount,
            None if type_ == "market" else adj_price,
            params or {},
        )

    ex.create_order = _guarded_create_order  # type: ignore[attr-defined]
    ex._lt_guard_wrapped = True  # type: ignore[attr-defined]


def print_intel_summary(ex) -> None:
    """Log a startup summary of exchange intel for visibility."""
    intel = get_profile(getattr(ex, "id", "unknown"), lambda: ex, force=False)
    print("\n================= EXCHANGE INTEL SUMMARY =================")
    print(f" Exchange:        {getattr(ex, 'id', '?')}")
    print(f" Rate limit:      {intel.get('rate_limit_ms', '?')} ms")
    print(f" Burst limit:     {intel.get('burst_limit', '?')}")
    print(f" Min notional:    {intel.get('min_notional_usdt', intel.get('min_cost', '?'))} USDT")
    print(f" Maker fee:       {intel.get('maker_fee_pct', '?')}%")
    print(f" Taker fee:       {intel.get('taker_fee_pct', '?')}%")
    print(f" Price precision: {intel.get('price_precision', 'auto')}")
    print(f" Amount precision:{intel.get('amount_precision', 'auto')}")
    print("===========================================================\n")


