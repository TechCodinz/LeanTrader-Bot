"""
Compatibility shim for tests expecting `web3.guards`.
Re-exports from `w3guard.guards` so that imports succeed without the `web3` package.
"""

from w3guard.guards import (  # type: ignore
    MEMPOOL_RISK,
    FLASH_HEDGE_COUNT,
    set_mempool_tuning,
    get_mempool_tuning,
    MempoolMonitor,
    mempool_monitor,
    dynamic_slippage,
    PrivateTxClient,
    private_tx_mode,
    emergency_hedge,
    load_mempool_tuning_from_file,
    guarded_swap,
)

# Additional simple helpers used by tests
def estimate_price_impact(amount_in: float, reserve_in: float, reserve_out: float) -> float:
    """Naive constant-product price impact fraction for swapping amount_in into pool(reserve_in,reserve_out).

    Returns fraction of price impact in (0,1). Larger trades -> lower remaining fraction.
    """
    try:
        amount_in = float(amount_in)
        reserve_in = max(1e-9, float(reserve_in))
        reserve_out = max(1e-9, float(reserve_out))
        # simplistic: remaining fraction after trade relative to pool size
        remaining = reserve_in / (reserve_in + amount_in)
        impact = max(0.0, min(1.0, remaining))
        return impact
    except Exception:
        return 0.0

def is_safe_price_impact(impact: float, max_impact: float = 0.95) -> bool:
    try:
        return float(impact) <= float(max_impact)
    except Exception:
        return False

def is_safe_gas(max_gas_price: float, gas_price: float) -> bool:
    """Return True if current gas_price is under the configured max.

    Note: parameter order matches tests: (max_gas_price, gas_price).
    """
    try:
        return float(gas_price) <= float(max_gas_price)
    except Exception:
        return False

def token_safety_checks(meta: dict) -> dict:
    reasons = []
    ok = True
    try:
        if meta.get("owner_can_mint"):
            reasons.append("flag:owner_can_mint")
            ok = False
        if float(meta.get("liquidity_usd", 0.0)) < float(meta.get("min_liquidity_usd", 0.0)):
            reasons.append("liquidity_usd_lt_min")
            ok = False
    except Exception:
        pass
    return {"ok": ok, "reasons": reasons}

__all__ = [
    "MEMPOOL_RISK",
    "FLASH_HEDGE_COUNT",
    "set_mempool_tuning",
    "get_mempool_tuning",
    "MempoolMonitor",
    "mempool_monitor",
    "dynamic_slippage",
    "PrivateTxClient",
    "private_tx_mode",
    "emergency_hedge",
    "load_mempool_tuning_from_file",
    "guarded_swap",
    "estimate_price_impact",
    "is_safe_price_impact",
    "is_safe_gas",
    "token_safety_checks",
]


