import math
import time
from typing import Dict, Any, Tuple


def round_to_precision(value: float, decimals) -> float:
    if decimals in ("auto", None):
        return value
    q = 10 ** int(decimals)
    return math.floor(value * q) / q


def enforce_rules(intel: Dict[str, Any], symbol: str, price: float, amount: float) -> Tuple[float, float, list]:
    warnings = []
    price = round_to_precision(price, intel.get("price_precision"))
    amount = round_to_precision(amount, intel.get("amount_precision"))
    min_cost = intel.get("min_cost") or 0
    min_notional = max(min_cost, intel.get("min_notional_usdt", 0))
    notional = price * amount
    if min_notional and notional < min_notional:
        target_amount = (min_notional / max(price, 1e-9)) * 1.01
        amount = round_to_precision(target_amount, intel.get("amount_precision"))
        warnings.append(f"amount bumped to meet min_notional {min_notional}")
    return price, amount, warnings


class Throttler:
    def __init__(self, rate_limit_ms: int = 50, burst_limit: int = 8, cooldown_secs: int = 2) -> None:
        self.rate_limit_ms = rate_limit_ms
        self.burst_limit = burst_limit
        self.cooldown_secs = cooldown_secs
        self._last = 0.0
        self._burst = 0

    def before_call(self) -> None:
        now = time.time()
        min_gap = self.rate_limit_ms / 1000.0
        if now - self._last < min_gap:
            time.sleep(min_gap - (now - self._last))
        self._last = time.time()
        self._burst += 1
        if self._burst >= self.burst_limit:
            time.sleep(self.cooldown_secs)
            self._burst = 0


