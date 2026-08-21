from __future__ import annotations

import math
from typing import Any

from .strategy_observatory_v141 import *  # noqa: F401,F403
from .strategy_observatory_v141 import StrategyObservatory as _V141StrategyObservatory


class StrategyObservatory(_V141StrategyObservatory):
    """v1.42 evidence wrapper preserving the exact v1.41 observatory core.

    Closed costed shadow episodes are enriched with the signal-open and
    outcome-close timestamps needed to prove purge/embargo boundaries. The
    aggregate v1.41 accounting and authority rules are unchanged.
    """

    VERSION = "3.1"

    def observe(
        self,
        symbol: str,
        price: float,
        signals: list[dict[str, Any]],
        timeframe_signals: dict[str, float],
    ) -> dict[str, Any]:
        before = {
            str(name): dict(row)
            for name, row in (
                (self.state.get("episodes") or {}).get(symbol, {}) or {}
            ).items()
            if isinstance(row, dict)
        }
        result = super().observe(symbol, price, signals, timeframe_signals)
        closed_at = float(self.state.get("last_observation_epoch") or 0.0)
        outcomes = result.get("outcomes") if isinstance(result, dict) else None
        if not isinstance(outcomes, list):
            return result
        for row in outcomes:
            if not isinstance(row, dict):
                continue
            strategy = str(row.get("strategy") or "")
            active = before.get(strategy) or {}
            try:
                opened_at = float(active.get("opened_at"))
            except (TypeError, ValueError):
                opened_at = math.nan
            complete = (
                math.isfinite(opened_at)
                and math.isfinite(closed_at)
                and opened_at > 0.0
                and closed_at >= opened_at
            )
            row.update(
                {
                    "opened_at": opened_at if complete else None,
                    "closed_at": closed_at if complete else None,
                    "feature_start": opened_at if complete else None,
                    "feature_end": opened_at if complete else None,
                    "label_end": closed_at if complete else None,
                    "evidence_interval_complete": complete,
                }
            )
        return result
