"""Compatibility facade for the canonical production intelligence engine."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .intelligence import AdaptiveIntelligence, IntelligenceDecision

Decision = IntelligenceDecision


def decide(frame: pd.DataFrame) -> IntelligenceDecision:
    """Evaluate one frame using persisted bounded weights without changing them."""
    state_path = Path(os.getenv("INTELLIGENCE_STATE_PATH", "runtime/vps_intelligence_state.json"))
    return AdaptiveIntelligence(state_path).evaluate(frame)
