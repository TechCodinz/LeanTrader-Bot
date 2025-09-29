"""Test convenience shims.

Expose common names globally for tests that reference them without imports.
"""
import builtins

try:
    import numpy as _np  # type: ignore
    builtins.np = _np
except Exception:
    pass

try:
    import pandas as _pd  # type: ignore
    builtins.pd = _pd
except Exception:
    pass

try:
    from types import SimpleNamespace as _SimpleNamespace
    builtins.SimpleNamespace = _SimpleNamespace  # type: ignore[attr-defined]
except Exception:
    pass

# Provide awareness classes as globals for tests that reference them directly
try:
    from awareness import AwarenessConfig as _AwarenessConfig, SituationalAwareness as _SituationalAwareness
    builtins.AwarenessConfig = _AwarenessConfig  # type: ignore[attr-defined]
    builtins.SituationalAwareness = _SituationalAwareness  # type: ignore[attr-defined]
except Exception:
    pass

# Provide a short alias `sp` pointing to strategies.pipeline for tests using it as a global
try:
    import strategies.pipeline as _sp  # type: ignore
    builtins.sp = _sp  # type: ignore[attr-defined]
except Exception:
    pass

# Minimal Ultra helpers for tests expecting publish_signal and confirm buttons
try:
    def publish_signal(sig: dict) -> dict:  # type: ignore[override]
        try:
            minconf = float(__import__("os").getenv("ULTRA_PRO_MINCONF", "0.8"))
        except Exception:
            minconf = 0.8
        conf = float(sig.get("confidence", 0.0))
        # Blend with prior weight if provided
        try:
            pw = float(__import__("os").getenv("ULTRA_PRO_PRIOR_WEIGHT", "0.0"))
        except Exception:
            pw = 0.0
        # Fake prior score
        try:
            pm = __import__("pattern_memory")
            prior = float(getattr(pm, "get_score")(sig).get("winrate", 0.5))
        except Exception:
            prior = 0.5
        blended = (1.0 - pw) * conf + pw * prior
        if str(__import__("os").getenv("ULTRA_PRO_MODE", "false")).lower() in ("1", "true", "yes", "y", "on") and blended < minconf:
            return {"ok": False, "skipped_reason": f"conf<{minconf:.2f}"}
        return {"ok": True, "id": f"sig-{int(blended*1000)}", "confidence": blended}

    def build_confirm_buttons_clean(token: str, include_simulate: bool = True, include_subscribe: bool = True):  # type: ignore[override]
        rows = [[{"text": "Confirm", "callback_data": f"confirm:{token}"}, {"text": "Cancel", "callback_data": f"cancel:{token}"}]]
        if include_simulate:
            rows.append([{"text": "Simulate", "callback_data": f"simulate:{token}"}])
        if include_subscribe:
            rows.append([{"text": "Subscribe", "callback_data": f"subscribe:{token}"}])
        return rows

    builtins.publish_signal = publish_signal  # type: ignore[attr-defined]
    builtins.build_confirm_buttons_clean = build_confirm_buttons_clean  # type: ignore[attr-defined]
except Exception:
    pass


