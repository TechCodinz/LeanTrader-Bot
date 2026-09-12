"""Why no trade happened, stated from evidence rather than from silence.

A quiet run has more than one honest explanation. "Nothing qualified" is a
normal, correct outcome for a system with a confidence gate and a venue
minimum; "the handoff between the decision layer and preflight is broken" is a
defect. Reporting the second when the first is true sends an operator looking
for a bug that is not there -- and reporting the first when the second is true
hides one that is.

This module reads the two pieces of persisted evidence -- the lifecycle
counters and the intent lineage journal -- and names the most specific reason
they jointly support. Where they support nothing, it says so: an unknown is
reported as an unknown, never as a break and never as "nothing qualified".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import lineage

# Idle reasons, from "working as designed" through to "genuinely broken".
EXECUTION_ACTIVE = "EXECUTION_ACTIVE"
NO_QUALIFIED_DECISION = "NO_QUALIFIED_DECISION"
NO_DECISION_FROM_SIGNALS = "NO_DECISION_FROM_SIGNALS"
NO_SIGNALS_PRODUCED = "NO_SIGNALS_PRODUCED"
CAPITAL_BELOW_EXECUTABLE_MINIMUM = "CAPITAL_BELOW_EXECUTABLE_MINIMUM"
NO_ELIGIBLE_VENUE = "NO_ELIGIBLE_VENUE"
NO_EXECUTION_AUTHORITY = "NO_EXECUTION_AUTHORITY"
BLOCKED_AT_PREFLIGHT = "BLOCKED_AT_PREFLIGHT"
PREPARED_BUT_NOT_SUBMITTED = "PREPARED_BUT_NOT_SUBMITTED"
HANDOFF_BROKEN_BEFORE_PREFLIGHT = "HANDOFF_BROKEN_BEFORE_PREFLIGHT"
NO_EVIDENCE_RECORDED = "NO_EVIDENCE_RECORDED"

# Preflight blocker class -> the idle reason it implies when it dominates.
# Anything not named here falls back to BLOCKED_AT_PREFLIGHT, which still
# carries the blocker class in the detail rather than guessing a cause.
_BLOCKER_REASONS: Dict[str, str] = {
    "STRATEGY_REJECT": NO_QUALIFIED_DECISION,
    "CONFIDENCE_BELOW_THRESHOLD": NO_QUALIFIED_DECISION,
    "SIGNAL_STALE": NO_QUALIFIED_DECISION,
    "RISK_REJECT": NO_QUALIFIED_DECISION,
    "CAPITAL_BELOW_EXECUTABLE_MINIMUM": CAPITAL_BELOW_EXECUTABLE_MINIMUM,
    "BELOW_MIN_NOTIONAL": CAPITAL_BELOW_EXECUTABLE_MINIMUM,
    "BELOW_MIN_AMOUNT": CAPITAL_BELOW_EXECUTABLE_MINIMUM,
    "INSUFFICIENT_FREE_BALANCE": CAPITAL_BELOW_EXECUTABLE_MINIMUM,
    "PRECISION_COLLAPSED_TO_ZERO": CAPITAL_BELOW_EXECUTABLE_MINIMUM,
    "MARKET_NOT_LISTED": NO_ELIGIBLE_VENUE,
    "MARKET_NOT_SPOT": NO_ELIGIBLE_VENUE,
    "MARKET_INACTIVE": NO_ELIGIBLE_VENUE,
    "VENUE_NOT_ELIGIBLE": NO_ELIGIBLE_VENUE,
    "NO_EXECUTION_AUTHORITY": NO_EXECUTION_AUTHORITY,
}

# Stages that mean an intent had been accepted for execution. If one of these
# was reached and no preflight attempt was ever recorded, the handoff really
# is broken -- that is the one case where claiming a break is warranted.
_HANDOFF_STAGES = (
    lineage.CANDIDATE_SELECTED,
    lineage.VENUE_RESOLVED,
    lineage.ECONOMICS_EVALUATED,
)


def _dominant_blocker(blockers: Dict[str, Any]) -> Optional[str]:
    best_name, best_count = None, 0
    for name, entry in (blockers or {}).items():
        try:
            count = int(entry.get("count", 0)) if isinstance(entry, dict) else int(entry)
        except (TypeError, ValueError):
            continue
        if count > best_count:
            best_name, best_count = str(name), count
    return best_name


def classify_idle_reason(
    counters: Optional[Dict[str, Any]] = None,
    journal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Name the reason nothing was submitted, and say what supports it.

    ``counters`` is a preflight telemetry snapshot; ``journal`` is a lineage
    journal summary. Both are read if not supplied. The returned ``reason`` is
    one of the constants above; ``detail`` says which evidence produced it and
    ``evidence`` carries the numbers a reader would want to check.
    """
    if counters is None:
        from . import preflight

        counters = preflight.telemetry_snapshot()
    if journal is None:
        journal = lineage.journal_summary()

    def count(name: str) -> int:
        try:
            return int(counters.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    attempts = count("attempts")
    prepared = count("prepared")
    submitted = count("submitted")
    blockers = counters.get("blockers") or {}
    stages = journal.get("intents_last_stage") or {}
    transitions = int(journal.get("transitions") or 0)

    evidence = {
        "attempts": attempts,
        "prepared": prepared,
        "submitted": submitted,
        "lineage_transitions": transitions,
        "furthest_stage_reached": journal.get("furthest_stage_reached", ""),
        "dominant_blocker": _dominant_blocker(blockers) or "",
    }

    if submitted > 0:
        return {
            "reason": EXECUTION_ACTIVE,
            "detail": f"{submitted} order(s) submitted through route_order",
            "evidence": evidence,
        }

    if prepared > 0:
        # Preflight produced a submittable order and nothing reached the
        # exchange. That is downstream of every check in this module.
        return {
            "reason": PREPARED_BUT_NOT_SUBMITTED,
            "detail": (
                f"{prepared} order(s) passed preflight but none were submitted; "
                "the break is between prepare_order and route_order"
            ),
            "evidence": evidence,
        }

    if attempts > 0:
        dominant = _dominant_blocker(blockers)
        if dominant:
            reason = _BLOCKER_REASONS.get(dominant, BLOCKED_AT_PREFLIGHT)
            return {
                "reason": reason,
                "detail": f"every attempt was blocked; most frequent: {dominant}",
                "evidence": evidence,
            }
        return {
            "reason": BLOCKED_AT_PREFLIGHT,
            "detail": (
                f"{attempts} attempt(s) recorded, none prepared, and no blocker "
                "class was recorded -- a counter is being incremented without a "
                "classification"
            ),
            "evidence": evidence,
        }

    # No attempt was ever made. The journal decides between "nothing
    # qualified" and "something qualified and never arrived".
    handoff_reached = any(int(stages.get(stage, 0) or 0) > 0 for stage in _HANDOFF_STAGES)
    if handoff_reached:
        return {
            "reason": HANDOFF_BROKEN_BEFORE_PREFLIGHT,
            "detail": (
                "intents reached the execution handoff but preflight recorded "
                "no attempt; this is an upstream break, not an absence of "
                "opportunity"
            ),
            "evidence": evidence,
        }

    if int(stages.get(lineage.THRESHOLD_EVALUATED, 0) or 0) > 0 or int(
        stages.get(lineage.DECISION_CREATED, 0) or 0
    ) > 0:
        return {
            "reason": NO_QUALIFIED_DECISION,
            "detail": (
                "decisions were formed and evaluated, and none passed the "
                "confidence gate; the gate is doing its job"
            ),
            "evidence": evidence,
        }

    if int(stages.get(lineage.SIGNAL_RECEIVED, 0) or 0) > 0:
        return {
            "reason": NO_DECISION_FROM_SIGNALS,
            "detail": (
                "signals were received and none became a decision; the break, "
                "if any, is in the decision layer"
            ),
            "evidence": evidence,
        }

    if transitions > 0:
        return {
            "reason": NO_QUALIFIED_DECISION,
            "detail": (
                "lineage transitions exist but none reached a decision stage"
            ),
            "evidence": evidence,
        }

    return {
        "reason": NO_EVIDENCE_RECORDED,
        "detail": (
            "no lifecycle counters and no lineage transitions were found. This "
            "is unknown, not idle and not broken: either nothing has run since "
            "this evidence store was created, or this process is not reading "
            "the same paths the trading loop writes"
        ),
        "evidence": evidence,
    }
