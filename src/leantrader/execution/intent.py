"""Identity for an execution intent, so a trace can be followed end to end.

The runtime showed signals generated, signals published and decisions
created, alongside candidates 0, preflight 0, submitted 0 -- while execution
telemetry separately recorded dozens of attempts. Both were true. More than
one producer reaches execution, and nothing tied an attempt back to whatever
produced it, so the funnel could not say which engine was doing what.

An intent carries its lineage from the signal that suggested it to the fill
that settled it, and every stage it passes records where it got to. A
rejection says exactly where it stopped, which is the difference between
"no qualified trade exists" -- an acceptable answer -- and "the handoff is
broken".
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from . import lineage

# Stages, in the order an intent passes through them.
SIGNAL = "signal"
DECISION = "decision"
THRESHOLD = "threshold"
CANDIDATE = "candidate"
VENUE_ELIGIBILITY = "venue_eligibility"
ECONOMIC_ELIGIBILITY = "economic_eligibility"
PREFLIGHT = "preflight"
ROUTE_ORDER = "route_order"
BROKER = "broker"
EXCHANGE_ORDER = "exchange_order"
FILL = "fill"
POSITION = "position"
CLOSE = "close"
RECONCILIATION = "reconciliation"
EVOLUTION = "evolution"

STAGES = (
    SIGNAL,
    DECISION,
    THRESHOLD,
    CANDIDATE,
    VENUE_ELIGIBILITY,
    ECONOMIC_ELIGIBILITY,
    PREFLIGHT,
    ROUTE_ORDER,
    BROKER,
    EXCHANGE_ORDER,
    FILL,
    POSITION,
    CLOSE,
    RECONCILIATION,
    EVOLUTION,
)

DECISION_REJECTED_CONFIDENCE = "DECISION_REJECTED_CONFIDENCE"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _journal(intent: "ExecutionIntent", stage: str, **kwargs: Any) -> None:
    """Persist one transition, and never let doing so break the pipeline.

    The in-memory trail above is what this process can see. The journal is
    what survives the process, which is the only thing that can answer "where
    did intents stop?" after a restart or from another container.
    """
    try:
        lineage.record_transition(intent, stage, **kwargs)
    except Exception:
        pass


@dataclass
class ExecutionIntent:
    """One attempt to trade, with the lineage that produced it."""

    symbol: str
    side: str = ""
    intent_id: str = field(default_factory=lambda: _new_id("int"))
    correlation_id: str = ""
    source_engine: str = ""
    source_strategy: str = ""
    signal_id: str = ""
    decision_id: str = ""
    candidate_id: str = ""
    venue: str = ""
    environment: str = ""
    confidence: float = 0.0
    timeframe: str = ""
    created_at: float = field(default_factory=time.time)

    stage: str = SIGNAL
    terminal: bool = False
    outcome: str = ""
    reason: str = ""
    trail: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            # One correlation id per lineage, so every stage of the same
            # attempt can be found together.
            self.correlation_id = self.intent_id

    def advance(self, stage: str, detail: str = "") -> "ExecutionIntent":
        self.stage = stage
        self.trail.append(
            {"stage": stage, "at": time.time(), "detail": detail, "ok": True}
        )
        _journal(self, stage, ok=True, detail=detail)
        return self

    def stop(self, stage: str, outcome: str, reason: str = "") -> "ExecutionIntent":
        """Record where this intent ended, and why.

        Every rejection names a stage. An intent that simply vanishes is the
        thing that made the funnel unreadable.
        """
        self.stage = stage
        self.terminal = True
        self.outcome = outcome
        self.reason = reason
        self.trail.append(
            {"stage": stage, "at": time.time(), "detail": reason, "ok": False}
        )
        _journal(
            self,
            stage,
            ok=False,
            detail=reason,
            outcome=outcome,
            reason=reason,
            terminal=True,
        )
        record_outcome(self)
        return self

    def succeed(self, stage: str, detail: str = "") -> "ExecutionIntent":
        self.advance(stage, detail)
        if stage in (FILL, RECONCILIATION, EVOLUTION):
            self.outcome = "COMPLETED"
        return self

    def journal_stage(self) -> str:
        """The canonical external name of the stage this intent is at."""
        return lineage.journal_stage(self.stage)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_payload(self) -> Dict[str, Any]:
        """Identity fields to carry alongside an order payload.

        Passed through route_order as params so a receipt can be tied back to
        the intent that produced it.
        """
        return {
            "intent_id": self.intent_id,
            "correlation_id": self.correlation_id,
            "source_engine": self.source_engine,
            "source_strategy": self.source_strategy,
            "signal_id": self.signal_id,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
        }


def new_signal_id() -> str:
    return _new_id("sig")


def new_decision_id() -> str:
    return _new_id("dec")


def new_candidate_id() -> str:
    return _new_id("cnd")


def intent_from_decision(
    decision: Dict[str, Any],
    source_engine: str = "",
    environment: str = "",
    venue: str = "",
) -> ExecutionIntent:
    """Build an intent that keeps whatever lineage the decision carried."""
    signal = decision.get("signal") if isinstance(decision, dict) else {}
    signal = signal if isinstance(signal, dict) else {}
    data = signal.get("data") if isinstance(signal.get("data"), dict) else {}

    symbol = (
        decision.get("symbol")
        or signal.get("symbol")
        or data.get("symbol")
        or ""
    )
    try:
        confidence = float(decision.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return ExecutionIntent(
        symbol=str(symbol),
        side=str(decision.get("action") or signal.get("side") or "").lower(),
        correlation_id=str(
            decision.get("correlation_id") or signal.get("correlation_id") or ""
        ),
        source_engine=source_engine
        or str(signal.get("source") or data.get("source") or ""),
        source_strategy=str(
            signal.get("strategy") or data.get("strategy") or ""
        ),
        signal_id=str(signal.get("signal_id") or data.get("signal_id") or ""),
        decision_id=str(decision.get("decision_id") or new_decision_id()),
        venue=venue,
        environment=environment,
        confidence=confidence,
        timeframe=str(signal.get("timeframe") or data.get("timeframe") or ""),
    )


# ------------------------------------------------------------- outcome log


_OUTCOMES: List[Dict[str, Any]] = []
_OUTCOME_LOCK = threading.Lock()


def _outcome_limit() -> int:
    try:
        return int(os.getenv("EXECUTION_TRACE_LIMIT", "500"))
    except (TypeError, ValueError):
        return 500


def record_outcome(intent: ExecutionIntent) -> None:
    """Keep a bounded record of where intents stopped."""
    entry = {
        "intent_id": intent.intent_id,
        "correlation_id": intent.correlation_id,
        "symbol": intent.symbol,
        "side": intent.side,
        "source_engine": intent.source_engine,
        "source_strategy": intent.source_strategy,
        "venue": intent.venue,
        "environment": intent.environment,
        "confidence": intent.confidence,
        "stage": intent.stage,
        "outcome": intent.outcome,
        "reason": intent.reason,
        "at": time.time(),
    }
    with _OUTCOME_LOCK:
        _OUTCOMES.append(entry)
        limit = _outcome_limit()
        if len(_OUTCOMES) > limit:
            del _OUTCOMES[: len(_OUTCOMES) - limit]

    try:
        from . import preflight

        preflight.record_event(f"intent_stopped_{intent.stage}")
    except Exception:
        pass


def recent_outcomes(limit: int = 50) -> List[Dict[str, Any]]:
    with _OUTCOME_LOCK:
        return list(_OUTCOMES[-limit:])


def outcome_summary() -> Dict[str, Any]:
    """Where intents are stopping, and which engines produced them."""
    with _OUTCOME_LOCK:
        entries = list(_OUTCOMES)

    by_stage: Dict[str, int] = {}
    by_outcome: Dict[str, int] = {}
    by_engine: Dict[str, int] = {}

    for entry in entries:
        by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0) + 1
        outcome = entry.get("outcome") or "UNCLASSIFIED"
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        engine = entry.get("source_engine") or "unattributed"
        by_engine[engine] = by_engine.get(engine, 0) + 1

    def ranked(counts: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "traced_intents": len(entries),
        "stopped_by_stage": ranked(by_stage),
        "stopped_by_outcome": ranked(by_outcome),
        "by_source_engine": ranked(by_engine),
    }


def reset_outcomes() -> None:
    with _OUTCOME_LOCK:
        _OUTCOMES.clear()
