"""A durable journal of execution intent transitions.

Intent lineage existed, but only inside the process that created it. When the
trading loop restarted, or when a separate process asked where intents were
stopping, the answer was an empty list -- so the same question ("did anything
qualify, and where did it stop?") could not be answered after the fact. An
in-memory trail cannot be evidence.

Every transition an intent makes is appended here as one JSON line carrying
the full identity set, so a lineage can be reassembled by correlation id from
any process, after any restart. The file is bounded: it rotates once, which
keeps recent history readable without letting a long-running container fill
its disk.

Stage names are the pipeline's own stages under stable external labels. No
stage is invented: each one below corresponds to a transition the code
actually performs.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 1

RUN_ID = f"{int(time.time())}-{os.getpid()}"

# Canonical external names for the stages an intent passes through, in order.
# The left-hand side is the internal stage constant from ``intent``; the
# right-hand side is what the journal and status reporting call it.
SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
DECISION_CREATED = "DECISION_CREATED"
THRESHOLD_EVALUATED = "THRESHOLD_EVALUATED"
CANDIDATE_SELECTED = "CANDIDATE_SELECTED"
VENUE_RESOLVED = "VENUE_RESOLVED"
ECONOMICS_EVALUATED = "ECONOMICS_EVALUATED"
PREFLIGHT_PREPARED = "PREFLIGHT_PREPARED"
ROUTE_ORDER_SUBMITTED = "ROUTE_ORDER_SUBMITTED"
BROKER_ACCEPTED = "BROKER_ACCEPTED"
EXCHANGE_ACKNOWLEDGED = "EXCHANGE_ACKNOWLEDGED"
FILL_RECORDED = "FILL_RECORDED"
POSITION_OPENED = "POSITION_OPENED"
POSITION_CLOSED = "POSITION_CLOSED"
RECONCILED = "RECONCILED"
EVOLUTION_INGESTED = "EVOLUTION_INGESTED"

JOURNAL_STAGES = (
    SIGNAL_RECEIVED,
    DECISION_CREATED,
    THRESHOLD_EVALUATED,
    CANDIDATE_SELECTED,
    VENUE_RESOLVED,
    ECONOMICS_EVALUATED,
    PREFLIGHT_PREPARED,
    ROUTE_ORDER_SUBMITTED,
    BROKER_ACCEPTED,
    EXCHANGE_ACKNOWLEDGED,
    FILL_RECORDED,
    POSITION_OPENED,
    POSITION_CLOSED,
    RECONCILED,
    EVOLUTION_INGESTED,
)

_STAGE_INDEX = {name: position for position, name in enumerate(JOURNAL_STAGES)}

# Internal stage constant -> canonical journal stage. Kept as literals rather
# than imported from ``intent`` so that module can import this one.
STAGE_LABELS: Dict[str, str] = {
    "signal": SIGNAL_RECEIVED,
    "decision": DECISION_CREATED,
    "threshold": THRESHOLD_EVALUATED,
    "candidate": CANDIDATE_SELECTED,
    "venue_eligibility": VENUE_RESOLVED,
    "economic_eligibility": ECONOMICS_EVALUATED,
    "preflight": PREFLIGHT_PREPARED,
    "route_order": ROUTE_ORDER_SUBMITTED,
    "broker": BROKER_ACCEPTED,
    "exchange_order": EXCHANGE_ACKNOWLEDGED,
    "fill": FILL_RECORDED,
    "position": POSITION_OPENED,
    "close": POSITION_CLOSED,
    "reconciliation": RECONCILED,
    "evolution": EVOLUTION_INGESTED,
}

# The identity that has to survive a process boundary for a trace to be
# reassembled later.
IDENTITY_FIELDS = (
    "intent_id",
    "correlation_id",
    "source_engine",
    "source_strategy",
    "signal_id",
    "decision_id",
    "candidate_id",
)

_LOCK = threading.Lock()


def journal_stage(stage: str) -> str:
    """The canonical name for an internal stage constant."""
    return STAGE_LABELS.get(str(stage or "").strip().lower(), "")


def stage_rank(stage: str) -> int:
    """How far through the pipeline a stage is. -1 for an unknown stage."""
    return _STAGE_INDEX.get(stage, -1)


def journal_path() -> Path:
    """Where the lineage journal lives.

    Defaults to the shared runtime data directory so a separate process can
    read it, and falls back to the local runtime directory outside the
    container. Same contract as the universe snapshot.
    """
    explicit = os.getenv("EXECUTION_LINEAGE_PATH", "").strip()
    if explicit:
        return Path(explicit)

    shared = Path(os.getenv("LEANTRADER_DATA_DIR", "/app/data"))
    if shared.is_dir() and os.access(shared, os.W_OK):
        runtime_dir = shared / "runtime"
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            return runtime_dir / "intent_lineage.jsonl"
        except OSError:
            return shared / "intent_lineage.jsonl"

    return Path("runtime/intent_lineage.jsonl")


def _max_bytes() -> int:
    try:
        return max(64 * 1024, int(os.getenv("EXECUTION_LINEAGE_MAX_BYTES", "8388608")))
    except (TypeError, ValueError):
        return 8388608


def _rotate_if_needed(path: Path) -> None:
    """Keep one previous generation, so the journal cannot grow without end."""
    try:
        if path.exists() and path.stat().st_size >= _max_bytes():
            os.replace(path, path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass


def _suppressed() -> bool:
    """True while the caller is answering a question rather than trading.

    A diagnostic walk reaches real stages, but recording them here would make
    the lineage describe intents the system never actually formed.
    """
    try:
        from . import preflight

        return preflight.is_diagnostic()
    except Exception:
        return False


def record_transition(
    intent: Any,
    stage: str,
    *,
    ok: bool = True,
    detail: str = "",
    outcome: str = "",
    reason: str = "",
    terminal: bool = False,
) -> bool:
    """Append one transition. Returns whether it reached the journal.

    Never raises: lineage is evidence about trading, not a participant in it.
    """
    if _suppressed():
        return False

    label = journal_stage(stage) or str(stage or "").strip().upper()

    entry: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "at": time.time(),
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "stage": label,
        "internal_stage": str(stage or ""),
        "stage_rank": stage_rank(label),
        "ok": bool(ok),
        "terminal": bool(terminal),
        "detail": str(detail or "")[:240],
        "outcome": str(outcome or ""),
        "reason": str(reason or "")[:240],
    }

    for name in IDENTITY_FIELDS:
        entry[name] = str(getattr(intent, name, "") or "")
    for name in ("symbol", "side", "venue", "environment", "timeframe"):
        entry[name] = str(getattr(intent, name, "") or "")
    try:
        entry["confidence"] = float(getattr(intent, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        entry["confidence"] = 0.0

    path = journal_path()
    line = json.dumps(entry, sort_keys=True)
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(path)
            # One line, one write, in append mode: a concurrent reader sees
            # either the whole record or none of it.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        return True
    except Exception:
        return False


def _iter_lines(path: Path) -> Iterable[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line
    except OSError:
        return


def read_transitions(limit: int = 500, include_rotated: bool = False) -> List[Dict[str, Any]]:
    """The most recent transitions, oldest first.

    Malformed or partially written lines are skipped rather than raising: a
    reader must never be the thing that breaks because a writer was mid-flush.
    """
    path = journal_path()
    paths = [path]
    if include_rotated:
        rotated = path.with_suffix(path.suffix + ".1")
        if rotated.exists():
            paths.insert(0, rotated)

    rows: List[Dict[str, Any]] = []
    for candidate in paths:
        for line in _iter_lines(candidate):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)

    if limit and limit > 0:
        return rows[-limit:]
    return rows


def lineage_for(correlation_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
    """Every recorded transition of one lineage, in order."""
    wanted = str(correlation_id or "").strip()
    if not wanted:
        return []
    rows = [
        row
        for row in read_transitions(limit=limit, include_rotated=True)
        if row.get("correlation_id") == wanted or row.get("intent_id") == wanted
    ]
    rows.sort(key=lambda row: float(row.get("at") or 0.0))
    return rows


def journal_summary(limit: int = 5000) -> Dict[str, Any]:
    """What the journal says about how far intents are getting.

    This is observational evidence. It grants no authority: nothing reads it
    to decide whether to trade.
    """
    rows = read_transitions(limit=limit, include_rotated=True)

    by_stage: Dict[str, int] = {}
    by_engine: Dict[str, int] = {}
    intents: Dict[str, Dict[str, Any]] = {}
    terminal_outcomes: Dict[str, int] = {}
    furthest = ""
    last_at = 0.0

    for row in rows:
        stage = str(row.get("stage") or "")
        by_stage[stage] = by_stage.get(stage, 0) + 1

        engine = str(row.get("source_engine") or "") or "unattributed"
        by_engine[engine] = by_engine.get(engine, 0) + 1

        intent_id = str(row.get("intent_id") or "")
        if intent_id:
            seen = intents.setdefault(
                intent_id,
                {
                    "correlation_id": row.get("correlation_id", ""),
                    "symbol": row.get("symbol", ""),
                    "stage": stage,
                    "rank": stage_rank(stage),
                    "terminal": bool(row.get("terminal")),
                    "outcome": row.get("outcome", ""),
                    "reason": row.get("reason", ""),
                    "at": float(row.get("at") or 0.0),
                },
            )
            rank = stage_rank(stage)
            if rank >= int(seen.get("rank", -1)):
                seen.update(
                    {
                        "stage": stage,
                        "rank": rank,
                        "terminal": bool(row.get("terminal")),
                        "outcome": row.get("outcome", ""),
                        "reason": row.get("reason", ""),
                        "at": float(row.get("at") or 0.0),
                    }
                )

        if bool(row.get("terminal")):
            outcome = str(row.get("outcome") or "") or "UNCLASSIFIED"
            terminal_outcomes[outcome] = terminal_outcomes.get(outcome, 0) + 1

        if stage_rank(stage) > stage_rank(furthest):
            furthest = stage
        last_at = max(last_at, float(row.get("at") or 0.0))

    def ranked(counts: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    stopped_at: Dict[str, int] = {}
    for record in intents.values():
        stopped_at[record["stage"]] = stopped_at.get(record["stage"], 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(journal_path()),
        "transitions": len(rows),
        "distinct_intents": len(intents),
        "transitions_by_stage": ranked(by_stage),
        "intents_last_stage": ranked(stopped_at),
        "terminal_outcomes": ranked(terminal_outcomes),
        "by_source_engine": ranked(by_engine),
        "furthest_stage_reached": furthest,
        "last_transition_at": last_at,
    }


def stage_reached(stage: str, limit: int = 5000) -> int:
    """How many distinct intents have reached this canonical stage."""
    label = str(stage or "").strip().upper()
    seen = {
        str(row.get("intent_id") or "")
        for row in read_transitions(limit=limit, include_rotated=True)
        if str(row.get("stage") or "") == label
    }
    seen.discard("")
    return len(seen)


def clear_journal() -> None:
    """Remove the journal. For tests and for deliberate operator resets."""
    path = journal_path()
    for candidate in (path, path.with_suffix(path.suffix + ".1")):
        try:
            candidate.unlink()
        except OSError:
            pass
