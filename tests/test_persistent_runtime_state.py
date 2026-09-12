"""Persistence and observability across process boundaries.

Every component here was already correct in-process. What was missing was that
another process -- the status reporter, or this process after a restart -- could
see any of it. A canonical universe that only one process can read is not a
canonical universe; an intent trail that dies with its process cannot say where
intents stopped; and a diagnostic that writes to the same counters as the
trading loop makes the funnel lie.

The specific regressions these pin down:

* discovery succeeded while nothing was ever written, because persistence
  lived in a maintenance loop that could fail before reaching it;
* a missing snapshot made every market report as NOT LISTED, turning absence
  of evidence into evidence of absence;
* ``execution_status --explain`` drove global attempts from 0 to 3, so asking
  a question made the system look like it had tried to trade.

Nothing here reaches the network.
"""

import json

import pytest

from src.leantrader.execution import idle, intent as execution_intent, lineage, preflight
from src.leantrader.universe import routing
from src.leantrader.universe import venues as venue_capabilities
from src.leantrader.universe.registry import universe as market_universe
from src.leantrader.universe.venues import CapabilityRegistry


def market(symbol, active=True):
    base, quote = symbol.split("/")
    return {
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "spot": True,
        "type": "spot",
        "active": active,
        "precision": {"amount": 3, "price": 4},
        "limits": {"amount": {"min": 1.0}, "cost": {"min": 5.0}},
    }


BYBIT_TESTNET = {
    symbol: market(symbol)
    for symbol in ("BTC/USDT", "ETH/USDT", "DOGE/USDT", "FTM/USDT")
}


@pytest.fixture
def registry(monkeypatch):
    """A capability registry nothing else has written to."""
    fresh = CapabilityRegistry()
    monkeypatch.setattr(venue_capabilities, "capabilities", fresh)
    monkeypatch.setattr(routing, "capabilities", fresh)
    return fresh


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANTRADER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(
        "EXECUTION_TELEMETRY_PATH", str(tmp_path / "execution_telemetry.json")
    )
    monkeypatch.setenv("EXECUTION_LINEAGE_PATH", str(tmp_path / "lineage.jsonl"))
    monkeypatch.delenv("UNIVERSE_SNAPSHOT_PATH", raising=False)
    yield


# ------------------------------------------------ the snapshot is written


def test_the_snapshot_lands_in_the_shared_data_directory(tmp_path, registry):
    """The contract is <LEANTRADER_DATA_DIR>/runtime/, created if absent."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    path = venue_capabilities.snapshot_path()

    assert path.parent.name == "runtime"
    assert str(tmp_path / "data") in str(path)


def test_persisting_discovery_state_writes_a_complete_snapshot(tmp_path, registry):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")

    assert venue_capabilities.persist_discovery_state() is True

    payload = json.loads(venue_capabilities.snapshot_path().read_text())
    assert payload["schema_version"] >= 3
    assert payload["generated_at"] > 0
    assert payload["source_run_id"]
    assert payload["source_pid"] > 0
    assert payload["venue_market_counts"]["bybit:testnet"] == 4
    assert payload["routing_counters"]["capability_refreshes"] >= 1
    # A full per-market row, not just a count.
    rows = {row["canonical_symbol"]: row for row in payload["markets"]}
    assert set(BYBIT_TESTNET) <= set(rows)
    doge = rows["DOGE/USDT"]
    assert doge["venue"] == "bybit" and doge["environment"] == "testnet"
    assert doge["listed"] is True
    assert doge["min_notional"] == 5.0
    assert doge["asset_class"] == "crypto"
    # And the readable summary a status report prints.
    assert payload["capabilities"]["listed_by_venue_environment"][
        "bybit:testnet"
    ] == 4


def test_the_snapshot_is_written_atomically(tmp_path, registry):
    """A reader must never see half a file, so no .tmp is left behind."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    venue_capabilities.persist_discovery_state()

    path = venue_capabilities.snapshot_path()
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    json.loads(path.read_text())  # complete and parseable


def test_the_snapshot_carries_no_credentials(tmp_path, registry, monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "must-not-appear-in-a-snapshot")
    monkeypatch.setenv("BYBIT_API_SECRET", "also-must-not-appear")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    venue_capabilities.persist_discovery_state()

    text = venue_capabilities.snapshot_path().read_text()
    assert "must-not-appear-in-a-snapshot" not in text
    assert "also-must-not-appear" not in text
    for hint in ("apiKey", "api_key", "secret", "password"):
        assert hint not in text


def test_a_written_snapshot_restores_into_another_process(tmp_path, registry):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    venue_capabilities.persist_discovery_state()

    reader = CapabilityRegistry()
    monkeypatch_target = venue_capabilities.capabilities
    try:
        venue_capabilities.capabilities = reader
        restored = venue_capabilities.load_persisted_state()
    finally:
        venue_capabilities.capabilities = monkeypatch_target

    assert restored >= 4
    assert reader.venue_metadata_known("bybit", "testnet") is True
    assert reader.resolve(
        "bybit", "DOGE/USDT", environment="testnet"
    ).classification != venue_capabilities.NOT_LISTED


def test_the_real_discovery_integration_point_persists(tmp_path, monkeypatch, registry):
    """The writer must be in the process that discovers, not a test helper.

    This instantiates the real scanner's persistence call rather than a
    stand-in, because the original defect was precisely that the only writer
    lived somewhere the discovering process never reached.
    """
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    import DYNAMIC_MARKET_SCANNER as scanner
    import inspect

    source = inspect.getsource(scanner)
    assert "persist_discovery_state" in source, (
        "the scanner that discovers markets must be the thing that persists them"
    )

    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    assert scanner.venue_capabilities.persist_discovery_state(
        {"discovered_active_pairs": len(BYBIT_TESTNET)}
    ) is True
    payload = json.loads(venue_capabilities.snapshot_path().read_text())
    assert payload["discovered_active_pairs"] == len(BYBIT_TESTNET)


# -------------------------------------- unknown must not become not-listed


def test_no_snapshot_yields_unknown_not_absent(registry):
    """Absence of evidence must never become authoritative evidence of absence."""
    decision = routing.select_venue("DOGE/USDT", "testnet", preferred="bybit")

    assert decision.classification == venue_capabilities.UNKNOWN_NO_RUNTIME_SNAPSHOT
    assert set(decision.considered.values()) == {
        venue_capabilities.UNKNOWN_NO_RUNTIME_SNAPSHOT
    }
    assert "unknown, not absent" in decision.reason


def test_a_market_genuinely_absent_from_read_metadata_is_not_listed(registry):
    """Once metadata has been read, "absent" is a real finding."""
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")

    decision = routing.select_venue("BCH/USDT", "testnet", preferred="bybit")

    assert decision.considered["bybit"] == routing.MARKET_NOT_LISTED_ON_VENUE
    assert (
        venue_capabilities.UNKNOWN_NO_RUNTIME_SNAPSHOT
        not in decision.considered.values()
    )


def test_venue_metadata_known_distinguishes_looked_from_listed(registry):
    assert registry.venue_metadata_known("bybit", "testnet") is False
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    assert registry.venue_metadata_known("bybit", "testnet") is True
    # A different environment is a different question.
    assert registry.venue_metadata_known("bybit", "live") is False


def test_failing_to_read_metadata_does_not_make_markets_absent(registry):
    """A failed load_markets is not a delisting event."""
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    registry.record_venue_markets("bybit", None, environment="testnet")

    decision = routing.select_venue("DOGE/USDT", "testnet", preferred="bybit")
    assert decision.considered["bybit"] != routing.MARKET_NOT_LISTED_ON_VENUE


# ------------------------------------------------- call-avoidance counters


def test_routing_counters_record_what_was_avoided(registry):
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    for _ in range(3):
        registry.resolve("bybit", "BCH/USDT", environment="testnet")
    registry.resolve("bybit", "DOGE/USDT", environment="testnet")

    counters = registry.routing_counters()
    assert counters["market_calls_considered"] >= 4
    assert counters["market_calls_suppressed_not_listed"] >= 1
    assert counters["capability_refreshes"] >= 1
    for required in (
        "market_calls_allowed",
        "market_calls_suppressed_asset_class",
        "market_calls_suppressed_timeframe",
        "market_calls_suppressed_cached_negative",
        "capability_revalidations",
    ):
        assert required in counters


def test_routing_counters_survive_a_snapshot_round_trip(tmp_path, registry):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    registry.record_venue_markets("bybit", BYBIT_TESTNET, environment="testnet")
    registry.resolve("bybit", "BCH/USDT", environment="testnet")
    venue_capabilities.persist_discovery_state()

    payload = json.loads(venue_capabilities.snapshot_path().read_text())
    reader = CapabilityRegistry()
    reader.load_snapshot(payload)

    assert reader.routing_counters()["market_calls_considered"] >= 1


def test_the_counters_grant_no_trading_authority(registry):
    """Observational evidence only: nothing consults them to decide to trade."""
    import inspect

    for module in (preflight, routing):
        source = inspect.getsource(module)
        assert "routing_counters" not in source or "market_calls_allowed" not in source


# ------------------------------------- diagnostics must not touch the funnel


def test_a_diagnostic_does_not_move_the_runtime_funnel():
    before = preflight.runtime_counters()

    with preflight.diagnostic_context():
        preflight.record_event("attempts")
        preflight.record_event("attempts")
        preflight.record_blocker("MARKET_NOT_LISTED", "FOO/USDT")

    assert preflight.runtime_counters() == before
    assert preflight.runtime_counters()["attempts"] == 0


def test_a_diagnostic_is_still_counted_somewhere():
    """Suppressed is not the same as invisible: the question is recorded."""
    with preflight.diagnostic_context():
        preflight.record_event("attempts")

    assert preflight.diagnostic_counters()["attempts"] == 1
    snapshot = preflight.telemetry_snapshot()
    assert "MARKET_NOT_LISTED" not in (snapshot.get("blockers") or {})


def test_record_telemetry_false_reuses_the_real_checks():
    """No second copy of the rules: the same function, booked differently."""
    intent = {"symbol": "", "side": "buy", "confidence": 0.9}

    prepared, blocked = preflight.prepare_order(intent, record_telemetry=False)

    assert prepared is None
    assert blocked.blocker == preflight.SYMBOL_NOT_NORMALIZED
    assert preflight.runtime_counters()["attempts"] == 0
    assert preflight.diagnostic_counters()["attempts"] == 1


def test_a_diagnostic_does_not_create_a_cooldown_that_blocks_real_trading():
    preflight.clear_infeasible_memory()
    with preflight.diagnostic_context():
        preflight.note_economically_infeasible(
            "bybit", "DOGE/USDT", 4.0, "too small", price=0.1, min_notional=5.0
        )

    assert (
        preflight.economically_infeasible(
            "bybit", "DOGE/USDT", 4.0, price=0.1, min_notional=5.0
        )
        is None
    )


def test_the_status_tool_never_writes_runtime_counters():
    """The whole reporter runs inside a diagnostic context, by construction."""
    import inspect
    import tools.execution_status as status

    source = inspect.getsource(status.main)
    assert "diagnostic_context" in source
    assert "record_telemetry=False" in inspect.getsource(status.report_preflight)


# ------------------------------------------------ intent lineage persistence


def test_an_intent_transition_survives_the_process_that_made_it():
    trade = execution_intent.ExecutionIntent(
        symbol="DOGE/USDT",
        side="buy",
        source_engine="ultra_core",
        source_strategy="momentum",
        signal_id="sig-1",
        decision_id="dec-1",
        candidate_id="cnd-1",
        confidence=0.72,
    )
    trade.advance(execution_intent.DECISION, "decision formed")
    trade.stop(
        execution_intent.THRESHOLD,
        execution_intent.DECISION_REJECTED_CONFIDENCE,
        "0.72 below gate",
    )

    rows = lineage.read_transitions()
    assert [row["stage"] for row in rows] == [
        lineage.DECISION_CREATED,
        lineage.THRESHOLD_EVALUATED,
    ]
    for field in lineage.IDENTITY_FIELDS:
        assert rows[0][field] == getattr(trade, field)
    assert rows[-1]["terminal"] is True
    assert rows[-1]["outcome"] == execution_intent.DECISION_REJECTED_CONFIDENCE


def test_a_lineage_can_be_reassembled_by_correlation_id():
    first = execution_intent.ExecutionIntent(symbol="BTC/USDT", side="buy")
    second = execution_intent.ExecutionIntent(symbol="ETH/USDT", side="buy")
    first.advance(execution_intent.DECISION)
    second.advance(execution_intent.DECISION)
    first.advance(execution_intent.CANDIDATE)

    trail = lineage.lineage_for(first.correlation_id)
    assert len(trail) == 2
    assert {row["symbol"] for row in trail} == {"BTC/USDT"}


def test_every_named_stage_maps_to_a_real_pipeline_stage():
    """No invented stages: the journal names what the code actually does."""
    assert set(lineage.STAGE_LABELS) == set(execution_intent.STAGES)
    assert tuple(
        lineage.STAGE_LABELS[stage] for stage in execution_intent.STAGES
    ) == lineage.JOURNAL_STAGES
    assert lineage.JOURNAL_STAGES[0] == lineage.SIGNAL_RECEIVED
    assert lineage.JOURNAL_STAGES[-1] == lineage.EVOLUTION_INGESTED


def test_the_journal_is_bounded(monkeypatch):
    monkeypatch.setenv("EXECUTION_LINEAGE_MAX_BYTES", "65536")
    for index in range(400):
        trade = execution_intent.ExecutionIntent(symbol=f"SYM{index}/USDT")
        trade.advance(execution_intent.DECISION, "x" * 200)

    path = lineage.journal_path()
    assert path.stat().st_size < 200_000
    # Rotation keeps one generation, so recent history is still readable.
    assert lineage.read_transitions(limit=10)


def test_a_malformed_line_does_not_break_a_reader():
    trade = execution_intent.ExecutionIntent(symbol="BTC/USDT")
    trade.advance(execution_intent.DECISION)
    with lineage.journal_path().open("a", encoding="utf-8") as handle:
        handle.write('{"partial": tru')

    rows = lineage.read_transitions()
    assert len(rows) == 1
    assert rows[0]["stage"] == lineage.DECISION_CREATED


def test_a_diagnostic_walk_writes_no_lineage():
    with preflight.diagnostic_context():
        trade = execution_intent.ExecutionIntent(symbol="FOO/USDT")
        trade.advance(execution_intent.DECISION)

    assert lineage.read_transitions() == []


# ---------------------------------------------------- why nothing happened


def test_a_gated_decision_is_reported_as_no_qualified_decision():
    """Not "the break is upstream" -- the gate working is not a break."""
    trade = execution_intent.ExecutionIntent(symbol="DOGE/USDT", confidence=0.55)
    trade.advance(execution_intent.DECISION)
    trade.stop(
        execution_intent.THRESHOLD,
        execution_intent.DECISION_REJECTED_CONFIDENCE,
        "0.55 below gate",
    )

    verdict = idle.classify_idle_reason()
    assert verdict["reason"] == idle.NO_QUALIFIED_DECISION


def test_an_intent_that_reached_the_handoff_and_vanished_is_a_real_break():
    trade = execution_intent.ExecutionIntent(symbol="DOGE/USDT", confidence=0.95)
    trade.advance(execution_intent.DECISION)
    trade.advance(execution_intent.CANDIDATE)

    verdict = idle.classify_idle_reason()
    assert verdict["reason"] == idle.HANDOFF_BROKEN_BEFORE_PREFLIGHT


def test_no_evidence_at_all_is_reported_as_unknown():
    """An empty store is not proof that nothing qualified."""
    verdict = idle.classify_idle_reason()
    assert verdict["reason"] == idle.NO_EVIDENCE_RECORDED
    assert "unknown" in verdict["detail"]


def test_capital_blockers_are_named_as_capital_not_as_a_break():
    preflight.record_event("attempts")
    preflight.record_blocker(preflight.CAPITAL_BELOW_EXECUTABLE_MINIMUM, "DOGE/USDT")

    verdict = idle.classify_idle_reason()
    assert verdict["reason"] == idle.CAPITAL_BELOW_EXECUTABLE_MINIMUM


def test_prepared_but_never_submitted_is_named_as_the_downstream_break():
    preflight.record_event("attempts")
    preflight.record_event("prepared")

    verdict = idle.classify_idle_reason()
    assert verdict["reason"] == idle.PREPARED_BUT_NOT_SUBMITTED


def test_a_submitted_order_is_not_reported_as_idle():
    for name in ("attempts", "prepared", "submitted"):
        preflight.record_event(name)

    assert idle.classify_idle_reason()["reason"] == idle.EXECUTION_ACTIVE


# ----------------------------------------------------------- no regressions


def test_the_confidence_gate_was_not_lowered():
    """Producing trades by lowering the bar is not producing trades.

    Runtime decisions were landing at 0.50-0.72 against a higher gate. The
    honest answer to that is NO_QUALIFIED_DECISION, not a smaller number here.
    """
    from ADAPTIVE_CONFIDENCE_ENGINE import AdaptiveConfidenceEngine

    engine = AdaptiveConfidenceEngine()
    assert engine.base_min_confidence >= 0.75
    # The adaptive band may move the gate, but never below this floor, and the
    # floor is what a "just make it trade" change would reach for first.
    assert engine.absolute_min >= 0.65
    assert engine.absolute_max <= 0.95
    # A 0.50 decision cannot clear the gate in any regime the band allows.
    assert 0.50 < engine.absolute_min

    # And preflight still has a class to report it with.
    assert preflight.CONFIDENCE_BELOW_THRESHOLD in preflight.BLOCKER_CLASSES
    assert (
        idle._BLOCKER_REASONS[preflight.CONFIDENCE_BELOW_THRESHOLD]
        == idle.NO_QUALIFIED_DECISION
    )


def test_the_sub_minimum_cooldown_still_suppresses_repeats():
    preflight.clear_infeasible_memory()
    preflight.note_economically_infeasible(
        "bybit", "DOGE/USDT", 4.0, "minimum 5.0 exceeds spendable 4.0",
        price=0.1, min_notional=5.0, min_amount=1.0, risk_budget=0.0,
    )

    assert preflight.economically_infeasible(
        "bybit", "DOGE/USDT", 4.0,
        price=0.1, min_notional=5.0, min_amount=1.0, risk_budget=0.0,
    )
