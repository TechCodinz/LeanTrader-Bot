"""Signals are produced; this traces whether they arrive.

The runtime shows substantial signal generation and publication without
matching preflight or submission events. Each boundary in the chain is now
counted, so a run that produces signals and no orders can be attributed to a
specific hop rather than guessed at.

    scalper -> data hub -> signal queue -> decision engine -> alert queue
      -> execution orchestrator -> preflight -> route_order

No order is submitted and no venue is contacted.
"""

import asyncio

import pytest

from src.leantrader.execution import preflight
from src.leantrader.universe.registry import universe


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_TELEMETRY_PATH", str(tmp_path / "t.json"))
    monkeypatch.setenv("UNIVERSE_SNAPSHOT_PATH", str(tmp_path / "snap.json"))
    universe.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    universe.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()


def signal(symbol="DOGE/USDT", side="buy", confidence=0.9):
    return {
        "symbol": symbol,
        "side": side,
        "confidence": confidence,
        "strategy": "scalp",
        "timeframe": "5m",
        "data": {"symbol": symbol, "price": 0.14, "confidence": confidence},
    }


# ----------------------------------- 15. signals reach the decision consumer


def test_a_published_signal_reaches_the_signal_queue():
    from COMPLETE_UNIFIED_ORCHESTRATOR import CentralDataHub

    hub = CentralDataHub()
    asyncio.run(hub.publish_signal(signal()))

    assert hub.signal_queue.qsize() == 1
    assert len(hub.recent_signals) == 1


def test_publication_is_counted_at_the_hub():
    from COMPLETE_UNIFIED_ORCHESTRATOR import CentralDataHub

    hub = CentralDataHub()
    for _ in range(3):
        asyncio.run(hub.publish_signal(signal()))

    snapshot = preflight.telemetry_snapshot()
    assert snapshot.get("signals_published") == 3
    assert universe.telemetry()["signals_generated"] == 3


def test_the_decision_engine_consumes_what_the_hub_published():
    """The hop the Telegram monitor used to win by draining the queue."""
    from COMPLETE_UNIFIED_ORCHESTRATOR import CentralDataHub, UnifiedDecisionEngine

    hub = CentralDataHub()
    for symbol in ("DOGE/USDT", "PEPE/USDT", "BONK/USDT"):
        asyncio.run(hub.publish_signal(signal(symbol)))

    class Brain:
        def engineer_features(self, data):
            return {}

    class Swarm:
        async def collective_decision(self):
            return None

    engine = UnifiedDecisionEngine(hub, Brain(), Swarm(), None)

    async def one_pass():
        engine.decision_active = True
        task = asyncio.create_task(engine.run_decision_loop())
        await asyncio.sleep(0.05)
        engine.decision_active = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_pass())

    snapshot = preflight.telemetry_snapshot()
    assert snapshot.get("signals_consumed") == 3
    assert snapshot.get("decisions_created") == 3
    assert hub.alert_queue.qsize() == 3


def test_a_hold_is_counted_as_a_rejection_not_a_decision():
    from COMPLETE_UNIFIED_ORCHESTRATOR import CentralDataHub, UnifiedDecisionEngine

    hub = CentralDataHub()
    asyncio.run(hub.publish_signal(signal(side="hold")))

    class Brain:
        def engineer_features(self, data):
            return {}

    class Swarm:
        async def collective_decision(self):
            return None

    engine = UnifiedDecisionEngine(hub, Brain(), Swarm(), None)

    async def one_pass():
        engine.decision_active = True
        task = asyncio.create_task(engine.run_decision_loop())
        await asyncio.sleep(0.05)
        engine.decision_active = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_pass())

    snapshot = preflight.telemetry_snapshot()
    assert snapshot.get("decisions_rejected") == 1
    assert snapshot.get("decisions_created", 0) == 0


# --------------------------------------- 16. valid decisions reach preflight


def test_a_valid_decision_reaches_preflight(monkeypatch):
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator

    reached = []

    class Hub:
        def __init__(self):
            self.trades = []

        async def publish_trade(self, trade):
            self.trades.append(trade)

    def fake_prepare(intent, **kwargs):
        reached.append(intent)
        return None, preflight.Blocked(
            preflight.BELOW_MIN_NOTIONAL, "balance too small", "sizing"
        )

    monkeypatch.setattr(preflight, "prepare_order", fake_prepare)

    orchestrator = ExecutionOrchestrator(Hub(), {}, None, None, mode="testnet")
    result = asyncio.run(
        orchestrator.execute_trade("DOGE/USDT", "buy", 0.9, signal())
    )

    assert reached, "the decision never reached preflight"
    assert reached[0]["symbol"] == "DOGE/USDT"
    assert result["blocker"] == preflight.BELOW_MIN_NOTIONAL


def test_a_low_confidence_decision_is_counted_with_its_reason():
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator

    class Hub:
        async def publish_trade(self, trade):
            pass

    orchestrator = ExecutionOrchestrator(Hub(), {}, None, None, mode="testnet")
    asyncio.run(
        orchestrator.process_decision(
            {"signal": signal(), "action": "buy", "confidence": 0.10}
        )
    )

    blockers = preflight.telemetry_snapshot().get("blockers") or {}
    assert preflight.CONFIDENCE_BELOW_THRESHOLD in blockers
    assert blockers[preflight.CONFIDENCE_BELOW_THRESHOLD]["count"] == 1


# ------------------------------------ 17. every rejection carries a reason


@pytest.mark.parametrize(
    "intent,expected",
    [
        ({"symbol": "DOGE/USDT", "side": "hold"}, preflight.INVALID_INTENT),
        ({"symbol": "???", "side": "buy"}, preflight.SYMBOL_NOT_NORMALIZED),
    ],
)
def test_every_preflight_rejection_names_a_structured_reason(intent, expected):
    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return {}

        def fetch_balance(self):
            return {}

        def fetch_ticker(self, symbol):
            return {}

        def _make_exchange(self, environment, authenticated):
            return None

    prepared, blocked = preflight.prepare_order(intent, broker=Broker())

    assert prepared is None
    assert blocked.blocker == expected
    assert blocked.stage
    assert blocked.blocker in preflight.BLOCKER_CLASSES


def test_rejections_are_counted_by_reason_and_survive_a_reread():
    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return {}

        def fetch_balance(self):
            return {}

        def fetch_ticker(self, symbol):
            return {}

        def _make_exchange(self, environment, authenticated):
            return None

    for _ in range(4):
        preflight.prepare_order({"symbol": "X/USDT", "side": "hold"}, broker=Broker())

    snapshot = preflight.telemetry_snapshot()
    assert snapshot["blockers"][preflight.INVALID_INTENT]["count"] == 4
    assert snapshot["attempts"] == 4
    assert snapshot["prepared"] == 0


def test_the_funnel_distinguishes_submitted_acknowledged_and_filled():
    """A rejected intent is not a fill; an acknowledgement is not a fill."""
    preflight.record_event("attempts", 10)
    preflight.record_event("prepared", 6)
    preflight.record_event("submitted", 4)
    preflight.record_event("acknowledged", 3)
    preflight.record_event("fills", 2)
    preflight.record_event("closes", 1)

    snapshot = preflight.telemetry_snapshot()

    assert snapshot["attempts"] > snapshot["prepared"] > snapshot["submitted"]
    assert snapshot["submitted"] > snapshot["acknowledged"] > snapshot["fills"]
    assert snapshot["fills"] > snapshot["closes"]


def test_every_pipeline_boundary_has_a_counter():
    """The chain must be attributable hop by hop."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    recorded = set()

    for name in (
        "COMPLETE_UNIFIED_ORCHESTRATOR.py",
        "EXECUTION_ORCHESTRATOR.py",
        "src/leantrader/universe/registry.py",
        "src/leantrader/execution/preflight.py",
    ):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"record_event", "_record_pipeline"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                recorded.add(node.args[0].value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"_record_pipeline", "record_event"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                recorded.add(node.args[0].value)

    required = {
        "signals_published",
        "signals_consumed",
        "decisions_created",
        "decisions_rejected",
        "decisions_consumed",
        "candidates_created",
        "candidates_ranked",
        "candidates_execution_eligible",
        "attempts",
        "prepared",
        "submitted",
        "acknowledged",
        "fills",
        "positions_opened",
        "close_orders_submitted",
        "close_orders_filled",
        "reconciled_cycles",
    }

    missing = required - recorded
    assert not missing, f"pipeline boundaries with no counter: {sorted(missing)}"
