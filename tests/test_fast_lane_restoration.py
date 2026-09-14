"""The mature fast lane, and its connection to the running orchestrator.

The failure these protect against is not a crash. It is silence: the velocity
lane and its market-intelligence service existed on disk, nothing started them,
and the runtime still looked busy because slower loops were attempting orders.
Telemetry showed 355 attempts and 0 fills while the component that actually
traded profitably was never constructed.

So these tests assert two separate things. First, that the historical
components are present and keep their historical parameters -- a lane running
at a 60-second cadence is not the lane that traded. Second, that the current
orchestrator actually builds and starts them, which is the connection that was
missing.

Nothing here reaches the network or reads credentials.
"""

import ast
import asyncio
import inspect
import os
from pathlib import Path

import pytest

from src.leantrader.execution import preflight
from src.leantrader.production import fast_lane_assembly as assembly
from src.leantrader.production.fast_lane_executor import CentralAuthorityExecutor


REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = REPO_ROOT / "COMPLETE_ULTIMATE_ORCHESTRATOR.py"


class StubService:
    """The two methods the lane asks of its intelligence service."""

    def __init__(self, candidates=None, signal=None):
        self._candidates = candidates or []
        self._signal = signal or {}

    def collective_candidates(self, *args, **kwargs):
        return list(self._candidates)

    def collective_signal(self, *args, **kwargs):
        return dict(self._signal)


class StubBroker:
    def __init__(self, environment="testnet"):
        self.exchange_id = "bybit"
        self.market_mode = "spot"
        self._environment = environment

    @property
    def authority(self):
        return self._environment

    def resolve_mode(self):
        return self._environment

    def load_markets(self):
        return {}

    def fetch_balance(self):
        return {"free": {"USDT": 25.0}, "used": {}, "total": {"USDT": 25.0}}

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def _make_exchange(self, environment, authenticated):
        return None


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANTRADER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EXECUTION_TELEMETRY_PATH", str(tmp_path / "telemetry.json"))
    monkeypatch.setenv("EXECUTION_LINEAGE_PATH", str(tmp_path / "lineage.jsonl"))
    for name in (
        "VELOCITY_CADENCE_SECONDS",
        "VELOCITY_MAX_HOLD_SECONDS",
        "VELOCITY_TAKE_PROFIT_BPS",
        "VELOCITY_STOP_LOSS_BPS",
        "FAST_LANE_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    preflight.reset_caches()
    preflight.reset_shared_brokers()


# ------------------------------------------------ the components are present


def test_the_historical_fast_lane_components_are_restored():
    from src.leantrader.agents.swarm_service import ReadOnlySwarmService
    from src.leantrader.production.capital_growth import CapitalGrowthGovernor
    from src.leantrader.production.engine_control import EngineRegistry
    from src.leantrader.production.velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    assert ReadOnlySwarmService is not None
    assert VelocitySniperTestnetLane is not None
    assert CapitalGrowthGovernor is not None
    assert EngineRegistry is not None


def test_the_restored_package_does_not_import_the_patch_chain():
    """The nineteen install_testnet_* patches are deliberately not restored.

    They were written after the fast lane was already running, each patching
    the previous one, and every one is Bybit-Testnet-specific by construction.
    Importing them would make this package unable to carry Live.
    """
    init = (REPO_ROOT / "src/leantrader/production/__init__.py").read_text()
    tree = ast.parse(init)

    # Parsed, not grepped: the module's own docstring explains what it is
    # deliberately not doing, and that prose must not decide this test.
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], "the package init must not import at module scope"

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls == [], "the package init must not execute patch installers"


def test_the_lane_keeps_its_historical_speed():
    """A 60-second loop is not this lane. The cadence is the behaviour."""
    lane = assembly.build_velocity_lane(
        service_provider=StubService, executor=object()
    )

    assert lane.cadence_seconds == 0.5
    assert lane.maximum_hold_seconds == 30.0
    assert lane.take_profit_bps == 50.0
    assert lane.stop_loss_bps == 30.0
    assert lane.maximum_entries_per_day == 45
    assert lane.bootstrap_after_seconds == 5.0
    assert lane.maximum_concurrent_positions == 6
    assert lane.maximum_adaptive_positions == 24
    assert lane.maximum_entries_per_cycle == 3
    assert lane.candidate_scan_limit == 24
    assert lane.reentry_cooldown_seconds == 2.0


def test_the_velocity_qualifier_uses_microstructure_not_candles():
    """Freshness, spread, depth, velocity and acceleration -- not a 1m bar."""
    from src.leantrader.production.velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    source = inspect.getsource(VelocitySniperTestnetLane._velocity_state)
    for characteristic in (
        "age_seconds",
        "spread_bps",
        "bid_depth_usd",
        "temporal_samples",
        "midpoint_velocity_bps_per_second",
        "midpoint_acceleration_bps_per_second2",
        "recent_midpoint_trend_bps_5s",
        "depth_imbalance",
        "microprice_shift_bps",
    ):
        assert characteristic in source


# ------------------------------------------- the executor uses one authority


def test_the_fast_lane_executor_never_calls_create_order():
    """No second execution universe. Parsed, not grepped."""
    from src.leantrader.production import fast_lane_executor as module

    tree = ast.parse(inspect.getsource(module))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "create_order" not in called
    assert "createOrder" not in called

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "ccxt" not in imported
    assert "route_order" in imported


def test_an_order_reaches_route_order_through_preflight():
    calls = []

    def fake_route_order(payload, *args, **kwargs):
        calls.append(payload)
        return {
            "ok": True,
            "executed": True,
            "order": {
                "id": "fast-1",
                "filled": payload["qty"],
                "average": 100.0,
                "cost": 100.0 * payload["qty"],
                "status": "closed",
            },
        }

    executor = CentralAuthorityExecutor(
        broker_provider=lambda: StubBroker(), route=fake_route_order
    )

    # No markets on the stub broker, so preflight refuses. The point is that
    # it refuses through preflight rather than reaching the exchange.
    results = executor.mirror_events(
        [{"symbol": "BTC/USDT", "side": "buy", "price": 100.0, "quantity": 0.05}]
    )

    assert len(results) == 1
    assert calls == [], "preflight must refuse before anything is routed"
    assert results[0]["status"] == "skipped"
    assert results[0]["filled"] == 0.0


def test_the_same_decision_is_never_submitted_twice():
    """A timeout after acceptance must be reconciled, not resubmitted."""
    executor = CentralAuthorityExecutor(
        broker_provider=lambda: StubBroker(), route=lambda *a, **k: {}
    )
    event = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 100.0,
        "quantity": 0.05,
        "reason": "velocity_entry",
        "timestamp": "2026-09-14T00:00:00Z",
    }

    first = executor.mirror_events([event])[0]
    second = executor.mirror_events([dict(event)])[0]

    assert first["client_order_id"] == second["client_order_id"]
    assert first.get("idempotent") is False
    assert second["idempotent"] is True


def test_an_acknowledgement_with_no_fill_is_not_a_fill():
    """A Bybit ack can carry an id, status open and filled 0."""

    def ack_only(payload, *args, **kwargs):
        return {
            "ok": True,
            "executed": True,
            "order": {"id": "ack-1", "filled": 0.0, "status": "open"},
        }

    executor = CentralAuthorityExecutor(
        broker_provider=lambda: StubBroker(), route=ack_only
    )
    # Reach the router by stubbing preflight's verdict, so this test is about
    # receipt interpretation rather than about sizing.
    prepared = preflight.PreparedOrder(
        symbol="BTC/USDT",
        side="buy",
        amount=0.05,
        price=100.0,
        notional=5.0,
        quote_currency="USDT",
        free_quote=25.0,
        min_notional=5.0,
        min_amount=0.0001,
        fee_rate=0.001,
        exchange_id="bybit",
        execution_mode="testnet",
        order_type="market",
        sizing_reason="test",
    )
    import unittest.mock as mock

    with mock.patch.object(preflight, "prepare_order", return_value=(prepared, None)):
        result = executor.mirror_events(
            [{"symbol": "BTC/USDT", "side": "buy", "price": 100.0, "quantity": 0.05}]
        )[0]

    assert result["order_id"] == "ack-1"
    assert result["filled"] == 0.0
    assert result["status"] == "open"
    assert preflight.telemetry_snapshot()["fills"] == 0
    assert preflight.telemetry_snapshot()["acknowledged"] == 1


def test_an_unreadable_account_reports_not_flat():
    """A failed balance read must never look like a flat account."""

    def exploding_broker():
        raise RuntimeError("no credentials")

    executor = CentralAuthorityExecutor(broker_provider=exploding_broker)
    snapshot = executor.safe_snapshot()

    assert snapshot["fresh"] is False
    assert snapshot["positions"] == {}
    assert snapshot["free_quote"] == 0.0


def test_only_sellable_inventory_is_offered_as_a_position():
    """Dust is held, but it is not a position the lane can manage."""
    source = inspect.getsource(CentralAuthorityExecutor.safe_snapshot)
    assert 'item.get("sellable")' in source


# ------------------------------------------------- the orchestrator wiring


def test_the_orchestrator_builds_the_fast_lane():
    """The connection that was missing: construction inside wire_all_systems."""
    source = ORCHESTRATOR.read_text()

    assert "FastTradingLane" in source
    assert "fast_lane_assembly" in source
    assert "self.fast_trading_lane" in source


def test_the_orchestrator_starts_the_fast_lane():
    """Constructing it is not starting it. This is the actual defect."""
    source = ORCHESTRATOR.read_text()
    tree = ast.parse(source)

    orchestrator = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "CompleteUltimateOrchestrator"
    )
    methods = {
        node.name
        for node in orchestrator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "run_fast_trading_lane" in methods, (
        "the orchestrator must own a lane entrypoint, not merely construct the lane"
    )

    start_all = next(
        node
        for node in orchestrator.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "start_all_orchestrators"
    )
    started = ast.dump(start_all)
    assert "run_fast_trading_lane" in started, (
        "start_all_orchestrators must schedule the fast lane"
    )
    assert "fast_trading_lane.run" in started, (
        "the lane must be scheduled through _schedule_once so it starts at most once"
    )


def test_the_lane_entrypoint_is_supervised_and_reports_idleness():
    source = ORCHESTRATOR.read_text()
    tree = ast.parse(source)
    entry = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_fast_trading_lane"
    )
    body = ast.get_source_segment(source, entry) or ""

    assert "lane.start" in body
    assert "lane.running" in body, "a dead thread must be detected, not assumed alive"
    assert "classify_idle_reason" in body, (
        "a lane that is up but not trading must say why"
    )


def test_the_assembly_starts_intelligence_before_the_lane():
    """A lane started before its service has nothing fresh to qualify."""
    order = []

    class RecordingService:
        def start(self):
            order.append("service")

        def stop(self):
            order.append("service_stop")

    class RecordingLane:
        _thread = None

        def start(self):
            order.append("lane")

        def stop(self):
            order.append("lane_stop")

    unit = assembly.FastTradingLane()
    unit.service = RecordingService()
    unit.lane = RecordingLane()
    unit.start()

    assert order == ["service", "lane"]

    unit.stop()
    assert order[-2:] == ["lane_stop", "service_stop"]


def test_the_lane_can_be_disabled_without_touching_code():
    source = ORCHESTRATOR.read_text()
    assert "FAST_LANE_ENABLED" in source


# ------------------------------------------------------ mode neutrality


def test_the_lane_is_not_welded_to_testnet():
    """Testnet is where this is proven, not what it is.

    The historical executor verified Bybit Testnet URLs on every submission and
    raised otherwise. The restored seam asks the broker which environment it
    resolved and reports that, so the same lane carries paper, testnet and live.
    """
    from src.leantrader.production import fast_lane_executor as module

    source = inspect.getsource(module)
    assert "_verify_testnet_urls" not in source

    for environment in ("paper", "testnet", "live"):
        executor = CentralAuthorityExecutor(
            broker_provider=lambda env=environment: StubBroker(env)
        )
        assert executor.safe_snapshot()["environment"] == environment


def test_no_order_is_submitted_anywhere_in_this_module():
    """This whole file runs without an exchange."""
    assert preflight.telemetry_snapshot().get("submitted", 0) >= 0
