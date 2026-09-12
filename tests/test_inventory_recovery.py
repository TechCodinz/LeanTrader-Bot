"""Recovery of unmanaged inventory, and the authority its exits must use.

Four assets were sitting in the account with no owner, no cost basis and no
way to be acted on. The coordinator under test adopts them. The two things
these tests exist to pin down are that adoption is not liquidation -- the
default posture sells nothing -- and that when an exit does happen it goes
through preflight and route_order like every other order, never through a
private exchange client.

Nothing here reaches the network.
"""

import pytest

from src.leantrader.execution import inventory as inv
from src.leantrader.execution import lineage, preflight, recovery


QUOTE = "USDT"


def market(symbol, min_amount, min_cost, amount_precision, active=True):
    base, quote = symbol.split("/")
    return {
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "spot": True,
        "active": active,
        "taker": 0.001,
        "precision": {"amount": amount_precision},
        "limits": {"amount": {"min": min_amount}, "cost": {"min": min_cost}},
    }


class StubExchange:
    def __init__(self, amount_step=8, price_step=4):
        self.amount_step = amount_step
        self.price_step = price_step

    def amount_to_precision(self, symbol, amount):
        factor = 10 ** self.amount_step
        return str(int(float(amount) * factor) / factor)

    def price_to_precision(self, symbol, price):
        factor = 10 ** self.price_step
        return str(int(float(price) * factor) / factor)


class StubBroker:
    """The surface preflight uses, with no network and no credentials."""

    def __init__(self, markets, balance, tickers):
        self.exchange_id = "bybit"
        self.market_mode = "spot"
        self._markets = markets
        self._balance = balance
        self._tickers = tickers
        self._exchange = StubExchange()

    @property
    def authority(self):
        return "testnet"

    def resolve_mode(self):
        return "testnet"

    def load_markets(self):
        return self._markets

    def fetch_balance(self):
        return self._balance

    def fetch_ticker(self, symbol):
        return self._tickers.get(symbol, {})

    def _make_exchange(self, environment, authenticated):
        return self._exchange

    # Deliberately absent: create_order. If the coordinator ever tried to
    # place an order itself, this stub would raise AttributeError rather than
    # quietly succeeding.


MARKETS = {
    "ETH/USDT": market("ETH/USDT", 0.0001, 5.0, 4),
    "SOL/USDT": market("SOL/USDT", 0.01, 5.0, 2),
    "CHIP/USDT": market("CHIP/USDT", 1.0, 5.0, 0),
    "CSPR/USDT": market("CSPR/USDT", 1.0, 5.0, 0, active=False),
}

TICKERS = {
    "ETH/USDT": {"last": 2400.0},
    "SOL/USDT": {"last": 140.0},
    "CHIP/USDT": {"last": 0.02},
    "CSPR/USDT": {"last": 0.015},
}

HELD = {"USDT": 12.34, "ETH": 0.0041, "SOL": 0.12, "CHIP": 900.0, "CSPR": 40.0}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("INVENTORY_RECOVERY_PATH", str(tmp_path / "recovery.json"))
    monkeypatch.setenv("EXECUTION_TELEMETRY_PATH", str(tmp_path / "telemetry.json"))
    monkeypatch.setenv("EXECUTION_LINEAGE_PATH", str(tmp_path / "lineage.jsonl"))
    monkeypatch.delenv("INVENTORY_RECOVERY_EXIT_ALLOWLIST", raising=False)
    monkeypatch.delenv("INVENTORY_DUST_THRESHOLD_QUOTE", raising=False)
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    preflight.clear_infeasible_memory()
    yield
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    preflight.clear_infeasible_memory()


def held_balance(**overrides):
    free = dict(HELD)
    free.update(overrides)
    return {"free": free, "used": {}, "total": dict(free)}


def reconciled(balance=None):
    return inv.reconcile(
        balance or held_balance(),
        MARKETS,
        TICKERS,
        venue="bybit",
        quote=QUOTE,
    )


def test_orphans_are_discovered_and_adopted_without_being_sold():
    """Adoption is the default. Selling is not.

    The regression: capital sat unmanaged because nothing owned it, and the
    obvious "fix" -- liquidate anything with no owner -- is the one thing that
    must not happen automatically.
    """
    coordinator = recovery.coordinate(
        reconciled(), venue="bybit", environment="testnet"
    )

    assets = {record.asset for record in coordinator.records()}
    assert assets == {"ETH", "SOL", "CHIP", "CSPR"}

    for record in coordinator.records():
        assert record.state != recovery.EXIT_SUBMITTED
        assert record.exit_order_id == ""
        assert record.recovered_quote == 0.0
        assert record.authorized is False


def test_a_holding_with_no_active_market_is_stranded_not_valued():
    """CSPR's market is inactive: there is no exit to authorize at all."""
    coordinator = recovery.coordinate(reconciled(), venue="bybit")
    cspr = coordinator.record_for("CSPR/USDT")

    assert cspr.state == recovery.RECOVERY_BLOCKED_NOT_EXECUTABLE
    assert cspr.exit_possible is False
    assert cspr.executable_exit_value == 0.0
    assert cspr.mark_to_market_value > 0.0
    assert "not an active market" in cspr.exit_blocked_reason


def test_a_holding_below_the_venue_minimum_is_reported_as_stranded():
    """A legal exit that does not exist is named, not approximated."""
    balance = held_balance(ETH=0.0000009)
    coordinator = recovery.coordinate(reconciled(balance), venue="bybit")
    eth = coordinator.record_for("ETH/USDT")

    assert eth.state in {
        recovery.RECOVERY_BLOCKED_BELOW_MINIMUM,
        recovery.RECOVERY_BLOCKED_NOT_EXECUTABLE,
    }
    assert eth.exit_possible is False


def test_provenance_is_reconstructed_from_venue_trades():
    trades = {
        "SOL/USDT": [
            {
                "side": "buy",
                "amount": 0.12,
                "price": 150.0,
                "cost": 18.0,
                "fee": {"cost": 0.018},
                "order": "order-1",
                "id": "trade-1",
                "timestamp": 1_700_000_000_000,
            }
        ]
    }
    coordinator = recovery.coordinate(
        reconciled(), venue="bybit", trades_by_symbol=trades
    )
    sol = coordinator.record_for("SOL/USDT")

    assert sol.provenance == "RECONSTRUCTED_FROM_VENUE_TRADES"
    assert sol.acquisition_price == pytest.approx(150.0)
    assert sol.acquisition_cost == pytest.approx(18.0)
    assert sol.order_ids == ["order-1"]
    assert sol.trade_ids == ["trade-1"]
    assert sol.acquisition_fees == pytest.approx(0.018)
    # Bought at 150, marked at 140: the loss is reported, not hidden.
    assert sol.unrealized_pnl == pytest.approx((140.0 - 150.0) * 0.12)


def test_an_unknown_cost_basis_stays_unknown():
    """Defaulting the entry to the mark would show zero PnL on a loss."""
    coordinator = recovery.coordinate(reconciled(), venue="bybit")
    chip = coordinator.record_for("CHIP/USDT")

    assert chip.provenance == "UNKNOWN"
    assert chip.acquisition_price is None
    assert chip.acquisition_cost is None
    assert chip.unrealized_pnl is None
    assert coordinator.summary()["holdings_with_unknown_cost_basis"] >= 1


def test_recovery_refuses_to_exit_without_explicit_authorization():
    coordinator = recovery.coordinate(reconciled(), venue="bybit")
    result = coordinator.recover("ETH/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS))

    assert result["ok"] is False
    assert result["reason"] == recovery.RECOVERY_AWAITING_AUTHORIZATION
    assert coordinator.record_for("ETH/USDT").exit_order_id == ""


def test_authorization_cannot_be_granted_where_no_legal_exit_exists():
    coordinator = recovery.coordinate(reconciled(), venue="bybit")

    assert coordinator.authorize_exit("CSPR/USDT", "operator") is False
    assert coordinator.record_for("CSPR/USDT").authorized is False


def test_an_authorized_exit_goes_through_preflight_and_route_order():
    """The one path out: intent -> preflight -> route_order.

    The router is captured rather than stubbed away, so the test asserts on
    what the coordinator actually handed to the single order authority.
    """
    calls = []

    def fake_route_order(payload, *args, **kwargs):
        calls.append(payload)
        return {
            "ok": True,
            "executed": True,
            "order": {
                "id": "recovery-order-1",
                "filled": payload["qty"] if "qty" in payload else payload.get("amount"),
                "average": 2400.0,
                "cost": 9.84,
                "fee": {"cost": 0.00984},
                "status": "closed",
            },
        }

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=fake_route_order
    )
    items = reconciled()
    for record in coordinator.discover(items):
        coordinator.assess_executability(record)
        coordinator.adopt(record)

    assert coordinator.authorize_exit("ETH/USDT", "operator") is True

    result = coordinator.recover(
        "ETH/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS)
    )

    assert result["ok"] is True
    assert len(calls) == 1, "exactly one order reached the central authority"
    payload = calls[0]
    assert payload["side"] == "sell"
    assert payload["symbol"] == "ETH/USDT"
    # Identity travels with the order, so the fill can be traced back.
    assert payload["params"]["intent_id"] == result["intent_id"]
    assert payload["params"]["source_engine"] == "inventory_recovery_coordinator"

    record = coordinator.record_for("ETH/USDT")
    assert record.exit_order_id == "recovery-order-1"
    assert record.state == recovery.EXIT_FILLED


def test_an_exit_sells_the_whole_holding_not_a_fraction_of_a_balance():
    """A close is sized by what is held, not by new-risk allocation.

    Sizing an exit from a fraction of the balance would leave most of the
    trapped capital exactly where it was.
    """
    captured = {}

    def fake_route_order(payload, *args, **kwargs):
        captured.update(payload)
        return {"ok": True, "executed": True, "order": {"id": "o", "filled": 0.0}}

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=fake_route_order
    )
    for record in coordinator.discover(reconciled()):
        coordinator.assess_executability(record)
        coordinator.adopt(record)
    coordinator.authorize_exit("CHIP/USDT", "operator")

    coordinator.recover(
        "CHIP/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS)
    )

    expected = coordinator.record_for("CHIP/USDT").executable_exit_amount
    assert expected == pytest.approx(900.0)
    assert float(captured["qty"]) == pytest.approx(expected)


def test_an_acknowledged_order_with_no_fill_is_not_reported_as_recovered():
    """Acknowledged is not filled, and no proceeds may be claimed from it."""

    def fake_route_order(payload, *args, **kwargs):
        return {
            "ok": True,
            "executed": True,
            "order": {"id": "ack-only", "filled": 0.0, "status": "open"},
        }

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=fake_route_order
    )
    for record in coordinator.discover(reconciled()):
        coordinator.assess_executability(record)
        coordinator.adopt(record)
    coordinator.authorize_exit("ETH/USDT", "operator")
    coordinator.recover(
        "ETH/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS)
    )

    record = coordinator.record_for("ETH/USDT")
    assert record.state == recovery.EXIT_ACKNOWLEDGED
    assert record.recovered_quote == 0.0
    assert record.exit_filled_amount == 0.0


def test_a_router_refusal_is_not_a_recovery():
    def refusing_route_order(payload, *args, **kwargs):
        return {"ok": False, "executed": False, "error": "insufficient balance"}

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=refusing_route_order
    )
    for record in coordinator.discover(reconciled()):
        coordinator.assess_executability(record)
        coordinator.adopt(record)
    coordinator.authorize_exit("ETH/USDT", "operator")

    result = coordinator.recover(
        "ETH/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS)
    )

    assert result["ok"] is False
    assert coordinator.record_for("ETH/USDT").state == recovery.RECOVERY_EXIT_REFUSED
    assert coordinator.record_for("ETH/USDT").recovered_quote == 0.0


def test_recovery_is_complete_only_when_the_balance_agrees():
    def fake_route_order(payload, *args, **kwargs):
        return {
            "ok": True,
            "executed": True,
            "order": {
                "id": "o-1",
                "filled": 0.0041,
                "average": 2400.0,
                "cost": 9.84,
                "status": "closed",
            },
        }

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=fake_route_order
    )
    for record in coordinator.discover(reconciled()):
        coordinator.assess_executability(record)
        coordinator.adopt(record)
    coordinator.authorize_exit("ETH/USDT", "operator")
    coordinator.recover(
        "ETH/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS)
    )

    # Balance still shows the asset: the order filled, the account has not
    # caught up, and the recovery is not finished.
    still_there = coordinator.reconcile_recovery("ETH/USDT", reconciled())
    assert still_there.state == recovery.EXIT_FILLED

    gone = coordinator.reconcile_recovery(
        "ETH/USDT", reconciled(held_balance(ETH=0.0))
    )
    assert gone.state == recovery.RECOVERED_RECONCILED


def test_recovery_state_survives_a_process_boundary():
    coordinator = recovery.coordinate(reconciled(), venue="bybit")
    assert coordinator.persist() is True

    fresh = recovery.InventoryRecoveryCoordinator()
    assert fresh.load() == 4
    assert {r.asset for r in fresh.records()} == {"ETH", "SOL", "CHIP", "CSPR"}

    snapshot = recovery.read_snapshot()
    assert snapshot["schema_version"] == recovery.SCHEMA_VERSION
    assert snapshot["summary"]["adopted"] == 4
    assert "source_run_id" in snapshot and "generated_at" in snapshot


def test_the_exit_allowlist_is_empty_by_default():
    """The default posture is not to sell. That default is the safety."""
    assert recovery.exit_allowlist() == frozenset()
    coordinator = recovery.coordinate(reconciled(), venue="bybit")
    assert coordinator.summary()["awaiting_authorization"] == 3


def test_an_allowlisted_asset_is_authorized_on_adoption(monkeypatch):
    monkeypatch.setenv("INVENTORY_RECOVERY_EXIT_ALLOWLIST", "eth , sol")
    coordinator = recovery.coordinate(reconciled(), venue="bybit")

    assert coordinator.record_for("ETH/USDT").state == recovery.EXIT_AUTHORIZED
    assert coordinator.record_for("SOL/USDT").authorized is True
    assert coordinator.record_for("CHIP/USDT").authorized is False


def test_an_actively_managed_position_is_not_adopted():
    """Adopting a position an engine already owns would give it two owners."""
    known = {
        "ETH/USDT": {
            "owner": "ultra_core",
            "owner_alive": True,
            "entry_price": 2300.0,
        }
    }
    items = inv.reconcile(
        held_balance(), MARKETS, TICKERS, known_positions=known,
        venue="bybit", quote=QUOTE,
    )
    coordinator = recovery.coordinate(items, venue="bybit")

    assert coordinator.record_for("ETH/USDT") is None
    assert "SOL" in {record.asset for record in coordinator.records()}


def test_the_recovery_exit_is_journalled_as_lineage():
    """A recovery exit is traceable end to end like any other order."""

    def fake_route_order(payload, *args, **kwargs):
        return {
            "ok": True,
            "executed": True,
            "order": {"id": "o-1", "filled": 0.0041, "average": 2400.0, "cost": 9.84},
        }

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=fake_route_order
    )
    for record in coordinator.discover(reconciled()):
        coordinator.assess_executability(record)
        coordinator.adopt(record)
    coordinator.authorize_exit("ETH/USDT", "operator")
    result = coordinator.recover(
        "ETH/USDT", broker=StubBroker(MARKETS, held_balance(), TICKERS)
    )

    trail = lineage.lineage_for(result["correlation_id"])
    stages = [row["stage"] for row in trail]
    assert lineage.PREFLIGHT_PREPARED in stages
    assert lineage.ROUTE_ORDER_SUBMITTED in stages
    assert lineage.FILL_RECORDED in stages
    assert all(
        row["source_engine"] == "inventory_recovery_coordinator" for row in trail
    )


# -------------------------------------- reclaimable capital is not cash


def test_reclaimable_capital_is_never_offered_as_spendable():
    """An exit that has not filled has returned nothing."""
    items = reconciled()
    capital = inv.spendable_capital(12.34, items)

    assert capital["cash_spendable_now"] == pytest.approx(12.34)
    assert capital["reclaimable_capital"] > 0
    assert capital["cash_spendable_now"] < capital["reclaimable_capital"]
    # The legacy alias must track cash, never the hypothetical proceeds.
    assert capital["spendable_quote"] == capital["cash_spendable_now"]
    assert capital["capital_recovered"] == 0.0
    for name in (
        "cash_spendable_now",
        "capital_committed",
        "capital_exit_eligible",
        "capital_exit_pending",
        "capital_recovered",
    ):
        assert name in capital


def test_a_new_buy_is_sized_from_cash_not_from_orphan_inventory():
    """The orchestrator's budget must read the cash field, by name."""
    import inspect

    import EXECUTION_ORCHESTRATOR as orchestrator

    source = inspect.getsource(orchestrator.ExecutionOrchestrator._spendable_budget)
    assert "cash_spendable_now" in source
    # It must not read any reclaimable/exit-proceeds field as if it were cash.
    for hypothetical in (
        '["reclaimable_capital"]',
        '["capital_exit_eligible"]',
        '["total_reclaimable_capital"]',
    ):
        assert hypothetical not in source


def test_the_running_orchestrator_adopts_orphans():
    """Adoption has to happen in the process that reads the account.

    A coordinator that only ever runs in a test is not a coordinator.
    """
    import inspect

    import EXECUTION_ORCHESTRATOR as orchestrator

    source = inspect.getsource(orchestrator.ExecutionOrchestrator._spendable_budget)
    assert "inventory_recovery.coordinate" in source


def test_totals_reconcile_against_their_rows():
    items = reconciled()
    summary = inv.summarize(items)

    assert inv.valuation_invariants(items, summary) == []

    mark_to_market = sum(item.raw_quote_value for item in items)
    assert summary["total_mark_to_market_value"] == pytest.approx(mark_to_market)
    executable = sum(
        item.executable_quote_value
        for item in items
        if item.included_in_inventory_total
    )
    assert summary["total_executable_inventory_value"] == pytest.approx(executable)


def test_the_valuation_invariant_catches_a_broken_total():
    """The invariant has to be able to fail, or it proves nothing."""
    items = reconciled()
    summary = inv.summarize(items)
    summary["total_mark_to_market_value"] += 1.0

    problems = inv.valuation_invariants(items, summary)
    assert problems
    assert any("mark" in problem.lower() for problem in problems)


# ------------------------------------------- the exit path has no back door


def test_the_recovery_module_never_calls_create_order():
    """One final order authority. No side-channel exchange client.

    Parsed rather than grepped, so the module's own prose about what it must
    not do cannot make this pass or fail for the wrong reason.
    """
    import ast
    import inspect

    from src.leantrader.execution import recovery as module

    tree = ast.parse(inspect.getsource(module))

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "create_order" not in called
    assert "createOrder" not in called
    assert not {"fetch_balance", "load_markets"} & called, (
        "the coordinator reads the account through the shared broker, not "
        "through an exchange client of its own"
    )

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "ccxt" not in imported
    # route_order is reached by import, not reimplemented.
    assert "route_order" in imported


def test_a_recovery_exit_still_passes_every_preflight_check():
    """Recovery is not a bypass: the same checks refuse the same things."""
    calls = []

    def fake_route_order(payload, *args, **kwargs):
        calls.append(payload)
        return {"ok": True, "order": {"id": "x", "filled": 0.0}}

    coordinator = recovery.InventoryRecoveryCoordinator(
        venue="bybit", environment="testnet", route=fake_route_order
    )
    for record in coordinator.discover(reconciled()):
        coordinator.assess_executability(record)
        coordinator.adopt(record)
    coordinator.authorize_exit("ETH/USDT", "operator")

    # A venue that does not list the market refuses at preflight, and the
    # order never reaches route_order.
    broker = StubBroker({}, held_balance(), TICKERS)
    result = coordinator.recover("ETH/USDT", broker=broker)

    assert result["ok"] is False
    assert calls == []
    assert coordinator.record_for("ETH/USDT").state == recovery.RECOVERY_EXIT_REFUSED
