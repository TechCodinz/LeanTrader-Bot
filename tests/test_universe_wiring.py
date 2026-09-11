"""The engines must take their pairs from the registry, not from a literal.

Discovery already worked. What was missing was the connection: trading_universe
was read in four places and assigned in none, so each engine fell back to
whatever was written inside it -- three majors in the swarm, six in the
continuous trader, an empty list in the micro bot. These check the connection
exists, and that having a broad universe never turns into permission to trade
a market on a venue we are not authenticated against.

Offline: a stub registry stands in for discovery, and route_order is patched.
"""

import ast
import asyncio
import pathlib

import pytest

from src.leantrader.execution import preflight
from src.leantrader.universe.registry import universe as registry

ROOT = pathlib.Path(__file__).resolve().parent.parent


def ticker(last=1.0, quote_volume=5_000_000.0, change=4.0):
    return {
        "last": last,
        "quoteVolume": quote_volume,
        "percentage": change,
        "bid": last * 0.999,
        "ask": last * 1.001,
    }


def meta(spot=True, active=True, min_cost=5.0):
    return {
        "spot": spot,
        "active": active,
        "taker": 0.001,
        "limits": {"amount": {"min": 0.0}, "cost": {"min": min_cost}},
        "precision": {"amount": 8, "price": 6},
    }


BYBIT_TICKERS = {
    "BTC/USDT": ticker(last=64_000.0, quote_volume=900_000_000.0, change=1.2),
    "DOGE/USDT": ticker(last=0.14, quote_volume=30_000_000.0, change=5.0),
    "PEPE/USDT": ticker(last=0.00000812, quote_volume=12_000_000.0, change=8.0),
    "BONK/USDT": ticker(last=0.00002140, quote_volume=6_000_000.0, change=6.0),
    "FLOKI/USDT": ticker(last=0.00014, quote_volume=3_000_000.0, change=4.5),
}
BYBIT_MARKETS = {s: meta() for s in BYBIT_TICKERS}


@pytest.fixture(autouse=True)
def _clean_registry(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "EXECUTION_TELEMETRY_PATH", str(tmp_path / "telemetry.json")
    )
    registry.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    registry.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()


@pytest.fixture
def populated():
    registry.ingest_venue("bybit", BYBIT_TICKERS, markets=BYBIT_MARKETS)
    # Something only another venue lists.
    registry.ingest_venue("binance", {"ONLYBIN/USDT": ticker()}, markets={})
    registry.apply_execution_venue("bybit", BYBIT_MARKETS)
    return registry


# ------------------------------------ 1. the micro bot does not stay at zero


def test_micro_bot_starts_empty_without_a_universe(monkeypatch):
    """The honest baseline: no discovery, no pairs -- and it says so."""
    import MICRO_TRADING_BOT

    monkeypatch.setattr(
        MICRO_TRADING_BOT.MICRO_GATE_BOT, "check_gate_balance", lambda self: 0.0
    )
    bot = MICRO_TRADING_BOT.MICRO_GATE_BOT()

    assert bot.crypto_pairs == []
    assert bot.universe_source in {"unset", "registry(capital=0.0000)"}


def test_micro_bot_populates_from_the_registry(populated, monkeypatch):
    """This is the defect: it initialised to [] and nothing ever filled it."""
    import MICRO_TRADING_BOT

    monkeypatch.setattr(
        MICRO_TRADING_BOT.MICRO_GATE_BOT,
        "check_gate_balance",
        lambda self: 13.95644171,
    )
    bot = MICRO_TRADING_BOT.MICRO_GATE_BOT()

    assert len(bot.crypto_pairs) > 0, "micro bot is still stuck at zero pairs"
    assert "registry" in bot.universe_source


def test_micro_bot_accepts_the_orchestrators_universe(populated, monkeypatch):
    import MICRO_TRADING_BOT

    monkeypatch.setattr(
        MICRO_TRADING_BOT.MICRO_GATE_BOT, "check_gate_balance", lambda self: 13.95
    )
    bot = MICRO_TRADING_BOT.MICRO_GATE_BOT()

    count = bot.set_universe(["doge/usdt", "PEPE-USDT", "garbage", ""])

    assert count == 2
    assert bot.crypto_pairs == ["DOGE/USDT", "PEPE/USDT"]
    assert bot.universe_source == "orchestrator"


def test_micro_bot_universe_changes_as_discovery_changes(populated, monkeypatch):
    import MICRO_TRADING_BOT

    monkeypatch.setattr(
        MICRO_TRADING_BOT.MICRO_GATE_BOT, "check_gate_balance", lambda self: 13.95
    )
    bot = MICRO_TRADING_BOT.MICRO_GATE_BOT()
    first = list(bot.crypto_pairs)

    registry.ingest_venue(
        "bybit",
        {"NEWLIST/USDT": ticker(last=0.002, quote_volume=40_000_000.0, change=12.0)},
        markets={"NEWLIST/USDT": meta()},
    )
    registry.apply_execution_venue(
        "bybit", {**BYBIT_MARKETS, "NEWLIST/USDT": meta()}
    )
    bot.refresh_universe(balance=13.95)

    assert "NEWLIST/USDT" in bot.crypto_pairs
    assert bot.crypto_pairs != first, "the universe is static, not dynamic"


def test_micro_bot_universe_includes_non_majors(populated, monkeypatch):
    import MICRO_TRADING_BOT

    monkeypatch.setattr(
        MICRO_TRADING_BOT.MICRO_GATE_BOT, "check_gate_balance", lambda self: 13.95
    )
    bot = MICRO_TRADING_BOT.MICRO_GATE_BOT()

    assert any(
        s in bot.crypto_pairs for s in ("PEPE/USDT", "BONK/USDT", "FLOKI/USDT")
    ), f"only majors reached the micro bot: {bot.crypto_pairs}"


def test_micro_bot_refuses_a_market_the_execution_venue_does_not_list(
    populated, monkeypatch
):
    import MICRO_TRADING_BOT

    monkeypatch.setattr(
        MICRO_TRADING_BOT.MICRO_GATE_BOT, "check_gate_balance", lambda self: 13.95
    )
    bot = MICRO_TRADING_BOT.MICRO_GATE_BOT()

    bot.set_universe(["ONLYBIN/USDT", "DOGE/USDT"])

    assert "ONLYBIN/USDT" not in bot.crypto_pairs
    assert "DOGE/USDT" in bot.crypto_pairs


# ---------------------- 2. the continuous trader is not six hardcoded majors


HARDCODED_SIX = {
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "ADA/USDT",
    "SOL/USDT",
    "MATIC/USDT",
}


def _continuous_orchestrator():
    from ultra_continuous_trading import UltraContinuousTradingOrchestrator

    return UltraContinuousTradingOrchestrator.__new__(
        UltraContinuousTradingOrchestrator
    )


def test_continuous_trader_no_longer_returns_the_hardcoded_majors(populated):
    import logging

    orchestrator = _continuous_orchestrator()
    orchestrator.logger = logging.getLogger("test")
    orchestrator.daily_balance = 13.95

    for timeframe in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
        symbols = asyncio.run(orchestrator._get_symbols_for_timeframe(timeframe))
        assert symbols, f"{timeframe} produced nothing"
        assert set(symbols) != HARDCODED_SIX
        assert set(symbols) <= set(registry.symbols(executable_only=True))


def test_continuous_trader_sees_non_major_markets(populated):
    import logging

    orchestrator = _continuous_orchestrator()
    orchestrator.logger = logging.getLogger("test")
    orchestrator.daily_balance = 13.95

    symbols = asyncio.run(orchestrator._get_symbols_for_timeframe("H1"))
    assert any(s in symbols for s in ("PEPE/USDT", "BONK/USDT", "FLOKI/USDT"))


def test_continuous_trader_breadth_is_bounded_per_timeframe(populated):
    """A 1m cycle must not try to analyse the whole universe."""
    import logging
    from ultra_continuous_trading import UltraContinuousTradingOrchestrator

    for index in range(200):
        registry.ingest_venue(
            "bybit",
            {f"SYM{index}/USDT": ticker(last=1.0 + index)},
            markets={f"SYM{index}/USDT": meta()},
        )
    registry.apply_execution_venue(
        "bybit",
        {m.symbol: meta() for m in registry.all_markets()},
    )

    orchestrator = _continuous_orchestrator()
    orchestrator.logger = logging.getLogger("test")
    orchestrator.daily_balance = 1000.0

    fast = asyncio.run(orchestrator._get_symbols_for_timeframe("M1"))
    slow = asyncio.run(orchestrator._get_symbols_for_timeframe("D1"))

    breadth = UltraContinuousTradingOrchestrator.TIMEFRAME_BREADTH
    assert len(fast) <= breadth["M1"]
    assert len(slow) <= breadth["D1"]
    assert len(fast) < len(slow), "a fast timeframe should take fewer symbols"


def test_continuous_trader_analyses_nothing_rather_than_falling_back(monkeypatch):
    """An empty registry must not resurrect a hardcoded list."""
    import logging

    orchestrator = _continuous_orchestrator()
    orchestrator.logger = logging.getLogger("test")
    orchestrator.daily_balance = 13.95
    orchestrator.universe = []

    symbols = asyncio.run(orchestrator._get_symbols_for_timeframe("H1"))
    assert symbols == []


def test_no_hardcoded_major_list_remains_in_the_timeframe_selector():
    """Checked against the source: the literals must be gone, not shadowed."""
    source = (ROOT / "ultra_continuous_trading.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_get_symbols_for_timeframe"
        ):
            body = ast.get_source_segment(source, node) or ""
            for major in ("'BTC/USDT'", "'ETH/USDT'", "'BNB/USDT'", "'MATIC/USDT'"):
                assert major not in body, f"{major} is still hardcoded here"
            break
    else:
        pytest.fail("_get_symbols_for_timeframe not found")


def test_the_swarm_no_longer_hardcodes_three_majors():
    source = (ROOT / "ultra_swarm_consciousness.py").read_text()
    assert "self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']" not in source

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_signal_collection_loop"
        ):
            body = ast.get_source_segment(source, node) or ""
            assert "next_work_items" in body, (
                "the swarm must take work from the registry, not iterate a list"
            )
            break
    else:
        pytest.fail("_signal_collection_loop not found")


# -------------------------------- 7 & 8. one execution authority, no bypasses


def test_every_engine_order_path_goes_through_route_order():
    """No engine may call create_order on an exchange client directly."""
    engines = (
        "MICRO_TRADING_BOT.py",
        "REAL_PROFIT_BOT.py",
        "EXECUTION_ORCHESTRATOR.py",
        "tools/callback_executor.py",
    )

    for name in engines:
        source = (ROOT / name).read_text()
        tree = ast.parse(source)

        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)

        for forbidden in (
            "create_order",
            "create_market_order",
            "create_limit_order",
            "create_market_buy_order",
            "create_market_sell_order",
        ):
            assert forbidden not in called, f"{name} calls {forbidden} directly"

        assert "route_order" in source, f"{name} has no path to the router"


def test_the_raw_exchange_bypass_guard_actually_blocks(monkeypatch):
    """Exercised, not grepped: the guard must refuse a raw create_order."""
    import logging
    import router as router_module

    class RawExchange:
        def create_order(self, *args, **kwargs):
            raise AssertionError("a raw exchange order reached the exchange")

        def create_market_order(self, *args, **kwargs):
            raise AssertionError("a raw exchange order reached the exchange")

    client = router_module.ExchangeRouter.__new__(router_module.ExchangeRouter)
    client.id = "bybit"
    client.testnet = True
    client.live = False
    client.ex = RawExchange()

    client.apply_runtime_order_block(logging.getLogger("test"))

    receipt = client.ex.create_order("BTC/USDT", "market", "buy", 0.001)

    assert receipt["ok"] is False
    assert receipt["executed"] is False
    assert "bypass_blocked" in receipt["error"]

    # The original is preserved rather than destroyed, so the guard is
    # reversible and the client stays usable for market data.
    assert hasattr(client.ex, "_leantrader_raw_create_order")


def test_the_router_is_the_only_module_that_calls_ccxt_create_order():
    """Repo-wide: the authority is one function, in one place."""
    allowed = {
        "src/leantrader/execution/broker_ccxt.py",  # the authority itself
        "router.py",                                 # installs the guard
    }
    offenders = []

    for path in ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".git", "__pycache__", "node_modules", "tests"}:
            continue
        relative = str(path.relative_to(ROOT))
        if relative in allowed:
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_order"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"exchange", "ex", "client"}
            ):
                offenders.append(f"{relative}:{node.lineno}")

    assert not offenders, "raw exchange order calls outside the router: " + ", ".join(
        offenders
    )


# ----------- 5 & 11. breadth of discovery is never breadth of execution


def test_a_foreign_market_cannot_be_prepared_for_the_execution_venue(populated):
    """End to end: discovered on Binance, refused by preflight on Bybit."""

    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return BYBIT_MARKETS

        def fetch_balance(self):
            return {"free": {"USDT": 13.95644171}}

        def fetch_ticker(self, symbol):
            return {"last": 1.0}

        def _make_exchange(self, environment, authenticated):
            return None

    prepared, blocked = preflight.prepare_order(
        {"symbol": "ONLYBIN/USDT", "side": "buy", "price": 1.0, "confidence": 0.9},
        broker=Broker(),
    )

    assert prepared is None
    assert blocked.blocker == preflight.MARKET_NOT_LISTED


def test_a_listed_market_is_prepared_with_small_account_sizing(populated):
    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return BYBIT_MARKETS

        def fetch_balance(self):
            return {"free": {"USDT": 13.95644171}}

        def fetch_ticker(self, symbol):
            return {"last": 0.14}

        def _make_exchange(self, environment, authenticated):
            return None

    prepared, blocked = preflight.prepare_order(
        {"symbol": "DOGE/USDT", "side": "buy", "price": 0.14, "confidence": 0.85},
        broker=Broker(),
    )

    assert blocked is None
    assert prepared.notional >= prepared.min_notional
    assert prepared.notional <= 13.95644171
    assert prepared.amount > 0


# --------------------- 9 & 10. refusals and closes, still honest downstream


def test_a_refusal_still_never_becomes_a_position(monkeypatch, tmp_path):
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator
    import EXECUTION_ORCHESTRATOR as module

    class Hub:
        def __init__(self):
            self.trades = []

        async def publish_trade(self, trade):
            self.trades.append(trade)

    hub = Hub()
    orchestrator = ExecutionOrchestrator(hub, {}, None, None, mode="testnet")

    monkeypatch.setattr(
        preflight,
        "prepare_order",
        lambda intent, **kw: (
            preflight.PreparedOrder(
                symbol="DOGE/USDT",
                side="buy",
                amount=40.0,
                price=0.14,
                order_type="market",
                exchange_id="bybit",
                execution_mode="testnet",
                notional=5.6,
                quote_currency="USDT",
                free_quote=13.95,
                min_notional=5.0,
                min_amount=0.0,
                fee_rate=0.001,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        module,
        "route_order",
        lambda payload: {"ok": False, "error": "bybit InsufficientFunds"},
    )

    result = asyncio.run(
        orchestrator.execute_trade("DOGE/USDT", "buy", 0.9, {"symbol": "DOGE/USDT"})
    )

    assert result["ok"] is False
    assert hub.trades == []
    assert orchestrator.risk_manager.open_positions == {}


def test_a_close_still_requires_an_authenticated_closing_lifecycle(monkeypatch):
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator
    import EXECUTION_ORCHESTRATOR as module

    class Hub:
        def __init__(self):
            self.trades = []

        async def publish_trade(self, trade):
            self.trades.append(trade)

    hub = Hub()
    orchestrator = ExecutionOrchestrator(hub, {}, None, None, mode="testnet")
    orchestrator.risk_manager.record_position("DOGE/USDT", "buy", 40.0, 0.14)

    monkeypatch.setattr(
        module, "route_order", lambda payload: {"ok": False, "error": "rejected"}
    )

    assert (
        asyncio.run(orchestrator.close_position("DOGE/USDT", 0.15, "take_profit"))
        is None
    )
    assert "DOGE/USDT" in orchestrator.risk_manager.open_positions
    assert hub.trades == []
    assert orchestrator.total_profit == 0.0


# ------------------------------- the orchestrator actually publishes the list


def test_the_orchestrator_publishes_the_universe_into_the_engines():
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

    orchestrator = CompleteUltimateOrchestrator.__new__(CompleteUltimateOrchestrator)

    class Engine:
        def __init__(self):
            self.crypto_pairs = []

        def set_universe(self, symbols):
            self.crypto_pairs = list(symbols)
            return len(symbols)

    class Core:
        pairs = []

    class Swarm:
        symbols = []
        agent_count = 100

    orchestrator.real_profit_bot = Engine()
    orchestrator.micro_wallet_grower = Engine()
    orchestrator.ultra_core = Core()
    orchestrator.swarm = Swarm()
    orchestrator.ai_systems = {}

    symbols = ["DOGE/USDT", "PEPE/USDT", "BONK/USDT"]
    updated = orchestrator.publish_trading_universe(symbols)

    assert updated >= 4
    assert orchestrator.trading_universe == symbols
    assert orchestrator.real_profit_bot.crypto_pairs == symbols
    assert orchestrator.micro_wallet_grower.crypto_pairs == symbols
    assert orchestrator.ultra_core.pairs == symbols
    assert orchestrator.swarm.symbols == symbols


def test_publishing_an_empty_universe_does_not_wipe_the_engines():
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

    orchestrator = CompleteUltimateOrchestrator.__new__(CompleteUltimateOrchestrator)
    orchestrator.trading_universe = ["DOGE/USDT"]

    assert orchestrator.publish_trading_universe([]) == 0
    assert orchestrator.trading_universe == ["DOGE/USDT"]


def test_the_universe_loop_is_scheduled_from_both_entry_points():
    source = (ROOT / "COMPLETE_ULTIMATE_ORCHESTRATOR.py").read_text()
    assert source.count("'universe_maintenance_loop', self.universe_maintenance_loop") == 2


# ------------------------------------ public discovery breadth, honestly


def test_public_discovery_covers_the_named_venues():
    from DYNAMIC_MARKET_SCANNER import DynamicMarketScanner

    scanner = DynamicMarketScanner({}, None)
    venues = set(scanner._configured_public_venues())

    for venue in ("bybit", "binance", "okx", "kucoin", "gateio", "mexc", "bitget"):
        assert venue in venues, f"{venue} is not observed for public discovery"


def test_public_discovery_needs_no_credentials(monkeypatch):
    """Ticker and market data are public; breadth must not depend on keys."""
    import DYNAMIC_MARKET_SCANNER as module

    built = []

    class FakeClient:
        def __init__(self, config):
            built.append(config)
            self.markets = {"BTC/USDT": {}, "DOGE/USDT": {}}

        async def load_markets(self):
            return self.markets

        async def close(self):
            pass

    monkeypatch.setattr(
        module, "resolve_exchange_class", lambda ccxt, venue: FakeClient
    )
    monkeypatch.setenv("PUBLIC_DISCOVERY_VENUES", "bybit,binance,okx")
    for name in ("GATE_API_KEY", "MEXC_API_KEY", "OKX_API_KEY", "KUCOIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    scanner = module.DynamicMarketScanner({}, None)
    clients = asyncio.run(scanner.public_exchanges())

    assert set(clients) == {"bybit", "binance", "okx"}
    for config in built:
        assert "apiKey" not in config, "a public observer must carry no credentials"
        assert "secret" not in config


def test_an_unreachable_venue_is_reported_not_invented(monkeypatch):
    import DYNAMIC_MARKET_SCANNER as module

    class Reachable:
        def __init__(self, config):
            self.markets = {"BTC/USDT": {}}

        async def load_markets(self):
            return self.markets

        async def close(self):
            pass

    class Unreachable:
        def __init__(self, config):
            pass

        async def load_markets(self):
            raise ConnectionError("venue down")

        async def close(self):
            pass

    def resolver(ccxt, venue):
        if venue == "binance":
            return Reachable
        return Unreachable

    monkeypatch.setattr(module, "resolve_exchange_class", resolver)
    monkeypatch.setenv("PUBLIC_DISCOVERY_VENUES", "binance,okx")

    scanner = module.DynamicMarketScanner({}, None)
    clients = asyncio.run(scanner.public_exchanges())

    assert set(clients) == {"binance"}
    assert scanner.venue_status["binance"].startswith("public:")
    assert scanner.venue_status["okx"].startswith("unreachable:")
    assert "ConnectionError" in scanner.venue_status["okx"]
    assert scanner.get_stats()["venue_status"] == scanner.venue_status


def test_observer_clients_are_reused_not_rebuilt(monkeypatch):
    """A fresh async client per scan leaks a session."""
    import DYNAMIC_MARKET_SCANNER as module

    constructions = []

    class Client:
        def __init__(self, config):
            constructions.append(config)
            self.markets = {"BTC/USDT": {}}

        async def load_markets(self):
            return self.markets

        async def close(self):
            pass

    monkeypatch.setattr(module, "resolve_exchange_class", lambda ccxt, v: Client)
    monkeypatch.setenv("PUBLIC_DISCOVERY_VENUES", "binance")

    scanner = module.DynamicMarketScanner({}, None)
    asyncio.run(scanner.public_exchanges())
    asyncio.run(scanner.public_exchanges())

    assert len(constructions) == 1


def test_observing_a_venue_grants_no_execution_there(populated, monkeypatch):
    """Discovery breadth is not execution breadth, end to end."""
    import DYNAMIC_MARKET_SCANNER as module

    class Client:
        def __init__(self, config):
            self.markets = {"FOREIGN/USDT": {"spot": True, "active": True}}

        async def load_markets(self):
            return self.markets

        async def close(self):
            pass

    monkeypatch.setattr(module, "resolve_exchange_class", lambda ccxt, v: Client)
    monkeypatch.setenv("PUBLIC_DISCOVERY_VENUES", "okx")

    scanner = module.DynamicMarketScanner({}, None)
    asyncio.run(scanner.public_exchanges())

    registry.ingest_venue("okx", {"FOREIGN/USDT": ticker()}, markets={})
    registry.apply_execution_venue("bybit", BYBIT_MARKETS)

    market = registry.get("FOREIGN/USDT")
    assert market is not None, "it is observed"
    assert market.execution_eligible is False, "but not executable here"
    assert "FOREIGN/USDT" not in registry.symbols(executable_only=True)


def test_signals_are_counted_when_published(monkeypatch):
    """The data hub is the one seam every producer passes through."""
    from COMPLETE_UNIFIED_ORCHESTRATOR import CentralDataHub

    hub = CentralDataHub()
    asyncio.run(
        hub.publish_signal(
            {
                "symbol": "PEPE/USDT",
                "side": "buy",
                "strategy": "scalp",
                "timeframe": "5m",
                "confidence": 0.9,
            }
        )
    )

    telemetry = registry.telemetry()
    assert telemetry["signals_generated"] == 1
    assert telemetry["signals_by_symbol"] == {"PEPE/USDT": 1}
    assert telemetry["signals_by_strategy"] == {"scalp": 1}
    assert telemetry["signals_by_timeframe"] == {"5m": 1}
    assert registry.get("PEPE/USDT") is None or True  # unknown market is fine
