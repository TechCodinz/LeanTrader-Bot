"""Multi-exchange support, cross-exchange arbitrage, and the FX/MT5 path.

Written against one question: when an API key is added, does the system use
it -- and when one is absent, does it refuse rather than invent?

The arbitrage tests matter most. Spot arbitrage sells on a second venue, so
it needs inventory there before it buys on the first. Without that check the
buy fills, the sell is rejected, and capital is stranded on one exchange with
open price exposure.

No network. No credentials. No orders.
"""

import ast
import inspect
import os

import pytest


# ------------------------------------------------- every venue is configured


def test_all_named_exchanges_are_supported(monkeypatch):
    from exchange_manager import ExchangeManager

    for venue in ("bybit", "binance", "okx", "kucoin", "gateio", "mexc", "bitget"):
        assert venue in ExchangeManager.SUPPORTED_EXCHANGES, venue


def test_adding_a_key_is_the_only_step(monkeypatch):
    """No code change should be needed to bring a venue online."""
    monkeypatch.setenv("OKX_API_KEY", "key-not-real")
    monkeypatch.setenv("OKX_API_SECRET", "secret-not-real")

    from exchange_manager import ExchangeManager

    manager = ExchangeManager.__new__(ExchangeManager)
    manager.configs = {}
    manager._create_default_config()

    assert manager.configs["okx"].api_key == "key-not-real"
    assert manager.configs["okx"].secret == "secret-not-real"
    assert manager._is_authenticated("okx") is True


def test_a_venue_without_keys_is_configured_but_not_authenticated(monkeypatch):
    for var in ("MEXC_API_KEY", "MEXC_API_SECRET", "MEXC_SECRET"):
        monkeypatch.delenv(var, raising=False)

    from exchange_manager import ExchangeManager

    manager = ExchangeManager.__new__(ExchangeManager)
    manager.configs = {}
    manager._create_default_config()

    assert "mexc" in manager.configs           # usable for public data
    assert manager._is_authenticated("mexc") is False


def test_testnet_keys_count_as_authenticated(monkeypatch):
    """The gate used _has_live_credentials, which is False on testnet.

    Every Testnet deployment therefore fell through to mock trades and mock
    orders -- fabricated fills, in the one environment this system validates
    in.
    """
    monkeypatch.setenv("BYBIT_API_KEY", "k")
    monkeypatch.setenv("BYBIT_API_SECRET", "s")
    monkeypatch.setenv("BYBIT_TESTNET", "1")

    from exchange_manager import ExchangeManager

    manager = ExchangeManager.__new__(ExchangeManager)
    manager.configs = {}
    manager._create_default_config()

    assert manager._is_authenticated("bybit") is True
    assert manager._has_live_credentials("bybit") is False   # still not live


def test_no_credential_value_is_hardcoded():
    src = open("exchange_manager.py").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 25 and node.value.isalnum():
                pytest.fail(f"possible hardcoded credential: {node.value[:8]}...")


# ---------------------------------------- nothing falls back to invented data


def test_the_manager_never_returns_mock_market_data():
    from exchange_manager import ExchangeManager

    tree = ast.parse(inspect.getsource(ExchangeManager))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr.startswith("_get_mock")
    ]
    assert calls == [], "a mock fallback still executes"


def test_an_unquotable_symbol_reports_nothing_not_a_price():
    src = open("exchange_manager.py").read()
    assert "No venue could quote" in src
    assert "return self._get_mock_ticker(symbol)" not in src


# ------------------------------------------------ arbitrage needs both legs


def _arb_source():
    return open("CROSS_EXCHANGE_ARBITRAGE.py").read()


def test_arbitrage_checks_inventory_on_the_sell_venue():
    """Without this the buy fills, the sell is rejected, capital is stranded."""
    src = _arb_source()
    assert "sell_balance = await sell_ex.fetch_balance()" in src
    assert "base_available" in src
    assert "one-legged trade" in src


def test_arbitrage_refuses_rather_than_opening_one_leg():
    src = _arb_source()
    buy_at = src.index("create_market_buy_order")
    guard_at = src.index("elif base_available < amount:")
    assert guard_at < buy_at, "the inventory guard must precede the buy"


def test_arbitrage_reconciles_the_buy_before_selling():
    src = _arb_source()
    assert "An acknowledgement is not a fill" in src
    assert "BUY acknowledged with no fill" in src
    fetch_at = src.index("confirmed = await buy_ex.fetch_order")
    sell_at = src.index("create_market_sell_order")
    assert fetch_at < sell_at, "the buy must be reconciled before the sell"


def test_arbitrage_sells_only_what_actually_filled():
    src = _arb_source()
    assert "min(filled, base_available)" in src


def test_arbitrage_never_defaults_its_revenue():
    """sell_revenue fell back to the profit it was hoping for."""
    src = _arb_source()
    assert "sell_order.get('cost', position_usd" not in src
    assert "buy_order.get('cost', position_usd)" not in src
    assert "No profit booked" in src


def test_arbitrage_books_net_of_fees():
    src = _arb_source()
    assert "buy_fee" in src and "sell_fee" in src
    assert "- buy_fee - sell_fee" in src


# ------------------------------------------------------------- FX and MT5


def test_the_mt5_broker_refuses_rather_than_pretending():
    """No MetaTrader5 module means no FX execution, stated plainly."""
    src = open("src/leantrader/execution/broker_fx_mt5.py").read()
    assert "MetaTrader5 module not installed" in src
    assert "raise RuntimeError" in src

    tree = ast.parse(src)
    fabricators = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"uniform", "randint", "gauss", "choice"}
    ]
    assert fabricators == [], "the FX broker must not invent fills"


def test_the_fx_brokers_exist_for_both_venues():
    for path in (
        "src/leantrader/execution/broker_fx.py",
        "src/leantrader/execution/broker_fx_mt5.py",
        "src/leantrader/execution/broker_fx_oanda.py",
    ):
        assert os.path.exists(path), path


# ----------------------------------------------------- pair coverage

def test_pair_discovery_has_no_hardcoded_ceiling():
    """The universe comes from the venue, not a fixed list."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "self.gate.fetch_tickers()" in src
    assert "dynamic_pair_min_volume_usd" in src
    # The static list survives only as a fallback.
    assert "def refresh_dynamic_pairs" in src


def test_the_volume_filter_is_tunable_not_baked_in():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "self.dynamic_pair_min_volume_usd" in src
