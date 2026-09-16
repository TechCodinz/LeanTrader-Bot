"""Alpha ensemble and risk governance connected to REAL_PROFIT_BOT.

Risk engines are the most likely thing to silently stop a working bot, so
most of this file is about what these two must NOT be able to do: veto a
signal, size a position to zero, or halt trading without an operator opting
in.

No network. No orders.
"""

import os

import pytest

from rpb_alpha_risk import EXECUTABLE_FLOOR_USD, AlphaEnsemble, RiskGovernor


def bars(n=140, start=100.0, step=0.15):
    rows, price = [], start
    for i in range(n):
        price += step
        rows.append([i * 60_000, price, price * 1.003, price * 0.997, price, 1000.0 + i])
    return rows


class FakeExchange:
    def __init__(self, ohlcv=None, error=None):
        self._ohlcv = ohlcv
        self._error = error
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe="5m", limit=120):
        self.calls += 1
        if self._error:
            raise self._error
        return self._ohlcv


@pytest.fixture
def alpha():
    return AlphaEnsemble()


@pytest.fixture
def risk():
    return RiskGovernor()


# ------------------------------------------------- both are actually reachable


def test_the_alpha_router_is_connected(alpha):
    assert alpha.available is True, alpha.reason
    names = alpha.strategy_names()
    assert len(names) >= 8, names


def test_the_risk_manager_is_connected(risk):
    """core/risk_manager.py had its import block stripped like strategy_engine."""
    assert risk.available is True, risk.reason


def test_core_modules_import_after_the_repair():
    from core.order_manager import OrderSide
    from core.risk_manager import RiskLevel, RiskManager

    assert RiskManager is not None
    assert RiskLevel.LOW.value == "low"
    assert OrderSide.BUY is not None


# --------------------------------------------- alpha cannot starve execution


@pytest.mark.parametrize(
    "exchange",
    [FakeExchange(ohlcv=None), FakeExchange(ohlcv=[]),
     FakeExchange(ohlcv=bars(20)), FakeExchange(error=RuntimeError("rate limit"))],
)
def test_alpha_returns_confidence_unchanged_when_it_cannot_compute(alpha, exchange):
    confidence, multiplier, detail = alpha.evaluate(exchange, "BTC/USDT", "BUY", 95.0)
    assert confidence == 95.0
    assert multiplier == 1.0
    assert detail["applied"] is False


def test_alpha_never_lowers_confidence(alpha):
    """It raises or abstains. It cannot penalise the bot's own signal."""
    ex = FakeExchange(ohlcv=bars())
    for _ in range(5):
        alpha._cache.clear()
        confidence, _, _ = alpha.evaluate(ex, "BTC/USDT", "BUY", 90.0)
        assert confidence >= 90.0


def test_alpha_ignores_sell_because_it_is_long_only(alpha):
    ex = FakeExchange(ohlcv=bars())
    confidence, multiplier, detail = alpha.evaluate(ex, "BTC/USDT", "SELL", 95.0)
    assert (confidence, multiplier) == (95.0, 1.0)
    assert detail["reason"] == "alpha_is_long_only"


def test_alpha_never_returns_a_signal(alpha):
    """It scores; it cannot flip direction."""
    ex = FakeExchange(ohlcv=bars())
    result = alpha.evaluate(ex, "BTC/USDT", "BUY", 95.0)
    assert len(result) == 3
    assert isinstance(result[0], float) and isinstance(result[1], float)


def test_the_size_multiplier_is_bounded(alpha):
    ex = FakeExchange(ohlcv=bars())
    _, multiplier, _ = alpha.evaluate(ex, "BTC/USDT", "BUY", 95.0)
    assert 0.5 <= multiplier <= 2.0


def test_alpha_caches_candles(alpha):
    ex = FakeExchange(ohlcv=bars())
    alpha.evaluate(ex, "BTC/USDT", "BUY", 95.0)
    first = ex.calls
    alpha.evaluate(ex, "BTC/USDT", "BUY", 95.0)
    assert ex.calls == first


# ---------------------------------------------- risk cannot starve execution


def test_risk_never_sizes_below_the_executable_floor(risk):
    risk.sync_wallet(13.56)
    for proposed in (0.0, 0.01, 1.0, 3.0):
        final, _ = risk.cap_position("BTC/USDT", proposed, confidence=95.0)
        assert final >= EXECUTABLE_FLOOR_USD


def test_risk_never_returns_zero_even_in_deep_drawdown(risk):
    risk.sync_wallet(1000.0)
    risk.sync_wallet(200.0)  # an 80% drawdown
    final, _ = risk.cap_position("BTC/USDT", 50.0, confidence=95.0)
    assert final >= EXECUTABLE_FLOOR_USD


def test_risk_caps_an_oversized_proposal(risk):
    risk.sync_wallet(100.0)
    final, detail = risk.cap_position("BTC/USDT", 10_000.0, confidence=95.0)
    assert final < 10_000.0
    assert detail["applied"] is True


def test_the_drawdown_halt_is_off_by_default(risk):
    risk.sync_wallet(1000.0)
    risk.sync_wallet(100.0)
    halted, reason = risk.should_halt()
    assert halted is False, f"a risk engine must not silently stop the bot ({reason})"


def test_a_broken_risk_engine_still_returns_a_tradeable_size():
    broken = RiskGovernor()
    broken.available = False
    broken.reason = "import_failed"
    final, detail = broken.cap_position("BTC/USDT", 5.0, confidence=95.0)
    assert final >= EXECUTABLE_FLOOR_USD
    assert detail["applied"] is False


def test_the_alpha_multiplier_applies_even_without_risk():
    broken = RiskGovernor()
    broken.available = False
    doubled, _ = broken.cap_position("BTC/USDT", 10.0, confidence=95.0, size_multiplier=2.0)
    halved, _ = broken.cap_position("BTC/USDT", 10.0, confidence=95.0, size_multiplier=0.5)
    assert doubled > halved


def test_risk_tracks_the_authenticated_wallet(risk):
    risk.sync_wallet(13.56)
    metrics = risk.metrics()
    assert metrics["available"] is True
    assert metrics["portfolio_value"] == pytest.approx(13.56)


def test_risk_sees_an_opened_and_closed_position(risk):
    risk.sync_wallet(100.0)
    risk.sync_position("BTC/USDT", 0.5, 100.0)
    assert risk.metrics()["num_positions"] >= 1
    risk.sync_position("BTC/USDT", 0.0, 100.0)
    assert risk.metrics()["total_exposure"] == pytest.approx(0.0, abs=1e-6)


def test_neither_module_places_orders():
    import ast
    import inspect

    import rpb_alpha_risk as mod

    tree = ast.parse(inspect.getsource(mod))
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in ("create_order", "create_market_buy_order", "create_market_sell_order"):
        assert forbidden not in called


# ---------------------------------------------------------- rejected engines


def test_the_synthetic_scout_was_not_connected():
    """ultra_scout fabricates volume, holders and transfers with random."""
    src = open("ultra_scout.py").read()
    assert "random.uniform(100000, 10000000)" in src, "fixture check: still synthetic"

    for connector in ("rpb_intelligence.py", "rpb_scalping.py", "rpb_alpha_risk.py"):
        assert "ultra_scout" not in open(connector).read()
    assert "ultra_scout" not in open("REAL_PROFIT_BOT.py").read()


# ------------------------------------------------------------- bot wiring


def test_the_bot_wires_alpha_and_risk():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "from rpb_alpha_risk import AlphaEnsemble, RiskGovernor" in src
    assert "self.alpha.evaluate(" in src
    assert "self.risk.cap_position(" in src
    assert "self.risk.sync_wallet(" in src
    assert "self.risk.sync_position(" in src


def test_risk_sizing_is_testnet_only():
    """The live sizing path keeps its historical limit."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "min(balance * 0.25, 12.0)" in src


def test_everything_preserved_still_present():
    src = open("REAL_PROFIT_BOT.py").read()
    for keep in ("self.telegram_bot_token", "self.vip_chat_id", "self.free_chat_id",
                 "ccxt.bybit", "ccxt.gate", "def send_telegram",
                 "self.total_profit", "Non-executable dust", "QUARANTINED PAIR"):
        assert keep in src, keep
