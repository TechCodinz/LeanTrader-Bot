"""Market intelligence connected to REAL_PROFIT_BOT.

The defect these protect against is the one that cost a week: intelligence
built beside the bot instead of into it, and then intelligence so strict that
the bot stops trading. Both failures are tested here.

The most important test in this file is
test_intelligence_can_never_starve_execution: whatever goes wrong -- no
candles, a raising exchange, NaN indicators, a dead order book -- the bot's own
signal must come back unchanged.

No network. No orders.
"""

import math

import pytest

from rpb_intelligence import MarketIntelligence


def candles(n=60, start=100.0, step=0.1, noise=0.0):
    rows = []
    price = start
    for i in range(n):
        price = price + step + (noise if i % 2 else -noise)
        rows.append([i * 60_000, price, price * 1.002, price * 0.998, price, 1000.0])
    return rows


class FakeExchange:
    def __init__(self, ohlcv=None, book=None, ohlcv_error=None, book_error=None):
        self._ohlcv = ohlcv
        self._book = book
        self._ohlcv_error = ohlcv_error
        self._book_error = book_error

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=60):
        if self._ohlcv_error:
            raise self._ohlcv_error
        return self._ohlcv

    def fetch_order_book(self, symbol, limit=20):
        if self._book_error:
            raise self._book_error
        return self._book


def book(bid=100.0, ask=100.02, bid_size=100.0, ask_size=100.0):
    return {
        "bids": [[bid, bid_size]] * 10,
        "asks": [[ask, ask_size]] * 10,
    }


@pytest.fixture
def intel():
    return MarketIntelligence()


# ------------------------------------------------- it is actually connected


def test_the_real_indicator_engine_is_reachable(intel):
    """core/strategy_engine.py had nine imports stripped; it must import now."""
    assert intel.available is True, intel.reason
    assert intel.reason == "ready"


def test_it_uses_the_existing_engine_rather_than_its_own_math():
    import ast
    import inspect

    import rpb_intelligence as mod

    src = inspect.getsource(mod)
    assert "from core.strategy_engine import" in src
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for reimplemented in ("calculate_rsi", "calculate_macd", "calculate_bollinger_bands"):
        assert reimplemented not in defined, "the indicator math must not be duplicated"


# ------------------------------------- it must never starve execution


@pytest.mark.parametrize(
    "exchange",
    [
        FakeExchange(ohlcv=None),
        FakeExchange(ohlcv=[]),
        FakeExchange(ohlcv=candles(5)),
        FakeExchange(ohlcv_error=RuntimeError("rate limited")),
        FakeExchange(ohlcv=candles(), book_error=RuntimeError("book down")),
    ],
)
def test_intelligence_can_never_starve_execution(intel, exchange):
    """Any failure returns the bot's own signal untouched."""
    signal, confidence, detail = intel.evaluate(exchange, "BTC/USDT", "BUY", 95, 100.0)
    assert signal == "BUY"
    assert confidence >= 95 or detail["applied"] is True


def test_a_broken_engine_disables_intelligence_without_blocking(monkeypatch):
    broken = MarketIntelligence()
    broken.available = False
    broken.reason = "engine_import_failed"

    signal, confidence, detail = broken.evaluate(FakeExchange(), "BTC/USDT", "BUY", 95, 100.0)

    assert (signal, confidence) == ("BUY", 95)
    assert detail["applied"] is False


def test_a_dead_order_book_does_not_block_the_trade(intel):
    ex = FakeExchange(ohlcv=candles(), book={"bids": [], "asks": []})
    signal, confidence, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 95, 100.0)
    assert signal == "BUY"
    assert detail["applied"] is True


def test_a_hold_signal_is_passed_through_untouched(intel):
    signal, confidence, _ = intel.evaluate(FakeExchange(), "BTC/USDT", "HOLD", 50, 100.0)
    assert (signal, confidence) == ("HOLD", 50)


# ------------------------------------------------- it actually computes


def test_a_clean_uptrend_confirms_a_buy(intel):
    ex = FakeExchange(ohlcv=candles(60, step=0.1), book=book(bid_size=300.0, ask_size=100.0))
    signal, confidence, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 85, 106.0)

    assert signal == "BUY"
    assert detail["applied"] is True
    assert "macd" in detail["agreements"]
    assert "book_imbalance" in detail["agreements"]
    assert confidence > 85


def test_a_downtrend_contradicts_a_buy_without_forbidding_it(intel):
    ex = FakeExchange(ohlcv=candles(60, start=120.0, step=-0.1), book=book(bid_size=50.0, ask_size=300.0))
    signal, confidence, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 95, 114.0)

    assert signal == "BUY", "a contradicted signal is penalised, not vetoed"
    assert confidence < 95
    assert detail["conflicts"]


def test_real_indicator_values_are_reported(intel):
    ex = FakeExchange(ohlcv=candles(), book=book())
    _, _, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 85, 106.0)

    assert 0.0 <= detail["rsi"] <= 100.0
    assert not math.isnan(detail["macd_histogram"])
    assert detail["atr_pct"] >= 0.0
    assert detail["spread_bps"] > 0.0


def test_order_book_imbalance_and_microprice_are_computed(intel):
    ex = FakeExchange(ohlcv=candles(), book=book(bid_size=300.0, ask_size=100.0))
    _, _, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 85, 106.0)

    assert detail["imbalance"] == pytest.approx(0.5, abs=0.01)
    assert detail["microprice_shift_bps"] < 0 or detail["microprice_shift_bps"] > 0


# --------------------------------------------------------- the few vetoes


def test_an_unexecutable_spread_is_vetoed(intel):
    ex = FakeExchange(ohlcv=candles(), book=book(bid=100.0, ask=101.5))
    signal, confidence, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 95, 100.0)

    assert signal == "HOLD"
    assert "spread" in detail["reason"]


def test_a_blowoff_top_buy_is_vetoed(intel):
    """Extreme RSI *and* momentum already rolling over -- both required."""
    # A long rally, then a few bars of rollover: still extremely overbought
    # (RSI ~88) while MACD has just turned negative. That combination is the
    # blow-off; either half alone is not.
    rows = candles(56, step=0.8) + candles(3, start=100 + 56 * 0.8, step=-0.4)
    ex = FakeExchange(ohlcv=rows, book=book())
    signal, _, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 95, 120.0)

    assert signal == "HOLD"
    assert "blowoff" in detail["reason"]
    assert detail["rsi"] >= 85 and detail["macd_histogram"] < 0


def test_strong_momentum_alone_is_never_vetoed(intel):
    """The bot enters on momentum; a high RSI is what momentum looks like."""
    ex = FakeExchange(ohlcv=candles(60, step=0.5), book=book())
    signal, confidence, detail = intel.evaluate(ex, "BTC/USDT", "BUY", 95, 130.0)

    assert signal == "BUY", "a stretched RSI alone must not block the entry"
    assert detail["rsi"] >= 85


def test_vetoes_are_few_and_named():
    """Gate explosion is the failure mode; count the gates."""
    import ast
    import inspect

    import rpb_intelligence as mod

    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(mod.MarketIntelligence.evaluate)))
    holds = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Tuple)
        and any(isinstance(e, ast.Constant) and e.value == "HOLD" for e in n.value.elts)
    )
    assert holds <= 3, f"{holds} veto paths is gate explosion"


# ------------------------------------------------------ bot integration


def test_the_bot_wires_intelligence_into_its_own_signal():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "from rpb_intelligence import MarketIntelligence" in src
    assert "self.intelligence.evaluate(" in src


def test_the_bot_still_produces_its_own_signal_first():
    """Intelligence scores the bot's signal; it never invents one."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "best_idx = confidences.index(max(confidences))" in src
    assert "signal = signals[best_idx]" in src


def test_intelligence_places_no_orders():
    import ast
    import inspect

    import rpb_intelligence as mod

    tree = ast.parse(inspect.getsource(mod))
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in ("create_order", "create_market_buy_order", "create_market_sell_order"):
        assert forbidden not in called


def test_candles_are_cached_so_a_fast_loop_does_not_hammer_the_api(intel):
    calls = {"n": 0}

    class Counting(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe="1m", limit=60):
            calls["n"] += 1
            return candles()

    ex = Counting(book=book())
    intel.evaluate(ex, "BTC/USDT", "BUY", 85, 106.0)
    intel.evaluate(ex, "BTC/USDT", "BUY", 85, 106.0)
    assert calls["n"] == 1
