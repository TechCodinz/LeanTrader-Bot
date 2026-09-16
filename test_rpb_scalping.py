"""Scalping confluence and micro-wallet growth connected to REAL_PROFIT_BOT.

The single most important test here is
test_the_repaired_analyzer_no_longer_always_says_sell. Before the repair,
SMART_SCALPING_ENGINE.analyze_timeframe held a hardcoded indicator dict and
ignored its ohlcv argument entirely: ema_fast and ema_slow were both 0.0, so
`0.0 > 0.0` was False and every call fell through to SELL at full strength --
for every symbol, every timeframe, forever. Wiring that into a spot bot would
have produced naked SELL attempts and no entries at all.

No network. No orders.
"""

import pytest

from rpb_scalping import (
    GROWTH_LADDER,
    ScalpingConfluence,
    growth_phase,
    target_position_usd,
)


def bars(n=60, start=100.0, step=0.2, volume=1000.0):
    rows = []
    price = start
    for i in range(n):
        price += step
        rows.append([i * 60_000, price, price * 1.002, price * 0.998, price, volume])
    return rows


class FakeExchange:
    def __init__(self, ohlcv=None, error=None):
        self._ohlcv = ohlcv
        self._error = error
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=60):
        self.calls += 1
        if self._error:
            raise self._error
        return self._ohlcv


# ------------------------------------------- the placeholder is actually gone


def test_the_repaired_analyzer_no_longer_always_says_sell():
    from SMART_SCALPING_ENGINE import MultiTimeframeAnalyzer

    analyzer = MultiTimeframeAnalyzer()
    rising = analyzer.analyze_timeframe("X/USDT", "5m", bars(60, step=0.2))
    falling = analyzer.analyze_timeframe("Y/USDT", "5m", bars(60, start=120.0, step=-0.2))

    # The old code returned SELL at strength 1.0 for both of these.
    assert not (rising.direction == "SELL" and rising.strength == 1.0)
    assert not (falling.direction == "SELL" and falling.strength == 1.0)

    # And the indicators are now computed, not constants.
    assert rising.indicators["real_data"] is True
    assert rising.indicators["ema_fast"] != 0.0
    assert rising.indicators["rsi"] != 50.0
    assert rising.indicators["rsi"] != falling.indicators["rsi"]


def test_the_analyzer_reads_the_candles_it_is_given():
    from SMART_SCALPING_ENGINE import MultiTimeframeAnalyzer

    analyzer = MultiTimeframeAnalyzer()
    a = analyzer.analyze_timeframe("A/USDT", "5m", bars(60, step=0.3))
    b = analyzer.analyze_timeframe("B/USDT", "5m", bars(60, start=200.0, step=-0.3))

    assert a.indicators["ema_fast"] != b.indicators["ema_fast"], (
        "different candles must produce different indicators"
    )


def test_absent_candles_yield_neutral_not_a_direction():
    """No data must never be reported as a tradeable direction."""
    from SMART_SCALPING_ENGINE import MultiTimeframeAnalyzer

    analyzer = MultiTimeframeAnalyzer()
    for rows in ([], None, bars(5)):
        signal = analyzer.analyze_timeframe("X/USDT", "5m", rows)
        assert signal.direction == "NEUTRAL"
        assert signal.strength == 0.0
        assert signal.indicators["real_data"] is False


def test_the_indicator_math_is_not_duplicated():
    import inspect

    import SMART_SCALPING_ENGINE as mod

    src = inspect.getsource(mod.MultiTimeframeAnalyzer._compute_indicators)
    assert "from core.strategy_engine import" in src
    # The repair must not have grown a second copy of RSI/MACD/Bollinger here.
    for reimplemented in ("def calculate_rsi", "def calculate_macd"):
        assert reimplemented not in inspect.getsource(mod)


# --------------------------------------------------- the micro-wallet ladder


@pytest.mark.parametrize(
    "balance,expected_phase",
    [(11.0, "Micro"), (48.0, "Micro"), (150.0, "Foundation"),
     (500.0, "Acceleration"), (1200.0, "Expansion"), (5000.0, "Maturity")],
)
def test_the_growth_ladder_selects_a_phase_from_real_balance(balance, expected_phase):
    phase, fraction = growth_phase(balance)
    assert phase == expected_phase
    assert 0.0 < fraction <= 0.35


def test_the_twelve_dollar_ceiling_is_gone():
    """The old cap stopped the wallet compounding past roughly $48."""
    target, _ = target_position_usd(1000.0)
    assert target > 12.0

    small, _ = target_position_usd(11.0)
    assert small == pytest.approx(max(3.50, 11.0 * 0.25))


def test_the_ladder_scales_with_the_wallet():
    previous = 0.0
    for balance in (11.0, 50.0, 150.0, 500.0, 1200.0, 5000.0):
        target, _ = target_position_usd(balance)
        assert target > previous, "a bigger wallet must commit more, not less"
        previous = target


def test_the_minimum_executable_floor_is_kept():
    """A position below the floor cannot execute at all."""
    target, _ = target_position_usd(0.50)
    assert target == pytest.approx(3.50)


def test_the_ladder_tapers_as_the_wallet_grows():
    _, micro = growth_phase(50.0)
    _, mature = growth_phase(5000.0)
    assert micro > mature


def test_the_ladder_is_bounded():
    for floor, fraction, _name in GROWTH_LADDER:
        assert 0.0 < fraction <= 0.35


# -------------------------------------------- confluence cannot block a trade


@pytest.fixture
def confluence():
    return ScalpingConfluence()


def test_confluence_is_connected(confluence):
    assert confluence.available is True, confluence.reason
    assert confluence.current_session() != ""


@pytest.mark.parametrize(
    "exchange",
    [FakeExchange(ohlcv=None), FakeExchange(ohlcv=[]),
     FakeExchange(ohlcv=bars(5)), FakeExchange(error=RuntimeError("rate limit"))],
)
def test_confluence_never_reduces_confidence_when_it_cannot_compute(confluence, exchange):
    adjusted, detail = confluence.evaluate(exchange, "BTC/USDT", "BUY", 95.0)
    assert adjusted == 95.0
    assert detail["applied"] is False


def test_confluence_never_changes_the_signal(confluence):
    ex = FakeExchange(ohlcv=bars())
    before = 95.0
    adjusted, detail = confluence.evaluate(ex, "BTC/USDT", "BUY", before)
    assert isinstance(adjusted, float)
    # There is no signal in the return at all -- it cannot flip direction.
    assert "signal" not in detail


def test_confluence_returns_a_bounded_confidence(confluence):
    ex = FakeExchange(ohlcv=bars())
    adjusted, _ = confluence.evaluate(ex, "BTC/USDT", "BUY", 98.0)
    assert 0.0 <= adjusted <= 99.0


def test_a_hold_signal_is_untouched(confluence):
    adjusted, detail = confluence.evaluate(FakeExchange(), "BTC/USDT", "HOLD", 50.0)
    assert adjusted == 50.0
    assert detail["applied"] is False


def test_candles_are_cached_across_calls(confluence):
    ex = FakeExchange(ohlcv=bars())
    confluence.evaluate(ex, "BTC/USDT", "BUY", 95.0)
    first = ex.calls
    confluence.evaluate(ex, "BTC/USDT", "BUY", 95.0)
    assert ex.calls == first


def test_an_authenticated_close_feeds_session_tracking(confluence):
    confluence.record_result("BTC/USDT", 0.42)
    confluence.record_result("BTC/USDT", -0.10)
    stats = confluence.session_stats("BTC/USDT")

    assert stats.get("trades") == 2
    assert stats.get("wins") == 1
    assert stats.get("losses") == 1
    assert stats.get("total_profit") == pytest.approx(0.32)


def test_scalping_places_no_orders():
    import ast
    import inspect

    import rpb_scalping as mod

    tree = ast.parse(inspect.getsource(mod))
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in ("create_order", "create_market_buy_order", "create_market_sell_order"):
        assert forbidden not in called


# ------------------------------------------------------------ bot wiring


def test_the_bot_wires_confluence_and_the_ladder():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "from rpb_scalping import ScalpingConfluence" in src
    assert "self.scalping.evaluate(" in src
    assert "from rpb_scalping import target_position_usd" in src
    assert "self.scalping.record_result(" in src


def test_the_live_sizing_path_is_untouched():
    """Only the testnet branch takes the ladder; live keeps its old limit."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "min(balance * 0.25, 12.0)" in src


def test_telegram_and_modes_still_intact():
    src = open("REAL_PROFIT_BOT.py").read()
    for keep in ("self.telegram_bot_token", "self.vip_chat_id", "self.free_chat_id",
                 "ccxt.bybit", "ccxt.gate", "def send_telegram"):
        assert keep in src
