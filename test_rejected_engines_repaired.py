"""The engines previously rejected for fabricating data, now repaired.

Three of them invented numbers a caller could not distinguish from
measurements. The repair rule throughout: read a real source where one
exists, and report unavailability where none does. Never a random number,
and never a constant standing in for a measurement.

Two of my earlier rejections were wrong and are recorded here as such:
IBM_QUANTUM_ENGINE's np.random is variational-circuit parameter
initialisation, which is correct practice, and divine_intelligence_core
consumes indicators rather than inventing them.

No network. No orders.
"""

import ast
import inspect

import pandas as pd
import pytest


class FakeExchange:
    rateLimit = 50
    enableRateLimit = True
    last_response_headers = {"x-ratelimit-remaining": "99"}

    def fetch_order_book(self, symbol, limit=50):
        return {"bids": [[100.0, 5.0]] * 30, "asks": [[100.1, 3.0]] * 30}

    def fetch_balance(self):
        return {"free": {"USDT": 13.56}, "total": {"USDT": 13.56}}

    def fetch_ticker(self, symbol):
        return {"quoteVolume": 1234567.0}

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=200):
        return [[i, 100 + i * 0.1, 101, 99, 100 + i * 0.1, 10] for i in range(60)]


# ------------------------------------------------------------- ultra_scout


def _scout(exchange=None):
    from ultra_scout import UltraScout

    return UltraScout(exchange=exchange)


def test_no_executable_fabrication_remains_in_the_scout():
    """Every random draw is gone; only comments describing them remain."""
    import ultra_scout

    source = inspect.getsource(ultra_scout)
    tree = ast.parse(source)

    fabricators = {"uniform", "randint", "gauss", "choice", "random"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in fabricators
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"random", "np"}
    ]
    assert calls == [], f"{len(calls)} fabrication call(s) still execute"


def test_absent_onchain_provider_reports_unavailable_not_numbers():
    scout = _scout()
    result = scout._fetch_etherscan("0xabc")
    assert result["available"] is False
    assert result["reason"] == "no_etherscan_api_key"
    for invented in ("total_supply", "holders", "transfers_24h"):
        assert invented not in result


def test_absent_providers_report_unavailable():
    scout = _scout()
    assert scout._analyze_exchange_flows("0xabc")["available"] is False
    assert scout._get_holder_distribution("0xabc")["available"] is False
    assert scout.satellite_data_fusion("BTC/USDT")["available"] is False


def test_whale_activity_is_unknown_rather_than_guessed():
    """It was a random count keyed off the hour of day."""
    assert _scout()._estimate_whale_activity() is None


def test_whale_movements_are_empty_without_a_provider():
    assert _scout()._detect_whale_movements("0xabc") == []


def test_liquidity_depth_is_measured_from_a_real_book():
    scout = _scout(FakeExchange())
    depth = scout._get_liquidity_depth("BTC/USDT")

    assert depth["available"] is True
    assert depth["bid_depth"] == pytest.approx(100.0 * 5.0 * 25)
    assert depth["ask_depth"] == pytest.approx(100.1 * 3.0 * 25)
    assert depth["spread_bps"] == pytest.approx(9.995, abs=0.01)
    assert depth["imbalance"] > 0


def test_liquidity_depth_without_a_client_is_unavailable():
    assert _scout()._get_liquidity_depth("BTC/USDT")["available"] is False


def test_balance_is_authenticated_or_unavailable():
    assert _scout()._check_balance("bybit")["available"] is False

    real = _scout(FakeExchange())._check_balance("bybit")
    assert real["available"] is True
    assert real["free_usd"] == pytest.approx(13.56)


def test_backtest_returns_come_from_real_candles():
    scout = _scout(FakeExchange())
    result = scout._run_period_backtest(None, {"name": "p", "symbol": "BTC/USDT"})

    assert result["available"] is True
    assert result["trades"] == 59
    assert result["total_return"] > 0  # the fixture rises monotonically


def test_a_backtest_without_candles_is_unavailable_not_scored():
    result = _scout()._run_period_backtest(None, {"name": "p", "symbol": "BTC/USDT"})
    assert result["available"] is False
    assert result["total_return"] is None


def test_swarm_votes_follow_the_signal_not_chance():
    scout = _scout()
    assert set(scout._derive_votes({"score": 0.9}, 5)) == {"buy"}
    assert set(scout._derive_votes({"score": -0.9}, 5)) == {"sell"}
    assert set(scout._derive_votes({"score": 0.0}, 5)) == {"hold"}
    # Deterministic: the same signal must always vote the same way.
    assert scout._derive_votes({"score": 0.4}, 5) == scout._derive_votes({"score": 0.4}, 5)


def test_votes_handle_a_zero_to_one_hundred_confidence():
    scout = _scout()
    assert set(scout._derive_votes({"confidence": 95}, 3)) == {"buy"}
    assert set(scout._derive_votes({"confidence": 5}, 3)) == {"sell"}


def test_trends_without_a_client_are_empty_not_synthetic():
    assert _scout()._real_trends() == []


# ------------------------------------------------------- ml_strategy_engine


def test_the_ml_engine_constructs_without_tensorflow():
    """LSTM = object meant LSTM(100, ...) raised TypeError before any use."""
    from ml_strategy_engine import MLStrategyEngine

    assert MLStrategyEngine() is not None


def test_the_backtest_uses_the_data_it_is_given():
    from ml_strategy_engine import MLStrategyEngine

    engine = MLStrategyEngine()
    rising = engine.backtest_strategy(pd.DataFrame({"close": [100 + i * 0.3 for i in range(80)]}))
    falling = engine.backtest_strategy(pd.DataFrame({"close": [130 - i * 0.3 for i in range(80)]}))

    assert rising["available"] is True
    assert rising["total_return"] > 0
    assert falling["total_return"] < 0
    assert rising["avg_pnl"] != falling["avg_pnl"], "results must depend on the input"


def test_the_backtest_is_deterministic():
    from ml_strategy_engine import MLStrategyEngine

    engine = MLStrategyEngine()
    frame = pd.DataFrame({"close": [100 + i * 0.2 for i in range(60)]})
    assert engine.backtest_strategy(frame) == engine.backtest_strategy(frame)


def test_insufficient_data_is_reported_not_scored():
    from ml_strategy_engine import MLStrategyEngine

    engine = MLStrategyEngine()
    for supplied in (None, pd.DataFrame({"close": [100.0]})):
        result = engine.backtest_strategy(supplied)
        assert result["available"] is False
        assert result["sharpe_ratio"] is None


# ---------------------------------------------------- nobel_risk_management


def test_risk_figures_respond_to_the_actual_book():
    """They were the constants 0.05 and 0.10 whatever was held."""
    from nobel_risk_management import QuantumRiskManager

    manager = QuantumRiskManager({})
    manager.portfolio_value = 1000.0

    assert manager.calculate_daily_risk() == 0.0, "an empty book risks nothing"

    manager.positions = {"BTC/USDT": {"size": 1.0, "price": 100.0, "stop_loss": 97.0}}
    with_one = manager.calculate_daily_risk()
    assert with_one == pytest.approx(0.003)

    manager.positions["ETH/USDT"] = {"size": 2.0, "price": 100.0, "stop_loss": 95.0}
    assert manager.calculate_daily_risk() > with_one


def test_portfolio_risk_is_measured_dispersion():
    from nobel_risk_management import QuantumRiskManager

    steady = QuantumRiskManager({})
    steady.portfolio_history = [1000, 1001, 1002, 1003, 1004]

    volatile = QuantumRiskManager({})
    volatile.portfolio_history = [1000, 1200, 850, 1300, 700]

    assert volatile.calculate_portfolio_risk() > steady.calculate_portfolio_risk()


def test_neither_risk_figure_is_a_constant():
    import inspect

    from nobel_risk_management import QuantumRiskManager

    for method in (QuantumRiskManager.calculate_daily_risk, QuantumRiskManager.calculate_portfolio_risk):
        src = inspect.getsource(method)
        assert "return 0.05" not in src
        assert "return 0.10" not in src


# ------------------------------------ corrections to two earlier rejections


def test_the_quantum_engine_randomness_is_legitimate_initialisation():
    """np.random there seeds variational circuit parameters, as it should.

    Flagging it as fabricated data was wrong. It is left exactly as it is.
    """
    source = open("IBM_QUANTUM_ENGINE.py").read()
    assert "np.random.random(self.ansatz.num_parameters)" in source


def test_divine_intelligence_consumes_indicators_rather_than_inventing_them():
    """It reads market_data keys with sensible fallbacks and returns HOLD
    when it has no trained model. That is honest, not fabricated."""
    source = open("divine_intelligence_core.py").read()
    assert "market_data.get('rsi', 50)" in source
    assert "'signal': 'HOLD'" in source

    tree = ast.parse(source)
    fabricators = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"uniform", "randint", "gauss"}
    ]
    assert fabricators == []
