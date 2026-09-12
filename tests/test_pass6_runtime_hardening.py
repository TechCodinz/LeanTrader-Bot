
import os

from router import ExchangeRouter
from critical_features_addon import EmergencyStop
from ultra_multi_platform_scanner import (
    CEXScanner,
    DeFiScanner,
    OtherPlatformScanner,
)


class FakeExchange:
    def __init__(self):
        self.ticker_calls = []
        self.ohlcv_calls = []

    def fetch_ticker(self, symbol):
        self.ticker_calls.append(symbol)
        return {"symbol": symbol, "last": 100.0}

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=200):
        self.ohlcv_calls.append((symbol, timeframe))
        return [[1, 1, 1, 1, 1, 1]]


def make_router():
    r = ExchangeRouter.__new__(ExchangeRouter)
    r.id = "bybit"
    r.mode = "spot"
    r.execution_mode = "testnet"
    r.testnet = True
    r.live = False
    r._exchange_malformed = False
    r.ex = FakeExchange()
    r.markets = {
        "BTC/USDT": {
            "symbol": "BTC/USDT",
            "spot": True,
            "active": True,
        },
        "DOGE/USDT": {
            "symbol": "DOGE/USDT",
            "spot": True,
            "active": True,
        },
    }
    return r


def test_fx_never_reaches_bybit():
    r = make_router()

    assert r.fetch_ticker("EUR/USD") == {}
    assert r.ex.ticker_calls == []


def test_equity_never_reaches_bybit():
    r = make_router()

    assert r.fetch_ticker("AAPL") == {}
    assert r.ex.ticker_calls == []


def test_commodity_never_reaches_bybit():
    r = make_router()

    assert r.fetch_ticker("GOLD") == {}
    assert r.ex.ticker_calls == []


def test_absent_crypto_never_reaches_bybit():
    r = make_router()

    assert r.fetch_ticker("BCH/USDT") == {}
    assert r.ex.ticker_calls == []


def test_absent_crypto_ohlcv_never_reaches_bybit():
    r = make_router()

    assert r.fetch_ohlcv("FIL/USDT", "1m", 10) == []
    assert r.ex.ohlcv_calls == []


def test_valid_listed_crypto_still_reaches_exchange():
    r = make_router()

    out = r.fetch_ticker("BTC/USDT")

    assert out["last"] == 100.0
    assert r.ex.ticker_calls == ["BTC/USDT"]


def test_valid_listed_crypto_ohlcv_still_reaches_exchange():
    r = make_router()

    out = r.fetch_ohlcv("DOGE/USDT", "1m", 10)

    assert len(out) == 1
    assert r.ex.ohlcv_calls == [("DOGE/USDT", "1m")]


def test_all_synthetic_scanners_share_default_off_guard(monkeypatch):
    monkeypatch.delenv("LEANTRADER_SYNTHETIC_SCANNER", raising=False)

    for cls in (CEXScanner, DeFiScanner, OtherPlatformScanner):
        scanner = cls(None)

        assert scanner._synthetic_opportunities_enabled() is False


def test_emergency_stop_does_not_treat_free_cash_as_total_equity():
    stop = EmergencyStop(max_loss=0.10, max_trades_per_min=10)

    # $1 free cash versus $100 initial capital would look like a 99% loss
    # if free cash were incorrectly treated as total portfolio equity.
    assert (
        stop.check_conditions(
            account_balance=1.0,
            initial_balance=100.0,
            balance_is_total_equity=False,
        )
        is False
    )


def test_emergency_stop_still_protects_real_total_equity_drawdown():
    stop = EmergencyStop(max_loss=0.10, max_trades_per_min=10)

    assert (
        stop.check_conditions(
            account_balance=80.0,
            initial_balance=100.0,
            balance_is_total_equity=True,
        )
        is True
    )


def test_tick_size_precision_eth_exact_vps_holding():
    from src.leantrader.execution.inventory import _round_down

    assert _round_down(
        0.00422016,
        1e-05,
    ) == 0.00422


def test_tick_size_precision_sol_exact_vps_holding():
    from src.leantrader.execution.inventory import _round_down

    assert _round_down(
        0.0176035,
        0.001,
    ) == 0.017


def test_eth_and_sol_are_sellable_with_live_bybit_constraints(monkeypatch):
    from src.leantrader.execution.inventory import reconcile

    monkeypatch.setenv(
        "INVENTORY_DUST_QUOTE",
        "1.0",
    )

    balance = {
        "free": {
            "ETH": 0.00422016,
            "SOL": 0.0176035,
        },
        "used": {
            "ETH": 0.0,
            "SOL": 0.0,
        },
        "total": {
            "ETH": 0.00422016,
            "SOL": 0.0176035,
        },
    }

    markets = {
        "ETH/USDT": {
            "active": True,
            "precision": {
                "amount": 1e-05,
                "price": 0.01,
            },
            "limits": {
                "amount": {
                    "min": 1e-05,
                },
                "cost": {
                    "min": 1.0,
                },
            },
        },
        "SOL/USDT": {
            "active": True,
            "precision": {
                "amount": 0.001,
                "price": 0.01,
            },
            "limits": {
                "amount": {
                    "min": 0.001,
                },
                "cost": {
                    "min": 1.0,
                },
            },
        },
    }

    tickers = {
        "ETH/USDT": {
            "last": 2575.24,
        },
        "SOL/USDT": {
            "last": 101.68,
        },
    }

    items = reconcile(
        balance=balance,
        markets=markets,
        tickers=tickers,
        venue="bybit",
    )

    by_symbol = {
        item.symbol: item
        for item in items
    }

    eth = by_symbol["ETH/USDT"]
    sol = by_symbol["SOL/USDT"]

    assert eth.rounded_sell_amount == 0.00422
    assert eth.executable_quote_value > 10.8
    assert eth.sellable is True
    assert eth.can_close_now is True

    assert sol.rounded_sell_amount == 0.017
    assert sol.executable_quote_value > 1.7
    assert sol.sellable is True
    assert sol.can_close_now is True


def test_authenticated_total_equity_parser():
    from critical_features_addon import (
        authenticated_total_equity,
    )

    balance = {
        "info": {
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalEquity": "32.57272157",
                        "totalWalletBalance": "1.43630428",
                    }
                ]
            }
        }
    }

    assert authenticated_total_equity(
        balance
    ) == 32.57272157


def test_equity_parser_never_falls_back_to_free_usdt():
    from critical_features_addon import (
        authenticated_total_equity,
    )

    balance = {
        "free": {
            "USDT": 1.43572181,
        },
        "total": {
            "USDT": 1.43572181,
        },
    }

    assert authenticated_total_equity(
        balance
    ) is None


def test_real_equity_drawdown_guard_uses_equity_not_free_cash():
    from critical_features_addon import EmergencyStop

    stop = EmergencyStop(
        max_loss=0.10,
        max_trades_per_min=10,
    )

    assert stop.check_conditions(
        account_balance=32.57,
        initial_balance=32.57,
        balance_is_total_equity=True,
    ) is False

    assert stop.check_conditions(
        account_balance=29.00,
        initial_balance=32.57,
        balance_is_total_equity=True,
    ) is True
