from __future__ import annotations

import ccxt

from leantrader.production import ccxt_compat
from leantrader.production.arbitrage_monitor import CrossVenueQuoteCollector
from leantrader.production.runner import MarketFeed


def test_bybit_market_feed_defaults_to_spot_without_credentials():
    feed = MarketFeed("bybit")

    assert feed.exchange.options["defaultType"] == "spot"
    assert not getattr(feed.exchange, "apiKey", "")
    assert not getattr(feed.exchange, "secret", "")


def test_bybit_arbitrage_adapter_uses_same_public_spot_default():
    exchange = CrossVenueQuoteCollector._ccxt_exchange("bybit")

    assert exchange.options["defaultType"] == "spot"


def test_explicit_bybit_market_type_is_preserved():
    exchange = ccxt.bybit({"options": {"defaultType": "swap"}})

    assert exchange.options["defaultType"] == "swap"


def test_public_spot_compat_installation_is_idempotent():
    exchange_class = ccxt.bybit

    ccxt_compat.install_public_spot_defaults()

    assert ccxt.bybit is exchange_class
