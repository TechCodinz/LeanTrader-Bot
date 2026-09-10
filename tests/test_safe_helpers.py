import os

import pytest

from order_utils import safe_create_order
from paper_broker import PaperBroker

def test_safe_create_order_market_requires_a_reference_price():
    """A paper market order with no price must be refused, not filled at zero.

    This previously asserted a successful fill for an order carrying no price.
    It only passed because the emulator had no zero-price guard: ref_price fell
    through as 0.0 and the order came back ok=True, status=filled, avg_px=0.0 --
    a fabricated fill at a fabricated price. The universal execution checkpoint
    added the guard, so the refusal below is the corrected behaviour.
    """
    ex = PaperBroker(1000.0)
    res = safe_create_order(ex, "market", "BTC/USDT", "buy", 0.001)
    assert isinstance(res, dict)
    assert res.get("ok") is False
    assert res.get("executed") is not True
    assert res.get("error") == "paper_reference_price_unavailable"


def test_safe_create_order_market_fills_when_a_price_is_supplied():
    """The same order succeeds deterministically once priced."""
    ex = PaperBroker(1000.0)
    res = safe_create_order(ex, "market", "BTC/USDT", "buy", 0.001, price=50_000.0)

    # safe_create_order flattens the router receipt, so order fields sit at the
    # top level rather than under an "order" key.
    assert res.get("ok") is True
    assert res["side"] == "buy"
    assert res["status"] == "filled"
    assert res["id"]
    assert res["filled"] == pytest.approx(0.001)
    # buy pays the emulator's 2 bps: 50000 * (1 + 2/10000)
    assert res["avg_px"] == pytest.approx(50_000.0 * 1.0002)

def test_exchange_router_paper_mode_fetch_and_order():
    # exercise the ExchangeRouter in paper mode: fetch_ohlcv fallback and create_order
    from router import ExchangeRouter

    # ensure we use paper backend
    os.environ["EXCHANGE_ID"] = "paper"
    ex = ExchangeRouter()
    # fetch ohlcv (PaperBroker returns synthetic or empty but should not raise)
    bars = ex.safe_fetch_ohlcv("BTC/USDT", "1m", limit=5)
    assert isinstance(bars, list)
    # create a paper market order (router.safe_place_order signature: symbol, side, amount, price=None, params=None)
    # Unpriced market orders are refused rather than filled at zero; see
    # test_safe_create_order_market_requires_a_reference_price.
    res = ex.safe_place_order("BTC/USDT", "buy", 0.001)
    assert isinstance(res, dict)
    assert res.get("ok") is False
    assert res.get("error") == "paper_reference_price_unavailable"

def test_tg_notifier_mocked(monkeypatch):
    # ensure notifier will call Telegram endpoints; mock requests.post to avoid network
    import notifier

    called = {}

    def fake_post(url, *args, **kwargs):
        called["url"] = url

        class R:
            status_code = 200

            def json(self):
                return {"ok": True, "result": {"message_id": 1}}

        return R()

    # ensure notifier instance thinks it's enabled
    import os

    import requests

    os.environ["TELEGRAM_ENABLED"] = "true"
    os.environ["TELEGRAM_BOT_TOKEN"] = "fake-token"
    os.environ["TELEGRAM_CHAT_ID"] = "1"
    monkeypatch.setattr(requests, "post", fake_post)
    tn = notifier.TelegramNotifier()
    # call the internal _send which uses requests.post under the hood
    tn._send("test message")
    assert called.get("url") is not None

def test_fetch_ohlcv_returns_nothing_rather_than_synthetic_bars():
    """A router that cannot reach its exchange must return no bars.

    This previously asserted that `limit` bars come back when the exchange
    never loaded its markets. Those bars were manufactured: open, high, low and
    close all set to the last ticker price, or to 0.0 when even that was
    unavailable, which is how 2001 all-zero candles reached the market cache.
    Callers cannot tell them from real candles, so indicators and backtests were
    running on fabricated history exactly when the exchange was known
    unreachable. Returning nothing is the honest answer and callers already
    handle it.
    """
    import os

    from router import ExchangeRouter

    os.environ["EXCHANGE_ID"] = "paper"
    ex = ExchangeRouter()
    ex._exchange_malformed = True
    bars = ex.safe_fetch_ohlcv("BTC/USDT", "1m", limit=3)
    assert isinstance(bars, list)
    assert bars == [], "no synthetic candles may be manufactured"

def test_place_oco_with_paper_broker():
    from order_utils import place_oco_ccxt
    from paper_broker import PaperBroker

    ex = PaperBroker(10000.0)
    res = place_oco_ccxt(ex, "BTC/USDT", "buy", 0.001, entry_px=100.0, stop_px=90.0, take_px=110.0)
    assert isinstance(res, dict)
    assert "entry" in res and "tp" in res and "sl" in res
    assert isinstance(res["entry"], dict)
