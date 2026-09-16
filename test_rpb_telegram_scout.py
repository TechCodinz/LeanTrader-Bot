"""Signal distribution to premium and free, and the connected scout.

The defect: REAL_PROFIT_BOT had admin, VIP and free channels configured and
every send_telegram call used the admin default, so the two product channels
had never received a signal. These tests pin the routing, and pin that no
channel failure can interrupt trading.

No network. No orders. No credential value appears in any message.
"""

import time

import pytest

from rpb_telegram import SignalDistributor


class Recorder:
    """Stands in for the bot's send_telegram, recording every delivery."""

    def __init__(self, fail_on=None):
        self.sent = []
        self.fail_on = fail_on

    def __call__(self, message, chat_id=None):
        if self.fail_on is not None and chat_id == self.fail_on:
            raise RuntimeError("telegram unreachable")
        self.sent.append((chat_id, message))
        return True

    def to(self, chat_id):
        return [m for c, m in self.sent if c == chat_id]


ADMIN, VIP, FREE = "admin-1", "vip-1", "free-1"


@pytest.fixture
def dist(monkeypatch):
    monkeypatch.setenv("RPB_FREE_DELAY_SECONDS", "0")
    monkeypatch.setenv("RPB_FREE_EVERY_N", "1")
    import importlib

    import rpb_telegram

    importlib.reload(rpb_telegram)
    recorder = Recorder()
    d = rpb_telegram.SignalDistributor(
        recorder, admin_chat_id=ADMIN, vip_chat_id=VIP, free_chat_id=FREE
    )
    d._recorder = recorder
    return d


# ------------------------------------------------------------- routing


def test_an_entry_reaches_the_premium_channel(dist):
    dist.broadcast_entry("DOT/USDT", 0.9422, 10.0, 9.42, 95.0,
                         take_profit_bps=50, stop_loss_bps=30)
    premium = dist._recorder.to(VIP)
    assert len(premium) == 1
    body = premium[0]
    assert "PREMIUM SIGNAL" in body and "DOT/USDT" in body
    assert "Target:" in body and "Stop:" in body


def test_the_free_channel_gets_the_signal_without_the_levels(dist):
    dist.broadcast_entry("DOT/USDT", 0.9422, 10.0, 9.42, 95.0,
                         take_profit_bps=50, stop_loss_bps=30)
    time.sleep(0.05)
    free = dist._recorder.to(FREE)
    assert len(free) == 1
    body = free[0]
    assert "DOT/USDT" in body
    # The actionable levels are the premium product.
    assert "Target:" not in body and "Stop:" not in body


def test_an_exit_reports_reconciled_net_pnl(dist):
    dist.broadcast_exit("DOT/USDT", 0.9422, 0.9500, 10.0,
                        gross_pnl=0.078, fees=0.0094,
                        realized_net_pnl=0.0686, hold_seconds=42.0,
                        exit_reason="take_profit", wallet_balance=13.62)
    body = dist._recorder.to(VIP)[0]
    assert "NET:" in body and "+0.068600" in body
    assert "take_profit" in body
    assert "Fees:" in body


def test_a_loss_is_reported_as_a_loss(dist):
    dist.broadcast_exit("X/USDT", 1.0, 0.99, 10.0,
                        gross_pnl=-0.1, fees=0.02, realized_net_pnl=-0.12,
                        hold_seconds=10.0, exit_reason="stop_loss",
                        wallet_balance=13.0)
    body = dist._recorder.to(VIP)[0]
    assert "🔴" in body and "-0.120000" in body
    time.sleep(0.05)
    assert "loss" in dist._recorder.to(FREE)[0].lower()


def test_performance_is_not_sent_before_any_close(dist):
    dist.broadcast_performance({"closed_positions": 0}, 13.56)
    assert dist._recorder.sent == []


def test_performance_reaches_both_channels(dist):
    dist.broadcast_performance(
        {"closed_positions": 4, "authentic_wins": 3, "authentic_win_rate": 75.0,
         "total_fees": 0.04, "realized_net_pnl": 0.21}, 13.77)
    assert len(dist._recorder.to(VIP)) == 1
    assert len(dist._recorder.to(FREE)) == 1
    assert "75.0%" in dist._recorder.to(VIP)[0]


# -------------------------------------------- delivery can never block trading


def test_a_dead_premium_channel_does_not_raise(monkeypatch):
    monkeypatch.setenv("RPB_FREE_DELAY_SECONDS", "0")
    import importlib

    import rpb_telegram

    importlib.reload(rpb_telegram)
    recorder = Recorder(fail_on=VIP)
    d = rpb_telegram.SignalDistributor(
        recorder, admin_chat_id=ADMIN, vip_chat_id=VIP, free_chat_id=FREE
    )
    d.broadcast_entry("X/USDT", 1.0, 1.0, 1.0, 90.0)  # must not raise
    assert d.delivered["failed"] >= 1


def test_an_unconfigured_channel_is_skipped_silently():
    recorder = Recorder()
    d = SignalDistributor(recorder, admin_chat_id=ADMIN, vip_chat_id="", free_chat_id="")
    d.broadcast_entry("X/USDT", 1.0, 1.0, 1.0, 90.0)
    assert recorder.sent == []


def test_broadcasting_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RPB_BROADCAST_ENABLED", "0")
    import importlib

    import rpb_telegram

    importlib.reload(rpb_telegram)
    recorder = Recorder()
    d = rpb_telegram.SignalDistributor(
        recorder, admin_chat_id=ADMIN, vip_chat_id=VIP, free_chat_id=FREE
    )
    d.broadcast_entry("X/USDT", 1.0, 1.0, 1.0, 90.0)
    assert recorder.sent == []


def test_no_credential_ever_appears_in_a_message(dist):
    dist.broadcast_entry("DOT/USDT", 0.94, 10.0, 9.4, 95.0)
    dist.broadcast_exit("DOT/USDT", 0.94, 0.95, 10.0, 0.1, 0.01, 0.09, 5.0, "tp", 13.6)
    time.sleep(0.05)
    for _chat, body in dist._recorder.sent:
        assert "bot" not in body.lower() or "token" not in body.lower()
        assert ":AA" not in body  # the shape of a telegram bot token


def test_the_distributor_places_no_orders():
    import ast
    import inspect

    import rpb_telegram as mod

    tree = ast.parse(inspect.getsource(mod))
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in ("create_order", "create_market_buy_order", "create_market_sell_order"):
        assert forbidden not in called


# ------------------------------------------------------------- bot wiring


def test_the_bot_routes_real_fills_to_both_channels():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "from rpb_telegram import SignalDistributor" in src
    assert "self.signals.broadcast_entry(" in src
    assert "self.signals.broadcast_exit(" in src
    assert "self.signals.broadcast_performance(" in src


def test_broadcasts_come_from_reconciled_events_only():
    """Entry broadcasts sit inside the recorded-fill branch, not the ack."""
    src = open("REAL_PROFIT_BOT.py").read()
    entry_at = src.index("self.signals.broadcast_entry(")
    guard_at = src.index('if record and getattr(self, "risk"')
    assert guard_at < entry_at, "entry broadcast must follow a recorded fill"

    exit_at = src.index("self.signals.broadcast_exit(")
    settle_at = src.index("settled = self.ledger.close_position(")
    assert settle_at < exit_at, "exit broadcast must follow settlement"


def test_every_existing_admin_message_is_preserved():
    src = open("REAL_PROFIT_BOT.py").read()
    for kept in ("REAL PROFIT BOT ACTIVATED", "POSITION OPENED", "POSITION CLOSED",
                 "REAL PROFIT SIGNAL", "def send_telegram"):
        assert kept in src, kept


def test_the_scout_is_connected_read_only():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "from ultra_scout import UltraScout" in src
    assert "UltraScout(exchange=self.gate)" in src
    # It annotates reasoning; it must not gate.
    scout_at = src.index("self.scout._get_liquidity_depth(symbol)")
    window = src[scout_at:scout_at + 700]
    assert "return" not in window.split("except Exception")[0], "scout must not veto"
