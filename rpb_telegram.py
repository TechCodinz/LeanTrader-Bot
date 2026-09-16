"""Signal distribution to the premium and free Telegram channels.

REAL_PROFIT_BOT has three channels configured -- admin, VIP and free -- and a
send_telegram(message, chat_id=None) that defaults to admin. Every call site
used that default, so the VIP and free channels were wired up and never
received anything. The product side of the bot was disconnected in exactly
the way the trading intelligence was.

This routes three audiences from one event:

* admin   -- operational detail: startup, errors, fills, reconciliation
* premium -- the full trade: entry, size, targets, reasoning, live result
* free    -- the same trade, delayed and without the actionable levels, so the
             channel is genuinely useful without being a substitute for premium

Nothing here decides or places a trade, and no send can interrupt trading:
every delivery is best-effort and a failure is logged, never raised. No token
or chat id is ever written into a message body.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

# How long the free channel lags premium. The point is that premium is
# actionable first; free still gets the full record.
FREE_CHANNEL_DELAY_SECONDS = float(os.getenv("RPB_FREE_DELAY_SECONDS", "90"))

# Free-channel volume control: not every trade, so the channel stays readable.
FREE_CHANNEL_EVERY_N = int(os.getenv("RPB_FREE_EVERY_N", "2") or 2)

BROADCAST_ENABLED = os.getenv("RPB_BROADCAST_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off",
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class SignalDistributor:
    """Fans one real trading event out to the three audiences.

    Takes the bot's own ``send_telegram`` and channel ids rather than building
    a second Telegram client, so there is one delivery path and one set of
    credentials, exactly where they already live.
    """

    def __init__(
        self,
        send: Callable[..., Any],
        *,
        admin_chat_id: str = "",
        vip_chat_id: str = "",
        free_chat_id: str = "",
    ) -> None:
        self._send = send
        self.admin_chat_id = str(admin_chat_id or "")
        self.vip_chat_id = str(vip_chat_id or "")
        self.free_chat_id = str(free_chat_id or "")
        self._free_counter = 0
        self._timers: List[threading.Timer] = []
        self.delivered = {"admin": 0, "premium": 0, "free": 0, "failed": 0}

    # ------------------------------------------------------------ delivery

    def _deliver(self, audience: str, chat_id: str, message: str) -> bool:
        if not BROADCAST_ENABLED or not chat_id or not message:
            return False
        try:
            self._send(message, chat_id)
            self.delivered[audience] = self.delivered.get(audience, 0) + 1
            return True
        except Exception as exc:
            # A channel being unreachable must never stop the bot trading.
            self.delivered["failed"] += 1
            print(f"⚠️ Telegram {audience} delivery failed: {type(exc).__name__}: {exc}")
            return False

    def _deliver_later(self, audience: str, chat_id: str, message: str, delay: float) -> None:
        if not BROADCAST_ENABLED or not chat_id or delay <= 0:
            self._deliver(audience, chat_id, message)
            return
        timer = threading.Timer(delay, self._deliver, args=(audience, chat_id, message))
        timer.daemon = True
        timer.start()
        self._timers = [t for t in self._timers if t.is_alive()][-50:] + [timer]

    # -------------------------------------------------------------- events

    def broadcast_entry(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        cost: float,
        confidence: float,
        *,
        order_id: str = "",
        reasoning: str = "",
        take_profit_bps: float = 0.0,
        stop_loss_bps: float = 0.0,
    ) -> None:
        """A real, authenticated entry. Never called from an unfilled ack."""
        entry = _f(entry_price)
        target = entry * (1 + take_profit_bps / 10_000.0) if take_profit_bps else 0.0
        stop = entry * (1 - stop_loss_bps / 10_000.0) if stop_loss_bps else 0.0

        premium = f"""🟢 <b>PREMIUM SIGNAL — ENTRY</b>

💎 <b>{symbol}</b>
📥 <b>Entry:</b> ${entry:.8f}
📦 <b>Size:</b> {_f(quantity):.8f}
💵 <b>Allocated:</b> ${_f(cost):.4f}
🔥 <b>Confidence:</b> {_f(confidence):.0f}%"""
        if target:
            premium += f"\n🎯 <b>Target:</b> ${target:.8f} (+{take_profit_bps:.0f} bps)"
        if stop:
            premium += f"\n🛡 <b>Stop:</b> ${stop:.8f} (−{stop_loss_bps:.0f} bps)"
        if reasoning:
            premium += f"\n\n🧠 <b>Why:</b> {reasoning}"
        premium += "\n\n<i>Live position — exit will be posted here.</i>"

        self._deliver("premium", self.vip_chat_id, premium)

        self._free_counter += 1
        if FREE_CHANNEL_EVERY_N > 0 and self._free_counter % FREE_CHANNEL_EVERY_N == 0:
            free = f"""📈 <b>SIGNAL — ENTRY</b>

💰 <b>{symbol}</b>
📥 <b>Entry zone:</b> ${entry:.6f}
🔥 <b>Confidence:</b> {_f(confidence):.0f}%

<i>Targets, stop and live exits go to premium members first.</i>"""
            self._deliver_later("free", self.free_chat_id, free, FREE_CHANNEL_DELAY_SECONDS)

    def broadcast_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        gross_pnl: float,
        fees: float,
        realized_net_pnl: float,
        hold_seconds: float,
        exit_reason: str,
        wallet_balance: float,
    ) -> None:
        """A real, authenticated close with reconciled net PnL."""
        net = _f(realized_net_pnl)
        entry = _f(entry_price)
        won = net >= 0
        emoji = "🟢" if won else "🔴"
        pct = ((_f(exit_price) - entry) / entry * 100.0) if entry > 0 else 0.0

        premium = f"""{emoji} <b>PREMIUM SIGNAL — CLOSED</b>

💎 <b>{symbol}</b>
📥 <b>Entry:</b> ${entry:.8f}
📤 <b>Exit:</b> ${_f(exit_price):.8f}  ({pct:+.3f}%)
📦 <b>Size:</b> {_f(quantity):.8f}
🚪 <b>Reason:</b> {exit_reason}
⏱ <b>Held:</b> {_f(hold_seconds):.1f}s

📊 <b>Gross:</b> ${_f(gross_pnl):+.6f}
🧾 <b>Fees:</b> ${_f(fees):.6f}
{emoji} <b>NET:</b> ${net:+.6f}
👛 <b>Wallet:</b> ${_f(wallet_balance):.6f}

<i>Reconciled from the exchange fill.</i>"""
        self._deliver("premium", self.vip_chat_id, premium)

        if FREE_CHANNEL_EVERY_N > 0 and self._free_counter % FREE_CHANNEL_EVERY_N == 0:
            free = f"""{emoji} <b>RESULT — {symbol}</b>

{'✅ Closed in profit' if won else '⚠️ Closed at a loss'}  ({pct:+.3f}%)
⏱ Held {_f(hold_seconds):.0f}s · {exit_reason}

<i>Entries and exits in real time on premium.</i>"""
            self._deliver_later("free", self.free_chat_id, free, FREE_CHANNEL_DELAY_SECONDS)

    def broadcast_performance(self, stats: Dict[str, Any], wallet_balance: float) -> None:
        """Periodic real performance. Authenticated figures only."""
        net = _f(stats.get("realized_net_pnl"))
        closed = int(stats.get("closed_positions") or 0)
        if closed <= 0:
            return
        emoji = "🟢" if net >= 0 else "🔴"

        body = f"""{emoji} <b>PERFORMANCE</b>

📊 <b>Closed trades:</b> {closed}
✅ <b>Wins:</b> {int(stats.get('authentic_wins') or 0)}
🎯 <b>Win rate:</b> {_f(stats.get('authentic_win_rate')):.1f}%
🧾 <b>Fees paid:</b> ${_f(stats.get('total_fees')):.6f}
{emoji} <b>Realized net:</b> ${net:+.6f}
👛 <b>Wallet:</b> ${_f(wallet_balance):.6f}

<i>All figures reconciled from exchange fills.</i>"""
        self._deliver("premium", self.vip_chat_id, body)
        self._deliver("free", self.free_chat_id, body)

    def health(self) -> Dict[str, Any]:
        return {
            "enabled": BROADCAST_ENABLED,
            "premium_channel": bool(self.vip_chat_id),
            "free_channel": bool(self.free_chat_id),
            "admin_channel": bool(self.admin_chat_id),
            "free_delay_seconds": FREE_CHANNEL_DELAY_SECONDS,
            "free_every_n": FREE_CHANNEL_EVERY_N,
            "delivered": dict(self.delivered),
        }
