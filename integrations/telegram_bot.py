from __future__ import annotations

import os
import time
import json
import logging
from typing import Dict, Any

import requests
from requests.adapters import HTTPAdapter, Retry

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_FREE_CHAT_ID, TELEGRAM_VIP_CHAT_ID

log = logging.getLogger("telegram_bot")


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def _send_text(text: str, chat_id: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or TELEGRAM_BOT_TOKEN
    if not token or not chat_id:
        log.warning("telegram credentials missing; skipping send")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        r = _session().post(url, json=payload, timeout=10)
        if r.status_code == 429:
            retry_after = (r.json().get("parameters", {}).get("retry_after") if r.headers.get("Content-Type", "").startswith("application/json") else None) or 1
            time.sleep(float(retry_after))
            r = _session().post(url, json=payload, timeout=10)
        if r.ok:
            return True
        log.warning("telegram send failed: %s %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("telegram send error: %s", e)
    return False


def send_signal(text: str, *, vip: bool = False) -> bool:
    chat_id = (os.getenv("TELEGRAM_VIP_CHAT_ID") if vip else os.getenv("TELEGRAM_FREE_CHAT_ID")) or (TELEGRAM_VIP_CHAT_ID if vip else TELEGRAM_FREE_CHAT_ID)
    return _send_text(text, chat_id)


def _fmt_trade(payload: Dict[str, Any]) -> str:
    sym = payload.get("symbol", "?")
    side = payload.get("side", "HOLD").upper()
    entry = payload.get("entry", "-")
    sl = payload.get("sl", "-")
    tp = payload.get("tp", [])
    rr = payload.get("rr", "-")
    pnl = payload.get("pnl", None)
    conf = payload.get("confidence", None)
    lines = [
        f"{('🟢' if side == 'BUY' else '🔴' if side == 'SELL' else '🟡')} *{side}* `{sym}`",
        f"🎯 Entry: {entry}",
        f"🛑 SL: {sl}",
        f"🏁 TP: {', '.join(map(str, tp)) if tp else '-'}",
        f"📐 R:R: {rr}",
    ]
    if conf is not None:
        lines.append(f"📊 Confidence: {float(conf):.1%}")
    if pnl is not None:
        lines.append(f"💰 PnL: {float(pnl):+.2f}%")
    return "\n".join(lines)


def send_trade_report(payload: Dict[str, Any], *, vip: bool = False) -> bool:
    text = _fmt_trade(payload)
    return send_signal(text, vip=vip)