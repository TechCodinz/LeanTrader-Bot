from __future__ import annotations

import os
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List

from notifier import TelegramNotifier


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _start_of_week(dt: datetime) -> datetime:
    # Monday 00:00 UTC
    d = dt - timedelta(days=dt.weekday())
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _collect_training_summaries(days: int = 7) -> List[Dict[str, Any]]:
    root = Path('runtime') / 'training_daily'
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    files = sorted(root.glob('*.json'))[-days:]
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out


def _summarize_week(payloads: List[Dict[str, Any]]) -> str:
    total = 0
    ok = 0
    agg_f1s: List[float] = []
    by_symbol: Dict[str, int] = {}
    for p in payloads:
        results = p.get('results', [])
        total += len(results)
        for r in results:
            if isinstance(r, dict) and 'ensemble' in r:
                ok += 1
                sym = r.get('symbol')
                by_symbol[sym] = by_symbol.get(sym, 0) + 1
                ens = r.get('ensemble', {})
                f1s = [float(m.get('f1', 0)) for m in ens.values() if isinstance(m, dict) and 'f1' in m]
                if f1s:
                    agg_f1s.append(sum(f1s)/len(f1s))
    avg_f1 = (sum(agg_f1s)/len(agg_f1s)) if agg_f1s else 0.0
    top_syms = sorted(by_symbol.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_txt = ', '.join([f"{s}({n})" for s, n in top_syms]) if top_syms else 'n/a'
    return (
        f"📅 Weekly training summary\n"
        f"Models trained: {ok}/{total}\n"
        f"Avg ensemble f1 (where available): {avg_f1:.2f}\n"
        f"Most-updated symbols: {top_txt}"
    )


def main() -> int:
    notif = TelegramNotifier()
    if not (notif.enabled and notif.token and notif.chat_id):
        print('[weekly_digest] Telegram not enabled; exiting')
        return 1
    # Send once per week on Monday 08:00 UTC
    target_hhmm = os.getenv('WEEKLY_DIGEST_UTC', '08:00')
    while True:
        now = _utc_now()
        dow = now.weekday()
        hh, mm = [int(x) for x in target_hhmm.split(':')]
        # next target time
        start_week = _start_of_week(now)
        target = start_week + timedelta(hours=hh, minutes=mm)
        if now > target:
            target = target + timedelta(days=7)
        # sleep until target
        time.sleep(max(1, int((target - now).total_seconds())))
        try:
            payloads = _collect_training_summaries(7)
            msg = _summarize_week(payloads)
            notif.note(msg)
        except Exception as e:
            notif.note(f"weekly digest error: {e}")
        # small pause to avoid double-send
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(main())

