from __future__ import annotations

import os
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

from notifier import TelegramNotifier
from tools.market_data import get_market_data_manager
from tools.ultra_trainer import UltraTrainer


DEFAULT_SYMBOLS: List[str] = [
    # crypto majors (concise)
    'BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT','DOGE/USDT','LINK/USDT','AVAX/USDT','MATIC/USDT','TON/USDT',
    # metals
    'XAUUSD','XAGUSD','XPTUSD','XPDUSD',
    # commodities
    'USOIL','UKOIL','NATGAS',
    # forex majors
    'EURUSD','GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD','NZDUSD',
    # forex minors
    'EURGBP','EURJPY','GBPJPY','AUDJPY','NZDJPY','EURAUD','GBPAUD','EURCAD','AUDCAD','CADJPY','CHFJPY','EURNZD','GBPCAD','EURCHF',
    # select exotics
    'USDTRY','USDZAR','USDMXN','USDNOK','USDSEK','USDCNH'
]


def _parse_list(env_name: str, default: List[str]) -> List[str]:
    v = os.getenv(env_name, '')
    if not v:
        return default
    out: List[str] = []
    for p in v.split(','):
        p = p.strip()
        if p:
            out.append(p)
    return out or default


def _tf_list() -> List[str]:
    return _parse_list('TRAIN_TIMEFRAMES', ['5m','15m','1h'])


def _limit_for(tf: str, days: int) -> int:
    tf = tf.lower()
    if tf.endswith('m'):
        mins = int(tf[:-1])
        per_day = int(60*24/mins)
        return max(100, days * per_day)
    if tf.endswith('h'):
        hours = int(tf[:-1])
        per_day = int(24/hours)
        return max(100, days * per_day)
    if tf.endswith('d'):
        d = int(tf[:-1])
        return max(30, int(days / d))
    return days * 288


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _next_run_at(when_utc: str) -> datetime:
    # when_utc format: HH:MM
    hh, mm = [int(x) for x in when_utc.split(':')]
    now = _utc_now()
    run = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if run <= now:
        run = run + timedelta(days=1)
    return run


def _sleep_until(dt_utc: datetime) -> None:
    now = _utc_now()
    secs = max(1, int((dt_utc - now).total_seconds()))
    time.sleep(secs)


def _summarize(results: List[Dict[str, Any]]) -> str:
    lines = []
    ok = 0
    for r in results:
        sym = r.get('symbol')
        ns = r.get('n_samples', 0)
        if 'ensemble' in r:
            ok += 1
            ens = r['ensemble'] or {}
            # try to compute a compact quality score
            f1s = []
            for m in ens.values():
                if isinstance(m, dict) and 'f1' in m:
                    f1s.append(float(m.get('f1') or 0))
            score = (sum(f1s)/len(f1s)) if f1s else 0.0
            lines.append(f"• {sym}: n={ns} f1≈{score:.2f}")
        else:
            err = r.get('error','err')
            lines.append(f"• {sym}: error {err}")
    head = f"📚 Daily training: {ok}/{len(results)} models trained"
    return head + "\n" + "\n".join(lines[:20])


def run_once() -> Dict[str, Any]:
    symbols = _parse_list('TRAIN_SYMBOLS', DEFAULT_SYMBOLS)
    tfs = _tf_list()
    days = int(os.getenv('TRAIN_DAYS', '30'))

    mdm = get_market_data_manager()
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            # Prefer 5m training, fallback to next tf on empty
            df = None
            for tf in tfs:
                limit = _limit_for(tf, days)
                dft = mdm.fetch_ohlcv(sym, tf, limit=limit)
                if not dft.empty:
                    df = dft
                    break
            if df is None or df.empty:
                out.append({'symbol': sym, 'error': 'no data'})
                continue
            tr = UltraTrainer()
            res = tr.train_full_system(data_path=df, symbol=sym, task='classification')
            out.append(res)
        except Exception as e:
            out.append({'symbol': sym, 'error': str(e)})
    return {'results': out}


def main() -> int:
    notif = TelegramNotifier()
    log_dir = Path('runtime') / 'training_daily'
    log_dir.mkdir(parents=True, exist_ok=True)

    # schedule
    run_at = os.getenv('TRAIN_DAILY_UTC', '03:00')
    while True:
        next_dt = _next_run_at(run_at)
        _sleep_until(next_dt)
        try:
            payload = run_once()
            day = _utc_now().strftime('%Y%m%d')
            (log_dir / f'{day}.json').write_text(json.dumps(payload, indent=2))
            if notif.enabled:
                notif.note(_summarize(payload.get('results', [])))
        except Exception as e:
            if notif.enabled:
                notif.note(f"training scheduler error: {e}")
        # brief pause to avoid immediate loop on same minute
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(main())

