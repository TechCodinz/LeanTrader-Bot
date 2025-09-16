#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data import get_market_data_manager

DEFAULT_SYMBOLS: List[str] = [
    'BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT',
    'XAUUSD','XAGUSD','USOIL','EURUSD','GBPUSD','USDJPY'
]
DEFAULT_TFS: List[str] = ['5m','15m','1h','4h','1d']


def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.getenv(name, '')
    if not v:
        return default
    out: List[str] = []
    for p in v.split(','):
        p = p.strip()
        if p:
            out.append(p)
    return out or default


def main() -> int:
    symbols = _env_list('COLLECT_SYMBOLS', DEFAULT_SYMBOLS)
    tfs = _env_list('COLLECT_TFS', DEFAULT_TFS)
    limit = int(os.getenv('COLLECT_LIMIT', '720'))  # about a month for 1h

    # Optional: set a proxy for this process
    http_proxy = os.getenv('HTTP_PROXY') or ''
    https_proxy = os.getenv('HTTPS_PROXY') or ''
    if http_proxy or https_proxy:
        print(f"[collector] proxies configured http={bool(http_proxy)} https={bool(https_proxy)}")

    mdm = get_market_data_manager()
    out = {"collected": []}
    for sym in symbols:
        for tf in tfs:
            try:
                df = mdm.fetch_ohlcv(sym, tf, limit)
                rows = 0 if df is None else len(df)
                print(f"[collector] {sym} {tf} -> rows={rows}")
                out["collected"].append({"symbol": sym, "tf": tf, "rows": rows})
            except Exception as e:
                print(f"[collector] error {sym} {tf}: {e}")
                out["collected"].append({"symbol": sym, "tf": tf, "error": str(e)})

    logdir = ROOT / 'runtime' / 'collector'
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / 'latest.json').write_text(json.dumps(out, indent=2))
    print(f"[collector] done. wrote {logdir / 'latest.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())