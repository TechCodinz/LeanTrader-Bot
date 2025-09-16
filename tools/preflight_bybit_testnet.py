from __future__ import annotations

import os
import json


def main() -> int:
    # Env expected
    cfg = {
        'EXCHANGE_ID': os.getenv('EXCHANGE_ID', 'bybit'),
        'BYBIT_TESTNET': os.getenv('BYBIT_TESTNET', 'true'),
        'ENABLE_LIVE': os.getenv('ENABLE_LIVE', 'false'),
        'ALLOW_LIVE': os.getenv('ALLOW_LIVE', 'false'),
        'LIVE_CONFIRM': os.getenv('LIVE_CONFIRM', 'NO'),
        'BYBIT_API_KEY': os.getenv('BYBIT_API_KEY', ''),
        'BYBIT_API_SECRET': os.getenv('BYBIT_API_SECRET', ''),
    }
    out = {'env_ok': True, 'checks': [], 'router_ok': False}

    # Basic env checks
    want = ['BYBIT_API_KEY', 'BYBIT_API_SECRET']
    for k in want:
        ok = bool(cfg.get(k))
        out['checks'].append({'name': k, 'ok': ok})
        if not ok:
            out['env_ok'] = False

    # Router sanity
    try:
        from router import ExchangeRouter
        r = ExchangeRouter()
        r._load_markets_safe()
        # public calls
        t = r.safe_fetch_ticker('BTC/USDT') or {}
        ohlcv = r.safe_fetch_ohlcv('BTC/USDT', '1m', limit=5) or []
        out['router_ok'] = True
        out['ticker_last'] = float(t.get('last') or 0.0)
        out['ohlcv_len'] = len(ohlcv)
    except Exception as e:
        out['router_ok'] = False
        out['router_error'] = str(e)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

