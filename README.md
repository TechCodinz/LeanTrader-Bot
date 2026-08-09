# LeanTrader

LeanTrader is an experimental trading research repository. The only currently supported production path is a **paper-only VPS runner** that consumes public market data, simulates fees and slippage, persists its ledger, and stops opening positions when daily-loss or drawdown limits are reached.

It does not promise profits and the supported runner cannot submit real exchange orders.

## Current status

- Supported: Bybit-compatible public OHLCV data through CCXT, EMA/Bollinger breakout signals, ATR stops, paper ledger persistence, simulated fees/slippage, daily-loss and drawdown halts, heartbeat health checks, Docker deployment, and blocking CI.
- Not approved for real funds: live order execution, exchange reconciliation, restart-safe live order state, native exchange stops, or verified multi-week strategy performance.
- Legacy: the repository still contains older research scripts and duplicated experimental systems. They are not part of the VPS image or supported release.

## Local preflight

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.runtime.txt
cp .env.production.example .env
PYTHONPATH=src python -m leantrader.production.runner --preflight
```

To run one cycle against public market data:

```bash
PYTHONPATH=src python -m leantrader.production.runner --once
```

## Docker

```bash
cp .env.production.example .env
docker compose config
docker compose up -d --build
docker compose ps
docker compose logs -f --tail=100 leantrader
```

See `VPS_RUNBOOK.md` for VPS sizing, deployment, security actions, and the paper-to-live promotion gate.

## Security warning

Credentials were previously committed to this public repository. They must be revoked before any exchange account is funded. The newest branch removes the credential files, but historical commits remain exposed until a separate coordinated history rewrite is completed.
