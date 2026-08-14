# LeanTrader

LeanTrader is an experimental trading research repository. The supported VPS path is a **paper-authority runner** that consumes public market data, simulates fees and slippage, persists its ledger, and stops opening positions when daily-loss or drawdown limits are reached. It can optionally mirror approved paper events to **Bybit Testnet only** to exercise authenticated order placement and reconciliation with test funds.

It does not promise profits and the supported runner cannot submit real exchange orders. Production endpoints and live-mode flags are rejected.

## Current status

- Supported: Bybit-compatible public OHLCV data through CCXT; strict candle validation; deterministic trend, momentum, and mean-reversion ensemble signals; measured market-regime detection; bounded evidence-gated weight adaptation; ATR stops; paper ledger persistence; simulated fees/slippage; daily-loss and drawdown halts; per-engine lifecycle/circuit breakers; explainable heartbeat decisions; Docker deployment; blocking CI; and an optional Bybit Testnet mirror with private credential validation, deterministic client IDs, durable event delivery, persistent order state, partial-fill accounting, restart reconciliation, exchange-balance and execution-performance evidence, endpoint verification, entry limits, and a local kill switch.
- Not approved for real funds: live order execution, production exchange credentials, native live exchange stops, or any claim of verified future profitability.
- Legacy: the repository still contains older research scripts and duplicated experimental systems. They are not part of the VPS image or supported release.

The supported intelligence can adapt only after closed paper trades, moves weights slowly, and keeps every component between 10% and 70%. It cannot rewrite code, deploy itself, enable live trading, or bypass risk limits. See `ENGINE_AUDIT.md` for the branch-by-branch audit and supported-engine boundary.

Wave 2 rehabilitates the former “ultra” concepts as deterministic shadow engines: smart scalping, MACD/ADX/Stochastic/OBV structure confirmation, spectral cycles, liquidity impact, news awareness, pattern memory, swarm consensus, moon/scout ranking, portfolio risk, net-cost arbitrage, walk-forward forecasting, champion/challenger rollback, FX normalization, execution reality, reconciliation, manipulation alerts, strategy capacity, provenance, Prometheus textfile metrics, and outbound-only Telegram alerts. See `ENGINE_CAPABILITY_LEDGER.md` for the exact name-to-responsibility mapping.

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
