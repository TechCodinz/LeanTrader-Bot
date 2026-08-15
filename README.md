# LeanTrader

For outbound-only ChatGPT/Codex administration of the supported VPS, see the [Secure VPS Operations Bridge](VPS_MCP_BRIDGE.md).

LeanTrader is an experimental trading research repository. The supported VPS path is a **paper-authority runner** that consumes public market data, simulates fees and slippage, persists its ledger, and stops opening positions when daily-loss or drawdown limits are reached. It can optionally mirror approved paper events to **Bybit Testnet only** to exercise authenticated order placement and reconciliation with test funds.

It does not promise profits and the supported runner cannot submit real exchange orders. Production endpoints and live-mode flags are rejected.

## Current status

- Supported: provider-aware public OHLCV and order-book data through any configured CCXT adapter that passes runtime attestation; a fail-closed exchange protection orchestrator that maps advertised API capabilities and market product rules to compatible research engines, and requires a complete protection contract before authenticated execution; exchange-server clock measurement; UTC-normalized timestamps; DST-aware FX sessions; only closed, fresh candles; dynamic discovery of every parseable exchange-advertised candle interval; exchange identity, data/order capabilities, market types, sandbox declaration, rate limits, precision, limits, fees and contract-rule coverage; the complete Bybit matrix (`1m,3m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M`) when Bybit is selected; timeframe-aware caching and fast/tactical/strategic consensus; dynamic discovery of active, sufficiently liquid quote markets; real CoinGecko market-cap/global/trending context; timestamped CoinDesk/Cointelegraph RSS ingestion with per-source freshness, robust symbol targeting, future-clock-skew rejection and high-impact blackouts; active Moon Scout cross-sectional ranking; read-only multi-venue arbitrage observation net of modeled costs; fair rotating coverage with persistent sweep progress; strict candle validation; deterministic adaptive and ultra-engine synthesis through a bounded decision router; ungated per-engine/timeframe paper evidence; structured OpenAI/Anthropic/Gemini research proposals with bounded schemas and no execution authority; market-specific exploration, qualification, probation, quarantine and retest; online Brier/ECE calibration from closed trades; measured drift; ATR stops; paper ledger persistence; simulated fees/slippage; daily-loss and drawdown halts; per-engine lifecycle/circuit breakers and activity counters; tiered admin/free/paid Telegram signal distribution and monitoring; explainable heartbeat decisions; Docker deployment; blocking CI; and an optional Bybit Testnet spot mirror with endpoint, exchange-identity, market-type, method-capability, API permission, IP-binding and credential validation, deterministic client IDs, durable event delivery, persistent order state, partial-fill accounting, restart reconciliation, exchange-balance and execution-performance evidence, entry limits, and a local kill switch.
- Not approved for real funds: live order execution, production exchange credentials, native live exchange stops, or any claim of verified future profitability.
- Legacy: the repository still contains older research scripts and duplicated experimental systems. They are not part of the VPS image or supported release.

The supported intelligence adapts from closed paper outcomes, moves weights slowly, and keeps every component between 10% and 70%. An optional external model can propose a bounded research challenger, but the proposal remains non-executing until causal replay and champion/challenger evidence qualify it for paper testing. No model can rewrite code, deploy itself, enable live trading, or bypass risk limits. See `ENGINE_AUDIT.md` for the branch-by-branch audit and supported-engine boundary.

The former “ultra” concepts are deterministic engines: smart scalping, MACD/ADX/Stochastic/OBV structure confirmation, spectral cycles, observed order-book liquidity, news awareness, pattern memory, swarm consensus, Moon Scout ranking, portfolio risk, net-cost cross-venue arbitrage, walk-forward forecasting, champion/challenger rollback, FX normalization, execution reality, reconciliation, manipulation alerts, strategy capacity, provenance, Prometheus textfile metrics, and tiered outbound Telegram alerts. Signal-producing ultra engines feed the bounded paper/Testnet decision router; monitoring/research-only engines remain non-executing and report why. See `ENGINE_CAPABILITY_LEDGER.md` for the exact boundary.

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

Set `DATA_EXCHANGE` to a CCXT exchange id, `CONFIRM_TIMEFRAMES=AUTO`, and `PAPER_SYMBOLS=AUTO` to attest the adapter and discover its full eligible quote-asset spot universe. The runtime removes inactive, leveraged-token, low-volume, invalid-spread, and excessive-spread markets and rotates through every remaining symbol. If the Bybit Testnet mirror is enabled, the public-data exchange must be Bybit and the universe is intersected with Testnet availability. `MARKET_SCAN_BATCH_SIZE` controls work per cycle, not the total universe; persistent `full_sweeps` telemetry proves complete eligible-set coverage over time. Existing positions are always evaluated even when outside the current rotation batch.

The `exchange_protection` engine builds a per-market research plan from the adapter's real capabilities. Candle engines require OHLCV support and the fluid-liquidity engine requires order-book support. Advertised trade-tape, funding-rate and open-interest methods are reported as available but unbound observations until real processing engines are integrated—availability is never presented as implementation. An authenticated API never inherits authority merely because CCXT can connect to it. Unknown venues and margin, swaps, futures, options, forex and cross-venue arbitrage are explicitly blocked from execution until their product-specific reconciliation, leverage/liquidation, settlement and recovery contracts are implemented and tested.

The `market_temporal_guard` heartbeat section proves server-clock agreement, candle-close filtering, per-timeframe freshness and current market session. `public_market_context.news_fresh` and `advanced_shadow_suite.capabilities.news_awareness` separately prove RSS collection freshness and locally ingested news freshness; a successful market-cap call cannot mask failed news feeds.

`cross_venue_arbitrage` collects public two-sided quotes from the configured venues (default: Bybit and OKX). The ultra arbitrage engine measures spreads net of taker fees and modeled slippage and reports whether top-of-book quantity is known. It never submits paired orders. Telegram can distribute gated summaries to separate free and paid channels, detailed alerts and health to an admin chat, and a one-tap Bybit Testnet link. Channel assignment is not payment verification; connect a real billing/subscription webhook before selling automated paid access.

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
