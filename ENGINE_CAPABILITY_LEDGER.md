# LeanTrader verified capability ledger

This ledger prevents a named engine from being mistaken for a working capability. A capability is complete only when it has deterministic logic, explicit inputs and outputs, tests, lifecycle health, and a declared authority level.

Canonical runtime release: `verified-multi-engine-v12.3-cns-brain-memory`.

## Canonical runtime authority

- **Trade authority:** the bounded decision router combines the adaptive ensemble with measured ultra-engine consensus, market-specific evidence, and capital/risk governors. Approved paper events may be mirrored on Bybit Testnet only.
- **Research authority:** scanners, memory, arbitrage, forecasting, quantum and experimental adapters may observe or score but cannot independently place orders. Unconfigured engines report an explicit blocked state.
- **Operations authority:** provenance, Prometheus monitoring and tiered outbound Telegram alerts; no inbound order commands.
- **Live authority:** none. All live flags are rejected during startup.

## Previous names mapped to real responsibilities

| Previous engine or request | Canonical verified responsibility | Runtime status |
|---|---|---|
| Central nervous system / unified runner / ultra launcher | Dedicated CNS state fusion over adaptive, Ultra/swarm, liquidity, bounded router, memory and exchange-protection evidence; dependency-aware registry, circuit breakers and health manifest | Active; CNS can only inform or restrict, never execute |
| Divine intelligence | Deterministic decision synthesis, rationale, provenance fingerprint | Active; no random or spiritual signals |
| Brain / Awareness / GloAware | Bounded meta-controller over CNS, causal memory and cost-adjusted strategy evidence; may preserve, downsize or veto upstream entries and quarantine persistent negative expectancy | Active; cannot increase risk, enable live, rewrite or deploy |
| Hivemind / swarm consciousness | Confidence-weighted consensus with disagreement penalty | Active router input |
| Photographic memory / pattern memory | Working + episodic + semantic memory; only closed outcomes are promoted, both wins and losses retained, similarity/recency/importance weighted, restart-persistent | Active Brain input; causal closed outcomes only |
| Frequency / harmonics / ultrasonic | Detrended FFT cycle measurement and concentration | Active router input |
| Fluid mechanics | Observed order-book pressure, spread, impact and safe quantity | Active router input when depth is available; failure visible |
| Dynamic scanner / Ultra Scout | Cross-sectional price, volume, liquidity and volatility anomaly ranking | Shadow |
| Moon spotter / hype radar | Measured cross-sectional momentum, volume, liquidity and volatility anomaly ranking with per-cycle activity, top-candidate telemetry and paid/admin alerts | Active scanner; no order authority |
| Smart scalping | EMA/RSI/volume signal with spread penalty | Active router input |
| Main technical/microstructure indicators | Deterministic MACD, ADX, Stochastic, OBV and liquidity-sweep confirmation | Active router input |
| Hedge-fund arsenal | Portfolio correlation, concentration, VaR and deterministic stress scenarios | Active risk observation |
| Evolution engine | Bounded adaptive weights, closed-trade market evidence, online calibration, plus champion/challenger promotion and rollback | Adaptive/evidence learning active; model promotion offline |
| Ultra ML pipeline | Cost-aware gradient-boosting walk-forward forecaster and calibration | Offline/shadow |
| Kronos | Validated external predictor adapter; unavailable is reported explicitly until configured | Adapter complete; model not bundled |
| Optuna | Seeded offline optimizer; unavailable is explicit when optional dependency is absent | Offline adapter |
| IBM / ultra quantum | Candidate optimizer benchmarked against a classical baseline | Benchmark only |
| Arbitrage engine | Cached read-only quotes from at least two runtime-attested CCXT venues; observed cross-venue bid/ask spreads net of fees, slippage and optional transfer costs; quote/source failures and liquidity verification visible | Active monitor, no execution |
| Ultra Forex Master | XAUUSD/EURUSD normalization, pip/session and risk-unit calculations for Oanda/MT5 formats | Shadow; no broker orders |
| Ultra business system | Win rate, profit factor and expectancy from realized paper events | Monitoring only |
| News intelligence | Timestamped RSS collection through public market context, per-source and latest-item freshness, robust full-pair/base matching, UTC normalization, future-skew rejection, local auditable event state, decayed sentiment and high-impact blackout | Active bounded router input; collector provenance visible |
| Telegram notifier | Secret-file token loading; separate admin, free and paid chats/channels; confidence thresholds; cooldown/deduplication; gated paper signals; Moon Scout, arbitrage and exchange-protection block alerts; periodic health digests; fill/halt alerts; one-tap Bybit Testnet link; per-audience delivery metrics | Optional outbound distribution; no inbound execution, live authority or automatic payment verification |
| Prometheus/Grafana monitoring | Atomic Prometheus textfile metrics from canonical heartbeat fields | Active monitoring |
| Bybit testnet execution | Sandbox-first CCXT initialization; endpoint, adapter identity, supported market type, method capability and API-key permission attestation; deterministic client order IDs; durable paper-event delivery; persistent state; partial fills; restart reconciliation; exchange-balance snapshots; execution slippage/P&L evidence; entry caps and kill switch | Optional testnet-only mirror; no strategy or live authority |
| Dynamic market universe | Discovers all active, liquid, spread-bounded Bybit USDT spot markets, removes leveraged/unsupported instruments, intersects with Testnet availability, persists cursor state, prioritizes open positions, and rotates fairly until every eligible symbol is evaluated | Required in the supported runtime; batch size limits per-cycle load, never total coverage |
| Bounded decision router | Combines adaptive score (70%) and confidence-weighted ultra consensus (30%), applies news/liquidity/evidence gates, records closed net outcomes, calibrates probabilities, quarantines persistent losers, and permits sparse probes for recovery evidence | Required; paper and Testnet authority only |
| Full timeframe matrix | Evaluates every verified Bybit kline interval in fast, tactical and strategic groups; measures disagreement and coverage instead of requiring unanimous direction; caches frames according to interval | Active router and diagnostics input |
| Public market context | Read-only CoinGecko market cap/rank/supply/global/trending data plus timestamped CoinDesk and Cointelegraph RSS; cached with provenance, freshness and explicit provider errors | Active bounded router input; never execution authority |
| Ungated strategy observatory | Records adaptive components, adaptive ensemble, ultra signals, swarm consensus, routed consensus, and every timeframe prediction independently; scores next-cycle direction net of configured round-trip costs; persists per-strategy/per-symbol evidence even when the shared router rejects an order | Active paper laboratory; no Testnet or live order authority |
| Exchange intelligence | Resolves any configured CCXT adapter by identity; discovers advertised timeframes, market/product types, data and order capabilities, sandbox declaration, quote assets, rate limits, precision, limits, fees and contract-rule coverage without loading credentials | Required public-data control plane; execution adapters remain separately attested |
| Exchange protection orchestrator | Converts each adapter's observed methods and each market's product rules into an engine plan; activates only compatible research engines and requires exchange identity, Testnet endpoint, IP-bound least-privilege key, precision/limits/fees, balance and order reconciliation, idempotency, recovery, caps, kill switch and healthy runtime dependencies before any authenticated mirror | Required fail-closed policy; Bybit Testnet spot is the only executable product, all other products remain research-only with missing protections listed |
| Structured model research | OpenAI, Anthropic or Gemini adapters receive compact measured evidence and may submit schema-validated bounded challengers; leverage/order/credential controls are rejected; proposals are journaled for causal replay and champion/challenger evidence | Optional research automation; never direct Testnet/live authority |
| Market temporal guard | Measures exchange-server time and round-trip delay, rejects excessive clock drift, removes still-forming candles, rejects stale candle series, keeps internal timestamps in UTC, and applies DST-aware New York FX weekly/rollover sessions while preserving 24/7 crypto observation | Required decision safety control; no execution authority |

## Core trading capabilities

| Capability | Status |
|---|---|
| EMA 50/200 trend | Active |
| Bollinger squeeze breakout | Restored and active as an ensemble component |
| Momentum and mean reversion | Active |
| ATR risk sizing | Active, bounded by order and position caps |
| ATR fixed and trailing stops | Active |
| Partial take-profit and final take-profit | Active in paper ledger |
| Paper OCO behavior | Active: any terminal exit removes the remaining position |
| Controlled multi-entry / scale-in | Active with entry and confidence caps |
| Complete Bybit timeframe matrix | All 13 verified intervals active; fast/tactical/strategic coverage and disagreement are measured |
| FX session gate | Active with America/New_York DST and rollover handling; crypto remains 24/7 subject to exchange-clock and data-freshness checks |
| Fees and slippage | Active and included in realized P&L |
| Daily loss and account drawdown halt | Active |
| Normal / defensive / recovery / halt states | Active with recovery hysteresis |
| Principal-protecting capital growth governor | Locks configured principal plus non-reinvested realized profit; compounds only bounded realized gains; drawdown reduces sizing; no martingale |

## Research and promotion gates

| Gate | Requirement |
|---|---|
| Causal replay | Evaluator sees no future candle |
| Walk-forward | Ordered train/test windows, explicit costs, and a canonical 30-day/7-day schedule |
| Calibration | Brier score and expected calibration error |
| Drift | Feature distribution shift measured before promotion |
| Challenger promotion | Minimum evidence, return advantage, drawdown and calibration limits |
| Rollback | Previous champion is retained and recoverable |
| Quantum/Kronos/Optuna | Never treated as available or superior without a configured provider and benchmark evidence |

## Required before live trading

The testnet mirror now provides broker-specific idempotent order IDs, authenticated reconciliation, restart recovery, partial-fill accounting, entry limits and a kill switch against test funds. Exchange intelligence can understand a newly configured CCXT venue, but that does not make its execution semantics safe automatically. Live orders remain intentionally absent. Each future live venue requires its own endpoint/permission attestation, client-order-id mapping, precision and minimum-order enforcement, native stop verification, websocket execution ingestion, restart recovery, rate-limit and disconnection testing, independent reconciliation/alerting, sustained sandbox burn-in, measured strategy evidence, and a separately reviewed capital-promotion gate.
