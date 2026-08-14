# LeanTrader verified capability ledger

This ledger prevents a named engine from being mistaken for a working capability. A capability is complete only when it has deterministic logic, explicit inputs and outputs, tests, lifecycle health, and a declared authority level.

Canonical runtime release: `verified-multi-engine-v6-attested-routing`.

## Canonical runtime authority

- **Trade authority:** the bounded decision router combines the adaptive ensemble with measured ultra-engine consensus, market-specific evidence, and capital/risk governors. Approved paper events may be mirrored on Bybit Testnet only.
- **Research authority:** scanners, memory, arbitrage, forecasting, quantum and experimental adapters may observe or score but cannot independently place orders. Unconfigured engines report an explicit blocked state.
- **Operations authority:** provenance and outbound Telegram alerts; no inbound commands.
- **Live authority:** none. All live flags are rejected during startup.

## Previous names mapped to real responsibilities

| Previous engine or request | Canonical verified responsibility | Runtime status |
|---|---|---|
| Central nervous system / unified runner / ultra launcher | Dependency-aware registry, startup rollback, reverse shutdown, circuit breakers, health manifest | Active |
| Divine intelligence | Deterministic decision synthesis, rationale, provenance fingerprint | Active; no random or spiritual signals |
| Brain / Awareness / GloAware | Data quality, regime, multi-timeframe, sessions, news context, capital state | Active |
| Hivemind / swarm consciousness | Confidence-weighted consensus with disagreement penalty | Active router input |
| Photographic memory / pattern memory | Outcome-labelled nearest-pattern retrieval with persistent evidence | Active learning/router input |
| Frequency / harmonics / ultrasonic | Detrended FFT cycle measurement and concentration | Active router input |
| Fluid mechanics | Observed order-book pressure, spread, impact and safe quantity | Active router input when depth is available; failure visible |
| Dynamic scanner / Ultra Scout | Cross-sectional price, volume, liquidity and volatility anomaly ranking | Shadow |
| Moon spotter / hype radar | Measured anomaly ranking; no invented tokens or opportunities | Shadow |
| Smart scalping | EMA/RSI/volume signal with spread penalty | Active router input |
| Main technical/microstructure indicators | Deterministic MACD, ADX, Stochastic, OBV and liquidity-sweep confirmation | Active router input |
| Hedge-fund arsenal | Portfolio correlation, concentration, VaR and deterministic stress scenarios | Active risk observation |
| Evolution engine | Bounded adaptive weights, closed-trade market evidence, online calibration, plus champion/challenger promotion and rollback | Adaptive/evidence learning active; model promotion offline |
| Ultra ML pipeline | Cost-aware gradient-boosting walk-forward forecaster and calibration | Offline/shadow |
| Kronos | Validated external predictor adapter; unavailable is reported explicitly until configured | Adapter complete; model not bundled |
| Optuna | Seeded offline optimizer; unavailable is explicit when optional dependency is absent | Offline adapter |
| IBM / ultra quantum | Candidate optimizer benchmarked against a classical baseline | Benchmark only |
| Arbitrage engine | Observed cross-venue bid/ask spreads net of fees and slippage | Shadow, no execution |
| Ultra Forex Master | XAUUSD/EURUSD normalization, pip/session and risk-unit calculations for Oanda/MT5 formats | Shadow; no broker orders |
| Ultra business system | Win rate, profit factor and expectancy from realized paper events | Monitoring only |
| News intelligence | Local auditable event state, decayed sentiment and high-impact blackout | Shadow gate |
| Telegram notifier | Outbound paper event/halt alerts only | Optional |
| Prometheus/Grafana monitoring | Atomic Prometheus textfile metrics from canonical heartbeat fields | Active monitoring |
| Bybit testnet execution | Sandbox-first CCXT initialization; endpoint, adapter identity, supported market type, method capability and API-key permission attestation; deterministic client order IDs; durable paper-event delivery; persistent state; partial fills; restart reconciliation; exchange-balance snapshots; execution slippage/P&L evidence; entry caps and kill switch | Optional testnet-only mirror; no strategy or live authority |
| Dynamic market universe | Discovers all active, liquid, spread-bounded Bybit USDT spot markets, removes leveraged/unsupported instruments, intersects with Testnet availability, persists cursor state, prioritizes open positions, and rotates fairly until every eligible symbol is evaluated | Required in the supported runtime; batch size limits per-cycle load, never total coverage |
| Bounded decision router | Combines adaptive score (70%) and confidence-weighted ultra consensus (30%), applies news/liquidity/evidence gates, records closed net outcomes, calibrates probabilities, quarantines persistent losers, and permits sparse probes for recovery evidence | Required; paper and Testnet authority only |

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
| 1h/4h confirmation | Active and configurable |
| FX session gate | Active; crypto remains 24/7 |
| Fees and slippage | Active and included in realized P&L |
| Daily loss and account drawdown halt | Active |
| Normal / defensive / recovery / halt states | Active with recovery hysteresis |

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

The testnet mirror now provides broker-specific idempotent order IDs, authenticated reconciliation, restart recovery, partial-fill accounting, entry limits and a kill switch against test funds. Live orders remain intentionally absent. A future live release still requires sustained testnet burn-in, exchange-native stop verification, websocket execution ingestion, rate-limit and disconnection testing, independent reconciliation/alerting, measured strategy evidence, and a separately reviewed capital-promotion gate.
