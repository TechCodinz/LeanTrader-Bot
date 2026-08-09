# LeanTrader engine and branch audit

Audit date: 2026-08-09
Supported runtime: `src/leantrader/production` (`verified-multi-engine-v3`)

The original audit established the safe boundary. `ENGINE_CAPABILITY_LEDGER.md` records the subsequent Wave 2 rehabilitation of the named advanced and ultra engines.

## Executive result

The repository contains a large research archive, not one coherent production bot. The audit found 643 tracked Python files, 124 engine-like class declarations (89 unique names), 17 duplicated engine class names, and four syntax-invalid legacy files. All four syntax failures are now repaired or safely retired, and the complete tree compiles.

The supported VPS path is deliberately narrower. It contains only engines that are deterministic, observable, tested, restart-safe for paper state, and incapable of placing live orders. Legacy files remain available for research but are not loaded by the supported image or runner.

## Remote branch archaeology

Every remote branch was fetched and compared with the hardened VPS branch.

| Branch group | Result | Decision |
|---|---|---|
| 12 ancestor branches | Already represented in current history; no missing branch-only files | No merge needed |
| `main`, `integration/all-in`, `copilot/fix-a…`, `cursor/*6078`, `cursor/*cff8`, `cursor/honest*` | Divergent from an older snapshot | Reviewed file-by-file; do not bulk merge |
| Older monitoring/status/systemd files | Status endpoints expose raw logs without authentication; units target obsolete runners and root paths | Replaced by Docker healthcheck, atomic heartbeat, engine health manifest, and canonical Prometheus textfile metrics |
| `tools/market_data_ultra.py` | Contains a syntax error, placeholder news/on-chain sources, broad exception swallowing, and in-sample pseudo-ML | Not promoted; its valid MACD/ADX/Stochastic/OBV/liquidity-sweep ideas were rebuilt deterministically in the shadow suite |
| Telegram integration | Mixed formatting, multiple credential surfaces, and broad retries | Rebuilt as an optional outbound-only paper event/halt alert engine |
| IBM/“quantum” hooks | Optional experiments without demonstrated execution advantage | Research-only |
| `divine_*` files on `cursor/*6078` | Signals include random “void whispers,” randomized timelines, and simulated karma | Rejected as non-measurable and unsafe for decisions |

Files named `.env` and `.env.recover` on old branches were intentionally not restored. Historical credentials must still be revoked before any account is funded.

### Final `main` reconciliation audit

Before reconciling the divergent `main` history, all 50 `main`-only commits and 36 `main`-only paths were inspected again. The useful ideas and their dispositions are:

| Main-only area | Result |
|---|---|
| Prometheus/Grafana monitoring | Preserved as atomic Prometheus textfile metrics derived from the canonical heartbeat; the old dashboard may be redesigned against these stable series |
| MACD, ADX, Stochastic, OBV and liquidity-sweep features | Reimplemented as deterministic shadow confirmation with no lookahead or execution authority |
| Training scheduler and weekly digest | Superseded by causal replay, evidence-gated model promotion, the canonical 30-day/7-day walk-forward path, and the existing weekly research workflow |
| Arbitrage daemon | Superseded by the net-cost arbitrage engine, which has no execution authority |
| IBM connectivity CLI | Retained only as historical research; connectivity did not implement or prove a superior optimizer, while the canonical adapter requires a benchmark against the classical baseline |
| Status API | Rejected because it exposed raw log tails without authentication; canonical health is the bounded heartbeat/healthcheck |
| Testnet preflight | Superseded by startup rejection of every live flag and the canonical paper preflight |
| Old Compose/systemd units | Rejected because they launch obsolete `ultra_launcher.py` paths and pass Telegram secrets on the command line |
| `.env`, `.env.recover`, live-enable commits | Rejected; they contain or enable credential-bearing/live behavior |

## Supported engine manifest

| Engine | Responsibility | Failure behavior | Evidence |
|---|---|---|---|
| Market data | Public OHLCV only; no API credentials or order methods | Per-symbol isolation; circuit opens after repeated failures | Integration and circuit-breaker tests |
| Data quality | Reject missing, short, non-finite, non-positive, malformed, duplicate, non-monotonic, or materially gapped candles | No signal and no entry | Invalid-data tests |
| Regime detector | Classifies trend, range, or high volatility from EMA separation and ATR | Deterministic fallback through isolated symbol error | Determinism tests |
| Strategy ensemble | Combines trend, momentum, and mean-reversion scores with regime-aware weights | Long entry requires quality and score threshold | Determinism and bounds tests |
| Evolution governor | Accumulates closed-paper-trade evidence, promotes only after five samples, limits learning rate, and bounds every weight to 10–70% | Corrupt optional state is ignored; no code mutation | Promotion-gate and persistence tests |
| Paper ledger | Atomic restart persistence, fees, slippage, realized return, and decision attribution | Rejects duplicate positions and overspending | Accounting and persistence tests |
| Risk governor | Position cap, notional cap, daily-loss halt, drawdown halt, fixed ATR stop, and trailing ATR stop | Halts entries and closes on risk halt | Existing risk and runner tests |
| Engine control plane | Dependency order, startup rollback, reverse shutdown, health snapshots, failure isolation, and cooldown recovery | Required startup failure aborts safely | Lifecycle tests |

## Graceful evolution boundary

“Evolution” means bounded parameter learning from measured closed paper trades. It does not mean random predictions, autonomous code changes, automated deployment, removal of risk limits, or automatic promotion to real money. The learning state is separate from the financial ledger and is atomically persisted. Every heartbeat includes the current regime, component scores, weights, rationale, engine state, failure count, and circuit state.

## Verification record

- Full test suite: 108 passed, 1 intentionally skipped.
- Python compile check: all tracked Python sources compile.
- Focused lint: supported production package and production tests pass.
- Paper preflight: safe defaults accepted; all live flags remain hard-rejected.
- Compose static validation: paper service exists, filesystem is read-only, and restart policy is `unless-stopped`.
- Changed-file secret-pattern scan: no findings.
- Docker daemon execution was unavailable in the audit environment, so the image must still be built on CI or the VPS before deployment.

## Remaining release gates

1. Revoke every credential ever committed in older Git history before funding any exchange account.
2. Resolve the GitHub account billing lock so Actions can build the container and run independent CI.
3. Run the supported container in paper mode for multiple weeks and review fills, data gaps, drawdown behavior, restarts, and adaptation history.
4. Do not add live execution merely because paper tests pass. Live trading needs exchange reconciliation, native stops, idempotent client order IDs, partial-fill handling, and a separate reviewed release.
