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
| Older monitoring/status/systemd files | Status endpoints expose raw logs without authentication; units target obsolete runners and root paths | Replaced by Docker healthcheck, atomic heartbeat, and engine health manifest |
| `tools/market_data_ultra.py` | Large experimental aggregator with placeholder sources and unverified fallbacks | Not promoted; supported feed remains public read-only CCXT plus strict validation |
| Telegram integration | Adds credential and outbound-message surface unrelated to safe trading correctness | Deferred until the paper runtime proves stable |
| IBM/“quantum” hooks | Optional experiments without demonstrated execution advantage | Research-only |
| `divine_*` files on `cursor/*6078` | Signals include random “void whispers,” randomized timelines, and simulated karma | Rejected as non-measurable and unsafe for decisions |

Files named `.env` and `.env.recover` on old branches were intentionally not restored. Historical credentials must still be revoked before any account is funded.

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

- Full test suite: 76 passed, 1 intentionally skipped.
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
