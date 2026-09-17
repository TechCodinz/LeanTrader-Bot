# RPB Unified Intelligence and Execution Handoff Design

Date: 2026-09-17

## Purpose

Make `REAL_PROFIT_BOT` (RPB) the single order-producing LeanTrader engine while preserving and channeling the full useful intelligence, research, evolution, and learning systems that have accumulated outside RPB.

The design combines the two requested directions:

1. a controlled RPB handoff with position/cost-basis preservation and closed-loop learning; and
2. the full production runner's intelligence/evolution capabilities made available to RPB.

It deliberately does **not** instantiate the production runner's own Testnet executor inside RPB, because that would create a second order authority rather than a unified bot.

## Non-negotiable constraints

- `REAL_PROFIT_BOT` remains the final execution owner.
- Only one process may submit orders for the Bybit Testnet wallet after cutover.
- Testnet remains the execution environment until the operator explicitly selects live mode. This project does not auto-promote Testnet to live.
- Existing Telegram admin/VIP/free routing is preserved.
- Existing evolution/research state is preserved; no reset or retraining-from-zero step is allowed.
- Existing authenticated position/cost-basis state is preserved before execution authority changes.
- Research, evolution, councils, specialists, scouts, world models, and training components may advise RPB but may not independently place orders.
- No fabricated fills, prices, PnL, signals, or exchange evidence may be introduced.
- Authoritative profit remains authenticated realized net PnL after actual exchange costs.

## Current evidence and handoff state

The current production runtime has an authenticated Bybit Testnet execution state with historical fills and open inventory. The latest audited state reported 81 filled orders and positions in CSPR/USDT, DOT/USDT, and JASMY/USDT, with persisted position-cost and reconciliation metadata. This state must be adopted/reconciled before the old production executor is retired.

The current production runner constructs a `FastCollectiveTestnetLane` around its Testnet executor. Therefore the runner cannot simply be imported and started inside RPB unchanged: doing so would recreate a competing order producer.

The current evolution fabric is active and persistent. Its state includes thousands of evolution cycles plus current capability packs, research demand, specialist/council context, microstructure and multi-timeframe context, cross-venue context, conditional-edge context, and other evidence. The existing production runner already calls `evolution_fabric.evidence_for(symbol)`; the new RPB bridge must preserve the semantic role of that evidence.

The standalone `leantrader-fast-lane-testnet` has produced no fills in the audited runtime and is not the source of authenticated execution history. It is a donor/reference and will not remain an order authority after cutover.

## Target architecture

```text
Public/real market data
        |
        v
Existing LeanTrader research + production intelligence
  - EvolutionFabric + inbox packs
  - world/context models
  - specialist/council evidence
  - microstructure / multi-timeframe evidence
  - qualification / governance outputs
  - existing alpha/scalping/intelligence/scout engines
        |
        | read-only/advisory evidence
        v
RPBProductionBridge
        |
        v
REAL_PROFIT_BOT  <-------------------------------+
        |                                         |
        | single execution authority              | authenticated outcome feedback
        v                                         |
Bybit Testnet (explicit mode)                     |
        |                                         |
        v                                         |
actual order -> actual fill -> owned position     |
        |                                         |
        v                                         |
rapid lifecycle / authenticated close             |
        |                                         |
        v                                         |
fees + realized net PnL -> wallet refresh --------+

Feedback targets:
  - persistent online learner
  - AlphaRouter reliability memory
  - scalping/session outcome tracker
  - RPB outcome journal / safe evolution feedback adapter
```

## Component 1: `RPBProductionBridge`

Add a focused read-only bridge module in `LeanTrader-Bot`, tentatively `rpb_production_bridge.py`.

Responsibilities:

- Read current production/evolution artifacts from configurable paths without importing or starting the production runner.
- Prefer canonical persisted artifacts over duplicating the production runner's internals.
- Load the evolution fabric snapshot and relevant symbol/context evidence.
- Read current heartbeat/world-model/qualified context where available.
- Expose one bounded API to RPB, for example:

```python
bridge.evaluate(symbol, signal, confidence, market_context) -> (signal, confidence, detail)
```

The bridge is advisory. It may enrich confidence, attach warnings/context, or veto only when an existing, explicit production evidence contract says the setup is invalid. Missing/stale/unparseable research evidence must fail open and be reported in detail; it must not silently starve the original RPB execution loop.

The bridge must not contain `ccxt.create_order`, `route_order`, broker credentials, an executor, or any order method.

## Component 2: full production intelligence without production execution

The existing production runner remains useful as a research/evolution/intelligence producer, but its Testnet execution path must be disabled before RPB becomes active.

The final runtime has two choices for the production intelligence process, in priority order:

1. **Preferred:** run the existing production stack with its Testnet execution adapter disabled while evolution/research/heartbeat state continues to refresh.
2. If the existing runner cannot be cleanly made advisory-only without invasive changes, preserve the evolution sidecar and other read-only producers and stop the execution-capable runner. RPB reads their persisted state directly.

In either case, importing `PaperRunner`/`FastCollectiveTestnetLane` into RPB is prohibited. The goal is full intelligence transfer, not duplicate execution transfer.

## Component 3: existing RPB intelligence stays in-process

Keep the already integrated RPB stack:

- original momentum/volume/breakout signal generation;
- RSI/MACD/Bollinger/ATR + order-book intelligence;
- multi-timeframe scalping confluence;
- nine-strategy alpha ensemble;
- risk/Kelly/exposure governor;
- real order-book/volume scout;
- dynamic Bybit market discovery;
- owned-position lifecycle;
- admin/VIP/free Telegram distribution.

`RPBProductionBridge` becomes an additional advisory layer, not a replacement for these components.

## Component 4: closed-loop learning feedback

After an authenticated position is settled, use only the actual realized result to update learning.

The close flow becomes:

```text
authenticated SELL fill
-> PositionLedger.close_position()
-> realized_net_pnl after actual fees
-> scalping.record_result()
-> online learner update_after_trade()
-> AlphaRouter reliability update for strategies that contributed
-> persistent RPB outcome journal
-> optional safe evolution feedback artifact
```

Rules:

- A submitted/acknowledged order is never a learning result.
- An open or zero-filled order is never a learning result.
- Synthetic legacy `total_profit` is never used as learner reward.
- Alpha reliability is updated only for strategies whose attribution was recorded at entry; do not reward every strategy indiscriminately.
- Online learner state is written to an RPB-owned writable runtime path.
- Research/evolution runtime may be mounted read-only into RPB; RPB must not casually mutate production state files.

## Component 5: position and cost-basis adoption

Before cutover, create a one-time, idempotent adoption/reconciliation utility.

Inputs:

- authenticated Bybit Testnet balances/open inventory;
- existing `/app/runtime/vps_testnet_execution.json` state, including positions and position cost basis;
- existing RPB position ledger, if present.

Process:

1. Snapshot production execution state before any process is stopped.
2. Query authenticated exchange inventory.
3. For each production-owned position, reconcile quantity against the exchange.
4. Import only reconciled quantity and known cost-basis/fee evidence into RPB ledger.
5. Keep unrelated dust/inventory distinct unless the production state proves ownership/cost basis.
6. Produce a migration report containing adopted, skipped, dust, ambiguous, and mismatched rows.
7. Refuse final cutover if a material production-owned position is ambiguous.

The migration is idempotent: running it twice must not duplicate positions or cost basis.

## Component 6: single execution authority cutover

Cutover order:

1. Create rollback refs/snapshots for the RPB source and current runtime state.
2. Run unit/integration tests in the actual LeanTrader container dependency environment.
3. Run the adoption utility in dry-run mode.
4. Verify authenticated inventory/cost-basis reconciliation.
5. Stop the standalone `leantrader-fast-lane-testnet`.
6. Disable/stop only the production runner's Testnet execution authority while keeping evolution/research producers alive where possible.
7. Re-run a single-authority process check.
8. Start `leantrader-old-bybit-profit` in **Testnet** mode only.
9. Verify RPB startup reports all required intelligence layers as connected/available.
10. Verify it restores/adopts owned positions and monitors them.
11. Observe authenticated Testnet cycle evidence: scan -> signal -> order -> fill -> ledger ownership -> exit -> fill -> realized net PnL -> wallet refresh -> learner feedback.

No live real-money activation is part of this cutover.

## Runtime mounts/state separation

RPB needs two classes of state:

### Read-only intelligence sources

Suggested mount:

```text
/opt/leantrader/app/runtime -> /intelligence/runtime:ro
```

This gives RPB current evolution/world/heartbeat evidence without allowing RPB to corrupt the production runtime.

### Writable RPB-owned state

Suggested persistent directory:

```text
/opt/leantrader/rpb-runtime -> /workspace/runtime
```

or an equivalent existing persistent directory after confirming current container paths.

This contains:

- `rpb_positions.json`
- `learn_state.json`
- RPB outcome/feedback journal
- any RPB-owned alpha/session persistence that should survive restarts

Existing production state is copied/adopted only through the explicit migration utility, not by sharing mutable JSON files between two processes.

## Testing strategy

Use TDD for the new bridge and migration behavior.

Required tests before cutover:

1. bridge reads valid evolution evidence and returns advisory detail;
2. bridge fails open on missing/stale/corrupt files;
3. bridge has no execution method/authority;
4. RPB calls bridge in the decision path;
5. authenticated close updates online learner;
6. authenticated close updates only attributed alpha strategy reliability;
7. zero-fill/ack does not update learning;
8. adoption dry-run maps known production positions and cost basis;
9. adoption is idempotent;
10. ambiguous/mismatched material inventory blocks cutover;
11. full RPB cycle integration still passes;
12. single-authority guard detects a competing Testnet order producer.

Run the existing RPB suite as a regression set in the correct container environment. Host-Python dependency failures are not accepted as proof of bot failure; validation must use the runtime dependency environment that will execute RPB.

## Acceptance criteria

The handoff is complete only when all of the following are evidenced:

- RPB is the only process with Bybit Testnet order authority.
- Standalone FastLane is not trading.
- Production intelligence/evolution continues to refresh or its equivalent read-only producers remain active.
- RPB can read current evolution/production intelligence and reports freshness/source status.
- Current production-owned Testnet positions are reconciled and owned by RPB ledger without duplication.
- RPB submits an authenticated Testnet order from its own execution path.
- ACK is reconciled to an actual fill before ownership is created.
- RPB monitors and closes owned quantity only.
- Actual fees and realized net PnL are recorded.
- Wallet balance is refreshed and next size is recalculated.
- Actual settled outcomes persist into learning feedback.
- Restart restores RPB positions and learner state.
- No live authority is enabled automatically.

## Rollback

If any acceptance check fails after cutover:

- stop RPB before starting any prior execution process;
- restore the preserved source/runtime refs;
- restore exactly one prior Testnet execution authority;
- keep evolution/research state intact;
- do not run old and new executors simultaneously.

## Out of scope for this handoff

- enabling real-money live trading;
- promising profitability or a particular return;
- deleting historical engines merely because they are old;
- rotating intentionally configured Telegram/exchange credentials;
- rewriting the evolution system from scratch;
- making every dormant experimental module an order-producing component.
