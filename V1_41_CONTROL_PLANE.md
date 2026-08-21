# LeanTrader v1.41 Unified Decision Control Plane

## Purpose

v1.41 does not add another independent signal engine. It introduces a shadow-only
control plane over the existing v1.35-v1.40 research and safety components. Its
job is to turn specialist outputs into one auditable recommendation while
preserving the canonical paper route unchanged.

The control plane has no order, sizing, paper-promotion, Testnet, live, or
real-money authority.

## Existing capability map

| Existing component | Retained role | v1.40 limitation addressed by v1.41 |
|---|---|---|
| Bounded decision router | Canonical paper route | Fixed 70/30 blend can double-count correlated specialists |
| Strategy observatory | Closed, 30-bps-costed outcome evidence | Evidence was not joined to a unified portfolio decision |
| Alpha Tournament / Strategy Foundry | Candidate generation and comparison | Remains research-only; no automatic promotion |
| Prospective validation lab | Walk-forward, Bonferroni-corrected prospective evidence | Purging, embargo, PBO, deflated statistics, and untouched holdout are not yet proven |
| Execution quality intelligence | Observed paper fill and cost drag | Does not simulate rejection, partial fill, latency, funding, liquidity, and impact before a decision |
| Capital stress simulator | Deterministic survival scenarios | Does not allocate a new order against symbol and correlation-bucket room |
| Net-profit attribution | Closed-trade cost attribution | Observational; no causal authority |
| Probability calibration lab | Closed-trade reliability diagnostics | Cannot rewrite probabilities and is immature until sample and regime gates pass |
| Evolution fabric / hypothesis lab | Bounded research generation | Must yield resources to the paper runtime and cannot promote itself |
| Cognitive governance / tail-risk sentinel | Veto or reduce only | Cannot increase upstream risk |

## v1.41 control contract

1. Raw specialists declare a correlation group, regime, timeframe, direction,
   confidence, calibration error, and expected gross edge.
2. Expected edge is populated only from sufficiently mature, closed, costed
   strategy-observatory episodes. Raw signal magnitude is never converted into
   a profit claim.
3. Specialists in the same declared correlation group collapse to one capped
   contribution. Duplicating a correlated specialist cannot add independent
   weight.
4. The deterministic paper order model accounts for spread, fees, base
   slippage, funding, latency, adverse selection, square-root market impact,
   rejection probability, liquidity participation, and partial fill. The
   round-trip research cost floor can never be less than 30 bps.
5. Portfolio allocation is bounded by gross, symbol, and correlation-bucket
   exposure room and then haircuts expected fill and calibrated confidence.
6. Stale or low-quality data, leakage uncertainty, exchange anomalies,
   survivorship uncertainty, drawdown, daily loss, loss streak, volatility
   shock, heartbeat, runtime integrity, Testnet, or live flags fail closed.
7. Every result includes a paper lifecycle plan: initial stop, target, trailing
   activation, trailing distance, time stop, and invalidation conditions.
8. Promotion review requires minimum independent samples, purged walk-forward,
   embargo, untouched holdout, multiple-testing control, acceptable PBO,
   positive deflated statistic, positive prospective net evidence, calibration,
   and drift stability. v1.41 does not weaken any threshold and cannot promote
   automatically.
9. Experiments use deterministic input/configuration hashes and an append-only
   SHA-256 lineage chain. A broken chain blocks recommendations.
10. Research is limited to one hypothesis per cycle only when runtime health,
    host load, and available-memory budgets permit it.

## Evidence partitions

Heartbeat diagnostics keep four partitions explicit:

- training
- validation
- prospective paper
- untouched holdout

At the v1.41 introduction point, the untouched holdout, purge/embargo proof,
PBO, deflated performance statistic, and drift-stability proof remain closed
gates. Passing unit or CI tests is not profitability evidence.

## Authority

The production paper runner records v1.41 recommendations for prospective
comparison only. It does not modify the existing route, order, size, risk
limits, Testnet state, or live authority.
