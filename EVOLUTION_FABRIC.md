# LeanTrader Continuous Evolution Fabric

LeanTrader v12.11 adds a **data-only hot capability fabric** so new research and market-sensor implementations can be attached without restarting the core paper runtime.

## Why this exists

Core trading state (paper ledger, Brain, memory, strategy evidence, capital governor and world/self models) should not be reset merely because a new sensor, research model or experimental algorithm is being tested. The Evolution Fabric separates **stable authority** from **replaceable intelligence experiments**.

## Operating model

1. A new implementation runs out-of-process, ideally as a separate container or service with only the public data access it needs.
2. It writes a JSON capability pack into `runtime/evolution/inbox/`.
3. LeanTrader discovers the pack at a cycle boundary through the already-mounted runtime volume. No core restart is required.
4. The pack is validated for schema, freshness, provenance and safety authority.
5. Research capabilities become available to the Active Research Planner as `available_external_shadow` evidence.
6. Directional challenger signals are measured as costed shadow episodes. They remain non-executing even after positive evidence.
7. Invalid, stale or authority-seeking packs are quarantined.

The core never imports or executes Python from the inbox.

## Safety boundary

Capability packs cannot:

- place orders;
- enable Testnet or live mode;
- add credentials;
- rewrite LeanTrader source;
- increase upstream risk;
- bypass Router, Brain, Cognitive Governance or Capital Governance.

A pack that requests any of those authorities is rejected.

## Capability-pack schema

```json
{
  "schema_version": 1,
  "pack_id": "macro-regime-sidecar",
  "version": "1.0.0",
  "producer": "macro-sidecar",
  "generated_at": 1786970000.0,
  "expires_at": 1786973600.0,
  "execution_authority": false,
  "can_enable_live": false,
  "can_increase_risk": false,
  "can_add_credentials": false,
  "capabilities": ["rates_fx_cross_asset", "macro_calendar"],
  "observations": [
    {
      "symbol": "BTC/USDT",
      "kind": "signal",
      "score": 0.35,
      "confidence": 0.72,
      "source": "documented-public-source",
      "provenance": "provider + endpoint + observation identifier",
      "observed_at": 1786970000.0,
      "horizon_seconds": 900,
      "metadata": {"note": "research-only example"}
    }
  ]
}
```

`kind` may be `context`, `risk`, `signal`, `relationship`, or `anomaly`. Signal scores are in `[-1, 1]`; confidence is in `[0, 1]`.

## Self-expansion request queue

LeanTrader writes `runtime/evolution/requests.json`. It contains the current adapter backlog, active external capabilities, research agenda and rare-scope context. This is the machine-readable contract for the next bounded sidecar implementation. A developer or coding agent can build the requested adapter without changing the running core.

## Champion/challenger evidence

External `signal` observations create non-executing shadow episodes. When their horizon expires, the fabric resolves them against observed prices and applies the same configured round-trip cost assumption used by LeanTrader's evidence system. Metrics include sample count, win rate, cumulative/average net return, EWMA net return and negative streak.

`research_validated=true` only means the pack crossed the configured shadow-sample and positive-net thresholds. It **does not grant execution authority**.

## What still requires a core release

Changes to authoritative execution logic, ledger accounting, Brain/Cognitive Governance safety contracts, or core market-data semantics still require a versioned release. Those deployments continue using state backup, candidate-image validation and rollback so evidence is preserved.
