# Execution modes

LeanTrader has three execution environments for one intelligence fabric:
**paper**, **testnet** and **live**. They are destinations, not tiers of
capability. The same discovery, universe, swarm, strategy, risk and
reconciliation code runs behind all three; what changes is where an order
goes and how much evidence stands behind it.

| | paper | testnet | live |
|---|---|---|---|
| Execution | `BrokerEmulator` | exchange sandbox | exchange production |
| Fills, fees, limits | simulated | exchange-real | exchange-real |
| Balances | declared (`PAPER_EQUITY_QUOTE`) | exchange-real | exchange-real |
| Purpose | broad experimentation | exchange-realistic validation | validated execution |

No strategy, venue, agent or engine is withheld by mode. Restricting
intelligence by destination would mean the thing validated on Testnet is not
the thing that runs live, which defeats the point of validating it.

## Precedence: execution mode

Resolved in `src/leantrader/execution/broker_ccxt.py::_legacy_mode`, highest
first. The first rule that matches wins; nothing below it can shadow it.

1. **`EXECUTION_MODE`** — the canonical selector. `paper`, `testnet`, `live`,
   or an alias (`sandbox`/`demo`/`practice` → testnet, `real`/`production`/
   `prod` → live, `sim`/`simulation`/`emu` → paper). Whatever it names wins,
   including live.
2. **The legacy live grant** — `ENABLE_LIVE=true` **and** `ALLOW_LIVE=true`
   **and** `LIVE_CONFIRM=YES`. All three are required, so this is never
   reached by accident. Any one missing is not a grant.
3. **Venue sandbox hints** — `CCXT_TESTNET` or `BYBIT_TESTNET` → testnet.
   These say which endpoint a venue should use. They are *below* the live
   grant deliberately: a per-venue endpoint flag must not silently override an
   operator's explicit decision. When both are set the live grant wins and a
   warning names the contradiction.
4. **`TRADING_MODE`** — `paper`/`simulation`/`sim` → paper. Recognised for old
   configuration files; `TRADING_MODE=live` is *not* a live grant.
5. **Nothing set** → `auto`.

### What `auto` does, and does not do

`auto` means no mode was named, so the broker probes with whatever credentials
it has. It arrives two ways, and they are **not** the same decision:

- **`EXECUTION_MODE=auto`, written deliberately** — the operator asked to be
  routed wherever their credentials work. Discovery probes Testnet first, then
  live. It can land on live.
- **Nothing set at all** — the operator selected nothing. Discovery probes
  **Testnet only** and will never resolve to live.

The second case is the one that mattered. Discovery used to probe Testnet then
live regardless, so an account whose credentials were valid only on production
resolved to live with no mode ever having been chosen — adding an exchange API
key silently moved the execution destination to real money.

Even with `auto` written explicitly, Testnet is probed first, so reaching live
is a fallback rather than a preference.

An operator who wants live directly says so: `EXECUTION_MODE=live`, the
three-flag grant, or `API_ENVIRONMENT=live` / `EXCHANGE_ENVIRONMENT=live`.

## Precedence: credentials

Resolved in `BrokerCCXT.__init__`, highest first:

1. An in-memory profile passed to the call (`auth_profile=`).
2. **Testnet secret files** — `<VENUE>_TESTNET_API_KEY_FILE` /
   `<VENUE>_TESTNET_API_SECRET_FILE`. Read only when the resolved mode is
   testnet. A path that does not exist raises rather than falling through, so
   a broken mount fails loudly instead of running unauthenticated.
3. **Testnet environment variables** — `<VENUE>_TESTNET_API_KEY` /
   `<VENUE>_TESTNET_API_SECRET`. Also testnet-only.
4. **Generic credentials** — `<VENUE>_API_KEY`, then `CCXT_API_KEY`, then
   `API_KEY` (and the matching secrets).

`<VENUE>` is the ccxt exchange id upper-cased: `BYBIT_TESTNET_API_KEY_FILE`.

Testnet credentials are scoped to Testnet. In live mode they are not read at
all, so a Testnet key cannot authenticate a live session — a broker in live
mode holding only Testnet keys reports `authority == "none"` and refuses.

## Credentials are not a mode

Adding an exchange API key authenticates an account. It does not choose a
destination. Explicit `paper` stays paper and explicit `testnet` stays testnet
no matter what credentials are present, and `auto` never reaches live.

This is what lets LeanTrader hold credentials for several venues — to read
balances, inventory and venue capabilities, and to see cross-venue
opportunities — without any of them becoming a live trading destination.

## A tracked file cannot override an operator

Every `load_dotenv()` call in this repository uses the default
`override=False`, so a real environment variable always beats the tracked
`.env`. A tracked file can only fill in what the operator left unset.

Tracked templates all ship fail-closed, enforced by
`tools/config_posture.py` in CI. That is a floor for what the repository
ships, not a ceiling on what an operator may configure at runtime.

## What is the same in every mode

- **`route_order` is the only order authority.** Paper, testnet and live all
  go through it. Raw `create_order` on a market-data client is blocked by the
  guard in `router.py`.
- **Preflight runs in all three modes.** Symbol normalization, venue
  eligibility, spot-market checks, real balance, sizing, minimum notional,
  minimum amount and precision. Live is not exempt from the venue minimums.
- **A refusal is never a fill.** An order counts as acknowledged only when the
  router reports execution *and* the exchange returns an order id.

## Venue capabilities

Run `python -m tools.venue_capabilities`. It probes the installed ccxt rather
than repeating a table that would go stale, and sends nothing.

One finding worth knowing: some venues accept `set_sandbox_mode(True)` and
change nothing that reaches the wire. `BrokerCCXT._make_exchange` verifies the
switch actually altered the URLs, hostname or headers, and refuses if it did
not — otherwise a caller who asked for Testnet would have sent real orders to
a production endpoint.

## Current posture

The validated runtime is **Testnet**. `LIVE_READINESS_NOT_ESTABLISHED` stands
until the separate live runtime validation is complete. The live path is
implemented and selectable; it has not been exercised against a real account.
