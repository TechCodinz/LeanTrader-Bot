# Credentials

Every credential this repository once carried in tracked files has been
redacted and must be treated as disclosed. They are rotated externally; the
old values are invalid. Nothing here is a substitute for that rotation.

## Where credentials live

Runtime credentials are read from **root-owned files mounted into the
container**, never from tracked source. The execution broker resolves them in
this order (`src/leantrader/execution/broker_ccxt.py`):

1. an in-memory profile passed to the call
2. `<VENUE>_TESTNET_API_KEY_FILE` / `<VENUE>_TESTNET_API_SECRET_FILE`
   — a path to a mounted file whose contents are the credential
3. `<VENUE>_TESTNET_API_KEY` / `<VENUE>_TESTNET_API_SECRET`
4. `<VENUE>_API_KEY` / `<VENUE>_API_SECRET`, then `CCXT_API_KEY` / `API_KEY`

`<VENUE>` is the ccxt exchange id upper-cased, so Bybit uses
`BYBIT_TESTNET_API_KEY_FILE`. The `_FILE` form is the supported one: it keeps
the value out of the process environment, out of `docker inspect`, and out of
anything that dumps `os.environ`.

A `_FILE` path that does not exist raises rather than silently falling back,
so a broken mount fails loudly instead of running unauthenticated.

Example mount:

```
-v /etc/leantrader/secrets:/run/secrets:ro
-e BYBIT_TESTNET_API_KEY_FILE=/run/secrets/bybit_testnet_api_key
-e BYBIT_TESTNET_API_SECRET_FILE=/run/secrets/bybit_testnet_api_secret
```

The files should be owned by root and mode `0400`.

## What tracked files may contain

Placeholders only, showing the shape of the configuration:

```
BYBIT_TESTNET_API_KEY=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
```

`.env` and the `*_ENV.env` templates are documentation of what to set. They
are not where real values go.

## The gate

`tools/secret_scan.py` fails on a concrete credential literal in any tracked
file. It runs as a test (`tests/test_no_committed_secrets.py`), so it also
runs in CI.

```
python -m tools.secret_scan          # tracked files
python -m tools.secret_scan --all    # whole working tree
```

It reports masked values only — it never prints a secret, including when it
finds one. If it flags something that is genuinely a public constant (an
event topic, a transaction hash), name it so on the line rather than
loosening the scanner.

## Execution authority is separate

Holding credentials for a venue is not authorization to trade there. The only
venue with authenticated execution authority is the one the runtime resolves
through the universal router, currently **Bybit Testnet**. Credentials for
other venues — where they exist at all — are for public and research use.
Real-money live authority stays disabled:

```
EXECUTION_MODE=testnet
ENABLE_LIVE=false
ALLOW_LIVE=false
LIVE_CONFIRM=NO
```
