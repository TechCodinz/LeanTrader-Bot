# LeanTrader VPS runbook

This runbook deploys the supported paper-authority runtime. It consumes public data from a runtime-attested CCXT exchange and simulates fills locally. An optional, separately confirmed mirror can place orders on Bybit Testnet with test funds only. Production endpoints and live flags are rejected.

The supported VPS configuration uses `CONFIRM_TIMEFRAMES=AUTO` and `PAPER_SYMBOLS=AUTO`. Exchange intelligence verifies the configured CCXT adapter and discovers its advertised intervals, products, methods, precision, limits, fees and contract rules without credentials. The exchange protection orchestrator then converts those observed capabilities into a per-market engine plan and refuses authenticated execution if identity, environment, least-privilege IP-bound permissions, market rules, reconciliation, recovery, caps, kill switch or required engine health is incomplete. The required temporal guard measures exchange-server clock offset, normalizes internal time to UTC, removes still-forming candles, rejects stale series and reports DST-aware FX sessions. It then discovers every active, sufficiently liquid USDT spot market, excludes instruments that cannot be tested responsibly, and rotates across the eligible set. Moon Scout ranks the current measured batch while the cross-venue monitor collects read-only Bybit/OKX quotes for cost-adjusted arbitrage observation. When Bybit Testnet is enabled, it additionally requires all 13 verified Bybit intervals and intersects the universe with Testnet availability. `MARKET_SCAN_BATCH_SIZE=18` limits API/CPU work in one cycle but does not cap the universe. Every engine and timeframe prediction is scored independently in the ungated paper observatory, even when the shared Testnet router rejects an order. Optional OpenAI, Anthropic or Gemini research is structured and journaled for causal validation; it has no order authority. Watch `exchange_intelligence`, `exchange_protection`, `market_temporal_guard`, `cross_venue_arbitrage`, `eligible_symbols`, `full_sweeps`, `timeframe_matrix`, `public_market_context`, `strategy_observatory`, and `model_research` to prove coverage and boundaries.

## Security incident first

Real-looking exchange and Telegram credentials were committed to this public repository in September 2025. Treat every credential ever stored in `.env`, `.env.recover`, or `api_config.json` as compromised.

Before funding an account:

1. Revoke all exposed exchange API keys.
2. Revoke and regenerate the Telegram bot token.
3. Check exchange login and API-access history.
4. Create replacement exchange keys only after the VPS has a fixed IP; restrict them to that IP and disable withdrawals.
5. Do not put replacement credentials into this paper release or commit them to Git.

Removing the files in a new commit does not remove them from old Git history. History cleanup and a coordinated force-push should be performed separately after all keys are revoked.

## VPS size

Use Ubuntu 24.04 LTS with 2 vCPU, 4 GB RAM, and at least 40 GB NVMe. The paper runtime is light, but 4 GB leaves enough room for Docker, OS updates, logs, and later monitoring.

## Deploy

### Verified one-command bootstrap (fresh Ubuntu 24.04 VPS)

The supported bootstrap is idempotent, pins the audited release commit and tree,
configures UFW and Fail2ban, installs Docker from Docker's official Ubuntu
repository, creates a small emergency swap, starts the paper runtime, and waits
for its container healthcheck. Download the script first so it can be inspected
before running it as root:

```bash
curl -fsSLo /root/leantrader-bootstrap.sh \
  https://raw.githubusercontent.com/TechCodinz/LeanTrader-Bot/main/scripts/bootstrap_verified_vps.sh
less /root/leantrader-bootstrap.sh
bash /root/leantrader-bootstrap.sh
```

Successful completion prints `LEANTRADER_BOOTSTRAP_OK`. The full bootstrap log
is stored at `/var/log/leantrader-bootstrap.log`. The bootstrap deliberately
keeps the initial SSH login method unchanged; migrate to a user-owned SSH key
and disable password login only after that key has been tested in a second
session.

### Manual deployment

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Sign out and back in so the Docker group applies, then:

```bash
git clone https://github.com/TechCodinz/LeanTrader-Bot.git
cd LeanTrader-Bot
cp .env.production.example .env
docker compose config
docker compose up -d --build
docker compose ps
docker compose logs -f --tail=100 leantrader
```

The service persists its ledger in `runtime/vps_paper_state.json`, writes its current health snapshot to `runtime/vps_heartbeat.json`, and appends fills to `logs/vps_trades.jsonl`.

## Optional structured AI research

LeanTrader can call one configured `openai`, `anthropic`, or `gemini` model on a scheduled interval. The model receives compact derived evidence—not exchange credentials or order authority—and must return a bounded JSON challenger. Unsupported controls such as leverage or direct orders are rejected. Accepted proposals remain `pending_causal_replay`; only measured validation can qualify them for paper challenger status, and no proposal can promote itself to Testnet or live trading.

Store the provider key only in `/opt/leantrader/app/secrets/model_research_api_key`, owned by `root:10001` with mode `0440`. Then set `MODEL_RESEARCH_ENABLED=true`, the matching `MODEL_RESEARCH_PROVIDER`, the provider's current `MODEL_RESEARCH_MODEL` identifier, and `MODEL_RESEARCH_INTERVAL_CYCLES`. Keep `MODEL_RESEARCH_ENDPOINT` empty for the official provider endpoint. Restart the container and verify `.engines.model_research.configured == true`, `.structured_output_validated == true`, and `.execution_authority == false` in the heartbeat. Do not place the key in `.env`, Git, chat, logs, or screenshots.

## Enable the bounded Bybit Testnet mirror

Create the key from a [Bybit Testnet](https://testnet.bybit.com/) account, not the production or Demo Trading account. Testnet keys are separate from production keys. Grant only the trading permissions needed for spot orders, disable withdrawals, and restrict the key to the VPS IP when Bybit offers that control. Never paste a key into chat, a GitHub issue, Git, or `.env`.

The supported interactive setup keeps entries halted until authentication and health checks pass:

```bash
cd /opt/leantrader/app
sudo ./scripts/enable_bybit_testnet_vps.sh
```

The manual equivalent is documented below for inspection and recovery.

On the VPS, collect the values without placing them in shell history and make them readable only by root and the container's fixed group:

```bash
cd /opt/leantrader/app
install -d -m 0750 -o root -g 10001 secrets
read -rsp "Bybit Testnet API key: " LT_TESTNET_KEY; echo
umask 027
printf '%s' "$LT_TESTNET_KEY" > secrets/bybit_testnet_api_key
unset LT_TESTNET_KEY
read -rsp "Bybit Testnet API secret: " LT_TESTNET_SECRET; echo
printf '%s' "$LT_TESTNET_SECRET" > secrets/bybit_testnet_api_secret
unset LT_TESTNET_SECRET
chown root:10001 secrets/bybit_testnet_api_key secrets/bybit_testnet_api_secret
chmod 0440 secrets/bybit_testnet_api_key secrets/bybit_testnet_api_secret
```

Edit `/opt/leantrader/app/.env` and set exactly:

```dotenv
BYBIT_TESTNET_ENABLED=true
BYBIT_TESTNET_CONFIRM=I_UNDERSTAND_TESTNET_ONLY
BYBIT_TESTNET_MAX_ORDER_USD=10
BYBIT_TESTNET_MAX_POSITION_USD=20
BYBIT_TESTNET_MAX_DAILY_SUBMITTED_USD=50
BYBIT_TESTNET_MAX_ORDERS_PER_DAY=20
```

Start the updated container and verify that the sandbox boundary and every required engine are healthy:

```bash
docker compose config --quiet
docker compose up -d --build --force-recreate
docker compose ps
sleep 70
jq '{healthy, runtime, errors, exchange_protection: .engines.exchange_protection, market_time: .engines.market_temporal_guard, news_collector: .engines.public_market_context, news_engine: .engines.advanced_shadow_suite.capabilities.news_awareness, market_universe: .engines.market_universe, testnet_execution, testnet_engine: .engines.bybit_testnet_execution}' runtime/vps_heartbeat.json
jq '[.engines | to_entries[] | select(.value.required == true and .value.healthy != true)]' runtime/vps_heartbeat.json
```

The second `jq` command must print `[]`. The exchange protection engine must report `capability_driven: true`, `fail_closed: true`, `authenticated_executor: "bybit_testnet_spot_only"`, and no missing protection on an authorized event. The temporal guard must report `clock.verified: true`, `clock.safe: true`, `closed_candles_only: true`, no stale timeframe, and all 13 Bybit timeframe checks. The public collector and local news engine must each report fresh news independently. The testnet engine must report `environment: "testnet"`, `authenticated: true`, `sandbox_endpoint_verified: true`, `api_attestation.verified: true`, `api_attestation.ip_bound: true`, `api_attestation.spot_trade: true`, `api_attestation.withdrawal_permission: false`, `exchange_capabilities.exchange_id: "bybit"`, `exchange_capabilities.execution_market_type: "spot"`, a fully true `protection_contract`, `execution_authority: "testnet_only"`, and `live_authority: false`. The required decision router must be healthy and every ultra/research engine must show an active, ready, degraded, or explicitly blocked state—never an invented success.

To stop new testnet entries immediately without blocking position-reducing sells:

```bash
touch /opt/leantrader/app/runtime/TESTNET_HALT
```

Remove that exact file only after reviewing open orders, positions, reconciliation errors, and the reason for the halt.

Watch paper learning, testnet authentication, positions, balances, execution performance, reconciliation, required-engine health, and the ten latest testnet order records from Termius:

```bash
cd /opt/leantrader/app
./scripts/watch_testnet_vps.sh
```

Press `Ctrl+C` to leave the monitor; stopping the monitor does not stop LeanTrader.

## Telegram signals and monitoring

Create a fresh bot through `@BotFather`, start a private conversation with it, and obtain the numeric chat ID. Never reuse the Telegram token previously exposed in repository history. Then run:

```bash
cd /opt/leantrader/app
bash scripts/configure_telegram_vps.sh
```

The script stores the token in a root-owned secret file, verifies the bot through Telegram, sends an admin attestation message, configures optional free and paid chats/channels, rebuilds the container, and checks the canonical heartbeat. Admin receives fills, halts, Moon Scout/arbitrage alerts and periodic health. The paid channel receives full gated signal details and a Bybit Testnet button; the free channel receives a higher-confidence summary without the button. Cooldowns prevent repeated spam.

This is distribution-tier configuration, not billing verification. Do not advertise a user as paid until a payment provider or Telegram Stars subscription webhook has verified the subscription. Telegram remains outbound-only: it does not accept order commands, cannot bypass the router, and cannot reach production exchange endpoints.

## Operate

```bash
docker compose ps
docker compose logs --tail=200 leantrader
docker compose restart leantrader
docker compose down
```

Back up `runtime/vps_paper_state.json` before rebuilding or moving the VPS.

## Promotion gate

Do not enable live trading from this release. Run paper plus testnet continuously for at least seven consecutive days, deliberately test container and VPS restarts, verify zero unresolved reconciliation errors, and evaluate at least 100 closed trades across more than one market regime. Profit in a previous version or a short test is not a promotion criterion. A separate reviewed live-execution release must still add exchange-native stop verification, independent monitoring/alerts, websocket execution ingestion, rate-limit/disconnection tests, and a manual capital gate before real funds are connected.
