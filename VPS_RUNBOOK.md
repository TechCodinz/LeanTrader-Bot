# LeanTrader VPS runbook

This runbook deploys the supported paper-authority runtime. It consumes public Bybit market data and simulates fills locally. An optional, separately confirmed mirror can place orders on Bybit Testnet with test funds only. Production endpoints and live flags are rejected.

The supported VPS configuration uses `PAPER_SYMBOLS=AUTO`. It discovers every active, sufficiently liquid USDT spot market, excludes instruments that cannot be tested responsibly, intersects the result with Bybit Testnet availability, and rotates across the entire eligible set. `MARKET_SCAN_BATCH_SIZE=18` limits API/CPU work in one cycle but does not cap the universe. Watch `eligible_symbols`, `last_scan`, and `full_sweeps` to prove coverage.

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
jq '{healthy, runtime, errors, market_universe: .engines.market_universe, testnet_execution, testnet_engine: .engines.bybit_testnet_execution}' runtime/vps_heartbeat.json
jq '[.engines | to_entries[] | select(.value.required == true and .value.healthy != true)]' runtime/vps_heartbeat.json
```

The second `jq` command must print `[]`. The testnet engine must report `environment: "testnet"`, `authenticated: true`, `sandbox_endpoint_verified: true`, `api_attestation.verified: true`, `api_attestation.spot_trade: true`, `api_attestation.withdrawal_permission: false`, `exchange_capabilities.exchange_id: "bybit"`, `exchange_capabilities.execution_market_type: "spot"`, `execution_authority: "testnet_only"`, and `live_authority: false`. The required decision router must be healthy and every ultra/research engine must show an active, ready, degraded, or explicitly blocked state—never an invented success.

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
