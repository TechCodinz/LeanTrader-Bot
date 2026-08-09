# LeanTrader VPS runbook

This runbook deploys the supported paper-only runtime. It consumes public Bybit market data and simulates fills locally. It cannot place exchange orders, even if an old environment file contains live flags.

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

## Operate

```bash
docker compose ps
docker compose logs --tail=200 leantrader
docker compose restart leantrader
docker compose down
```

Back up `runtime/vps_paper_state.json` before rebuilding or moving the VPS.

## Promotion gate

Do not enable live trading from this release. Run paper mode continuously for at least seven days, verify data continuity and restarts, then evaluate at least 100 closed simulated trades. A separate reviewed live-execution release must add exchange reconciliation, idempotent client order IDs, tested stop handling, alerting, and a manual kill switch before real funds are connected.
