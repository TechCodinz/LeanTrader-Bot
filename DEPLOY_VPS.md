# VPS Purchase & Deployment Guide

## Where to buy (budget-friendly)

- Hetzner Cloud (best value EU): https://console.hetzner.cloud/
  - Start: CPX21 (3 vCPU, 4 GB, €8–9/mo), region: Falkenstein/Frankfurt/Helsinki
- OVH VPS: https://www.ovhcloud.com/
  - Start: 2 vCPU, 4 GB (€7–10/mo), region: Gravelines/Frankfurt
- Vultr: https://www.vultr.com/
  - Start: 2 vCPU, 4 GB ($24/mo), region: New Jersey/London/Singapore
- DigitalOcean: https://www.digitalocean.com/
  - Start: 2 vCPU, 4 GB ($24/mo)

Prefer EU region (Frankfurt/London) for Bybit/testnet/Jupiter reachability.

## Steps (Hetzner example)

1) Create project → Create server (Ubuntu 22.04 LTS)
2) Size: CPX21 (3 vCPU, 4 GB, 80 GB NVMe)
3) Networking: enable IPv4; add SSH key
4) Deploy

## Install bot

```bash
# on the new server
sudo apt-get update && sudo apt-get install -y python3-venv git curl
sudo mkdir -p /opt/ultrabot && sudo chown $USER:$USER /opt/ultrabot
cd /opt/ultrabot

# fetch repo
git clone <YOUR_REPO_URL> .
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || true
pip install yfinance prometheus-client python-telegram-bot ccxt aiohttp web3 matplotlib scikit-learn || true

# env
cp .env.live.conservative.example .env
# edit .env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BYBIT keys if needed

# systemd units
sudo cp -a deploy/systemd/*.service /etc/systemd/system/
sudo cp -a deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ultrabot ultrabot-status ultrabot-training ultrabot-weekly ultrabot-news ultrabot-collector ultrabot-collector.timer
sudo systemctl start ultrabot ultrabot-status ultrabot-training ultrabot-weekly ultrabot-news ultrabot-collector ultrabot-collector.timer
```

## Monitoring

Follow `monitoring/RUNBOOK.md` to install Prometheus + Grafana and import the dashboard.

## Proxy (optional)

If your region is restricted, set in `/opt/ultrabot/.env`:
```
PROXY_URL=http://USER:PASS@EU_PROXY_HOST:PORT
```
Use Telegram `/proxy_on` to enable for the running process; `/proxy_off` to disable.