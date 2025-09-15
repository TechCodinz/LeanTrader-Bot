# Ultra Bot Deployment

## Option A: systemd (Ubuntu/Debian)

1) Create a dedicated user and directories (optional):

```bash
sudo useradd -r -s /bin/false ultrabot || true
sudo mkdir -p /opt/ultrabot
sudo chown -R $USER:$USER /opt/ultrabot
```

2) Copy repo into /opt/ultrabot and set up a Python venv (or use system Python if allowed):

```bash
sudo rsync -a --delete . /opt/ultrabot/
cd /opt/ultrabot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || true
pip install feedparser python-dotenv ccxt aiohttp web3 python-telegram-bot matplotlib scikit-learn || true
```

3) Create an `.env` from `.env.example` and fill in tokens/keys.

4) Create systemd units:

`/etc/systemd/system/ultrabot.service`
```ini
[Unit]
Description=Ultra Trading Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ultrabot
EnvironmentFile=/opt/ultrabot/.env
ExecStart=/opt/ultrabot/.venv/bin/python /opt/ultrabot/ultra_launcher.py --mode paper --god-mode --moon-spotter --evolution --forex --telegram --telegram-token ${TELEGRAM_BOT_TOKEN} --telegram-channel ${TELEGRAM_CHAT_ID}
Restart=always
RestartSec=5
User=ultrabot

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/ultrabot-status.service`
```ini
[Unit]
Description=Ultra Status Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ultrabot
EnvironmentFile=/opt/ultrabot/.env
ExecStart=/opt/ultrabot/.venv/bin/python /opt/ultrabot/services/status_daemon.py
Restart=always
RestartSec=5
User=ultrabot

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/ultrabot-training.service`
```ini
[Unit]
Description=Ultra Daily Training Scheduler
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ultrabot
EnvironmentFile=/opt/ultrabot/.env
ExecStart=/opt/ultrabot/.venv/bin/python /opt/ultrabot/services/training_scheduler.py
Restart=always
RestartSec=5
User=ultrabot

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/ultrabot-weekly.service`
```ini
[Unit]
Description=Ultra Weekly Digest
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ultrabot
EnvironmentFile=/opt/ultrabot/.env
ExecStart=/opt/ultrabot/.venv/bin/python /opt/ultrabot/services/weekly_digest.py
Restart=always
RestartSec=5
User=ultrabot

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/ultrabot-news.service`
```ini
[Unit]
Description=Ultra News Harvester
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ultrabot
EnvironmentFile=/opt/ultrabot/.env
ExecStart=/opt/ultrabot/.venv/bin/python /opt/ultrabot/news_harvest.py
Restart=always
RestartSec=60
User=ultrabot

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ultrabot ultrabot-status ultrabot-training ultrabot-weekly ultrabot-news
sudo systemctl start ultrabot ultrabot-status ultrabot-training ultrabot-weekly ultrabot-news
```

## Option B: Docker + Compose

1) Dockerfile (CPU):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt || true \
 && pip install --no-cache-dir feedparser python-dotenv ccxt aiohttp web3 python-telegram-bot matplotlib scikit-learn || true
ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/ultra_launcher.py", "--mode", "paper", "--god-mode", "--moon-spotter", "--evolution", "--forex", "--telegram", "--telegram-token", "${TELEGRAM_BOT_TOKEN}", "--telegram-channel", "${TELEGRAM_CHAT_ID}"]
```

2) docker-compose.yml:

```yaml
version: "3.9"
services:
  ultrabot:
    build: .
    restart: always
    env_file: .env
    volumes:
      - ./:/app
    command: ["python", "/app/ultra_launcher.py", "--mode", "paper", "--god-mode", "--moon-spotter", "--evolution", "--forex", "--telegram", "--telegram-token", "${TELEGRAM_BOT_TOKEN}", "--telegram-channel", "${TELEGRAM_CHAT_ID}"]
  status:
    build: .
    restart: always
    env_file: .env
    command: ["python", "/app/services/status_daemon.py"]
  training:
    build: .
    restart: always
    env_file: .env
    command: ["python", "/app/services/training_scheduler.py"]
  weekly:
    build: .
    restart: always
    env_file: .env
    command: ["python", "/app/services/weekly_digest.py"]
  news:
    build: .
    restart: always
    env_file: .env
    command: ["python", "/app/news_harvest.py"]
```

Run:
```bash
docker compose up -d --build
```

## Environment (.env.example)

Copy `.env.example` to `.env` and fill in values (see `.env.example` in repo).