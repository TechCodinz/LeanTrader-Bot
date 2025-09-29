#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/leantrader
LOG_DIR=/var/log/leantrader
SERVICE=/etc/systemd/system/leantrader.service

sudo mkdir -p "$APP_DIR" "$LOG_DIR"
sudo rsync -a --delete ./ "$APP_DIR"/

# Prefer Python 3.11 for best wheel support
if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

cd "$APP_DIR"
if [ ! -d venv ]; then
  $PY -m venv venv
fi
source venv/bin/activate || true
pip install --upgrade pip wheel
if [ -f requirements-full.txt ]; then
  pip install -r requirements-full.txt || true
else
  pip install -r requirements.txt || true
fi

sudo cp "$APP_DIR/deploy/leantrader.service" "$SERVICE"
sudo systemctl daemon-reload
sudo systemctl enable leantrader.service
sudo systemctl restart leantrader.service

echo "Deployed. Service status:"
systemctl --no-pager status leantrader.service || true


