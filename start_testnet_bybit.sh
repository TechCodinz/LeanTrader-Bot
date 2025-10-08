#!/usr/bin/env bash
set -euo pipefail
cd /opt/ultra
source /opt/ultra/.venv/bin/activate
export $(grep -v '^#' /etc/ultra-testnet.env | xargs -d '\n') || true

# Sensible defaults for testnet
export EXCHANGE_ID=bybit
export BYBIT_TESTNET=true
export ENABLE_LIVE=true
export ALLOW_LIVE=true
export LIVE_CONFIRM=YES
export LIVE_ORDER_USD=${LIVE_ORDER_USD:-25}
export DAILY_MAX_LOSS=${DAILY_MAX_LOSS:-100}
export SESSION_MIN_WEIGHT=${SESSION_MIN_WEIGHT:-0.70}
export LIVE_SYMBOLS=${LIVE_SYMBOLS:-'BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT'}
export TELEGRAM_ENABLED=${TELEGRAM_ENABLED:-true}

tmux kill-session -t ultra_testnet || true
tmux new -d -s ultra_testnet -n daemon "python -u tools/run_bot_daemon.py 2>&1 | tee -a /opt/ultra/reports/daemon_console_testnet.log"
tmux split-window -t ultra_testnet -v "python -u -m tools.supervisor 2>&1 | tee -a /opt/ultra/runtime/logs/supervisor_testnet.log"
tmux split-window -t ultra_testnet -h "EXCHANGE_ID=bybit TELEGRAM_ENABLED=true python -u /opt/ultra/signals_scanner.py --publish 2>&1 | tee -a /opt/ultra/runtime/logs/scanner_testnet.log"
tmux select-layout -t ultra_testnet tiled
echo "Started tmux session 'ultra_testnet'. Attach with: tmux attach -t ultra_testnet"
