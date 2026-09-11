#!/usr/bin/env bash
set -euo pipefail
cd /opt/ultra
source /opt/ultra/.venv/bin/activate
export $(grep -v '^#' /etc/ultra-testnet.env | xargs -d '\n') || true

# Sensible defaults for testnet.
#
# This script exported ENABLE_LIVE=true, ALLOW_LIVE=true and LIVE_CONFIRM=YES
# despite its name. BYBIT_TESTNET=true happens to win in the current mode
# resolver, so it did route to Testnet -- but those three flags together are
# the live-authority grant, and a tracked script that sets all three is one
# unset variable or one resolver change away from real money. A Testnet
# launcher asks for Testnet explicitly and leaves live fail-closed.
export EXCHANGE_ID=bybit
export BYBIT_TESTNET=true
export EXECUTION_MODE=testnet
export ENABLE_LIVE=false
export ALLOW_LIVE=false
export LIVE_CONFIRM=NO
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
