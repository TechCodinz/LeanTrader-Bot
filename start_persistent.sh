#!/usr/bin/env bash
set -euo pipefail

SESSION="ultra_live"
# Load service env (Gate + Telegram)
export $(grep -v '^#' /etc/ultra-live.env | xargs -d '\n' || true)

# Use the venv
export PATH="/opt/ultra/.venv/bin:$PATH"

# Kill existing session if any
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

cd /opt/ultra

# Create session
tmux new-session -d -s "$SESSION" -n live

# Pane 1: live daemon
tmux send-keys -t "$SESSION":0.0 "python -u tools/run_bot_daemon.py 2>&1 | tee -a reports/daemon_console.log" C-m

# Pane 2: supervisor (learning + publishers)
tmux split-window -h -t "$SESSION":0
tmux send-keys -t "$SESSION":0.1 "python -u -m tools.supervisor 2>&1 | tee -a runtime/logs/supervisor_console.log" C-m

# Pane 3: periodic scanner signals (publish)
tmux split-window -v -t "$SESSION":0.1
tmux send-keys -t "$SESSION":0.2 "SCAN_TF=5m TOP_N=7 SCAN_LIMIT=250 python -u signals_scanner.py --publish --repeat 300 2>&1 | tee -a runtime/logs/scanner_console.log" C-m

echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
