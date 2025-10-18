#!/bin/bash
echo "🚀 Starting Trading Systems in Container Environment..."

# Start Redis in background
redis-server --daemonize yes

# Start brain systems
/workspace/activate_brain.sh

# Start ultra features
/workspace/activate_ultra.sh

# Start main trading bot
cd /workspace/systems
python3 RUN_BOT.py --testnet --auto-confirm &

# Keep container running
wait