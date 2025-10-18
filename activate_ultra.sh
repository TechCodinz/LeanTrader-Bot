#!/bin/bash
echo "🚀 Activating Ultra Features..."

# Start evolution engine
cd /workspace/engines
python3 EVOLUTION_ENGINE.py &
EVOLUTION_PID=$!
echo "Evolution Engine PID: $EVOLUTION_PID"

# Start ultra arsenal
cd /workspace/ultra
python3 FIXED_FULL_ULTRA_ARSENAL.py &
ULTRA_PID=$!
echo "Ultra Arsenal PID: $ULTRA_PID"

# Start continuous ultra bot
python3 continuous_ultra_bot.py &
CONTINUOUS_PID=$!
echo "Continuous Ultra Bot PID: $CONTINUOUS_PID"

# Save PIDs for monitoring
echo "$EVOLUTION_PID" > /workspace/logs/evolution.pid
echo "$ULTRA_PID" > /workspace/logs/ultra_arsenal.pid
echo "$CONTINUOUS_PID" > /workspace/logs/continuous_ultra.pid

echo "✅ Ultra features activated"