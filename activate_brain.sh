#!/bin/bash
echo "🧠 Activating Brain Systems..."

# Start brain loop
cd /workspace/brain
python3 brain_loop.py &
BRAIN_PID=$!
echo "Brain Loop PID: $BRAIN_PID"

# Start ultra moon spotter
python3 ultra_moon_spotter.py &
MOON_PID=$!
echo "Ultra Moon Spotter PID: $MOON_PID"

# Save PIDs for monitoring
echo "$BRAIN_PID" > /workspace/logs/brain.pid
echo "$MOON_PID" > /workspace/logs/moon_spotter.pid

echo "✅ Brain systems activated"