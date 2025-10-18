#!/bin/bash
echo "🛑 Stopping All Trading Systems..."

# Stop main trading bot
systemctl stop trading-bot-production.service

# Stop brain systems
if [ -f /workspace/logs/brain.pid ]; then
    kill $(cat /workspace/logs/brain.pid) 2>/dev/null || true
fi

if [ -f /workspace/logs/moon_spotter.pid ]; then
    kill $(cat /workspace/logs/moon_spotter.pid) 2>/dev/null || true
fi

# Stop ultra features
if [ -f /workspace/logs/evolution.pid ]; then
    kill $(cat /workspace/logs/evolution.pid) 2>/dev/null || true
fi

if [ -f /workspace/logs/ultra_arsenal.pid ]; then
    kill $(cat /workspace/logs/ultra_arsenal.pid) 2>/dev/null || true
fi

if [ -f /workspace/logs/continuous_ultra.pid ]; then
    kill $(cat /workspace/logs/continuous_ultra.pid) 2>/dev/null || true
fi

echo "✅ All systems stopped"