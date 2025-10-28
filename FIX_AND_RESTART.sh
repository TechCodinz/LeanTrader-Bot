#!/bin/bash
echo "🔧 FIX AND RESTART BOT"
echo "="*70

cd ~/bot

echo ""
echo "1️⃣  Killing any existing processes..."
pkill -9 -f COMPLETE_ULTIMATE_ORCHESTRATOR
pkill -9 -f "python.*bot"
sleep 2

echo ""
echo "2️⃣  Clearing old log..."
> bot.log

echo ""
echo "3️⃣  Pulling latest code..."
git pull

echo ""
echo "4️⃣  Starting bot in background..."
nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &
BOT_PID=$!
echo "✅ Bot started with PID: $BOT_PID"

echo ""
echo "5️⃣  Waiting 30 seconds for initialization..."
sleep 30

echo ""
echo "6️⃣  Checking status..."
if ps -p $BOT_PID > /dev/null; then
    echo "✅ Bot is still running!"
else
    echo "❌ Bot crashed! Showing error:"
    tail -50 bot.log
    exit 1
fi

echo ""
echo "7️⃣  Checking for signals..."
grep -E "ACTIVE|publish_signal|MICRO.*using" bot.log | tail -20

echo ""
echo "8️⃣  Watching live (Ctrl+C to stop)..."
tail -f bot.log | grep --line-buffered -E "MICRO|execute_trade|Decision:|ACTIVE|ERROR"
