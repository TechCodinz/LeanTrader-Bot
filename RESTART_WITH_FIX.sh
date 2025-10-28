#!/bin/bash
echo "🚀 RESTARTING BOT WITH ALL FIXES"
echo "="*70

cd ~/bot

echo ""
echo "1️⃣  Pulling CRITICAL FIX (start() method that runs all tasks)..."
git pull

echo ""
echo "2️⃣  Killing old bot..."
pkill -9 -f COMPLETE_ULTIMATE_ORCHESTRATOR
sleep 3

echo ""
echo "3️⃣  Clearing old log..."
> bot.log

echo ""
echo "4️⃣  Starting bot with ALL ENGINES ACTIVE..."
nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &
BOT_PID=$!
echo "✅ Bot started (PID: $BOT_PID)"

echo ""
echo "5️⃣  Waiting 45 seconds for full initialization..."
sleep 45

echo ""
echo "6️⃣  Checking if tasks were created..."
if grep -q "ACTIVE TASK LOOPS CREATED" bot.log; then
    TASK_COUNT=$(grep "ACTIVE TASK LOOPS CREATED" bot.log | grep -oP '\d+' | head -1)
    echo "✅ SUCCESS! $TASK_COUNT task loops created!"
else
    echo "❌ FAILED - Tasks not created. Showing error:"
    tail -50 bot.log
    exit 1
fi

echo ""
echo "7️⃣  Checking if MICRO has dynamic pairs..."
if grep -q "MICRO using.*pairs" bot.log; then
    echo "✅ MICRO has dynamic pairs!"
    grep "MICRO using" bot.log | tail -1
else
    echo "⚠️  MICRO might still be loading pairs (check in 1 minute)"
fi

echo ""
echo "8️⃣  Checking for signal engines..."
grep -E "ACTIVE.*loop|✅.*ACTIVE" bot.log | head -20

echo ""
echo "9️⃣  Watching for trades (Ctrl+C to stop)..."
echo "Looking for: 'execute_trade', 'MICRO GROWTH', 'ORDER PLACED'..."
echo ""
tail -f bot.log | grep --line-buffered -E "execute_trade|MICRO GROWTH|💎 MICRO|ORDER PLACED|⚡ EXECUTING|ACTIVE TASK LOOPS"
