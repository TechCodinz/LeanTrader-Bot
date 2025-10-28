#!/bin/bash
echo "🚨 EMERGENCY DIAGNOSTIC - Finding Why Bot Crashed"
echo "="*70

cd ~/bot

echo ""
echo "1️⃣  Check if bot is running:"
ps aux | grep python | grep -E "COMPLETE_ULTIMATE|bot" | grep -v grep
if [ $? -eq 0 ]; then
    echo "✅ Bot is running"
else
    echo "❌ Bot is NOT running (crashed?)"
fi

echo ""
echo "2️⃣  Check bot.log size and recent lines:"
if [ -f bot.log ]; then
    ls -lh bot.log
    echo ""
    echo "--- LAST 50 LINES ---"
    tail -50 bot.log
else
    echo "❌ bot.log doesn't exist!"
fi

echo ""
echo "3️⃣  Try starting bot in foreground (will show error):"
echo "Running: python COMPLETE_ULTIMATE_ORCHESTRATOR.py"
echo ""
timeout 20 python COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>&1 | head -100
EXIT_CODE=$?

echo ""
echo "="*70
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 124 ]; then
    echo "✅ Bot started successfully (killed after 20s for diagnostic)"
else
    echo "❌ Bot crashed with exit code: $EXIT_CODE"
    echo "   Check the error above!"
fi
