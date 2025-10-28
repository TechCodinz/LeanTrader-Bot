#!/bin/bash
# 🚀 EMERGENCY COMMAND GUIDE - If Agent Stops Responding
# Run this to understand current bot status

echo "════════════════════════════════════════════════════"
echo "  🚀 MICRO WALLET GROWER - STATUS CHECK"
echo "════════════════════════════════════════════════════"
echo ""

# 1. Is bot running?
echo "1️⃣  BOT PROCESS:"
if ps aux | grep -q "COMPLETE_ULTIMATE_ORCHESTRATOR.py" | grep -v grep; then
    echo "   ✅ Bot is RUNNING"
    ps aux | grep COMPLETE_ULTIMATE_ORCHESTRATOR | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, MEM: "$4"%"}'
else
    echo "   ❌ Bot is STOPPED"
    echo "   To restart: cd ~/bot && nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &"
fi

echo ""
echo "2️⃣  CURRENT BALANCE:"
strings bot.log | grep "Gate.io USDT Balance" | tail -1 | grep -o "[0-9]*\.[0-9]*" | awk '{print "   💰 $"$1" USDT"}'

echo ""
echo "3️⃣  RECENT TRADES (Last 10):"
strings bot.log | grep -E "MICRO BUY|MICRO SELL|Freed up" | tail -10

echo ""
echo "4️⃣  ACTIVE SYSTEMS:"
strings bot.log | grep -E "MICRO.*ACTIVE|Quantum.*initialized|Swarm.*initialized|Adaptive.*initialized" | tail -5

echo ""
echo "5️⃣  CURRENT GIT COMMIT:"
git log --oneline -1

echo ""
echo "════════════════════════════════════════════════════"
echo "  📋 USEFUL COMMANDS"
echo "════════════════════════════════════════════════════"
echo ""
echo "Watch live trades:"
echo "  tail -f bot.log | grep --line-buffered 'MICRO BUY\\|MICRO SELL\\|Balance'"
echo ""
echo "Check for errors:"
echo "  tail -100 bot.log | strings | grep -i error"
echo ""
echo "Restart bot:"
echo "  pkill -f COMPLETE_ULTIMATE_ORCHESTRATOR && sleep 3 && nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &"
echo ""
echo "Check which engines are running:"
echo "  strings bot.log | grep -E 'initialized|ACTIVE|STARTED' | tail -30"
echo ""
echo "════════════════════════════════════════════════════"
echo "  ⚠️  CRITICAL: DO NOT git reset --hard (kills venv!)"
echo "════════════════════════════════════════════════════"
