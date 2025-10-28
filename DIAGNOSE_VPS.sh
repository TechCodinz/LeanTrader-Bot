#!/bin/bash
echo "🔍 VPS DIAGNOSTICS"
echo "==================="
echo ""

echo "1️⃣ Python processes:"
ssh root@75.119.149.117 "ps aux | grep python | grep -v grep"
echo ""

echo "2️⃣ Bot directory files:"
ssh root@75.119.149.117 "ls -la ~/bot/*.py | head -15"
echo ""

echo "3️⃣ Current branch:"
ssh root@75.119.149.117 "cd ~/bot && git branch --show-current && git log --oneline -3"
echo ""

echo "4️⃣ Recent bot activity (last 30 lines):"
ssh root@75.119.149.117 "tail -30 ~/bot/bot.log"
echo ""

echo "5️⃣ MICRO files:"
ssh root@75.119.149.117 "ls -la ~/bot/MICRO* 2>/dev/null || echo 'No MICRO files found'"
echo ""

echo "6️⃣ MICRO in orchestrator:"
ssh root@75.119.149.117 "grep -c 'MICRO_GATE_BOT\|micro_wallet' ~/bot/COMPLETE_ULTIMATE_ORCHESTRATOR.py && echo 'MICRO references found' || echo 'No MICRO references'"
echo ""
