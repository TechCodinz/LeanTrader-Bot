#!/bin/bash
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              COMPLETE SYSTEM STATUS CHECK                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

echo "1️⃣ API KEYS:"
grep -E "Telegram Bot Token|Bybit API" bot.log | head -3
echo ""

echo "2️⃣ UNIVERSE BREAKDOWN:"
grep "Using orchestrator universe\|categorized:" bot.log | head -5
echo ""

echo "3️⃣ ALL SYSTEMS:"
grep "initialized\|ready\|loaded" bot.log | grep -E "✅|🚀" | tail -20
echo ""

echo "4️⃣ TELEGRAM SIGNALS:"
grep -E "VIP signal|FREE signal|Sent.*signal" bot.log | tail -10
echo ""

echo "5️⃣ LEARNING PROGRESS:"
grep -E "Evolution cycle|Collective Intelligence|trained" bot.log | tail -5
echo ""

echo "6️⃣ ACTIVE TRADES:"
grep "Trade executed\|Position opened" bot.log | tail -5 || echo "No trades executed yet"
echo ""

echo "7️⃣ DATABASE FILES:"
ls -lh *.db 2>/dev/null || echo "No .db files found"
