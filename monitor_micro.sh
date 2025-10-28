#!/bin/bash
clear
echo "═══════════════════════════════════════════════════"
echo "     💎 MICRO WALLET GROWER - LIVE MONITOR 💎"
echo "═══════════════════════════════════════════════════"
echo ""

# Current balance
echo "💰 CURRENT BALANCE:"
strings bot.log | grep "Gate.io USDT Balance" | tail -1 | grep -o "[0-9]*\.[0-9]*"

echo ""
echo "📊 RECENT ACTIVITY (Last 10):"
strings bot.log | grep -E "Creating BUY|Creating SELL" | tail -10

echo ""
echo "🛡️ SAFETY STATUS:"
strings bot.log | grep "Checking daily loss" | tail -1

echo ""
echo "⚡ TOTAL TRADE ATTEMPTS (Last 5 min):"
strings bot.log | grep "🚀 Calling execute_trade" | tail -20 | wc -l

echo ""
echo "═══════════════════════════════════════════════════"
echo "Refreshing every 30 seconds... (Ctrl+C to stop)"
echo "═══════════════════════════════════════════════════"
