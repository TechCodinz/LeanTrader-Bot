#!/bin/bash

echo "Checking why bot crashes in LIVE mode..."
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 CRASH LOGS"
echo "═══════════════════════════════════════════════════════════════════════"

journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -A 20 "Error:\|Traceback\|Failed\|CRITICAL"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 LAST 30 LINES"
echo "═══════════════════════════════════════════════════════════════════════"

journalctl -u trading-bot --since "2 minutes ago" --no-pager | tail -30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 Checking RUN_BOT.py for --live support"
echo "═══════════════════════════════════════════════════════════════════════"

grep -n "live\|testnet" /root/trading_bot/RUN_BOT.py | head -10

echo ""
