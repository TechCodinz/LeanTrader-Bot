#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          CHECKING CURRENT DEPLOYMENT STATUS                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

echo "1. Checking latest commit on VPS:"
echo "   Run on VPS: cd /root/trading_bot && git log --oneline -1"
echo ""

echo "2. Checking if Telegram monitor is running:"
echo "   Run on VPS: journalctl -u trading-bot --since '5 minutes ago' | grep 'Starting Telegram signal monitor'"
echo ""

echo "3. Checking if signals are being generated:"
echo "   Run on VPS: journalctl -u trading-bot --since '5 minutes ago' | grep -E 'Scalper generated|signals'"
echo ""

echo "4. Checking if monitor is routing signals:"
echo "   Run on VPS: journalctl -u trading-bot --since '5 minutes ago' | grep -E 'signal sent|VIP signal|Free signal'"
echo ""

echo "5. Check signal_queue and recent_signals sizes:"
echo "   Run on VPS: journalctl -u trading-bot --since '2 minutes ago' | grep -E 'Signals: [0-9]+ recent'"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "AFTER RUNNING THOSE COMMANDS, PASTE OUTPUT HERE"
echo "═══════════════════════════════════════════════════════════════════════"

