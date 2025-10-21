#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    BOT STATUS                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

if pgrep -f RUN_BOT.py > /dev/null; then
    echo "✅ Bot is RUNNING (PID: $(pgrep -f RUN_BOT.py))"
else
    echo "❌ Bot is NOT running"
    exit 1
fi

echo ""
echo "📱 Telegram Signals:"
echo "   VIP: $(grep -c '✅ VIP #' bot.log 2>/dev/null || echo 0)"
echo "   FREE: $(grep -c '✅ FREE #' bot.log 2>/dev/null || echo 0)"

echo ""
echo "📊 Unique Pairs: $(grep 'Decision:' bot.log 2>/dev/null | grep -oE '[A-Z]{2,5}/[A-Z]{2,5}' | sort -u | wc -l)"

echo ""
echo "💰 Latest Signals:"
grep "✅ VIP #\|✅ FREE #" bot.log 2>/dev/null | tail -5

echo ""
echo "📈 Latest Trading Activity:"
grep "Decision:" bot.log 2>/dev/null | tail -5
