#!/bin/bash
# Diagnostic script to find ALL missing imports

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              DIAGNOSING ALL MISSING IMPORTS                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot
source venv/bin/activate

echo "🔍 Testing ALL imports from the bot..."
echo ""

# Try to run the bot and capture the error
python3 RUN_BOT.py 2>&1 | head -100 > /tmp/bot_error.log

# Extract the actual error
ERROR=$(grep "No module named" /tmp/bot_error.log | head -1)

if [ -n "$ERROR" ]; then
    echo "❌ FOUND ERROR:"
    echo "   $ERROR"
    echo ""
    
    # Extract module name
    MODULE=$(echo "$ERROR" | sed "s/.*No module named '\([^']*\)'.*/\1/")
    
    echo "📦 Missing module: $MODULE"
    echo ""
    echo "🚀 QUICK FIX:"
    echo "   /root/trading_bot/venv/bin/pip install $MODULE"
    echo ""
    echo "OR install everything:"
    echo "   bash INSTALL_EVERYTHING.sh"
else
    echo "✅ No import errors found (or different error)"
    echo ""
    echo "Full error log:"
    cat /tmp/bot_error.log
fi
