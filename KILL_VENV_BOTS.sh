#!/bin/bash
# KILL BOTS RUNNING IN VENV
# Special commands for virtual environment bots

echo "================================================================================"
echo "🛑 KILLING VENV BOTS"
echo "================================================================================"
echo ""

# Find all venv directories
echo "1️⃣  Searching for venv directories..."
echo ""
find ~ -type d -name "venv" -o -name "env" -o -name ".venv" -o -name "virtualenv" 2>/dev/null | head -20
echo ""

# Find Python processes from venv
echo "2️⃣  Searching for venv Python processes..."
echo ""
ps aux | grep -E "venv|env/bin" | grep -v grep
echo ""

# Kill all venv Python processes
echo "3️⃣  Killing all venv Python processes..."
echo ""

# Method 1: Kill by venv path pattern
pkill -9 -f "venv/bin/python"
pkill -9 -f "env/bin/python"
pkill -9 -f ".venv/bin/python"

# Method 2: Kill all Python in home directory
pkill -9 -f "/home/.*python"

# Method 3: Kill by common bot names
pkill -9 -f "REAL_PROFIT"
pkill -9 -f "trading_bot"
pkill -9 -f "enhanced_bot"
pkill -9 -f "bot.py"

sleep 3

echo "✅ Processes killed"
echo ""

# Check what's still running
echo "4️⃣  Checking remaining processes..."
echo ""
ps aux | grep python | grep -v grep | grep -v "ps aux"

if ps aux | grep python | grep -v grep | grep -v "ps aux" | grep -q .; then
    echo ""
    echo "⚠️  STILL RUNNING! Kill manually:"
    echo ""
    ps aux | grep python | grep -v grep | grep -v "ps aux" | awk '{print "kill -9 " $2}'
else
    echo "✅ All Python processes stopped!"
fi

echo ""
echo "5️⃣  Killing screen sessions..."
echo ""
screen -wipe
screen -ls 2>&1 | grep -E "[0-9]+\." | cut -d. -f1 | xargs -I {} screen -X -S {} quit 2>/dev/null

echo "✅ Screen sessions killed"
echo ""

echo "6️⃣  Removing venv directories..."
echo ""
cd ~

# Remove all venv directories
rm -rf venv env .venv virtualenv python-env trading-env bot-env
find . -maxdepth 3 -type d -name "venv" -exec rm -rf {} \; 2>/dev/null
find . -maxdepth 3 -type d -name "env" -exec rm -rf {} \; 2>/dev/null

echo "✅ Venv directories removed"
echo ""

echo "7️⃣  Removing bot directories..."
echo ""
rm -rf bot trading_bot old_bot lean-trader leantrader Lean-Trader
rm -rf REAL_PROFIT* MICRO* enhanced*

echo "✅ Bot directories removed"
echo ""

echo "================================================================================"
echo "✅ CLEANUP COMPLETE"
echo "================================================================================"
echo ""

# Final verification
echo "Final check:"
echo ""
echo "Python processes:"
ps aux | grep python | grep -v grep | wc -l
echo ""
echo "Screen sessions:"
screen -ls 2>&1
echo ""
echo "Bot directories:"
ls -d *bot* *trade* *lean* 2>/dev/null | wc -l
echo ""

if ! ps aux | grep python | grep -v grep | grep -q .; then
    echo "✅✅✅ ALL BOTS STOPPED! ✅✅✅"
    echo ""
    echo "Wait 5 minutes and check Telegram."
    echo "If no messages = Success!"
else
    echo "⚠️  Some processes still running. Check above."
fi

echo ""
echo "================================================================================"
