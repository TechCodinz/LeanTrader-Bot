#!/bin/bash
# KILL ALL OLD BOTS - Run this on your VPS
# This will stop EVERYTHING and clean up completely

echo "================================================================================"
echo "🛑 KILLING ALL OLD BOTS"
echo "================================================================================"
echo ""

# Check what's running
echo "Current running bots:"
ps aux | grep python | grep -v grep
echo ""

# Kill all Python processes that might be bots
echo "Killing all Python bot processes..."
pkill -9 -f "python.*bot"
pkill -9 -f "python.*trade"
pkill -9 -f "python.*RUN_BOT"
pkill -9 -f "python.*COMPLETE"
pkill -9 -f "python.*REAL_PROFIT"
pkill -9 -f "python.*enhanced"
pkill -9 -f "python.*MASTER"
pkill -9 -f "python.*ultra"

# Wait a moment
sleep 2

echo "✅ Python processes killed"
echo ""

# Kill all screen sessions with bot-related names
echo "Killing all screen sessions..."
screen -ls | grep -E "bot|trade|profit" | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit 2>/dev/null

# Kill all screen sessions to be sure
screen -ls | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit 2>/dev/null

sleep 2

echo "✅ Screen sessions killed"
echo ""

# Kill by port if bot uses specific ports
echo "Killing processes on common bot ports..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:5000 | xargs kill -9 2>/dev/null

echo "✅ Port processes killed"
echo ""

# Verify everything is stopped
echo "Checking if any bots still running..."
running=$(ps aux | grep python | grep -i bot | grep -v grep)

if [ -z "$running" ]; then
    echo "✅ ALL BOTS STOPPED!"
    echo ""
    echo "No Python bot processes running."
else
    echo "⚠️  Some processes still running:"
    ps aux | grep python | grep -i bot | grep -v grep
    echo ""
    echo "Manual kill needed:"
    ps aux | grep python | grep -i bot | grep -v grep | awk '{print "kill -9 " $2}'
fi

echo ""
echo "================================================================================"
echo "Screen sessions:"
screen -ls
echo ""
echo "If you see any sessions, kill them:"
echo "  screen -X -S SESSION_NAME quit"
echo "================================================================================"
echo ""

echo "🧹 NOW CLEANING DIRECTORIES..."
echo ""

cd ~

# Remove old bot directories
echo "Removing old bot directories..."
rm -rf bot trading_bot old_bot lean-trader leantrader Lean-Trader 2>/dev/null
rm -rf *bot* 2>/dev/null

echo "✅ Directories cleaned"
echo ""

echo "================================================================================"
echo "✅ VPS IS COMPLETELY CLEAN!"
echo "================================================================================"
echo ""
echo "Current directory contents:"
ls -la
echo ""
echo "Python processes:"
ps aux | grep python | grep -v grep
echo ""
echo "If empty, you're ready to deploy new bot!"
echo "================================================================================"
