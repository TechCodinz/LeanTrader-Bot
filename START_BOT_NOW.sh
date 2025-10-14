#!/bin/bash
# Quick Start Script for Trading Bot
# Run this on your VPS after uploading files

echo "================================================================================"
echo "🚀 TRADING BOT - QUICK START"
echo "================================================================================"
echo ""

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found: $(python3 --version)"
else
    echo "❌ Python3 not found. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-pip
fi

# Check pip
echo ""
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 found"
else
    echo "Installing pip..."
    sudo apt install -y python3-pip
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Verify installation
echo ""
echo "Verifying dependencies..."
python3 -c "import ccxt, web3, telegram; print('✅ Core dependencies OK')" 2>/dev/null || echo "⚠️  Some dependencies missing - check above"

# Verify API keys
echo ""
echo "Verifying API keys..."
python3 load_env.py

# Ask for confirmation
echo ""
echo "================================================================================"
echo "Ready to start bot!"
echo "================================================================================"
echo ""
echo "This will start the bot in TESTNET mode (safe, fake money)"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Start bot in screen
echo ""
echo "Starting bot in screen session..."
screen -dmS trading_bot python3 RUN_BOT.py --testnet

echo ""
echo "================================================================================"
echo "✅ BOT STARTED!"
echo "================================================================================"
echo ""
echo "View logs:"
echo "  screen -r trading_bot"
echo ""
echo "Detach from screen:"
echo "  Press Ctrl+A then D"
echo ""
echo "Check status:"
echo "  screen -ls"
echo ""
echo "Stop bot:"
echo "  screen -X -S trading_bot quit"
echo ""
echo "================================================================================"
echo "Bot is running in background. Check Telegram for notifications!"
echo "================================================================================"
