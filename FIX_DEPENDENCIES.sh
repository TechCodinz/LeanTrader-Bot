#!/bin/bash
# FIX DEPENDENCY INSTALLATION
# Run this on VPS to install all dependencies

echo "================================================================================"
echo "📦 INSTALLING BOT DEPENDENCIES"
echo "================================================================================"
echo ""

cd ~/trading_bot || cd /root/trading_bot

# Method 1: Use --break-system-packages (safe for dedicated trading VPS)
echo "Method 1: Installing with pip (--break-system-packages)..."
pip3 install --break-system-packages -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed via pip"
else
    echo "⚠️  Pip install had issues, trying apt..."
    
    # Method 2: Install via apt (Debian packages)
    echo "Method 2: Installing via apt..."
    sudo apt update
    sudo apt install -y \
        python3-pandas \
        python3-numpy \
        python3-requests \
        python3-aiohttp \
        python3-bs4 \
        python3-dateutil \
        python3-pytz
    
    # Install ccxt and web3 via pip (not in apt)
    pip3 install --break-system-packages ccxt web3 python-telegram-bot qiskit tensorflow scikit-learn
    
    echo "✅ Dependencies installed via apt + pip"
fi

echo ""
echo "================================================================================"
echo "✅ DEPENDENCIES INSTALLED"
echo "================================================================================"
echo ""

# Restart bot service
echo "Restarting bot service..."
sudo systemctl restart trading-bot

sleep 3

# Check status
echo "Bot status:"
systemctl status trading-bot --no-pager

echo ""
echo "================================================================================"
echo "View logs with: tail -f ~/trading_bot/bot.log"
echo "================================================================================"
