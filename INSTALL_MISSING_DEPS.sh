#!/bin/bash

echo "📦 Installing missing dependencies for new features..."
echo ""

cd /root/trading_bot

# Activate venv
source venv/bin/activate

echo "1️⃣  Installing aiohttp (for news trading)..."
pip install aiohttp

echo ""
echo "2️⃣  Verifying numpy (should already be installed)..."
pip install --upgrade numpy

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "New features now fully enabled:"
echo "  ✅ News Trading (CoinGecko trending)"
echo "  ✅ Hedge Fund Arsenal (statistical arbitrage)"
echo ""
