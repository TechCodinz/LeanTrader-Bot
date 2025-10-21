#!/bin/bash
###############################################################################
# COMPLETE VPS SETUP - RUN THIS ONCE
###############################################################################

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              INSTALLING DEPENDENCIES                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# Install system packages
pip3 install --upgrade pip

# Core packages
pip3 install ccxt python-telegram-bot python-dotenv asyncio aiohttp

# ML/AI packages
pip3 install pandas numpy scikit-learn tensorflow || echo "⚠️ TF optional"

# TA packages
pip3 install matplotlib mplfinance || echo "⚠️ Charts optional"

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                  SETUP COMPLETE                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Run: ./start_bot.sh"

