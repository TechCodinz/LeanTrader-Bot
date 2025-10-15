#!/bin/bash
# Quick fix for missing bs4 (BeautifulSoup4) package

echo "🔧 Installing missing package: beautifulsoup4..."

/root/trading_bot/venv/bin/pip install beautifulsoup4 lxml -q

echo "✅ beautifulsoup4 installed!"
echo ""
echo "🔄 Restarting bot..."

sudo systemctl restart trading-bot

sleep 10

echo "✅ Done! Checking status..."
systemctl status trading-bot --no-pager | head -15
