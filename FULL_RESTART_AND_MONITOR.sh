#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    🔄 FULL RESTART WITH LIVE MONITORING                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "Pulling latest code..."
cd /root/trading_bot
git pull origin cursor/check-and-update-trading-bot-service-0f23

echo ""
echo "Stopping bot completely..."
sudo systemctl stop trading-bot
sleep 5

echo "Starting bot with fresh code..."
sudo systemctl start trading-bot

echo ""
echo "Waiting 30 seconds for initialization..."
sleep 30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 BOT STATUS"
echo "═══════════════════════════════════════════════════════════════════════"
systemctl status trading-bot --no-pager | head -15

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 COMPREHENSIVE DIAGNOSTIC"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "1️⃣ Exchanges:"
journalctl -u trading-bot --since "60 seconds ago" --no-pager | grep "added to arbitrage"

echo ""
echo "2️⃣ Arbitrage:"
journalctl -u trading-bot --since "60 seconds ago" --no-pager | grep -i "arbitrage.*start\|arbitrage.*active"

echo ""
echo "3️⃣ Telegram Config:"
journalctl -u trading-bot --since "60 seconds ago" --no-pager | grep "Telegram config"

echo ""
echo "4️⃣ Signals Generated:"
journalctl -u trading-bot --since "60 seconds ago" --no-pager | grep "Scalper generated\|signal.*published"

echo ""
echo "5️⃣ Signal Sending:"
journalctl -u trading-bot --since "60 seconds ago" --no-pager | grep "🔍 Sending"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📱 LIVE SIGNAL MONITOR (Watch for 2-3 minutes)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C when done watching"
echo ""
sleep 3

journalctl -u trading-bot -f | grep --line-buffered -E "🔍 Sending|🔵 send_signal|📊 Extracted|📊 Prices|❌ Skipping|📤 Attempting|✅✅✅ SUCCESS|arbitrage found|Scalper generated"
