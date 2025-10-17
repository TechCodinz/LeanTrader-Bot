#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    🚀 SETUP DUAL-MODE TRADING BOTS                                   ║
║                                                                      ║
║    TESTNET: Learn, test, experiment                                  ║
║    LIVE: Trade with proven strategies ($62 capital)                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📥 Step 1/5: Pull Latest Code"
echo "═══════════════════════════════════════════════════════════════════════"

cd /root/trading_bot
git pull origin cursor/check-and-update-trading-bot-service-0f23

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 Step 2/5: Install Service Files"
echo "═══════════════════════════════════════════════════════════════════════"

# Backup old service
sudo cp /etc/systemd/system/trading-bot.service /etc/systemd/system/trading-bot.service.old

# Install testnet service
sudo cp trading-bot-testnet.service /etc/systemd/system/
echo "✅ Testnet service installed"

# Install live service
sudo cp trading-bot-live.service /etc/systemd/system/
echo "✅ Live service installed"

# Reload systemd
sudo systemctl daemon-reload
echo "✅ Systemd reloaded"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 Step 3/5: Stop Old Service"
echo "═══════════════════════════════════════════════════════════════════════"

sudo systemctl stop trading-bot
sudo systemctl disable trading-bot
echo "✅ Old service stopped"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 Step 4/5: Start Both Bots"
echo "═══════════════════════════════════════════════════════════════════════"

# Enable and start testnet
sudo systemctl enable trading-bot-testnet
sudo systemctl start trading-bot-testnet
echo "✅ Testnet bot started"

sleep 5

# Enable and start live
sudo systemctl enable trading-bot-live
sudo systemctl start trading-bot-live
echo "✅ Live bot started"

echo ""
echo "Waiting 30 seconds for both bots to initialize..."
sleep 30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 Step 5/5: Check Status"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "TESTNET Bot:"
systemctl status trading-bot-testnet --no-pager | head -10

echo ""
echo "LIVE Bot:"
systemctl status trading-bot-live --no-pager | head -10

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ BOTH BOTS RUNNING!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "🧪 TESTNET Bot:"
echo "  - Tests strategies"
echo "  - Learns from experiments"
echo "  - Tries new ideas"
echo "  - No real money"
echo "  - Signals may have price=\$0"
echo ""
echo "💰 LIVE Bot:"
echo "  - Trades with proven strategies"
echo "  - Uses your \$62.77 capital"
echo "  - Real prices, real trades"
echo "  - Sends signals to channels"
echo "  - Makes real profits!"
echo ""
echo "Monitor commands:"
echo "  Testnet: journalctl -u trading-bot-testnet -f"
echo "  Live:    journalctl -u trading-bot-live -f"
echo "  Both:    journalctl -u 'trading-bot-*' -f"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "📱 Checking for LIVE signals..."
echo "═══════════════════════════════════════════════════════════════════════"

timeout 60 bash -c 'journalctl -u trading-bot-live -f | grep --line-buffered -E "📊 Prices: entry=\$[1-9]|✅✅✅ SUCCESS" | head -3'

echo ""
echo "Check your Telegram channels - signals should be appearing!"
echo ""
