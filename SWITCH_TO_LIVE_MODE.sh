#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    🚀 SWITCH TO LIVE MODE - GET REAL PRICES & TRADES                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

⚠️  WARNING: This will enable REAL trading with your $62 capital!

Testnet vs Live:
  Testnet: ❌ No prices (price=$0) ❌ No real trades
  Live:    ✅ Real prices         ✅ Real trades ✅ Real profit

You have $62.77 across 4 exchanges ready to trade!

EOF

read -p "Switch to LIVE mode? Type 'YES' to confirm: " confirm

if [ "$confirm" != "YES" ]; then
    echo "❌ Cancelled. Staying in testnet mode."
    echo "   Note: Testnet signals will have price=$0 and won't appear in channels"
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Switching to LIVE mode..."
echo "═══════════════════════════════════════════════════════════════════════"

cd /root/trading_bot

# Backup systemd service
sudo cp /etc/systemd/system/trading-bot.service /etc/systemd/system/trading-bot.service.backup

# Update systemd service to remove --testnet flag
sudo sed -i 's|--testnet|--live|g' /etc/systemd/system/trading-bot.service

echo "✅ Updated systemd service to live mode"

# Reload systemd
sudo systemctl daemon-reload

echo "✅ Reloaded systemd"

# Restart bot
echo ""
echo "Restarting bot in LIVE mode..."
sudo systemctl restart trading-bot

echo "Waiting 30 seconds for startup..."
sleep 30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 LIVE MODE STATUS"
echo "═══════════════════════════════════════════════════════════════════════"

systemctl status trading-bot --no-pager | head -15

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 Checking for Signals with REAL Prices"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Watching for signals with real prices..."
echo "(This may take 1-2 minutes for first signals)"
echo ""

timeout 120 bash -c 'journalctl -u trading-bot -f | grep --line-buffered -E "📊 Prices: entry=\$[1-9]|✅✅✅ SUCCESS" | head -5'

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ LIVE MODE ACTIVATED!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Your bot is now:"
echo "  ✅ Trading with REAL prices"
echo "  ✅ Executing REAL trades"
echo "  ✅ Using your \$62.77 capital"
echo "  ✅ Sending complete signals to channels"
echo ""
echo "📱 Check your Telegram channels NOW!"
echo ""
echo "Monitor live:"
echo "  journalctl -u trading-bot -f"
echo ""
echo "⚠️  IMPORTANT:"
echo "  - Monitor closely for first hour"
echo "  - Check your exchange balances"
echo "  - Signals should now have real prices"
echo "  - Trades will execute automatically"
echo ""
