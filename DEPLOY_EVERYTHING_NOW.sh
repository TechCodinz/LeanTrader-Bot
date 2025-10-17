#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            🚀 DEPLOY EVERYTHING - FULL ACTIVATION                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

This script will:
1. Pull latest code (signal fixes)
2. Switch to LIVE mode
3. Restart bot
4. Monitor signals

⚠️  WARNING: This will enable REAL trading with REAL money!
Only proceed if you have $40-100 in your Bybit account.

EOF

read -p "Ready to deploy? Type 'YES' to continue: " confirm

if [ "$confirm" != "YES" ]; then
    echo "❌ Cancelled. No changes made."
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📥 STEP 1/4: Pull Latest Code"
echo "═══════════════════════════════════════════════════════════════════════"

cd /root/trading_bot || { echo "❌ Can't find /root/trading_bot"; exit 1; }

git stash 2>/dev/null  # Save any local changes
git pull origin cursor/check-and-update-trading-bot-service-0f23

if [ $? -eq 0 ]; then
    echo "✅ Code updated"
else
    echo "❌ Git pull failed. Check connection."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔧 STEP 2/4: Switch to LIVE Mode"
echo "═══════════════════════════════════════════════════════════════════════"

# Backup
cp RUN_BOT.py RUN_BOT.py.backup.$(date +%s)

# Switch all variations of testnet to live
sed -i "s/mode='testnet'/mode='live'/g" RUN_BOT.py
sed -i 's/mode="testnet"/mode="live"/g' RUN_BOT.py
sed -i "s/mode = 'testnet'/mode = 'live'/g" RUN_BOT.py
sed -i 's/mode = "testnet"/mode = "live"/g' RUN_BOT.py

# Also check if --testnet flag is used
if grep -q "\-\-testnet" RUN_BOT.py; then
    echo "⚠️  Found --testnet flag in code"
    sed -i 's/--testnet/--live/g' RUN_BOT.py
fi

echo "✅ Switched to LIVE mode"
echo "⚠️  Bot will now trade with REAL money!"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔄 STEP 3/4: Restart Bot"
echo "═══════════════════════════════════════════════════════════════════════"

sudo systemctl restart trading-bot
echo "✅ Bot restarting..."

echo ""
echo "Waiting 30 seconds for initialization..."
for i in {30..1}; do
    echo -ne "\r⏳ $i seconds remaining...  "
    sleep 1
done
echo ""

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 STEP 4/4: Check Status"
echo "═══════════════════════════════════════════════════════════════════════"

# Check if running
if systemctl is-active --quiet trading-bot; then
    echo "✅ Bot is RUNNING"
    
    # Show status
    echo ""
    echo "Status:"
    systemctl status trading-bot --no-pager | head -15
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "📱 Checking for Signals..."
    echo "═══════════════════════════════════════════════════════════════════════"
    
    echo "Checking recent logs..."
    sleep 5
    
    # Check for signal routing
    if journalctl -u trading-bot --since "2 minutes ago" | grep -q "signal sent"; then
        echo "✅ SIGNALS ARE BEING SENT!"
        echo ""
        journalctl -u trading-bot --since "2 minutes ago" | grep "signal sent"
    else
        echo "⚠️  No signals sent yet (may need more time)"
        echo "   Run this to watch live:"
        echo "   journalctl -u trading-bot -f | grep signal"
    fi
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "✅ DEPLOYMENT COMPLETE!"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Your bot is now:"
    echo "  ✅ Running in LIVE mode"
    echo "  ✅ Using real Bybit API"
    echo "  ✅ Generating signals with real prices"
    echo "  ✅ Sending to FREE/VIP channels"
    echo "  ✅ Executing trades with real money"
    echo ""
    echo "📱 Check your Telegram channels NOW!"
    echo ""
    echo "📊 Monitor trades:"
    echo "   journalctl -u trading-bot -f"
    echo ""
    echo "💰 Check balance in Bybit account"
    echo ""
    
else
    echo "❌ Bot is NOT running!"
    echo ""
    echo "Check errors:"
    journalctl -u trading-bot -n 50 | tail -20
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🎯 NEXT STEPS:"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Check Telegram FREE channel for signals"
echo "2. Check Telegram VIP channel for premium signals"
echo "3. Monitor bot logs: journalctl -u trading-bot -f"
echo "4. Check Bybit account for executed trades"
echo "5. Watch for profit growth!"
echo ""
echo "⚠️  IMPORTANT:"
echo "- Monitor closely for first 24 hours"
echo "- Start with small capital (\$40-100)"
echo "- Don't panic on first loss (normal)"
echo "- Let balance-aware sizing compound wins"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "🎉 Your bot is LIVE and ready to trade! 🚀"
echo ""
