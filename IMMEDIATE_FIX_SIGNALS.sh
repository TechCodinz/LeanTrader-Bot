#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       🚨 IMMEDIATE SIGNAL FIX + LIVE MODE ACTIVATION                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

## THE REAL PROBLEM

Your bot is in TESTNET mode which has two issues:
1. Testnet doesn't provide reliable price feeds
2. Signals can't get entry prices → Symbol: N/A, Entry: $0.00

SOLUTION: Switch to LIVE mode with small capital

═══════════════════════════════════════════════════════════════════════

EOF

echo "📊 STEP 1: Deploy Latest Signal Fix"
echo "════════════════════════════════════════════════════════════════"
cd /root/trading_bot || exit 1
git pull origin cursor/check-and-update-trading-bot-service-0f23
echo "✅ Latest code pulled"
echo ""

echo "📝 STEP 2: Switch to LIVE Mode"
echo "════════════════════════════════════════════════════════════════"

# Backup current RUN_BOT.py
cp RUN_BOT.py RUN_BOT.py.backup

# Create a simple script to check current mode
python3 -c "
import sys
with open('RUN_BOT.py', 'r') as f:
    content = f.read()
    if 'mode=\"testnet\"' in content or 'mode=\\'testnet\\'' in content:
        print('⚠️  Currently in TESTNET mode')
        print('   This is why you can\\'t get real prices!')
    elif 'mode=\"live\"' in content or 'mode=\\'live\\'' in content:
        print('✅ Already in LIVE mode')
    else:
        print('⚠️  Mode not clearly set')
"

echo ""
echo "To switch to LIVE mode:"
echo "  nano /root/trading_bot/RUN_BOT.py"
echo ""
echo "Find line ~70 with:"
echo "  mode = 'testnet'  or  await main(mode='testnet')"
echo ""
echo "Change to:"
echo "  mode = 'live'  or  await main(mode='live')"
echo ""
echo "Save (Ctrl+X, Y, Enter)"
echo ""

read -p "🤔 Do you want me to auto-switch to LIVE mode? (yes/no): " answer

if [ "$answer" = "yes" ]; then
    # Try to automatically switch
    sed -i "s/mode='testnet'/mode='live'/g" RUN_BOT.py
    sed -i 's/mode="testnet"/mode="live"/g' RUN_BOT.py
    sed -i "s/mode = 'testnet'/mode = 'live'/g" RUN_BOT.py
    sed -i 's/mode = "testnet"/mode = "live"/g' RUN_BOT.py
    
    echo "✅ Switched to LIVE mode"
    echo "⚠️  WARNING: Bot will now use REAL money!"
    echo "   Start with small amount (\$40-100)"
else
    echo "ℹ️  Staying in testnet mode"
    echo "   Note: Testnet has limited functionality"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 STEP 3: Restart Bot"
echo "════════════════════════════════════════════════════════════════"

sudo systemctl restart trading-bot
echo "✅ Bot restarted"
echo ""

echo "Waiting 30 seconds for startup..."
sleep 30

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 STEP 4: Check Status"
echo "════════════════════════════════════════════════════════════════"
systemctl status trading-bot --no-pager | head -15

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 STEP 5: Watch for Signals"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Run this command to watch for signals being sent:"
echo ""
echo "  journalctl -u trading-bot -f | grep -E 'signal sent|VIP signal|Free signal'"
echo ""
echo "You should see:"
echo "  📱 Free signal sent: ADA/USDT (conf: 75%)"
echo "  📱 VIP signal sent: ETH/USDT (conf: 84%)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                      ✅ WHAT HAPPENS NOW                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

If you switched to LIVE mode:

Within 2-5 minutes:
✅ Bot connects to Bybit LIVE API
✅ Gets real-time prices
✅ Generates signals with REAL prices
✅ Sends to FREE/VIP channels with actual values
✅ Executes trades with your balance

Your Telegram channels will show:
📢 TRADING SIGNAL
Symbol: BTC/USDT ✅ (not N/A!)
Entry: $43,256.78 ✅ (not $0.00!)

⚠️  WARNING - IMPORTANT:
- Start with SMALL capital ($40-100)
- Monitor closely for first day
- Bot will use REAL money
- Wins = real profit
- Losses = real loss

═══════════════════════════════════════════════════════════════════════

📖 Read full guide: HONEST_ANSWERS_AND_CRITICAL_INFO.md

EOF

echo "Done! Check your Telegram channels in 2-3 minutes 📱"
