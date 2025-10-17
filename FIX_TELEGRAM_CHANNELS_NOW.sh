#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           FIX TELEGRAM CHANNELS - INTERACTIVE SETUP                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

This script will help you:
1. Get correct channel IDs
2. Add bot to channels as admin
3. Update .env with correct IDs
4. Test that signals reach channels

EOF

cd /root/trading_bot || exit 1

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📱 TELEGRAM CHANNEL SETUP"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Do you have FREE and VIP Telegram channels created?"
read -p "(yes/no): " has_channels

if [ "$has_channels" != "yes" ]; then
    echo ""
    echo "Create channels first:"
    echo "1. Open Telegram"
    echo "2. Create new channel (FREE signals)"
    echo "3. Create another channel (VIP signals)"
    echo "4. Come back here"
    echo ""
    read -p "Press Enter when channels are created..."
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 1: Add Bot to Channels"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "For BOTH channels:"
echo "1. Open channel in Telegram"
echo "2. Click channel name/info"
echo "3. Click 'Administrators'"
echo "4. Click 'Add Administrator'"
echo "5. Search for your bot"
echo "6. Add it as admin"
echo "7. Enable 'Post Messages' permission"
echo ""
read -p "Press Enter when bot is added to both channels..."

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 2: Get Channel IDs"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "For EACH channel:"
echo "1. Send a message to the channel"
echo "2. Forward that message to: @getidsbot"
echo "3. It will reply with the channel ID"
echo "4. Channel ID looks like: -1001234567890"
echo ""
echo "Enter FREE channel ID:"
read -p "ID: " free_id

echo "Enter VIP channel ID:"
read -p "ID: " vip_id

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 3: Update .env File"
echo "═══════════════════════════════════════════════════════════════════════"

# Update .env
if grep -q "^TELEGRAM_FREE_CHANNEL=" .env; then
    sed -i "s|^TELEGRAM_FREE_CHANNEL=.*|TELEGRAM_FREE_CHANNEL=$free_id|" .env
else
    echo "TELEGRAM_FREE_CHANNEL=$free_id" >> .env
fi

if grep -q "^TELEGRAM_VIP_CHANNEL=" .env; then
    sed -i "s|^TELEGRAM_VIP_CHANNEL=.*|TELEGRAM_VIP_CHANNEL=$vip_id|" .env
else
    echo "TELEGRAM_VIP_CHANNEL=$vip_id" >> .env
fi

echo "✅ Channel IDs updated in .env"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 4: Test Channels"
echo "═══════════════════════════════════════════════════════════════════════"

source venv/bin/activate

python3 <<PYTHON
import asyncio
from telegram import Bot
import os
from dotenv import load_dotenv

load_dotenv()

async def test_channels():
    bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
    
    free_id = "$free_id"
    vip_id = "$vip_id"
    
    print('Testing FREE channel...')
    try:
        result = await bot.send_message(
            chat_id=free_id,
            text='📢 Test message - FREE channel is working!'
        )
        print(f'✅ FREE channel works! (msg_id: {result.message_id})')
    except Exception as e:
        print(f'❌ FREE channel failed: {e}')
        print('   Make sure bot is admin with Post Messages permission')
    
    print('')
    print('Testing VIP channel...')
    try:
        result = await bot.send_message(
            chat_id=vip_id,
            text='🌟 Test message - VIP channel is working!'
        )
        print(f'✅ VIP channel works! (msg_id: {result.message_id})')
    except Exception as e:
        print(f'❌ VIP channel failed: {e}')
        print('   Make sure bot is admin with Post Messages permission')

asyncio.run(test_channels())
PYTHON

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 5: Restart Bot"
echo "═══════════════════════════════════════════════════════════════════════"

sudo systemctl restart trading-bot

echo "✅ Bot restarted with new channel IDs"
echo ""
echo "Waiting 30 seconds..."
sleep 30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 Checking for Signals"
echo "═══════════════════════════════════════════════════════════════════════"

journalctl -u trading-bot --since "30 seconds ago" | grep -E "FREE channel signal sent|VIP channel signal sent" | tail -10

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ SETUP COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Check your Telegram channels now!"
echo "You should see signals appearing."
echo ""
echo "Monitor live:"
echo "  journalctl -u trading-bot -f | grep 'signal sent'"
echo ""
