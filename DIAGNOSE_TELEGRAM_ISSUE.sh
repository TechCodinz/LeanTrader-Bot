#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         🔍 TELEGRAM SIGNAL DIAGNOSTIC                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "1️⃣ Check if Bot Restarted"
echo "═══════════════════════════════════════════════════════════════════════"
systemctl status trading-bot --no-pager | head -10
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "2️⃣ Check Telegram Config in .env"
echo "═══════════════════════════════════════════════════════════════════════"
grep "TELEGRAM" /root/trading_bot/.env | grep -v "^#"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "3️⃣ Check if Telegram Initialized"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "3 minutes ago" --no-pager | grep -i "telegram.*init\|premium.*vip\|channel config" | head -10
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "4️⃣ Check Recent Signal Attempts"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep "signal sent" | tail -10
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "5️⃣ Check for Telegram Errors"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -i "telegram.*error\|forbidden\|badrequest\|channel.*fail" | head -10
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "6️⃣ Test Direct Send (Manual)"
echo "═══════════════════════════════════════════════════════════════════════"

cd /root/trading_bot
source venv/bin/activate

python3 <<'PYTHON'
import asyncio
import os
from pathlib import Path
from telegram import Bot

# Load .env
env_path = Path('.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

async def send_test_signal():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    free_id = os.getenv('TELEGRAM_FREE_CHANNEL')
    vip_id = os.getenv('TELEGRAM_VIP_CHANNEL')
    
    bot = Bot(token=token)
    
    # Send real-looking signal to FREE
    if free_id and free_id.startswith('-100'):
        try:
            message = """📢 <b>TEST TRADING SIGNAL</b>

Symbol: BTC/USDT
Side: BUY
Confidence: 85%

Entry: $66,345.67
Stop Loss: $65,019.95
Take Profit: $67,671.39

🌟 VIP members can trade with ONE CLICK!
Use /subscribe to join VIP"""
            
            result = await bot.send_message(
                chat_id=free_id,
                text=message,
                parse_mode='HTML'
            )
            print(f'✅ Test signal sent to FREE channel (msg_id: {result.message_id})')
            print(f'   Check channel: {free_id}')
        except Exception as e:
            print(f'❌ Failed to send to FREE: {e}')
    else:
        print(f'⚠️  FREE channel ID invalid: {free_id}')
    
    # Send to VIP
    if vip_id and vip_id.startswith('-100'):
        try:
            message = """🌟 <b>VIP PREMIUM TEST SIGNAL</b>

Symbol: ETH/USDT
Action: BUY
Confidence: 92% 🔥

Entry: $2,543.21
Stop Loss: $2,492.74
Take Profit: $2,593.68

Risk/Reward: 2.0:1"""
            
            result = await bot.send_message(
                chat_id=vip_id,
                text=message,
                parse_mode='HTML'
            )
            print(f'✅ Test signal sent to VIP channel (msg_id: {result.message_id})')
            print(f'   Check channel: {vip_id}')
        except Exception as e:
            print(f'❌ Failed to send to VIP: {e}')
    else:
        print(f'⚠️  VIP channel ID invalid: {vip_id}')

asyncio.run(send_test_signal())
PYTHON

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DIAGNOSTIC COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Check your Telegram channels NOW!"
echo "You should see test signals above."
echo ""
