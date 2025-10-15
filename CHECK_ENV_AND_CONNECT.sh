#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        CHECKING .env FILE & CONNECTING EVERYTHING                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 1/5: Checking .env file"
echo "═══════════════════════════════════════════════════════════════════════"

cd /root/trading_bot || exit 1

if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "   Creating from template..."
    cp .env.example .env 2>/dev/null || touch .env
fi

echo "✅ .env file exists"
echo ""

# Check for Telegram credentials
echo "Checking Telegram configuration..."
echo ""

TELEGRAM_BOT_TOKEN=$(grep "^TELEGRAM_BOT_TOKEN=" .env | cut -d'=' -f2)
TELEGRAM_ADMIN_CHAT_ID=$(grep "^TELEGRAM_ADMIN_CHAT_ID=" .env | cut -d'=' -f2)
TELEGRAM_FREE_CHANNEL=$(grep "^TELEGRAM_FREE_CHANNEL=" .env | cut -d'=' -f2)
TELEGRAM_VIP_CHANNEL=$(grep "^TELEGRAM_VIP_CHANNEL=" .env | cut -d'=' -f2)

if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ "$TELEGRAM_BOT_TOKEN" = "your_telegram_bot_token_here" ]; then
    echo "❌ TELEGRAM_BOT_TOKEN not set!"
    echo "   Current: Empty or default"
    echo ""
    read -p "Enter your Telegram Bot Token: " token
    if [ ! -z "$token" ]; then
        if grep -q "^TELEGRAM_BOT_TOKEN=" .env; then
            sed -i "s|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$token|" .env
        else
            echo "TELEGRAM_BOT_TOKEN=$token" >> .env
        fi
        echo "✅ Bot token added"
    fi
else
    echo "✅ TELEGRAM_BOT_TOKEN: Set (${TELEGRAM_BOT_TOKEN:0:10}...)"
fi

if [ -z "$TELEGRAM_ADMIN_CHAT_ID" ] || [ "$TELEGRAM_ADMIN_CHAT_ID" = "your_admin_chat_id_here" ]; then
    echo "❌ TELEGRAM_ADMIN_CHAT_ID not set!"
    echo ""
    read -p "Enter your Telegram Admin Chat ID (your user ID): " admin_id
    if [ ! -z "$admin_id" ]; then
        if grep -q "^TELEGRAM_ADMIN_CHAT_ID=" .env; then
            sed -i "s|^TELEGRAM_ADMIN_CHAT_ID=.*|TELEGRAM_ADMIN_CHAT_ID=$admin_id|" .env
        else
            echo "TELEGRAM_ADMIN_CHAT_ID=$admin_id" >> .env
        fi
        echo "✅ Admin chat ID added"
    fi
else
    echo "✅ TELEGRAM_ADMIN_CHAT_ID: Set ($TELEGRAM_ADMIN_CHAT_ID)"
fi

if [ -z "$TELEGRAM_FREE_CHANNEL" ] || [ "$TELEGRAM_FREE_CHANNEL" = "@your_free_channel" ]; then
    echo "⚠️  TELEGRAM_FREE_CHANNEL not set"
    echo "   You can add it later"
else
    echo "✅ TELEGRAM_FREE_CHANNEL: Set ($TELEGRAM_FREE_CHANNEL)"
fi

if [ -z "$TELEGRAM_VIP_CHANNEL" ] || [ "$TELEGRAM_VIP_CHANNEL" = "@your_vip_channel" ]; then
    echo "⚠️  TELEGRAM_VIP_CHANNEL not set"
    echo "   You can add it later"
else
    echo "✅ TELEGRAM_VIP_CHANNEL: Set ($TELEGRAM_VIP_CHANNEL)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 2/5: Checking Exchange API Keys"
echo "═══════════════════════════════════════════════════════════════════════"

# Check each exchange
for exchange in MEXC BITGET OKX KUCOIN GATE BINANCE BYBIT; do
    key=$(grep "^${exchange}_API_KEY=" .env | cut -d'=' -f2)
    if [ ! -z "$key" ] && [ "$key" != "your_${exchange,,}_api_key_here" ]; then
        echo "✅ $exchange: API key configured"
    else
        echo "⚠️  $exchange: No API key (will skip)"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 3/5: Testing Telegram Connection"
echo "═══════════════════════════════════════════════════════════════════════"

source venv/bin/activate

python3 <<'PYTHON'
import asyncio
import os
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

async def test_telegram():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    admin_id = os.getenv('TELEGRAM_ADMIN_CHAT_ID')
    
    if not token or token == 'your_telegram_bot_token_here':
        print('❌ Bot token not configured')
        return False
    
    try:
        bot = Bot(token=token)
        me = await bot.get_me()
        print(f'✅ Bot connected: @{me.username}')
        
        if admin_id and admin_id != 'your_admin_chat_id_here':
            try:
                result = await bot.send_message(
                    chat_id=admin_id,
                    text='🎉 Trading bot connected! All systems ready.'
                )
                print(f'✅ Test message sent to admin (msg_id: {result.message_id})')
                return True
            except Exception as e:
                print(f'❌ Could not send to admin: {e}')
                print('   Make sure you\'ve started a chat with the bot first')
                return False
        else:
            print('⚠️  Admin chat ID not set, skipping message test')
            return True
            
    except Exception as e:
        print(f'❌ Telegram connection failed: {e}')
        return False

result = asyncio.run(test_telegram())
exit(0 if result else 1)
PYTHON

TELEGRAM_OK=$?

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 4/5: Testing Exchange Connections"
echo "═══════════════════════════════════════════════════════════════════════"

python3 <<'PYTHON'
import asyncio
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

async def test_exchanges():
    exchanges_config = {
        'mexc': {'key_env': 'MEXC_API_KEY', 'secret_env': 'MEXC_SECRET'},
        'bitget': {'key_env': 'BITGET_API_KEY', 'secret_env': 'BITGET_SECRET'},
        'okx': {'key_env': 'OKX_API_KEY', 'secret_env': 'OKX_SECRET'},
        'kucoin': {'key_env': 'KUCOIN_API_KEY', 'secret_env': 'KUCOIN_SECRET'},
        'gateio': {'key_env': 'GATE_API_KEY', 'secret_env': 'GATE_SECRET'},
        'binance': {'key_env': 'BINANCE_API_KEY', 'secret_env': 'BINANCE_SECRET'},
        'bybit': {'key_env': 'BYBIT_API_KEY', 'secret_env': 'BYBIT_SECRET'}
    }
    
    connected = []
    
    for name, config in exchanges_config.items():
        api_key = os.getenv(config['key_env'])
        secret = os.getenv(config['secret_env'])
        
        if not api_key or api_key.startswith('your_'):
            print(f'⚠️  {name.upper()}: Not configured')
            continue
        
        try:
            exchange_class = getattr(ccxt, name)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True
            })
            
            balance = await exchange.fetch_balance()
            total_usd = balance.get('total', {}).get('USDT', 0)
            
            print(f'✅ {name.upper()}: Connected! Balance: ${total_usd:.2f} USDT')
            connected.append(name)
            
            await exchange.close()
            
        except Exception as e:
            print(f'❌ {name.upper()}: Failed - {str(e)[:50]}')
    
    print(f'\n📊 Summary: {len(connected)}/7 exchanges connected')
    return len(connected)

num_connected = asyncio.run(test_exchanges())
PYTHON

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📋 STEP 5/5: Deploying with Arbitrage"
echo "═══════════════════════════════════════════════════════════════════════"

# Pull latest code
echo "Pulling latest code..."
git pull origin cursor/check-and-update-trading-bot-service-0f23

# Restart bot
echo "Restarting bot..."
sudo systemctl restart trading-bot

echo "Waiting 30 seconds for startup..."
sleep 30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 FINAL STATUS"
echo "═══════════════════════════════════════════════════════════════════════"

systemctl status trading-bot --no-pager | head -15

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🔍 Recent Logs"
echo "═══════════════════════════════════════════════════════════════════════"

journalctl -u trading-bot --since "1 minute ago" --no-pager | tail -30

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Check for:"
echo "  ✅ Telegram bot connected"
echo "  ✅ Exchanges connected"
echo "  ✅ Arbitrage scanner started"
echo "  ✅ Signals being generated"
echo ""
echo "Monitor live:"
echo "  journalctl -u trading-bot -f | grep -E 'Arbitrage|signal sent|Connected'"
echo ""
