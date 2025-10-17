#!/bin/bash

echo "Fixing .env file and connecting all exchanges..."

cd /root/trading_bot || exit 1

# Backup
cp .env .env.backup.$(date +%s)

# Create clean .env file
cat > .env.new <<'EOF'
# Telegram
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
TELEGRAM_ADMIN_CHAT_ID=5329503447
TELEGRAM_FREE_CHANNEL=@your_free_channel
TELEGRAM_VIP_CHANNEL=@your_vip_channel

# MEXC
MEXC_API_KEY=mx0vgl7ytNbnU44V5G
MEXC_SECRET=68562da9963e4666a32a4a73cda61062

# Bitget (requires passphrase)
BITGET_API_KEY=bg_a76be18966412e3f95b11eac379edf91
BITGET_SECRET=7507beac89f798ea88f469747e5c8fd0094fc3c3887afc671c777380d9c95cff
BITGET_PASSPHRASE=your_bitget_passphrase

# OKX (requires password)
OKX_API_KEY=9b6e8a19-5a9c-44ca-942f-e98cc36d0354
OKX_SECRET=A59EF8A73CC7462F9B2C20FECB4C6723
OKX_PASSWORD=your_okx_password

# KuCoin (requires password)
KUCOIN_API_KEY=68d494bd54d53500017383ed
KUCOIN_SECRET=e25a93de-01c1-4d4a-8d70-cc33c47d89ab
KUCOIN_PASSWORD=your_kucoin_password

# Gate.io
GATE_API_KEY=a0508d8aadf3bcb76e16f4373e1f3a76
GATE_SECRET=451770a07dbede1b87bb92f5ce98e24029d2fe91e0053be2ec41771c953113f9

# Binance
BINANCE_API_KEY=uxMw38StLFlWpqzi9OpFMMj4H7m3dWy8jnR2EAl2raL0n465jtxnlK9S2CYBflyf
BINANCE_SECRET=k6dCSRQfCiNYHn3PjWtORNUKP69EvnbyAHmEIio9my8qRBHbzNHbXdWV2HilzRrO

# Bybit
BYBIT_API_KEY=fX0py6Av5dFPmCPOMX
BYBIT_SECRET=P9lkTCsxMWhmnqmCeoZzjll0kR2Db7ykgek0
EOF

mv .env.new .env
chmod 600 .env

echo "✅ .env file cleaned and updated"
echo ""

echo "Testing connections..."
echo ""

source venv/bin/activate

python3 <<'PYTHON'
import asyncio
import ccxt.async_support as ccxt
import os
from dotenv import load_dotenv

load_dotenv()

async def test_all():
    results = []
    
    # Test MEXC
    try:
        exchange = ccxt.mexc({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_SECRET'),
            'enableRateLimit': True
        })
        balance = await exchange.fetch_balance()
        total = balance.get('total', {}).get('USDT', 0)
        print(f'✅ MEXC: Connected! Balance: ${total:.2f} USDT')
        results.append('mexc')
        await exchange.close()
    except Exception as e:
        print(f'❌ MEXC: {str(e)[:60]}')
    
    # Test Bitget (needs passphrase)
    passphrase = os.getenv('BITGET_PASSPHRASE')
    if passphrase and passphrase != 'your_bitget_passphrase':
        try:
            exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET'),
                'password': passphrase,
                'enableRateLimit': True
            })
            balance = await exchange.fetch_balance()
            total = balance.get('total', {}).get('USDT', 0)
            print(f'✅ Bitget: Connected! Balance: ${total:.2f} USDT')
            results.append('bitget')
            await exchange.close()
        except Exception as e:
            print(f'❌ Bitget: {str(e)[:60]}')
    else:
        print('⚠️  Bitget: Need passphrase (from Bitget API settings)')
    
    # Test OKX (needs password)
    okx_pass = os.getenv('OKX_PASSWORD')
    if okx_pass and okx_pass != 'your_okx_password':
        try:
            exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET'),
                'password': okx_pass,
                'enableRateLimit': True
            })
            balance = await exchange.fetch_balance()
            total = balance.get('total', {}).get('USDT', 0)
            print(f'✅ OKX: Connected! Balance: ${total:.2f} USDT')
            results.append('okx')
            await exchange.close()
        except Exception as e:
            print(f'❌ OKX: {str(e)[:60]}')
    else:
        print('⚠️  OKX: Need password (from OKX API settings)')
    
    # Test KuCoin (needs password)
    kucoin_pass = os.getenv('KUCOIN_PASSWORD')
    if kucoin_pass and kucoin_pass != 'your_kucoin_password':
        try:
            exchange = ccxt.kucoin({
                'apiKey': os.getenv('KUCOIN_API_KEY'),
                'secret': os.getenv('KUCOIN_SECRET'),
                'password': kucoin_pass,
                'enableRateLimit': True
            })
            balance = await exchange.fetch_balance()
            total = balance.get('total', {}).get('USDT', 0)
            print(f'✅ KuCoin: Connected! Balance: ${total:.2f} USDT')
            results.append('kucoin')
            await exchange.close()
        except Exception as e:
            print(f'❌ KuCoin: {str(e)[:60]}')
    else:
        print('⚠️  KuCoin: Need password (from KuCoin API settings)')
    
    # Test Gate.io
    try:
        exchange = ccxt.gateio({
            'apiKey': os.getenv('GATE_API_KEY'),
            'secret': os.getenv('GATE_SECRET'),
            'enableRateLimit': True
        })
        balance = await exchange.fetch_balance()
        total = balance.get('total', {}).get('USDT', 0)
        print(f'✅ Gate.io: Connected! Balance: ${total:.2f} USDT')
        results.append('gateio')
        await exchange.close()
    except Exception as e:
        print(f'❌ Gate.io: {str(e)[:60]}')
    
    # Test Binance
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'enableRateLimit': True
        })
        balance = await exchange.fetch_balance()
        total = balance.get('total', {}).get('USDT', 0)
        print(f'✅ Binance: Connected! Balance: ${total:.2f} USDT')
        results.append('binance')
        await exchange.close()
    except Exception as e:
        print(f'❌ Binance: {str(e)[:60]}')
    
    # Test Bybit
    try:
        exchange = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_SECRET'),
            'enableRateLimit': True
        })
        balance = await exchange.fetch_balance()
        total = balance.get('total', {}).get('USDT', 0)
        print(f'✅ Bybit: Connected! Balance: ${total:.2f} USDT')
        results.append('bybit')
        await exchange.close()
    except Exception as e:
        print(f'❌ Bybit: {str(e)[:60]}')
    
    print(f'\n📊 Result: {len(results)}/7 exchanges connected')
    
    if len(results) >= 2:
        print(f'\n✅ Arbitrage will work with {len(results)} exchanges!')
        print(f'   Connected: {", ".join([e.upper() for e in results])}')
    else:
        print('\n⚠️  Need at least 2 exchanges for arbitrage')
    
    return len(results)

num = asyncio.run(test_all())
exit(0 if num >= 2 else 1)
PYTHON

CONNECTED=$?

echo ""

if [ $CONNECTED -eq 0 ]; then
    echo "✅ Enough exchanges connected for arbitrage!"
    echo ""
    echo "Restarting bot..."
    sudo systemctl restart trading-bot
    
    echo "Waiting 30 seconds..."
    sleep 30
    
    echo ""
    echo "Checking for arbitrage..."
    journalctl -u trading-bot --since "30 seconds ago" | grep -i "arbitrage.*active\|exchange.*connected" | head -5
else
    echo "⚠️  Not enough exchanges connected yet"
    echo ""
    echo "To connect more exchanges, you need to add passwords/passphrases:"
    echo ""
    echo "For Bitget:"
    echo "  1. Go to Bitget → API Management"
    echo "  2. Find your API passphrase"
    echo "  3. Add to .env: BITGET_PASSPHRASE=your_passphrase"
    echo ""
    echo "For OKX:"
    echo "  1. Go to OKX → API Management"  
    echo "  2. Find your API password"
    echo "  3. Add to .env: OKX_PASSWORD=your_password"
    echo ""
    echo "For KuCoin:"
    echo "  1. Go to KuCoin → API Management"
    echo "  2. Find your API passphrase"
    echo "  3. Add to .env: KUCOIN_PASSWORD=your_passphrase"
    echo ""
    echo "Then restart: sudo systemctl restart trading-bot"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DONE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
