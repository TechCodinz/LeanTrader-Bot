#!/bin/bash

echo "Adding all exchange API keys to .env..."

cd /root/trading_bot || exit 1

# Backup .env
cp .env .env.backup.$(date +%s)

# Add MEXC
if ! grep -q "^MEXC_API_KEY=" .env; then
    echo "MEXC_API_KEY=mx0vgl7ytNbnU44V5G" >> .env
    echo "MEXC_SECRET=68562da9963e4666a32a4a73cda61062" >> .env
    echo "✅ MEXC keys added"
else
    sed -i "s|^MEXC_API_KEY=.*|MEXC_API_KEY=mx0vgl7ytNbnU44V5G|" .env
    sed -i "s|^MEXC_SECRET=.*|MEXC_SECRET=68562da9963e4666a32a4a73cda61062|" .env
    echo "✅ MEXC keys updated"
fi

# Add Bitget
if ! grep -q "^BITGET_API_KEY=" .env; then
    echo "BITGET_API_KEY=bg_a76be18966412e3f95b11eac379edf91" >> .env
    echo "BITGET_SECRET=7507beac89f798ea88f469747e5c8fd0094fc3c3887afc671c777380d9c95cff" >> .env
    echo "✅ Bitget keys added"
else
    sed -i "s|^BITGET_API_KEY=.*|BITGET_API_KEY=bg_a76be18966412e3f95b11eac379edf91|" .env
    sed -i "s|^BITGET_SECRET=.*|BITGET_SECRET=7507beac89f798ea88f469747e5c8fd0094fc3c3887afc671c777380d9c95cff|" .env
    echo "✅ Bitget keys updated"
fi

# Add OKX
if ! grep -q "^OKX_API_KEY=" .env; then
    echo "OKX_API_KEY=9b6e8a19-5a9c-44ca-942f-e98cc36d0354" >> .env
    echo "OKX_SECRET=A59EF8A73CC7462F9B2C20FECB4C6723" >> .env
    echo "✅ OKX keys added"
else
    sed -i "s|^OKX_API_KEY=.*|OKX_API_KEY=9b6e8a19-5a9c-44ca-942f-e98cc36d0354|" .env
    sed -i "s|^OKX_SECRET=.*|OKX_SECRET=A59EF8A73CC7462F9B2C20FECB4C6723|" .env
    echo "✅ OKX keys updated"
fi

# Add KuCoin
if ! grep -q "^KUCOIN_API_KEY=" .env; then
    echo "KUCOIN_API_KEY=68d494bd54d53500017383ed" >> .env
    echo "KUCOIN_SECRET=e25a93de-01c1-4d4a-8d70-cc33c47d89ab" >> .env
    echo "✅ KuCoin keys added"
else
    sed -i "s|^KUCOIN_API_KEY=.*|KUCOIN_API_KEY=68d494bd54d53500017383ed|" .env
    sed -i "s|^KUCOIN_SECRET=.*|KUCOIN_SECRET=e25a93de-01c1-4d4a-8d70-cc33c47d89ab|" .env
    echo "✅ KuCoin keys updated"
fi

# Add Gate.io
if ! grep -q "^GATE_API_KEY=" .env; then
    echo "GATE_API_KEY=a0508d8aadf3bcb76e16f4373e1f3a76" >> .env
    echo "GATE_SECRET=451770a07dbede1b87bb92f5ce98e24029d2fe91e0053be2ec41771c953113f9" >> .env
    echo "✅ Gate.io keys added"
else
    sed -i "s|^GATE_API_KEY=.*|GATE_API_KEY=a0508d8aadf3bcb76e16f4373e1f3a76|" .env
    sed -i "s|^GATE_SECRET=.*|GATE_SECRET=451770a07dbede1b87bb92f5ce98e24029d2fe91e0053be2ec41771c953113f9|" .env
    echo "✅ Gate.io keys updated"
fi

# Add Binance
if ! grep -q "^BINANCE_API_KEY=" .env; then
    echo "BINANCE_API_KEY=uxMw38StLFlWpqzi9OpFMMj4H7m3dWy8jnR2EAl2raL0n465jtxnlK9S2CYBflyf" >> .env
    echo "BINANCE_SECRET=k6dCSRQfCiNYHn3PjWtORNUKP69EvnbyAHmEIio9my8qRBHbzNHbXdWV2HilzRrO" >> .env
    echo "✅ Binance keys added"
else
    sed -i "s|^BINANCE_API_KEY=.*|BINANCE_API_KEY=uxMw38StLFlWpqzi9OpFMMj4H7m3dWy8jnR2EAl2raL0n465jtxnlK9S2CYBflyf|" .env
    sed -i "s|^BINANCE_SECRET=.*|BINANCE_SECRET=k6dCSRQfCiNYHn3PjWtORNUKP69EvnbyAHmEIio9my8qRBHbzNHbXdWV2HilzRrO|" .env
    echo "✅ Binance keys updated"
fi

echo ""
echo "✅ All exchange keys added to .env"
echo ""
echo "Testing connections..."

source venv/bin/activate

python3 - <<'PYTHON'
import asyncio
import ccxt
import os
from dotenv import load_dotenv

load_dotenv('.env')

async def test_all():
    exchanges_config = {
        'mexc': ccxt.mexc,
        'bitget': ccxt.bitget,
        'okx': ccxt.okx,
        'kucoin': ccxt.kucoin,
        'gateio': ccxt.gateio,
        'binance': ccxt.binance,
        'bybit': ccxt.bybit
    }
    
    connected = []
    
    for name, exchange_class in exchanges_config.items():
        key_name = name.upper() if name != 'gateio' else 'GATE'
        api_key = os.getenv(f'{key_name}_API_KEY')
        secret = os.getenv(f'{key_name}_SECRET')
        
        if not api_key:
            print(f'⚠️  {name.upper()}: No API key')
            continue
        
        try:
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True
            })
            
            balance = await exchange.fetch_balance()
            total_usd = balance.get('total', {}).get('USDT', 0) or balance.get('total', {}).get('USD', 0)
            
            if total_usd > 0:
                print(f'✅ {name.upper()}: Connected! Balance: ${total_usd:.2f} USDT')
            else:
                print(f'✅ {name.upper()}: Connected! (Balance: $0 or non-USDT)')
            
            connected.append(name)
            
            await exchange.close()
            
        except Exception as e:
            error_msg = str(e)[:80]
            print(f'❌ {name.upper()}: {error_msg}')
    
    print(f'\n📊 Result: {len(connected)}/7 exchanges connected')
    
    if len(connected) >= 2:
        print('\n✅ Arbitrage will work with', len(connected), 'exchanges!')
    else:
        print('\n⚠️  Need at least 2 exchanges for arbitrage')

asyncio.run(test_all())
PYTHON

echo ""
echo "Restarting bot with all exchanges..."
sudo systemctl restart trading-bot

echo ""
echo "Waiting 30 seconds..."
sleep 30

echo ""
echo "Checking logs for arbitrage..."
journalctl -u trading-bot --since "30 seconds ago" | grep -i "arbitrage\|exchange.*connected" | tail -10

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DONE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Your bot now has access to all 7 exchanges!"
echo ""
echo "Next step: Set up Telegram channels"
echo "Run: bash FIX_TELEGRAM_CHANNELS_NOW.sh"
echo ""
