#!/usr/bin/env python3
import asyncio
import ccxt.async_support as ccxt
import os
from pathlib import Path

# Load .env explicitly
env_path = Path('/root/trading_bot/.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

async def test_all():
    results = []
    
    # Test MEXC
    print('Testing MEXC...')
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
        print(f'❌ MEXC: {str(e)[:80]}')
    
    # Test Gate.io
    print('\nTesting Gate.io...')
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
        print(f'❌ Gate.io: {str(e)[:80]}')
    
    # Test Binance
    print('\nTesting Binance...')
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
        print(f'❌ Binance: {str(e)[:80]}')
    
    # Test Bybit
    print('\nTesting Bybit...')
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
        print(f'❌ Bybit: {str(e)[:80]}')
    
    print(f'\n{"="*70}')
    print(f'📊 Result: {len(results)}/4 exchanges connected (tested without passwords)')
    
    if len(results) >= 2:
        print(f'\n✅ SUCCESS! Arbitrage will work with {len(results)} exchanges!')
        print(f'   Connected: {", ".join([e.upper() for e in results])}')
        return 0
    else:
        print(f'\n⚠️  Only {len(results)} exchange(s) connected')
        print('   Need at least 2 for arbitrage')
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(test_all())
    exit(exit_code)
