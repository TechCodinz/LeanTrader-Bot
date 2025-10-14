#!/usr/bin/env python3
"""
Download Historical OHLCV Data for Training ML Models
Uses ccxt to fetch data from Bybit (FREE!)
"""
import ccxt
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("📊 OHLCV DATA DOWNLOADER")
print("=" * 80)
print()

# Create data directory
data_dir = Path('training_data')
data_dir.mkdir(exist_ok=True)
print(f"✅ Data directory: {data_dir}/")
print()

# Initialize exchange (Bybit - free, no API key needed for public data)
exchange = ccxt.bybit()
print("✅ Connected to Bybit")
print()

# Symbols to download (add more if you want)
symbols = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'SOL/USDT',
    'XRP/USDT',
    'ADA/USDT',
    'DOGE/USDT',
    'MATIC/USDT',
]

# Timeframes to download
timeframes = ['1h', '4h', '1d']

# Start date (how far back to get data)
start_date = '2023-01-01T00:00:00Z'

print(f"📥 Downloading data:")
print(f"   Symbols: {len(symbols)}")
print(f"   Timeframes: {timeframes}")
print(f"   Start date: {start_date}")
print()

total_downloaded = 0

for symbol in symbols:
    for timeframe in timeframes:
        print(f"📊 {symbol} {timeframe}...", end=" ")
        
        try:
            # Get historical data
            all_data = []
            since = exchange.parse8601(start_date)
            
            while since < exchange.milliseconds():
                try:
                    # Fetch 1000 candles at a time (max)
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, 1000)
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    
                    # Rate limit (be nice to Bybit)
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    break
            
            if not all_data:
                print("❌ No data")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add useful columns
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Calculate returns (for ML features)
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Save to CSV
            filename = data_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(filename, index=False)
            
            total_downloaded += len(df)
            print(f"✅ {len(df)} candles → {filename.name}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

print()
print("=" * 80)
print(f"✅ DOWNLOAD COMPLETE!")
print(f"   Total candles: {total_downloaded:,}")
print(f"   Files created: {len(list(data_dir.glob('*.csv')))}")
print(f"   Location: {data_dir.absolute()}")
print("=" * 80)
print()
print("🎓 Your ML models can now train on this data!")
print()
print("Next steps:")
print("  1. Check the training_data/ folder")
print("  2. Run your bot - it will use this data")
print("  3. Models will train automatically!")
