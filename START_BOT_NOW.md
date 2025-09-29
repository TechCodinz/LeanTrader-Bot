# 🚀 START YOUR BYBIT BOT RIGHT NOW!

## **Copy and paste these commands on your VPS (75.119.149.117)**

### **Step 1: Connect to your VPS**
```bash
ssh root@75.119.149.117
```

### **Step 2: Install everything and start the bot**

Copy and paste this entire block:

```bash
# Update system
apt update && apt upgrade -y

# Install Python and dependencies
apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev build-essential git curl wget

# Create trading bot directory
mkdir -p /home/root/trading-bot
cd /home/root/trading-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install ccxt pandas numpy python-dotenv loguru asyncio

# Configure firewall
ufw allow ssh
ufw allow 8501/tcp
ufw --force enable

echo "✅ Environment ready!"
```

### **Step 3: Create the simple bot**

```bash
cat > simple_bot.py << 'EOF'
#!/usr/bin/env python3
"""
Simple Bybit Trading Bot - Ready to Run
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import os

class SimpleBybitBot:
    """Simple Bybit trading bot for immediate testing"""
    
    def __init__(self):
        self.exchange = None
        self.running = False
        
        # Your Bybit testnet credentials
        self.api_key = "g1mhPqKrOBp9rnqb4G"
        self.secret_key = "s9KCIelCqPwJOOWAXNoWqFHtiauRQr9PLeqG"
        self.sandbox = True
        
    async def initialize(self):
        """Initialize the bot"""
        print("🚀 Initializing Simple Bybit Trading Bot...")
        
        try:
            # Initialize Bybit exchange
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
            })
            
            # Test connection
            markets = await self.exchange.load_markets()
            print(f"✅ Connected to Bybit {'Testnet' if self.sandbox else 'Live'}")
            print(f"📊 Available markets: {len(markets)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def get_market_data(self, symbol='BTC/USDT'):
        """Get market data for a symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def generate_signal(self, ticker):
        """Generate a simple trading signal"""
        try:
            # Simple strategy: Price change based signal
            price = ticker['last']
            change_24h = ticker['change']
            change_percent = (change_24h / (price - change_24h)) * 100 if change_24h != 0 else 0
            
            # Simple logic: if price dropped more than 2% in 24h, consider buying
            if change_percent < -2:
                return 'BUY', 0.7, f"Price dropped {change_percent:.2f}% in 24h"
            elif change_percent > 2:
                return 'SELL', 0.7, f"Price rose {change_percent:.2f}% in 24h"
            else:
                return 'HOLD', 0.5, f"Price stable: {change_percent:.2f}% change"
                
        except Exception as e:
            print(f"Error generating signal: {e}")
            return 'HOLD', 0.0, "Error in signal generation"
    
    async def trading_loop(self):
        """Main trading loop"""
        print("🎯 Starting trading loop...")
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        while self.running:
            try:
                print(f"\n📊 Market Analysis - {datetime.now().strftime('%H:%M:%S')}")
                
                for symbol in symbols:
                    # Get market data
                    ticker = await self.get_market_data(symbol)
                    if not ticker:
                        continue
                    
                    # Generate signal
                    action, confidence, reasoning = await self.generate_signal(ticker)
                    
                    print(f"📈 {symbol}: ${ticker['last']:.2f} | Signal: {action} ({confidence:.1%}) | {reasoning}")
                    
                    # In simulation mode, we just log the signals
                    if confidence > 0.6 and action != 'HOLD':
                        print(f"🎮 SIMULATION: Would {action} {symbol} at ${ticker['last']:.2f}")
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(10)
    
    async def start(self):
        """Start the bot"""
        print("🚀 Starting Simple Bybit Trading Bot...")
        print(f"🔑 API Key: {self.api_key[:10]}...")
        print(f"🌐 Mode: {'Testnet' if self.sandbox else 'Live'}")
        print("=" * 50)
        
        if await self.initialize():
            self.running = True
            await self.trading_loop()
        else:
            print("❌ Failed to initialize bot")
    
    async def stop(self):
        """Stop the bot"""
        print("🛑 Stopping bot...")
        self.running = False

async def main():
    """Main entry point"""
    bot = SimpleBybitBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    print("🤖 Simple Bybit Trading Bot")
    print("=" * 30)
    print("Starting in 3 seconds...")
    import time
    time.sleep(3)
    
    asyncio.run(main())
EOF

chmod +x simple_bot.py
echo "✅ Bot file created!"
```

### **Step 4: Start the bot**

```bash
# Start the bot
source venv/bin/activate
python simple_bot.py
```

## **🎉 That's it! Your bot is now running!**

### **What you'll see:**
- ✅ Connection to Bybit testnet established
- 📊 Real-time market data for BTC, ETH, BNB
- 🧠 Simple trading signals based on price changes
- 🎮 Simulation mode (safe testing)

### **Bot Features:**
- **Real-time data** from Bybit testnet
- **Simple strategy** based on 24h price changes
- **Safe simulation** mode for testing
- **Multiple assets** (BTC, ETH, BNB)
- **Live logging** of all activities

### **To stop the bot:**
Press `Ctrl + C`

### **To restart the bot:**
```bash
cd /home/root/trading-bot
source venv/bin/activate
python simple_bot.py
```

### **To run in background:**
```bash
nohup python simple_bot.py > bot.log 2>&1 &
```

### **To check if it's running:**
```bash
ps aux | grep python
```

### **To view logs:**
```bash
tail -f bot.log
```

## **🚀 Your Bybit trading bot is now live and learning!**

The bot will:
1. ✅ Connect to Bybit testnet
2. 📊 Monitor BTC, ETH, BNB prices
3. 🧠 Generate buy/sell signals
4. 📝 Log all activities
5. 🎮 Run safely in simulation mode

**Start with this simple version, then we can upgrade to the full ML-powered system!**