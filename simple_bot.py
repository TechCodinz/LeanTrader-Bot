#!/usr/bin/env python3
"""
Simple Bybit Trading Bot - Ready to Run
This is a minimal working version that will start immediately
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleBybitBot:
    """Simple Bybit trading bot for immediate testing"""
    
    def __init__(self):
        self.exchange = None
        self.running = False
        
        # Configuration
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
            # Simple strategy: RSI-based signal
            price = ticker['last']
            change_24h = ticker['change']
            change_percent = (change_24h / (price - change_24h)) * 100
            
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
    
    async def execute_trade(self, symbol, action, quantity, price):
        """Execute a trade (simulation mode)"""
        try:
            if self.sandbox:
                print(f"🎮 SIMULATION: {action} {quantity} {symbol} at ${price}")
                return {'id': f'sim_{datetime.now().timestamp()}', 'status': 'filled'}
            else:
                if action == 'BUY':
                    order = await self.exchange.create_market_buy_order(symbol, quantity)
                else:
                    order = await self.exchange.create_market_sell_order(symbol, quantity)
                return order
                
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None
    
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
                    
                    # Execute trade if confidence is high enough
                    if confidence > 0.6 and action != 'HOLD':
                        quantity = 0.001  # Small quantity for testing
                        order = await self.execute_trade(symbol, action, quantity, ticker['last'])
                        if order:
                            print(f"✅ Trade executed: {order}")
                
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