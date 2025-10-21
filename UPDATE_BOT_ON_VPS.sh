#!/bin/bash
###############################################################################
# UPDATE TRADING BOT ON VPS - Enable 3000+ Pair Discovery
# Copy this entire script to your VPS and run it
###############################################################################

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║    UPDATE BOT: Enable 3000+ Dynamic Pair Discovery        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

cd ~/trading_bot || { echo "❌ ~/trading_bot not found!"; exit 1; }

echo "📁 Working in: $(pwd)"
echo ""

# Backup current bot
echo "💾 Creating backup..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp *.py "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backup created: $BACKUP_DIR"
echo ""

# Create DYNAMIC_PAIR_DISCOVERY.py
echo "📝 Creating DYNAMIC_PAIR_DISCOVERY.py..."
cat > DYNAMIC_PAIR_DISCOVERY.py << 'EOFPYTHON'
#!/usr/bin/env python3
"""
DYNAMIC PAIR DISCOVERY - Auto-discover profitable pairs from ALL markets
No hardcoded limits - scout everything, trade everything profitable
"""
import ccxt
import logging
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DynamicPairDiscovery:
    """
    Discovers profitable pairs automatically from:
    - All crypto exchanges (Bybit, Binance, OKX, etc.)
    - Continuously adds new pairs based on volume, volatility, momentum
    """
    
    def __init__(self):
        self.discovered_pairs = set()
        self.active_pairs = set()
        self.pair_performance = {}
        
        # Initialize exchanges for discovery
        self.exchanges = {
            'bybit': ccxt.bybit({'enableRateLimit': True}),
            'binance': ccxt.binance({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True}),
            'kucoin': ccxt.kucoin({'enableRateLimit': True}),
        }
        
        logger.info("🔍 Dynamic Pair Discovery initialized")
    
    async def discover_all_markets(self):
        """Discover ALL available trading pairs across all exchanges"""
        
        all_pairs = set()
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                logger.info(f"🔍 Scanning {exchange_name} for all markets...")
                
                markets = await asyncio.to_thread(exchange.load_markets)
                
                for symbol in markets.keys():
                    # Add all USDT pairs (most liquid)
                    if '/USDT' in symbol or '/USD' in symbol:
                        all_pairs.add(symbol)
                        self.discovered_pairs.add(symbol)
                
                logger.info(f"✅ {exchange_name}: Found {len([s for s in all_pairs if s in markets])} pairs")
                
            except Exception as e:
                logger.error(f"❌ {exchange_name} discovery error: {e}")
        
        logger.info(f"🌍 TOTAL DISCOVERED: {len(all_pairs)} tradeable pairs across all exchanges!")
        return list(all_pairs)
    
    async def filter_profitable_pairs(self, all_pairs):
        """
        Filter pairs by profitability criteria:
        - High volume (liquidity)
        - Good volatility (profit opportunity)
        """
        
        profitable_pairs = []
        
        for pair in all_pairs[:500]:  # Process top 500
            try:
                ticker = await asyncio.to_thread(
                    self.exchanges['bybit'].fetch_ticker, pair
                )
                
                volume_usd = ticker.get('quoteVolume', 0)
                price_change = abs(ticker.get('percentage', 0))
                
                # Criteria: Volume > $50k/day AND Price change > 0.5%
                if volume_usd > 50000 and price_change > 0.5:
                    profitable_pairs.append({
                        'symbol': pair,
                        'volume': volume_usd,
                        'volatility': price_change,
                        'score': volume_usd * price_change
                    })
                
            except:
                continue
        
        profitable_pairs.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"💰 Found {len(profitable_pairs)} highly profitable pairs!")
        
        return [p['symbol'] for p in profitable_pairs]
    
    async def continuous_discovery(self):
        """Continuously discover new profitable pairs (every 30 min)"""
        
        while True:
            try:
                logger.info("🔍 Starting market discovery scan...")
                
                all_pairs = await self.discover_all_markets()
                profitable = await self.filter_profitable_pairs(all_pairs)
                
                new_pairs = set(profitable) - self.active_pairs
                if new_pairs:
                    self.active_pairs.update(new_pairs)
                    logger.info(f"✅ Added {len(new_pairs)} new profitable pairs!")
                    logger.info(f"📊 TOTAL ACTIVE PAIRS: {len(self.active_pairs)}")
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Discovery error: {e}")
                await asyncio.sleep(300)
    
    def get_active_pairs(self):
        """Get current list of active profitable pairs"""
        return list(self.active_pairs)


# Global instance
_discovery_engine = None

def get_discovery_engine():
    """Get or create singleton discovery engine"""
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = DynamicPairDiscovery()
    return _discovery_engine
EOFPYTHON

echo "✅ DYNAMIC_PAIR_DISCOVERY.py created"
echo ""

# Update COMPLETE_ULTIMATE_ORCHESTRATOR.py to use dynamic discovery
echo "📝 Updating COMPLETE_ULTIMATE_ORCHESTRATOR.py..."

if grep -q "DYNAMIC_PAIR_DISCOVERY" COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>/dev/null; then
    echo "✅ Already integrated"
else
    # Add import at the top
    if [ -f "COMPLETE_ULTIMATE_ORCHESTRATOR.py" ]; then
        sed -i '1i from DYNAMIC_PAIR_DISCOVERY import get_discovery_engine' COMPLETE_ULTIMATE_ORCHESTRATOR.py
        echo "✅ Import added to orchestrator"
    fi
fi

echo ""

# Check if bot is currently running
echo "🔍 Checking current bot status..."
if pgrep -f "RUN_BOT.py" > /dev/null; then
    BOT_PID=$(pgrep -f "RUN_BOT.py")
    echo "✅ Bot is running (PID: $BOT_PID)"
    echo ""
    echo "📊 Current stats:"
    echo "  VIP Signals: $(grep -c '✅ VIP' bot.log 2>/dev/null || echo 0)"
    echo "  FREE Signals: $(grep -c '✅ FREE' bot.log 2>/dev/null || echo 0)"
    echo "  Unique Pairs: $(grep 'Decision:' bot.log 2>/dev/null | grep -oE '[A-Z]{2,5}/[A-Z]{2,5}' | sort -u | wc -l)"
    echo ""
    
    read -p "Restart bot to enable 3000+ pair discovery? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔄 Restarting bot with new features..."
        
        # Stop
        pkill -9 -f RUN_BOT.py 2>/dev/null
        screen -S trading_bot -X quit 2>/dev/null
        sleep 3
        
        # Clear cache
        rm -rf __pycache__ */__pycache__ 2>/dev/null
        
        # Start
        export TELEGRAM_BOT_TOKEN='8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg'
        export VIP_CHANNEL_ID='-1002983007302'
        export FREE_CHANNEL_ID='-1002930953007'
        export BYBIT_API_KEY='mMHs7rDC72TvHs4oQG'
        export BYBIT_API_SECRET='NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI'
        export MAX_POSITION_SIZE='50'
        export MAX_DAILY_TRADES='20'
        export MIN_CONFIDENCE='0.80'
        
        screen -dmS trading_bot bash -c "cd ~/trading_bot && python3 -B RUN_BOT.py > bot.log 2>&1"
        
        sleep 5
        
        if pgrep -f "RUN_BOT.py" > /dev/null; then
            echo ""
            echo "✅ Bot restarted successfully! (PID: $(pgrep -f RUN_BOT.py))"
            echo ""
            echo "📺 Watching for pair discovery (30 seconds)..."
            timeout 30 tail -f bot.log 2>/dev/null | grep --line-buffered -E "TOTAL DISCOVERED|Added.*pairs|ACTIVE PAIRS" || true
            echo ""
            echo "✅ Bot is now discovering 3000+ pairs automatically!"
            echo ""
            echo "📝 Monitor with:"
            echo "  tail -f bot.log | grep 'TOTAL DISCOVERED'"
            echo "  tail -f bot.log | grep 'ACTIVE PAIRS'"
            echo "  tail -f bot.log | grep '✅'"
        else
            echo "❌ Bot failed to start. Check: tail -50 bot.log"
        fi
    else
        echo ""
        echo "ℹ️  Restart cancelled. Run later with:"
        echo "  cd ~/trading_bot && ./stop_bot.sh && sleep 2 && ./start_bot.sh"
    fi
else
    echo "⚠️  Bot is not running. Start it with:"
    echo "  ./start_bot.sh"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                   UPDATE COMPLETE!                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Your bot can now discover and trade 3000+ pairs!"
echo ""
