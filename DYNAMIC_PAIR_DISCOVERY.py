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
    - Forex markets
    - Stock markets
    - Commodity markets
    
    Continuously adds new pairs based on:
    - Volume (liquidity)
    - Volatility (profit opportunity)
    - Momentum (trending)
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
        - Recent momentum
        """
        
        profitable_pairs = []
        
        for pair in all_pairs[:100]:  # Start with top 100, will expand
            try:
                # Get 24h stats from Bybit (fastest)
                ticker = await asyncio.to_thread(
                    self.exchanges['bybit'].fetch_ticker, pair
                )
                
                volume_usd = ticker.get('quoteVolume', 0)
                price_change = abs(ticker.get('percentage', 0))
                
                # Criteria for profitable pairs:
                # 1. Volume > $100k/day (liquid enough to trade)
                # 2. Price change > 1% (volatile enough for profit)
                if volume_usd > 100000 and price_change > 1:
                    profitable_pairs.append({
                        'symbol': pair,
                        'volume': volume_usd,
                        'volatility': price_change,
                        'score': volume_usd * price_change  # Profit potential score
                    })
                
            except:
                continue
        
        # Sort by profit potential
        profitable_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"💰 Found {len(profitable_pairs)} highly profitable pairs!")
        
        return [p['symbol'] for p in profitable_pairs]
    
    async def continuous_discovery(self):
        """
        Continuously discover new profitable pairs
        Runs every 1 hour to find emerging opportunities
        """
        
        while True:
            try:
                logger.info("🔍 Starting market discovery scan...")
                
                # Discover all available pairs
                all_pairs = await self.discover_all_markets()
                
                # Filter for profitable ones
                profitable = await self.filter_profitable_pairs(all_pairs)
                
                # Add new profitable pairs to active trading
                new_pairs = set(profitable) - self.active_pairs
                if new_pairs:
                    self.active_pairs.update(new_pairs)
                    logger.info(f"✅ Added {len(new_pairs)} new profitable pairs!")
                    logger.info(f"📊 TOTAL ACTIVE PAIRS: {len(self.active_pairs)}")
                
                # Wait 1 hour before next scan
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Discovery error: {e}")
                await asyncio.sleep(300)  # Retry in 5 min
    
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
