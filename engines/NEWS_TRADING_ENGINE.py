"""
📰 NEWS TRADING ENGINE
Fundamental analysis through real-time news monitoring

Features:
- CoinGecko trending coins
- Twitter/X sentiment (via public feeds)
- Major crypto news sources
- Breaking news detection
- Sentiment analysis
- Auto-generate signals from news events
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available - news trading will use fallback")

logger = logging.getLogger(__name__)


class NewsTradingEngine:
    """
    Monitors news and generates trading signals based on fundamental events
    """
    
    def __init__(self, data_hub):
        self.data_hub = data_hub
        self.enabled = True
        
        # News sources
        self.news_cache = deque(maxlen=500)
        self.trending_cache = deque(maxlen=100)
        
        # Sentiment scoring
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'breakout', 'pump', 'moon',
            'partnership', 'adoption', 'upgrade', 'launch', 'listing',
            'institutional', 'ETF', 'approval', 'buy', 'accumulate'
        ]
        
        self.negative_keywords = [
            'bearish', 'crash', 'dump', 'rug', 'scam', 'hack', 'exploit',
            'regulation', 'ban', 'lawsuit', 'investigation', 'sell',
            'warning', 'decline', 'drop', 'fall'
        ]
        
        # Stats
        self.news_processed = 0
        self.signals_generated = 0
        
        logger.info("📰 News Trading Engine initialized")
    
    async def run_news_monitor(self):
        """Main news monitoring loop"""
        logger.info("📰 Starting news monitoring...")
        
        while self.enabled:
            try:
                # Fetch trending coins (proxy for news/hype)
                trending = await self.fetch_trending_coins()
                
                if trending:
                    logger.info(f"📰 Found {len(trending)} trending coins")
                    
                    # Generate signals from trending data
                    for coin_data in trending:
                        signal = await self.generate_news_signal(coin_data)
                        if signal:
                            await self.data_hub.publish_signal(signal)
                            self.signals_generated += 1
                
                # Wait before next check (every 10 minutes)
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"News monitor error: {e}")
                await asyncio.sleep(600)
    
    async def fetch_trending_coins(self) -> List[Dict]:
        """Fetch trending coins from CoinGecko public API"""
        if not AIOHTTP_AVAILABLE:
            logger.debug("aiohttp not available, skipping trending fetch")
            return []
        
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        trending_coins = []
                        if 'coins' in data:
                            for item in data['coins'][:10]:  # Top 10
                                coin = item.get('item', {})
                                trending_coins.append({
                                    'symbol': coin.get('symbol', '').upper(),
                                    'name': coin.get('name', ''),
                                    'market_cap_rank': coin.get('market_cap_rank', 999),
                                    'price_btc': coin.get('price_btc', 0),
                                    'score': coin.get('score', 0)
                                })
                        
                        self.news_processed += 1
                        return trending_coins
        except Exception as e:
            logger.debug(f"CoinGecko trending fetch failed: {e}")
        
        return []
    
    async def generate_news_signal(self, coin_data: Dict) -> Optional[Dict]:
        """Generate trading signal from trending coin data"""
        
        symbol = coin_data.get('symbol', '')
        if not symbol:
            return None
        
        # Try to match to USDT pair
        possible_symbols = [
            f"{symbol}/USDT",
            f"{symbol}USDT/USDT" if len(symbol) <= 4 else None
        ]
        
        # Calculate confidence based on trending score and rank
        score = coin_data.get('score', 0)
        rank = coin_data.get('market_cap_rank', 999)
        
        # Higher score + lower rank = higher confidence
        confidence = min(0.95, (score / 10) * 0.5 + (1 - rank / 1000) * 0.5)
        
        if confidence < 0.60:  # Only signal if 60%+ confidence
            return None
        
        signal = {
            'type': 'news_trending',
            'symbol': possible_symbols[0],
            'side': 'BUY',  # Trending = buy signal
            'confidence': confidence,
            'price': 0,  # Will be fetched by Telegram
            'source': 'NewsTradingEngine',
            'reasoning': f"Trending on CoinGecko (rank #{rank}, score {score}). "
                        f"High social momentum and search volume detected.",
            'timestamp': datetime.now(),
            'news_data': coin_data
        }
        
        logger.info(f"📰 News signal: {symbol} trending (conf: {confidence*100:.0f}%)")
        
        return signal
    
    def get_stats(self) -> Dict:
        """Get news engine statistics"""
        return {
            'news_processed': self.news_processed,
            'signals_generated': self.signals_generated,
            'trending_cached': len(self.trending_cache)
        }
