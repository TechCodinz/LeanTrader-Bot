"""
🔍 DYNAMIC MARKET SCANNER
Automatically discovers trending pairs and expands trading universe in real-time

Features:
- Scans exchanges for high-volume pairs
- Detects trending coins (24h volume spike, price movement)
- Auto-adds new profitable opportunities
- Removes dead/low-volume pairs
- Updates universe every hour
"""

import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Set
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DynamicMarketScanner:
    """
    Continuously scans exchanges for trending pairs
    Expands trading universe beyond static 5 coins
    """
    
    def __init__(self, exchanges: Dict[str, ccxt.Exchange], data_hub):
        self.exchanges = exchanges
        self.data_hub = data_hub
        
        # Dynamic universe
        self.active_pairs = set()
        self.trending_pairs = deque(maxlen=50)  # Top 50 trending
        self.volume_leaders = deque(maxlen=30)  # Top 30 by volume
        
        # Scanning config
        self.min_24h_volume_usd = 5_000_000  # $5M minimum
        self.min_price_change_pct = 3.0  # 3% move
        self.rescan_interval = 3600  # 1 hour
        
        # Stats
        self.last_scan = None
        self.scan_count = 0
        
        logger.info("🔍 Dynamic Market Scanner initialized")
        logger.info(f"   Min volume: ${self.min_24h_volume_usd:,}")
        logger.info(f"   Min price change: {self.min_price_change_pct}%")
    
    async def run_continuous_scanning(self):
        """Main scanning loop - runs forever"""
        logger.info("🔍 Starting continuous market scanning...")
        
        while True:
            try:
                await self.scan_all_exchanges()
                
                self.scan_count += 1
                self.last_scan = datetime.now()
                
                logger.info(f"🔍 Scan #{self.scan_count} complete")
                logger.info(f"   Active pairs: {len(self.active_pairs)}")
                logger.info(f"   Trending: {len(self.trending_pairs)}")
                logger.info(f"   Volume leaders: {len(self.volume_leaders)}")
                
                # Wait before next scan
                await asyncio.sleep(self.rescan_interval)
                
            except Exception as e:
                logger.error(f"Dynamic scanner error: {e}")
                await asyncio.sleep(300)  # Wait 5 min on error
    
    async def scan_all_exchanges(self):
        """Scan all connected exchanges for opportunities"""
        
        if not self.exchanges:
            logger.warning("⚠️  No exchanges connected for dynamic scanning")
            return
        
        new_opportunities = set()
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                logger.info(f"🔍 Scanning {exchange_name}...")
                
                # Get all tickers
                tickers = await exchange.fetch_tickers()
                
                # Filter for USDT pairs only
                usdt_pairs = {
                    symbol: ticker for symbol, ticker in tickers.items()
                    if '/USDT' in symbol and ':' not in symbol  # Exclude futures
                }
                
                logger.info(f"   Found {len(usdt_pairs)} USDT pairs on {exchange_name}")
                
                # Analyze each pair
                for symbol, ticker in usdt_pairs.items():
                    try:
                        # Get volume and price change
                        volume_24h = ticker.get('quoteVolume', 0)  # Volume in USDT
                        price_change_pct = ticker.get('percentage', 0)
                        
                        # Check if it meets criteria
                        if volume_24h >= self.min_24h_volume_usd:
                            # High volume pair
                            new_opportunities.add(symbol)
                            
                            # Track as volume leader
                            if len(self.volume_leaders) < 30 or volume_24h > min([
                                t[1] for t in self.volume_leaders
                            ], default=0):
                                self.volume_leaders.append((symbol, volume_24h, exchange_name))
                            
                            # Check if trending (big price move)
                            if abs(price_change_pct) >= self.min_price_change_pct:
                                self.trending_pairs.append({
                                    'symbol': symbol,
                                    'exchange': exchange_name,
                                    'volume_24h': volume_24h,
                                    'price_change_pct': price_change_pct,
                                    'discovered_at': datetime.now()
                                })
                                
                                logger.info(
                                    f"   🔥 TRENDING: {symbol} "
                                    f"{price_change_pct:+.1f}% "
                                    f"(${volume_24h/1e6:.1f}M volume)"
                                )
                    
                    except Exception as e:
                        # Skip individual ticker errors
                        continue
                
            except Exception as e:
                logger.error(f"Error scanning {exchange_name}: {e}")
                continue
        
        # Update active pairs
        old_count = len(self.active_pairs)
        self.active_pairs.update(new_opportunities)
        new_count = len(self.active_pairs)
        
        if new_count > old_count:
            logger.info(f"✅ Added {new_count - old_count} new pairs to universe")
            logger.info(f"   Total active pairs: {new_count}")
    
    def get_active_universe(self) -> List[str]:
        """Get current active trading universe"""
        return sorted(list(self.active_pairs))
    
    def get_trending_opportunities(self, limit: int = 20) -> List[Dict]:
        """Get top trending pairs"""
        # Sort by absolute price change
        sorted_trending = sorted(
            self.trending_pairs,
            key=lambda x: abs(x['price_change_pct']),
            reverse=True
        )
        return list(sorted_trending)[:limit]
    
    def get_volume_leaders(self, limit: int = 20) -> List[tuple]:
        """Get pairs with highest volume"""
        sorted_volume = sorted(
            self.volume_leaders,
            key=lambda x: x[1],  # Sort by volume
            reverse=True
        )
        return list(sorted_volume)[:limit]
    
    def get_stats(self) -> Dict:
        """Get scanner statistics"""
        return {
            'active_pairs': len(self.active_pairs),
            'trending_count': len(self.trending_pairs),
            'volume_leaders': len(self.volume_leaders),
            'scan_count': self.scan_count,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'exchanges_scanned': len(self.exchanges)
        }
    
    async def publish_trending_signals(self):
        """Publish trending pairs as signals to data hub"""
        
        top_trending = self.get_trending_opportunities(limit=10)
        
        for trend in top_trending:
            signal = {
                'type': 'trending',
                'symbol': trend['symbol'],
                'side': 'BUY' if trend['price_change_pct'] > 0 else 'SELL',
                'confidence': min(abs(trend['price_change_pct']) / 10, 0.95),  # Cap at 95%
                'source': 'DynamicScanner',
                'reasoning': f"Trending {trend['price_change_pct']:+.1f}% with ${trend['volume_24h']/1e6:.1f}M volume",
                'timestamp': datetime.now()
            }
            
            await self.data_hub.publish_signal(signal)
        
        if top_trending:
            logger.info(f"📢 Published {len(top_trending)} trending signals to data hub")
