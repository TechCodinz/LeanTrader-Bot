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
from ccxt_exchange_compat import resolve_exchange_class

import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Set
from datetime import datetime, timedelta
from collections import deque
import logging

from src.leantrader.universe.registry import configured_quotes, universe
from src.leantrader.universe import venues as venue_capabilities
from src.leantrader.universe.venues import capabilities

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

        # Scanning config - RELAXED for more opportunities
        self.min_24h_volume_usd = 1_000_000  # $1M minimum (was $5M)
        self.min_price_change_pct = 1.5  # 1.5% move (was 3%)
        self.rescan_interval = 3600  # 1 hour

        # Stats
        self.last_scan = None
        self.scan_count = 0

        # Public observer clients, built once and reused. venue_status records
        # why a venue is not in that set, so an unreachable venue is reported
        # rather than silently missing.
        self._public_exchanges = {}
        self.venue_status = {}

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

        # Whatever the arbitrage engine injected, plus the public observers.
        # These used to be alternatives -- the public set was built only when
        # the injected objects did not look like ccxt clients -- so discovery
        # breadth depended on which engine happened to be wired first.
        actual_exchanges = {}

        for name, candidate in (self.exchanges or {}).items():
            if hasattr(candidate, 'fetch_tickers'):
                actual_exchanges[name] = candidate

        for name, client in (await self.public_exchanges()).items():
            actual_exchanges.setdefault(name, client)

        if not actual_exchanges:
            logger.warning("⚠️  No reachable venues for discovery this scan")
            return

        for exchange_name, exchange in actual_exchanges.items():
            try:
                logger.info(f"🔍 Scanning {exchange_name}...")

                # Get all tickers (check if method exists)
                if not hasattr(exchange, 'fetch_tickers'):
                    logger.warning(f"   {exchange_name} doesn't have fetch_tickers, skipping")
                    continue

                tickers = await exchange.fetch_tickers()

                # Everything this venue lists goes into the canonical registry,
                # not just what clears the volume threshold below. The
                # threshold decides what is worth trading now; the registry is
                # what LeanTrader studies, and narrowing it here is how the
                # engines ended up back on a handful of majors.
                try:
                    markets = getattr(exchange, 'markets', None) or {}
                    recorded = universe.ingest_venue(
                        exchange_name,
                        tickers,
                        markets=markets,
                        quote_filter=configured_quotes(),
                    )

                    # The capability matrix records what this venue lists, in
                    # its own notation, with its own precision and limits --
                    # separately from the intelligence universe above. That
                    # is what lets the router know a market is absent here
                    # before calling and finding out the expensive way.
                    capability_count = capabilities.record_venue_markets(
                        exchange_name,
                        markets,
                        environment='live',
                        exchange_has=getattr(exchange, 'has', {}) or {},
                    )

                    logger.info(
                        f"   Registry: {recorded} markets recorded from "
                        f"{exchange_name} ({capability_count} capabilities)"
                    )
                except Exception as e:
                    logger.warning(
                        f"   Registry ingest failed for {exchange_name}: "
                        f"{type(e).__name__}: {e}"
                    )

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

        # Persist from the process that actually discovered. This used to
        # happen only in the orchestrator's maintenance loop, so a scan could
        # succeed while nothing was ever written and another process saw an
        # empty universe.
        try:
            written = venue_capabilities.persist_discovery_state(
                {"discovered_active_pairs": new_count}
            )
            logger.info(
                "🌍 Canonical snapshot %s: %s",
                "written" if written else "NOT written",
                venue_capabilities.snapshot_path(),
            )
        except Exception as e:
            logger.warning(
                f"Canonical snapshot write failed: {type(e).__name__}: {e}"
            )

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

    def _extract_ccxt_exchanges(self) -> Dict:
        """Extract actual ccxt exchange objects from engines"""
        ccxt_exchanges = {}

        for name, obj in self.exchanges.items():
            try:
                # Check if it's already a ccxt exchange
                if hasattr(obj, 'fetch_tickers') and callable(obj.fetch_tickers):
                    ccxt_exchanges[name] = obj
                # Check if it has an 'exchange' attribute (engine with embedded exchange)
                elif hasattr(obj, 'exchange') and obj.exchange:
                    if hasattr(obj.exchange, 'fetch_tickers'):
                        ccxt_exchanges[name] = obj.exchange
                        logger.info(f"   ✅ Extracted exchange from {name} engine")
            except Exception as e:
                logger.debug(f"   Skipping {name}: {str(e)[:50]}")
                continue

        return ccxt_exchanges

    # Venues to observe for public market data. Ticker and market metadata
    # need no credentials anywhere here, so discovery breadth is not gated on
    # holding keys -- it used to be, which left public discovery on Binance
    # alone unless an operator happened to have keys for the others.
    #
    # Observing a venue is not permission to trade on it. Execution stays with
    # whichever venue the universal router is authenticated against, and the
    # registry marks everything else NOT_LISTED_ON_EXECUTION_VENUE.
    PUBLIC_DISCOVERY_VENUES = (
        'bybit',
        'binance',
        'okx',
        'kucoin',
        'gateio',
        'mexc',
        'bitget',
    )

    def _configured_public_venues(self):
        import os

        raw = os.getenv('PUBLIC_DISCOVERY_VENUES', '').strip()
        if not raw:
            return self.PUBLIC_DISCOVERY_VENUES
        venues = tuple(v.strip().lower() for v in raw.split(',') if v.strip())
        return venues or self.PUBLIC_DISCOVERY_VENUES

    async def public_exchanges(self) -> Dict:
        """Unauthenticated clients for every venue we can actually reach.

        Built once and reused: a fresh ccxt.async_support client per scan
        leaks an aiohttp session, and these are long-lived observers.

        A venue that cannot be constructed or whose markets will not load is
        recorded in self.venue_status with the reason and left out. Nothing is
        assumed to be available.
        """
        import ccxt.async_support as ccxt

        if getattr(self, '_public_exchanges', None):
            return self._public_exchanges

        self._public_exchanges = {}
        self.venue_status = getattr(self, 'venue_status', {})

        for venue in self._configured_public_venues():
            try:
                exchange_class = resolve_exchange_class(ccxt, venue)
            except Exception as e:
                self.venue_status[venue] = f'unavailable: {type(e).__name__}'
                logger.warning(f"   ⚠️  {venue}: not provided by this ccxt build")
                continue

            try:
                client = exchange_class({'enableRateLimit': True})
                await client.load_markets()
            except Exception as e:
                self.venue_status[venue] = f'unreachable: {type(e).__name__}'
                logger.warning(
                    f"   ⚠️  {venue}: {type(e).__name__}: {str(e)[:80]}"
                )
                try:
                    await client.close()
                except Exception:
                    pass
                continue

            self._public_exchanges[venue] = client
            self.venue_status[venue] = f'public: {len(client.markets)} markets'
            logger.info(
                f"   ✅ {venue}: {len(client.markets)} markets (public)"
            )

        reachable = len(self._public_exchanges)
        total = len(self._configured_public_venues())
        logger.info(f"🔭 Public discovery: {reachable}/{total} venues reachable")

        return self._public_exchanges

    async def close_public_exchanges(self) -> None:
        """Release the observer clients."""
        for client in (getattr(self, '_public_exchanges', None) or {}).values():
            try:
                await client.close()
            except Exception:
                pass
        self._public_exchanges = {}

    async def _create_fallback_exchanges(self) -> Dict:
        """Kept for callers that expect this name; public clients now."""
        return await self.public_exchanges()

    def get_stats(self) -> Dict:
        """Get scanner statistics"""
        return {
            'active_pairs': len(self.active_pairs),
            'trending_count': len(self.trending_pairs),
            'volume_leaders': len(self.volume_leaders),
            'scan_count': self.scan_count,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'exchanges_scanned': len(self.exchanges),
            'venue_status': dict(getattr(self, 'venue_status', {})),
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
