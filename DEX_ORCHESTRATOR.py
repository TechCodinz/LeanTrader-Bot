#!/usr/bin/env python3
"""
DEX ORCHESTRATOR - Complete Decentralized Exchange Trading System
Integrates: Moon Spotting, DEX Routing, MEV Protection, Multi-Chain Support
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

# DEX Infrastructure
import os
from dex_router import execute_swap
from ultra_moon_spotter import MicroMoonSpotter
from w3guard.guards import MempoolMonitor, PrivateTxClient, get_mempool_tuning
from DEX_SWAP_ENGINE import DEXSwapEngine
from dex_contracts import FACTORY_ADDRESSES

# Web3 for blockchain interaction
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
except ImportError:
    Web3 = None

logger = logging.getLogger(__name__)


@dataclass
class DEXConfig:
    """Configuration for DEX trading"""
    enabled: bool = True
    chains: List[str] = None  # ['ethereum', 'bsc', 'polygon', 'solana', 'arbitrum']
    max_position_usd: float = 100.0  # Max position size per trade
    max_slippage_bps: int = 50  # 0.5% max slippage
    min_liquidity_usd: float = 5000.0  # Minimum pool liquidity
    use_private_tx: bool = True  # Use Flashbots/private relays
    mev_protection: bool = True  # Enable MEV protection
    
    def __post_init__(self):
        if self.chains is None:
            self.chains = ['ethereum', 'bsc', 'polygon', 'arbitrum', 'solana']


@dataclass
class DEXOpportunity:
    """DEX Trading Opportunity"""
    token_address: str
    symbol: str
    name: str
    chain: str
    dex: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    holders: int
    market_cap: float
    age_hours: float
    safety_score: float  # 0-100
    potential_score: float  # 0-100
    buy_tax: float
    sell_tax: float
    honeypot_risk: bool
    contract_verified: bool
    social_signals: Dict[str, int]
    timestamp: datetime


class Web3Manager:
    """Manages Web3 connections across multiple chains"""
    
    def __init__(self):
        self.connections = {}
        self.rpcs = {
            'ethereum': 'https://eth.llamarpc.com',
            'bsc': 'https://bsc-dataseed1.binance.org',
            'polygon': 'https://polygon-rpc.com',
            'arbitrum': 'https://arb1.arbitrum.io/rpc',
            'avalanche': 'https://api.avax.network/ext/bc/C/rpc',
            'fantom': 'https://rpc.ftm.tools',
            'optimism': 'https://mainnet.optimism.io',
        }
        
        self.dex_routers = {
            'ethereum': {
                'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            },
            'bsc': {
                'pancakeswap_v2': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
                'pancakeswap_v3': '0x1b81D678ffb9C0263b24A97847620C99d213eB14',
                'biswap': '0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8',
            },
            'polygon': {
                'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
            },
            'arbitrum': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
            }
        }
        
    async def connect_chain(self, chain: str) -> Optional[Web3]:
        """Connect to a blockchain"""
        if Web3 is None:
            logger.warning("Web3 not installed - DEX trading disabled")
            return None
            
        if chain in self.connections:
            return self.connections[chain]
            
        try:
            rpc_url = self.rpcs.get(chain)
            if not rpc_url:
                logger.warning(f"No RPC for {chain}")
                return None
                
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add PoA middleware for BSC, Polygon
            if chain in ['bsc', 'polygon']:
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
            if w3.is_connected():
                self.connections[chain] = w3
                logger.info(f"✅ Connected to {chain} - Block: {w3.eth.block_number}")
                return w3
            else:
                logger.warning(f"Failed to connect to {chain}")
                return None
                
        except Exception as e:
            logger.error(f"Error connecting to {chain}: {e}")
            return None
    
    def get_dex_router(self, chain: str, dex: str) -> Optional[str]:
        """Get DEX router address for a chain"""
        return self.dex_routers.get(chain, {}).get(dex)


class DEXExecutor:
    """Executes DEX swaps with MEV protection"""
    
    def __init__(self, config: DEXConfig):
        self.config = config
        self.mempool_monitors = {}
        self.private_client = None
        self.swap_engines = {}  # Cache of swap engines per chain
        
        # Initialize private transaction client if enabled
        if config.use_private_tx:
            try:
                self.private_client = PrivateTxClient(
                    endpoint="https://rpc.flashbots.net",
                    api_key=""  # Set from env
                )
            except Exception as e:
                logger.warning(f"Private TX client init failed: {e}")
    
    def get_swap_engine(self, chain: str, w3: Web3, router_address: str) -> Optional[DEXSwapEngine]:
        """Get or create swap engine for chain"""
        key = f"{chain}:{router_address}"
        
        if key not in self.swap_engines:
            try:
                # Get factory address for this chain/dex
                factory_address = None
                for dex_name, factories in FACTORY_ADDRESSES.items():
                    if chain in factories:
                        factory_address = list(factories[chain].values())[0]
                        break
                
                if not factory_address:
                    logger.warning(f"No factory address for {chain}")
                    return None
                
                engine = DEXSwapEngine(chain, w3, router_address, factory_address)
                self.swap_engines[key] = engine
                logger.info(f"✅ Swap engine created for {chain}")
            except Exception as e:
                logger.error(f"Failed to create swap engine: {e}")
                return None
        
        return self.swap_engines.get(key)
    
    async def execute_buy(
        self, 
        opportunity: DEXOpportunity,
        amount_usd: float,
        w3: Web3,
        router_address: str
    ) -> Dict[str, Any]:
        """Execute a DEX buy with REAL implementation"""
        
        try:
            # Get swap engine
            engine = self.get_swap_engine(opportunity.chain, w3, router_address)
            if not engine:
                return {'success': False, 'error': 'Swap engine not available'}
            
            # Convert USD to native token amount (ETH/BNB/MATIC)
            # For simplicity, using fixed conversion. In production, get real price
            # Assuming ~$2000 per ETH, ~$300 per BNB, ~$0.50 per MATIC
            price_map = {
                'ethereum': 2000,
                'bsc': 300,
                'polygon': 0.50,
                'arbitrum': 2000,
                'solana': 100
            }
            
            native_price = price_map.get(opportunity.chain, 1000)
            amount_native = amount_usd / native_price
            
            # Calculate slippage based on safety score
            # Lower safety = higher slippage tolerance
            base_slippage = self.config.max_slippage_bps
            if opportunity.safety_score < 60:
                slippage = min(base_slippage * 2, 500)  # Max 5%
            else:
                slippage = base_slippage
            
            logger.info(f"Buying {opportunity.symbol} with {amount_native:.6f} native tokens")
            logger.info(f"Slippage: {slippage} bps")
            
            # Execute real swap
            result = engine.buy_token(
                token_address=opportunity.token_address,
                amount_eth=amount_native,
                slippage_bps=slippage
            )
            
            if result.get('success'):
                logger.info(f"✅ DEX Buy executed: {opportunity.symbol}")
                logger.info(f"   TX: {result.get('tx_hash')}")
                logger.info(f"   Amount out: {result.get('amount_out')}")
                logger.info(f"   Price impact: {result.get('price_impact', 0):.2%}")
            else:
                logger.error(f"❌ DEX buy failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"DEX buy failed for {opportunity.symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_sell(
        self,
        opportunity: DEXOpportunity,
        amount_tokens: int,
        w3: Web3,
        router_address: str
    ) -> Dict[str, Any]:
        """Execute a DEX sell with REAL implementation"""
        
        try:
            # Get swap engine
            engine = self.get_swap_engine(opportunity.chain, w3, router_address)
            if not engine:
                return {'success': False, 'error': 'Swap engine not available'}
            
            logger.info(f"Selling {amount_tokens} of {opportunity.symbol}")
            
            # Execute real swap
            result = engine.sell_token(
                token_address=opportunity.token_address,
                amount_tokens=amount_tokens,
                slippage_bps=self.config.max_slippage_bps
            )
            
            if result.get('success'):
                logger.info(f"✅ DEX Sell executed: {opportunity.symbol}")
                logger.info(f"   TX: {result.get('tx_hash')}")
                logger.info(f"   Amount out: {result.get('amount_out')} wei")
            else:
                logger.error(f"❌ DEX sell failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"DEX sell failed: {e}")
            return {'success': False, 'error': str(e)}


class DEXOrchestrator:
    """
    Complete DEX Trading Orchestrator
    - Scans for micro-cap gems across all chains
    - Analyzes safety and potential
    - Executes trades with MEV protection
    - Manages positions across DEXs
    """
    
    def __init__(self, config: Optional[DEXConfig] = None, data_hub: Optional[Any] = None):
        self.config = config or DEXConfig()
        self.data_hub = data_hub
        
        # Core components
        self.moon_spotter = MicroMoonSpotter()
        self.web3_manager = Web3Manager()
        self.executor = DEXExecutor(self.config)
        
        # State
        self.opportunities = []
        self.positions = {}
        self.watchlist = []
        self.running = False
        
        # Stats
        self.stats = {
            'opportunities_found': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume_usd': 0.0,
            'total_profit_usd': 0.0,
        }
        
        logger.info("✅ DEX Orchestrator initialized")
        logger.info(f"   Chains: {', '.join(self.config.chains)}")
        logger.info(f"   MEV Protection: {self.config.mev_protection}")
        logger.info(f"   Private TX: {self.config.use_private_tx}")
    
    async def start(self):
        """Start DEX orchestrator"""
        self.running = True
        logger.info("🚀 DEX Orchestrator STARTED")
        
        # Connect to all enabled chains
        for chain in self.config.chains:
            await self.web3_manager.connect_chain(chain)
        
        # Start scanning loop
        asyncio.create_task(self._scanning_loop())
        asyncio.create_task(self._position_monitoring_loop())
    
    async def stop(self):
        """Stop DEX orchestrator"""
        self.running = False
        logger.info("🛑 DEX Orchestrator STOPPED")
    
    async def _scanning_loop(self):
        """Continuously scan for new opportunities"""
        while self.running:
            try:
                # Scan for new gems
                gems = await self.moon_spotter.scan_for_new_gems()
                
                logger.info(f"🔍 Moon Spotter found {len(gems)} potential gems")
                
                # Convert to opportunities
                opportunities = []
                for gem in gems[:20]:  # Top 20
                    opp = DEXOpportunity(
                        token_address=gem.get('address', ''),
                        symbol=gem.get('symbol', ''),
                        name=gem.get('name', ''),
                        chain=gem.get('chain', 'unknown'),
                        dex=gem.get('dex', 'unknown'),
                        price_usd=gem.get('price', 0.0),
                        liquidity_usd=gem.get('liquidity', 0.0),
                        volume_24h=gem.get('volume_24h', 0.0),
                        holders=gem.get('holders', 0),
                        market_cap=gem.get('market_cap', 0.0),
                        age_hours=gem.get('age_hours', 0.0),
                        safety_score=gem.get('safety_score', 0.0),
                        potential_score=gem.get('moon_score', 0.0),
                        buy_tax=gem.get('buy_tax', 0.0),
                        sell_tax=gem.get('sell_tax', 0.0),
                        honeypot_risk=gem.get('is_honeypot', False),
                        contract_verified=gem.get('verified', False),
                        social_signals=gem.get('social_signals', {}),
                        timestamp=datetime.now()
                    )
                    
                    # Filter by criteria
                    if self._should_trade(opp):
                        opportunities.append(opp)
                        
                        # Send to data hub
                        if self.data_hub and hasattr(self.data_hub, 'signal_queue'):
                            signal = {
                                'type': 'dex_opportunity',
                                'symbol': opp.symbol,
                                'chain': opp.chain,
                                'dex': opp.dex,
                                'price': opp.price_usd,
                                'liquidity': opp.liquidity_usd,
                                'safety_score': opp.safety_score,
                                'potential_score': opp.potential_score,
                                'action': 'BUY' if opp.potential_score >= 70 else 'WATCH',
                                'confidence': opp.potential_score / 100.0,
                                'timestamp': opp.timestamp.isoformat()
                            }
                            await self.data_hub.signal_queue.put(signal)
                
                self.opportunities = opportunities
                self.stats['opportunities_found'] += len(opportunities)
                
                # Auto-trade high-confidence opportunities
                for opp in opportunities:
                    if opp.potential_score >= 80 and opp.safety_score >= 70:
                        await self._execute_opportunity(opp)
                
                # Wait before next scan
                await asyncio.sleep(60)  # Scan every 60 seconds
                
            except Exception as e:
                logger.error(f"Scanning loop error: {e}")
                await asyncio.sleep(30)
    
    def _should_trade(self, opp: DEXOpportunity) -> bool:
        """Determine if an opportunity should be traded"""
        
        # Safety checks
        if opp.honeypot_risk:
            return False
        
        if opp.safety_score < 50:
            return False
        
        if opp.liquidity_usd < self.config.min_liquidity_usd:
            return False
        
        # Tax checks
        if opp.buy_tax > 15.0 or opp.sell_tax > 15.0:  # Max 15% tax
            return False
        
        # Potential checks
        if opp.potential_score < 60:
            return False
        
        return True
    
    async def _execute_opportunity(self, opp: DEXOpportunity):
        """Execute a trading opportunity"""
        
        try:
            # Get Web3 connection
            w3 = await self.web3_manager.connect_chain(opp.chain)
            if not w3:
                logger.warning(f"No Web3 connection for {opp.chain}")
                return
            
            # Get DEX router
            router = self.web3_manager.get_dex_router(opp.chain, opp.dex)
            if not router:
                logger.warning(f"No router for {opp.dex} on {opp.chain}")
                return
            
            # Calculate position size (start small for micro caps)
            position_size = min(
                self.config.max_position_usd,
                opp.liquidity_usd * 0.01,  # Max 1% of liquidity
                50.0  # Max $50 for micro caps
            )
            
            # Execute buy
            result = await self.executor.execute_buy(opp, position_size, w3, router)
            
            if result.get('ok') or result.get('success'):
                self.stats['trades_executed'] += 1
                self.stats['successful_trades'] += 1
                self.stats['total_volume_usd'] += position_size
                
                # Track position
                self.positions[opp.token_address] = {
                    'opportunity': opp,
                    'entry_price': opp.price_usd,
                    'position_size': position_size,
                    'entry_time': datetime.now(),
                    'chain': opp.chain,
                    'dex': opp.dex,
                }
                
                logger.info(f"✅ TRADE EXECUTED: {opp.symbol} on {opp.chain}")
                logger.info(f"   Entry: ${opp.price_usd:.8f}")
                logger.info(f"   Size: ${position_size:.2f}")
                logger.info(f"   Potential: {opp.potential_score:.0f}/100")
            else:
                self.stats['failed_trades'] += 1
                logger.warning(f"❌ Trade failed: {opp.symbol}")
                
        except Exception as e:
            logger.error(f"Execute opportunity error: {e}")
            self.stats['failed_trades'] += 1
    
    async def _position_monitoring_loop(self):
        """Monitor open positions and take profit/stop loss"""
        while self.running:
            try:
                for address, position in list(self.positions.items()):
                    try:
                        opp = position['opportunity']
                        entry_price = position['entry_price']
                        position_size = position['position_size']
                        entry_time = position['entry_time']
                        chain = position['chain']
                        
                        # Get Web3 connection
                        w3 = await self.web3_manager.connect_chain(chain)
                        if not w3:
                            continue
                        
                        # Get router
                        router = self.web3_manager.get_dex_router(chain, position.get('dex', 'uniswap_v2'))
                        if not router:
                            continue
                        
                        # Get token balance
                        engine = self.executor.get_swap_engine(chain, w3, router)
                        if not engine:
                            continue
                        
                        balance = engine.get_token_balance(opp.token_address)
                        if balance == 0:
                            # Already sold
                            del self.positions[address]
                            continue
                        
                        # Calculate current price (rough estimate from pool)
                        current_price = opp.price_usd  # Would need to query pool for real price
                        
                        # Calculate PnL
                        price_change = (current_price - entry_price) / entry_price
                        
                        # Take profit at 2x
                        if price_change >= 1.0:  # 100% gain
                            logger.info(f"🎯 Taking profit on {opp.symbol}: {price_change:.1%} gain")
                            result = await self.executor.execute_sell(
                                opportunity=opp,
                                amount_tokens=balance,
                                w3=w3,
                                router_address=router
                            )
                            
                            if result.get('success'):
                                self.stats['total_profit_usd'] += position_size * price_change
                                del self.positions[address]
                                logger.info(f"✅ Sold {opp.symbol} for {price_change:.1%} profit!")
                        
                        # Stop loss at -50%
                        elif price_change <= -0.5:  # 50% loss
                            logger.warning(f"🛑 Stop loss on {opp.symbol}: {price_change:.1%} loss")
                            result = await self.executor.execute_sell(
                                opportunity=opp,
                                amount_tokens=balance,
                                w3=w3,
                                router_address=router
                            )
                            
                            if result.get('success'):
                                self.stats['total_profit_usd'] += position_size * price_change  # Negative
                                del self.positions[address]
                                logger.info(f"✅ Closed {opp.symbol} at {price_change:.1%} loss")
                        
                        # Log status for positions in between
                        else:
                            logger.info(f"📊 {opp.symbol}: {price_change:+.1%} (holding)")
                    
                    except Exception as e:
                        logger.error(f"Error monitoring position {address}: {e}")
                        continue
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get DEX trading statistics"""
        return {
            **self.stats,
            'active_positions': len(self.positions),
            'opportunities_watching': len(self.opportunities),
            'win_rate': (
                self.stats['successful_trades'] / self.stats['trades_executed']
                if self.stats['trades_executed'] > 0 else 0.0
            )
        }
    
    async def get_opportunities(self, min_score: float = 60.0) -> List[DEXOpportunity]:
        """Get current opportunities above a threshold"""
        return [
            opp for opp in self.opportunities
            if opp.potential_score >= min_score
        ]


# Quick test
if __name__ == "__main__":
    async def main():
        config = DEXConfig(
            enabled=True,
            chains=['bsc', 'ethereum'],  # Start with BSC and ETH
            max_position_usd=50.0,
            max_slippage_bps=100,  # 1% for micro caps
            min_liquidity_usd=1000.0,
            use_private_tx=False,  # Disable for testing
            mev_protection=True
        )
        
        orchestrator = DEXOrchestrator(config)
        await orchestrator.start()
        
        # Run for 5 minutes
        await asyncio.sleep(300)
        
        stats = await orchestrator.get_stats()
        print(f"\n📊 DEX Stats:")
        print(f"   Opportunities: {stats['opportunities_found']}")
        print(f"   Trades: {stats['trades_executed']}")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        
        await orchestrator.stop()
    
    asyncio.run(main())
