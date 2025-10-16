"""
🌐 EXPANDED MARKET UNIVERSE
Complete trading universe for full market coverage

From 5 coins → 70+ high-volume trading pairs
Includes: Major, DeFi, Meme, Gaming, AI, Layer-1/2, and trending pairs
"""

# ============================================================================
# FULL TRADING UNIVERSE - 70+ PAIRS
# ============================================================================

FULL_MARKET_UNIVERSE = [
    # ===== MAJOR COINS (Top 20 by market cap) =====
    'BTC/USDT',    # Bitcoin - King
    'ETH/USDT',    # Ethereum - Smart contracts
    'BNB/USDT',    # Binance Coin
    'SOL/USDT',    # Solana - Fast L1
    'XRP/USDT',    # Ripple - Payments
    'ADA/USDT',    # Cardano
    'DOGE/USDT',   # Dogecoin - OG meme
    'MATIC/USDT',  # Polygon - L2
    'DOT/USDT',    # Polkadot
    'AVAX/USDT',   # Avalanche
    'SHIB/USDT',   # Shiba Inu - Top meme
    'LTC/USDT',    # Litecoin
    'LINK/USDT',   # Chainlink - Oracles
    'UNI/USDT',    # Uniswap - DEX
    'ATOM/USDT',   # Cosmos
    'ETC/USDT',    # Ethereum Classic
    'XLM/USDT',    # Stellar
    'ICP/USDT',    # Internet Computer
    'FIL/USDT',    # Filecoin
    'VET/USDT',    # VeChain
    
    # ===== DEFI ECOSYSTEM (15 pairs) =====
    'AAVE/USDT',   # Lending
    'MKR/USDT',    # Maker DAO
    'SNX/USDT',    # Synthetics
    'SUSHI/USDT',  # SushiSwap
    'CAKE/USDT',   # PancakeSwap
    'CRV/USDT',    # Curve
    'BAL/USDT',    # Balancer
    'COMP/USDT',   # Compound
    '1INCH/USDT',  # 1inch
    'YFI/USDT',    # Yearn Finance
    'RUNE/USDT',   # THORChain
    'LUNA/USDT',   # Terra (if available)
    'FTM/USDT',    # Fantom
    'ONE/USDT',    # Harmony
    'CELO/USDT',   # Celo
    
    # ===== LAYER 1/2 & INFRA (10 pairs) =====
    'ARB/USDT',    # Arbitrum - L2
    'OP/USDT',     # Optimism - L2
    'NEAR/USDT',   # NEAR Protocol
    'APT/USDT',    # Aptos
    'SUI/USDT',    # Sui
    'SEI/USDT',    # Sei
    'INJ/USDT',    # Injective
    'TIA/USDT',    # Celestia
    'ALGO/USDT',   # Algorand
    'EGLD/USDT',   # MultiversX
    
    # ===== MEME COINS (10 pairs) =====
    'PEPE/USDT',   # Pepe - Top meme 2023-2024
    'FLOKI/USDT',  # Floki Inu
    'BONK/USDT',   # Bonk (Solana)
    'WIF/USDT',    # dogwifhat
    'POPCAT/USDT', # Popcat
    'BRETT/USDT',  # Brett
    'MOG/USDT',    # Mog Coin
    'BABYDOGE/USDT', # Baby Doge
    'ELON/USDT',   # Dogelon Mars
    'AKITA/USDT',  # Akita Inu
    
    # ===== AI & GAMING (5 pairs) =====
    'FET/USDT',    # Fetch.ai
    'AGIX/USDT',   # SingularityNET
    'RNDR/USDT',   # Render
    'GRT/USDT',    # The Graph
    'SAND/USDT',   # Sandbox
    
    # ===== HIGH VOLATILITY / TRENDING (5 pairs) =====
    'GMT/USDT',    # STEPN
    'APE/USDT',    # ApeCoin
    'GALA/USDT',   # Gala Games
    'AXS/USDT',    # Axie Infinity
    'MANA/USDT',   # Decentraland
]

# ============================================================================
# PRIORITY TIERS FOR DIFFERENT STRATEGIES
# ============================================================================

# High liquidity, safe for arbitrage
ARBITRAGE_SAFE_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'AVAX/USDT',
    'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'SHIB/USDT'
]

# High volatility for scalping
SCALPING_OPTIMAL_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT',
    'SHIB/USDT', 'WIF/USDT', 'BONK/USDT', 'FTM/USDT', 'INJ/USDT',
    'APT/USDT', 'SUI/USDT', 'ARB/USDT', 'OP/USDT', 'FLOKI/USDT'
]

# Momentum & trend following
MOMENTUM_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'AVAX/USDT',
    'NEAR/USDT', 'APT/USDT', 'SUI/USDT', 'INJ/USDT', 'TIA/USDT',
    'PEPE/USDT', 'WIF/USDT', 'RNDR/USDT', 'FET/USDT', 'AGIX/USDT'
]

# Moon hunting (low cap, high risk/reward)
MOON_HUNTING_PAIRS = [
    'BONK/USDT', 'WIF/USDT', 'POPCAT/USDT', 'BRETT/USDT', 'MOG/USDT',
    'FET/USDT', 'AGIX/USDT', 'SEI/USDT', 'SUI/USDT', 'APT/USDT'
]

# ============================================================================
# DYNAMIC MARKET SCANNER CONFIG
# ============================================================================

SCANNER_CONFIG = {
    'min_24h_volume_usd': 5_000_000,  # $5M minimum daily volume
    'max_pairs': 100,                  # Monitor up to 100 pairs
    'rescan_interval': 3600,           # Update universe every hour
    'trending_lookback': 24,           # Check 24h trends
    'min_exchanges': 2,                # Available on at least 2 exchanges
}

def get_market_universe_for_balance(balance_usd: float):
    """
    Returns appropriate trading pairs based on account balance
    Small balances → fewer, safer pairs
    Large balances → full market coverage
    """
    if balance_usd < 100:
        # Small account: Focus on major liquid pairs only
        return ARBITRAGE_SAFE_PAIRS[:10]
    elif balance_usd < 500:
        # Medium account: Add scalping opportunities
        return ARBITRAGE_SAFE_PAIRS + SCALPING_OPTIMAL_PAIRS[:10]
    elif balance_usd < 2000:
        # Growing account: Add momentum plays
        return list(set(ARBITRAGE_SAFE_PAIRS + SCALPING_OPTIMAL_PAIRS + MOMENTUM_PAIRS))
    else:
        # Large account: Full market coverage!
        return FULL_MARKET_UNIVERSE

def get_priority_pairs(strategy: str = 'balanced'):
    """Get pairs optimized for specific strategy"""
    strategies = {
        'arbitrage': ARBITRAGE_SAFE_PAIRS,
        'scalping': SCALPING_OPTIMAL_PAIRS,
        'momentum': MOMENTUM_PAIRS,
        'moon': MOON_HUNTING_PAIRS,
        'balanced': list(set(ARBITRAGE_SAFE_PAIRS + SCALPING_OPTIMAL_PAIRS[:10] + MOMENTUM_PAIRS[:10]))
    }
    return strategies.get(strategy, FULL_MARKET_UNIVERSE)

# ============================================================================
# TIMEFRAMES FOR MULTI-TIMEFRAME ANALYSIS
# ============================================================================

TIMEFRAMES = {
    'scalping': ['1m', '5m', '15m'],
    'day_trading': ['15m', '1h', '4h'],
    'swing': ['4h', '1d', '3d'],
    'all': ['1m', '5m', '15m', '1h', '4h', '1d']
}

def get_timeframes_for_pair(pair: str):
    """Returns relevant timeframes based on pair characteristics"""
    # High volatility pairs → focus on shorter timeframes
    if pair in ['PEPE/USDT', 'BONK/USDT', 'WIF/USDT', 'SHIB/USDT', 'DOGE/USDT']:
        return TIMEFRAMES['scalping'] + ['1h']
    # Major pairs → all timeframes
    elif pair in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']:
        return TIMEFRAMES['all']
    # Everything else → day trading timeframes
    else:
        return TIMEFRAMES['day_trading']
