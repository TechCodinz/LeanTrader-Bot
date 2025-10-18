#!/usr/bin/env python3
"""
🧠 ADAPTIVE PAIR SELECTOR
Smart pair selection based on wallet size - grows with your balance!

Small balance → Few pairs, high liquidity, tight spreads
Growing balance → More pairs, more opportunities
Large balance → All pairs, maximum diversification
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AdaptivePairSelector:
    """
    Intelligently selects trading pairs based on current wallet size
    Grows with your balance for optimal capital efficiency
    """
    
    def __init__(self):
        # Define pairs by tier based on liquidity & volatility
        self.tier1_pairs = [
            # TIER 1: Ultra-liquid, low spread - Perfect for small balance ($10-50 trades)
            'BTC/USDT',   # King - highest liquidity
            'ETH/USDT',   # Queen - second highest liquidity
            'SOL/USDT',   # High volume, good volatility
            'BNB/USDT',   # Exchange coin, stable
            'XRP/USDT',   # High volume, tight spreads
        ]
        
        self.tier2_pairs = [
            # TIER 2: High-liquid mid-caps - Good for growing balance ($50-100 trades)
            'ADA/USDT',   # Cardano - solid volume
            'DOGE/USDT',  # Meme with huge volume
            'MATIC/USDT', # Polygon - good for scalping
            'DOT/USDT',   # Polkadot - quality project
            'AVAX/USDT',  # Avalanche - high volatility
            'LINK/USDT',  # Chainlink - consistent
            'TRX/USDT',   # Tron - high volume
            'UNI/USDT',   # Uniswap - DeFi leader
            'LTC/USDT',   # Litecoin - stable
            'ATOM/USDT',  # Cosmos - good momentum
        ]
        
        self.tier3_pairs = [
            # TIER 3: Mid-caps with growth potential - For diversification ($100-200 trades)
            'SHIB/USDT',  # Shiba - meme with volume
            'FTM/USDT',   # Fantom - good tech
            'ALGO/USDT',  # Algorand - quality
            'XLM/USDT',   # Stellar - payments
            'NEAR/USDT',  # Near Protocol - growing
            'APT/USDT',   # Aptos - new L1
            'ARB/USDT',   # Arbitrum - L2 leader
            'OP/USDT',    # Optimism - L2
            'INJ/USDT',   # Injective - DeFi
            'SUI/USDT',   # Sui - new L1
        ]
        
        self.tier4_pairs = [
            # TIER 4: Smaller caps with high potential - For experienced larger balance
            'FIL/USDT',   # Filecoin - storage
            'VET/USDT',   # VeChain - supply chain
            'SAND/USDT',  # Sandbox - metaverse
            'MANA/USDT',  # Decentraland - metaverse
            'AXS/USDT',   # Axie Infinity - gaming
            'GALA/USDT',  # Gala Games - gaming
            'ICP/USDT',   # Internet Computer
            'ETC/USDT',   # Ethereum Classic
            'HBAR/USDT',  # Hedera
            'QNT/USDT',   # Quant
        ]
        
        # Balance thresholds for tier activation
        self.tier1_threshold = 0      # Always active
        self.tier2_threshold = 100    # Activate at $100
        self.tier3_threshold = 300    # Activate at $300
        self.tier4_threshold = 1000   # Activate at $1000
        
        logger.info("🧠 Adaptive Pair Selector initialized")
        logger.info(f"   Tier 1: {len(self.tier1_pairs)} pairs (Always active)")
        logger.info(f"   Tier 2: {len(self.tier2_pairs)} pairs (Unlocks at ${self.tier2_threshold})")
        logger.info(f"   Tier 3: {len(self.tier3_pairs)} pairs (Unlocks at ${self.tier3_threshold})")
        logger.info(f"   Tier 4: {len(self.tier4_pairs)} pairs (Unlocks at ${self.tier4_threshold})")
    
    def get_active_pairs(self, balance: float) -> List[str]:
        """
        Get list of pairs to trade based on current balance
        
        Args:
            balance: Current USDT balance
            
        Returns:
            List of trading pairs appropriate for balance size
        """
        active_pairs = []
        active_tiers = []
        
        # Tier 1 - Always active
        active_pairs.extend(self.tier1_pairs)
        active_tiers.append(1)
        
        # Tier 2 - Unlock at $100
        if balance >= self.tier2_threshold:
            active_pairs.extend(self.tier2_pairs)
            active_tiers.append(2)
        
        # Tier 3 - Unlock at $300
        if balance >= self.tier3_threshold:
            active_pairs.extend(self.tier3_pairs)
            active_tiers.append(3)
        
        # Tier 4 - Unlock at $1000
        if balance >= self.tier4_threshold:
            active_pairs.extend(self.tier4_pairs)
            active_tiers.append(4)
        
        logger.info(f"💰 Balance: ${balance:.2f} → {len(active_pairs)} pairs active (Tiers: {active_tiers})")
        
        return active_pairs
    
    def get_pair_allocation(self, balance: float) -> Dict[str, float]:
        """
        Get recommended allocation percentage for each pair
        
        Small balance: Concentrate on top pairs
        Large balance: Spread across all pairs
        """
        active_pairs = self.get_active_pairs(balance)
        total_pairs = len(active_pairs)
        
        allocations = {}
        
        if balance < 100:
            # Small balance: Concentrate 60% in Tier 1, 40% in others
            tier1_allocation = 0.60 / len(self.tier1_pairs)
            for pair in active_pairs:
                if pair in self.tier1_pairs:
                    allocations[pair] = tier1_allocation
                else:
                    allocations[pair] = 0.40 / (total_pairs - len(self.tier1_pairs))
        
        elif balance < 500:
            # Medium balance: Balanced allocation
            equal_allocation = 1.0 / total_pairs
            for pair in active_pairs:
                allocations[pair] = equal_allocation
        
        else:
            # Large balance: Slight preference for top tiers but mostly equal
            tier1_allocation = 0.40 / len(self.tier1_pairs)
            other_allocation = 0.60 / (total_pairs - len(self.tier1_pairs))
            for pair in active_pairs:
                if pair in self.tier1_pairs:
                    allocations[pair] = tier1_allocation
                else:
                    allocations[pair] = other_allocation
        
        return allocations
    
    def should_trade_pair(self, pair: str, balance: float, current_positions: int = 0) -> Tuple[bool, str]:
        """
        Determine if a pair should be traded given current balance and positions
        
        Returns:
            (should_trade: bool, reason: str)
        """
        active_pairs = self.get_active_pairs(balance)
        
        # Check if pair is in active tier
        if pair not in active_pairs:
            tier_needed = self._get_required_tier(pair)
            threshold_needed = self._get_tier_threshold(tier_needed)
            return False, f"Unlock at ${threshold_needed} (currently ${balance:.0f})"
        
        # Check if we have too many positions for this balance
        max_positions = self._get_max_positions(balance)
        if current_positions >= max_positions:
            return False, f"Max positions ({max_positions}) reached for ${balance:.0f} balance"
        
        return True, "Ready to trade"
    
    def _get_required_tier(self, pair: str) -> int:
        """Get which tier a pair belongs to"""
        if pair in self.tier1_pairs:
            return 1
        elif pair in self.tier2_pairs:
            return 2
        elif pair in self.tier3_pairs:
            return 3
        elif pair in self.tier4_pairs:
            return 4
        return 1  # Default to tier 1
    
    def _get_tier_threshold(self, tier: int) -> float:
        """Get balance threshold for a tier"""
        thresholds = {
            1: self.tier1_threshold,
            2: self.tier2_threshold,
            3: self.tier3_threshold,
            4: self.tier4_threshold
        }
        return thresholds.get(tier, 0)
    
    def _get_max_positions(self, balance: float) -> int:
        """
        Calculate maximum concurrent positions based on balance
        
        Small balance: Fewer positions (focus)
        Large balance: More positions (diversification)
        """
        if balance < 50:
            return 2   # Very small: Focus on 2 best opportunities
        elif balance < 100:
            return 3   # Small: 3 positions
        elif balance < 300:
            return 5   # Growing: 5 positions
        elif balance < 1000:
            return 8   # Medium: 8 positions
        else:
            return 12  # Large: Maximum diversification
    
    def get_strategy_mode(self, balance: float) -> str:
        """
        Recommend trading strategy based on balance
        
        Small: Conservative, high-probability only
        Medium: Balanced
        Large: Aggressive, more opportunities
        """
        if balance < 100:
            return "CONSERVATIVE"  # Only trade high-confidence (>85%)
        elif balance < 500:
            return "BALANCED"      # Trade medium-confidence (>75%)
        else:
            return "AGGRESSIVE"    # Trade all signals (>70%)
    
    def get_status_report(self, balance: float, open_positions: int) -> str:
        """Get a formatted status report"""
        active_pairs = self.get_active_pairs(balance)
        max_positions = self._get_max_positions(balance)
        strategy_mode = self.get_strategy_mode(balance)
        
        # Calculate progress to next tier
        next_tier_balance = None
        if balance < self.tier2_threshold:
            next_tier_balance = self.tier2_threshold
            next_tier_num = 2
        elif balance < self.tier3_threshold:
            next_tier_balance = self.tier3_threshold
            next_tier_num = 3
        elif balance < self.tier4_threshold:
            next_tier_balance = self.tier4_threshold
            next_tier_num = 4
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              🧠 ADAPTIVE TRADING STATUS                          ║
╚══════════════════════════════════════════════════════════════════╝

💰 Current Balance: ${balance:.2f}
📊 Trading {len(active_pairs)} pairs (Max: {max_positions} positions)
🎯 Strategy Mode: {strategy_mode}
📈 Open Positions: {open_positions}/{max_positions}
"""
        
        if next_tier_balance:
            progress = ((balance / next_tier_balance) * 100)
            needed = next_tier_balance - balance
            report += f"\n🔓 Next Unlock (Tier {next_tier_num}): ${next_tier_balance} ({progress:.1f}% complete)\n"
            report += f"   Need ${needed:.2f} more to unlock {len(self._get_tier_pairs(next_tier_num))} new pairs!\n"
        else:
            report += f"\n🎉 ALL TIERS UNLOCKED! Trading {len(active_pairs)} pairs!\n"
        
        return report
    
    def _get_tier_pairs(self, tier: int) -> List[str]:
        """Get pairs for a specific tier"""
        if tier == 1:
            return self.tier1_pairs
        elif tier == 2:
            return self.tier2_pairs
        elif tier == 3:
            return self.tier3_pairs
        elif tier == 4:
            return self.tier4_pairs
        return []


if __name__ == "__main__":
    # Demo the adaptive selector
    selector = AdaptivePairSelector()
    
    test_balances = [42, 100, 300, 500, 1000, 5000]
    
    for balance in test_balances:
        print(selector.get_status_report(balance, 2))
        print()
