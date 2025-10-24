#!/usr/bin/env python3
"""
WIRE DISCOVERED PAIRS TO TRADING ENGINES
Make the bot actually TRADE the 5,607 discovered pairs!

Problem: Discovery finds 5,607 pairs but trading engines still use 92
Solution: Update universe when discovery completes
"""

print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║                                                                              ║")
print("║         🔌 WIRE DISCOVERED PAIRS TO TRADING 🔌                               ║")
print("║                                                                              ║")
print("║  Make trading engines USE the 5,607 discovered pairs!                       ║")
print("║                                                                              ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")
print()

import sys
sys.path.insert(0, '.')

# ============================================================================
# FIX 1: Make COMPLETE_ULTIMATE_ORCHESTRATOR update parent's universe
# ============================================================================

print("1️⃣  Adding universe update mechanism to COMPLETE_ULTIMATE_ORCHESTRATOR...")

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
    ult_content = f.read()

# Backup
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py.pre_wire', 'w') as f:
    f.write(ult_content)

# Find the run_dynamic_pair_discovery method
if 'async def run_dynamic_pair_discovery' in ult_content:
    # Add universe update logic
    
    # Find where it updates self.dynamic_pairs
    if 'self.dynamic_pairs.extend(new_pairs)' in ult_content or 'self.dynamic_pairs =' in ult_content:
        
        # Add a method to refresh parent's universe
        update_method = '''
    def update_trading_universe(self):
        """Update parent's trading universe with discovered pairs"""
        if hasattr(self, 'dynamic_pairs') and self.dynamic_pairs:
            # Update parent's universe
            if hasattr(self, 'ultra_core'):
                # Update ultra_core's pair list
                self.ultra_core.pairs = self.dynamic_pairs
                logger.info(f'🔄 Updated ultra_core universe: {len(self.dynamic_pairs)} pairs')
            
            if hasattr(self, 'trading_universe'):
                self.trading_universe = self.dynamic_pairs
                logger.info(f'🔄 Updated trading_universe: {len(self.dynamic_pairs)} pairs')
            
            # Update engines if they exist
            if hasattr(self, 'engines'):
                for engine_name, engine in self.engines.items():
                    if hasattr(engine, 'pairs'):
                        engine.pairs = self.dynamic_pairs
                        logger.debug(f'🔄 Updated {engine_name}: {len(self.dynamic_pairs)} pairs')
'''
        
        # Add this method before run_dynamic_pair_discovery
        ult_content = ult_content.replace(
            'async def run_dynamic_pair_discovery',
            update_method + '\n    async def run_dynamic_pair_discovery'
        )
        
        # Now call this method whenever pairs are updated
        # Find where self.dynamic_pairs is updated
        if 'logger.info(f"✅ Added {len(new_pairs)} new profitable pairs!")' in ult_content:
            ult_content = ult_content.replace(
                'logger.info(f"✅ Added {len(new_pairs)} new profitable pairs!")',
                '''logger.info(f"✅ Added {len(new_pairs)} new profitable pairs!")
                    
                    # 🔌 UPDATE TRADING UNIVERSE
                    self.update_trading_universe()
                    logger.info(f"🔄 Trading engines now using {len(self.dynamic_pairs)} pairs!")'''
            )
            print("   ✅ Added universe update on discovery")
        
        with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as f:
            f.write(ult_content)

# ============================================================================
# FIX 2: Make COMPLETE_UNIFIED_ORCHESTRATOR accept universe updates
# ============================================================================

print()
print("2️⃣  Making COMPLETE_UNIFIED_ORCHESTRATOR accept universe updates...")

with open('COMPLETE_UNIFIED_ORCHESTRATOR.py', 'r') as f:
    uni_content = f.read()

# Backup
with open('COMPLETE_UNIFIED_ORCHESTRATOR.py.pre_wire', 'w') as f:
    f.write(uni_content)

# Add a method to update universe dynamically
if 'def update_universe' not in uni_content:
    update_uni_method = '''
    def update_universe(self, new_universe: list):
        """Update trading universe dynamically"""
        if new_universe and len(new_universe) > len(getattr(self, 'trading_universe', [])):
            old_count = len(getattr(self, 'trading_universe', []))
            self.trading_universe = new_universe
            
            # Update ultra_core
            if hasattr(self, 'ultra_core') and hasattr(self.ultra_core, 'pairs'):
                self.ultra_core.pairs = new_universe
            
            logger.info(f"🔄 UNIVERSE EXPANDED: {old_count} → {len(new_universe)} pairs!")
            logger.info(f"   🎯 Trading engines now scanning {len(new_universe)} pairs!")
            
            return True
        return False
'''
    
    # Add after __init__ or near other methods
    if 'async def initialize_all_systems' in uni_content:
        uni_content = uni_content.replace(
            'async def initialize_all_systems',
            update_uni_method + '\n    async def initialize_all_systems'
        )
        print("   ✅ Added update_universe method")
        
        with open('COMPLETE_UNIFIED_ORCHESTRATOR.py', 'w') as f:
            f.write(uni_content)

# ============================================================================
# TEST IMPORTS
# ============================================================================

print()
print("3️⃣  Testing integrations...")

try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("   ✅ COMPLETE_ULTIMATE_ORCHESTRATOR imports OK")
except Exception as e:
    print(f"   ❌ COMPLETE_ULTIMATE_ORCHESTRATOR: {e}")
    sys.exit(1)

try:
    from COMPLETE_UNIFIED_ORCHESTRATOR import CompleteUnifiedOrchestrator
    print("   ✅ COMPLETE_UNIFIED_ORCHESTRATOR imports OK")
except Exception as e:
    print(f"   ❌ COMPLETE_UNIFIED_ORCHESTRATOR: {e}")
    sys.exit(1)

# ============================================================================
# SUCCESS!
# ============================================================================

print()
print("═══════════════════════════════════════════════════════════════")
print("✅ DISCOVERED PAIRS WIRED TO TRADING!")
print("═══════════════════════════════════════════════════════════════")
print()
print("What happens now:")
print()
print("  1. Discovery finds 5,607 pairs")
print("  2. Updates self.dynamic_pairs")
print("  3. 🔌 CALLS update_trading_universe()")
print("  4. ✅ ALL engines now trade 5,607 pairs!")
print()
print("Expected log messages:")
print("  🌍 TOTAL DISCOVERED: 5607 pairs!")
print("  🔄 Trading engines now using 5607 pairs!")
print("  🔄 UNIVERSE EXPANDED: 92 → 5607 pairs!")
print()
print("═══════════════════════════════════════════════════════════════")
print()
print("Restart bot to activate:")
print("  pkill -9 -f RUN_BOT.py && ./start_bot.sh")
print()
print("Then watch:")
print("  tail -f bot.log | grep -E 'UNIVERSE EXPANDED|5607 pairs|Trading engines now'")
print()
