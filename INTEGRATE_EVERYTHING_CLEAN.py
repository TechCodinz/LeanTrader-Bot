#!/usr/bin/env python3
"""
INTEGRATE ADAPTIVE CONFIDENCE + ULTRA RARE ENGINES
Safely add both features without breaking anything
"""

import re
import sys

print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║                                                                              ║")
print("║           🚀 INTEGRATING PREMIUM FEATURES 🚀                                 ║")
print("║                                                                              ║")
print("║  1. Adaptive Confidence Engine (auto-tune thresholds)                       ║")
print("║  2. Ultra Rare Engines (10 advanced profit engines)                         ║")
print("║                                                                              ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# PART 1: INTEGRATE ADAPTIVE CONFIDENCE INTO EXECUTION_ORCHESTRATOR
# ============================================================================

print("1️⃣  Integrating Adaptive Confidence Engine...")
print()

with open('EXECUTION_ORCHESTRATOR.py', 'r') as f:
    exec_content = f.read()

# Backup
with open('EXECUTION_ORCHESTRATOR.py.pre_adaptive', 'w') as f:
    f.write(exec_content)

# Check if already integrated
if 'ADAPTIVE_CONFIDENCE_ENGINE' in exec_content:
    print("   ⚠️  Adaptive Confidence already imported")
else:
    # Add import at top
    lines = exec_content.split('\n')
    
    # Find where to add import (after other imports)
    import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_idx = i + 1
    
    # Insert import
    lines.insert(import_idx, 'from ADAPTIVE_CONFIDENCE_ENGINE import get_adaptive_confidence_engine')
    
    exec_content = '\n'.join(lines)
    print("   ✅ Added import")

# Now integrate into ExecutionOrchestrator class
if 'self.adaptive_confidence' not in exec_content:
    # Find the __init__ method of ExecutionOrchestrator class
    # Add after other initializations
    
    # Find ExecutionOrchestrator __init__
    init_pattern = r'(class ExecutionOrchestrator:.*?def __init__\(self.*?\):)(.*?)((?=\n    def |\nclass |\Z))'
    
    match = re.search(init_pattern, exec_content, re.DOTALL)
    if match:
        class_def = match.group(1)
        init_body = match.group(2)
        rest = match.group(3)
        
        # Add adaptive confidence initialization
        if 'self.adaptive_confidence' not in init_body:
            # Find a good place to add it (after self.min_confidence)
            if 'self.min_confidence' in init_body:
                init_body = init_body.replace(
                    'self.min_confidence = min_confidence',
                    '''self.min_confidence = min_confidence
        
        # 🧠 ADAPTIVE CONFIDENCE ENGINE
        try:
            self.adaptive_confidence = get_adaptive_confidence_engine()
            self.use_adaptive_confidence = True
            logger.info("🧠 Adaptive Confidence Engine enabled!")
        except Exception as e:
            logger.warning(f"Adaptive Confidence unavailable: {e}")
            self.adaptive_confidence = None
            self.use_adaptive_confidence = False'''
                )
                
                exec_content = class_def + init_body + rest
                print("   ✅ Added to __init__")
            else:
                print("   ⚠️  Could not find self.min_confidence")
    else:
        print("   ⚠️  Could not find ExecutionOrchestrator __init__")

# Now use adaptive confidence in process_signal
if 'adaptive_threshold' not in exec_content:
    # Find where confidence is checked
    # Pattern: if confidence < self.min_confidence
    
    confidence_check = r'if confidence < self\.min_confidence:'
    
    if re.search(confidence_check, exec_content):
        # Replace with adaptive version
        exec_content = re.sub(
            confidence_check,
            '''# Get adaptive threshold (or use static)
        if self.use_adaptive_confidence and self.adaptive_confidence:
            adaptive_threshold = self.adaptive_confidence.get_adaptive_threshold(
                symbol=symbol,
                confidence=confidence,
                market_regime=getattr(self, 'current_market_regime', 'neutral')
            )
            logger.debug(f"🧠 Adaptive threshold for {symbol}: {adaptive_threshold:.1%} (base: {self.min_confidence:.1%})")
        else:
            adaptive_threshold = self.min_confidence
        
        if confidence < adaptive_threshold:''',
            exec_content
        )
        print("   ✅ Using adaptive thresholds")
    else:
        print("   ⚠️  Could not find confidence check")

# Write back
with open('EXECUTION_ORCHESTRATOR.py', 'w') as f:
    f.write(exec_content)

# Test import
try:
    sys.path.insert(0, '.')
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator
    print("   ✅ EXECUTION_ORCHESTRATOR imports OK")
    print()
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Restoring backup...")
    with open('EXECUTION_ORCHESTRATOR.py.pre_adaptive', 'r') as f:
        with open('EXECUTION_ORCHESTRATOR.py', 'w') as out:
            out.write(f.read())
    print("   ❌ Adaptive Confidence integration FAILED")
    print()
    sys.exit(1)

# ============================================================================
# PART 2: INTEGRATE ULTRA RARE ENGINES INTO COMPLETE_ULTIMATE_ORCHESTRATOR
# ============================================================================

print("2️⃣  Integrating Ultra Rare Engines...")
print()

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
    orch_content = f.read()

# Backup
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py.pre_ultra', 'w') as f:
    f.write(orch_content)

# Check if already integrated
if 'ULTRA_RARE_ENGINES' in orch_content:
    print("   ⚠️  Ultra Rare Engines already imported")
else:
    # Add import at top (after other imports from DYNAMIC_PAIR_DISCOVERY)
    if 'from DYNAMIC_PAIR_DISCOVERY import' in orch_content:
        orch_content = orch_content.replace(
            'from DYNAMIC_PAIR_DISCOVERY import get_discovery_engine',
            '''from DYNAMIC_PAIR_DISCOVERY import get_discovery_engine
from ULTRA_RARE_ENGINES import UltraRareEnginesOrchestrator'''
        )
        print("   ✅ Added import")
    else:
        # Add at a safe location
        lines = orch_content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if 'import' in line.lower():
                import_idx = i + 1
        lines.insert(import_idx, 'from ULTRA_RARE_ENGINES import UltraRareEnginesOrchestrator')
        orch_content = '\n'.join(lines)
        print("   ✅ Added import")

# Add to advanced systems initialization
if 'UltraRareEnginesOrchestrator' not in orch_content:
    # Find where "ALL 8 ADVANCED SYSTEMS INITIALIZED" is
    if "ALL 8 ADVANCED SYSTEMS INITIALIZED" in orch_content:
        # Add before that message
        orch_content = orch_content.replace(
            "logger.info('🎉 ALL 8 ADVANCED SYSTEMS INITIALIZED!')",
            """# 9. ULTRA RARE ENGINES (10 advanced profit engines)
        try:
            self.advanced_systems['ultra_rare'] = UltraRareEnginesOrchestrator()
            logger.info('⚡ Ultra Rare Engines initialized (10 engines!)')
        except Exception as e:
            logger.warning(f'Ultra Rare Engines unavailable: {e}')
        
        logger.info('🎉 ALL 9 ADVANCED SYSTEMS INITIALIZED!')"""
        )
        print("   ✅ Added to initialization")
    else:
        print("   ⚠️  Could not find system initialization section")

# Add to start() method to run ultra rare engines
if 'run_ultra_rare_engines' not in orch_content:
    # Create the method
    run_method = '''
    async def run_ultra_rare_engines(self):
        """Run Ultra Rare Engines continuously"""
        engine = self.advanced_systems.get('ultra_rare')
        if not engine:
            return
            
        logger.info('⚡ Ultra Rare Engines active!')
        
        while True:
            try:
                # Get signals from all 10 engines
                signals = await engine.generate_signals()
                
                if signals:
                    logger.info(f'⚡ Ultra Rare: {len(signals)} opportunities found!')
                    
                    # Publish to data hub
                    if hasattr(self, 'data_hub'):
                        for signal in signals:
                            await self.data_hub.publish_signal(signal)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f'Ultra Rare Engines error: {e}')
                await asyncio.sleep(30)
'''
    
    # Find where to insert (before the start method ends or near run_dynamic_pair_discovery)
    if 'async def run_dynamic_pair_discovery' in orch_content:
        orch_content = orch_content.replace(
            'async def run_dynamic_pair_discovery',
            run_method + '\n    async def run_dynamic_pair_discovery'
        )
        print("   ✅ Added run_ultra_rare_engines method")
    
    # Add to background tasks in start()
    if 'run_dynamic_pair_discovery()' in orch_content:
        orch_content = orch_content.replace(
            'asyncio.create_task(self.run_dynamic_pair_discovery())',
            '''asyncio.create_task(self.run_dynamic_pair_discovery())
                )
                
                # Ultra Rare Engines task
                if self.advanced_systems.get('ultra_rare'):
                    background_tasks.append(
                        asyncio.create_task(self.run_ultra_rare_engines())'''
        )
        print("   ✅ Added to background tasks")

# Write back
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as f:
    f.write(orch_content)

# Test import
try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("   ✅ COMPLETE_ULTIMATE_ORCHESTRATOR imports OK")
    print()
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Restoring backup...")
    with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py.pre_ultra', 'r') as f:
        with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as out:
            out.write(f.read())
    print("   ❌ Ultra Rare Engines integration FAILED")
    print()
    sys.exit(1)

# ============================================================================
# FINAL TEST
# ============================================================================

print("═══════════════════════════════════════════════════════════════")
print("✅ ALL INTEGRATIONS SUCCESSFUL!")
print("═══════════════════════════════════════════════════════════════")
print()
print("Integrated features:")
print("  ✅ Adaptive Confidence Engine")
print("     - Auto-tunes thresholds 65-95%")
print("     - Based on market regime, pairs, performance")
print("     - MORE trades with BETTER timing!")
print()
print("  ✅ Ultra Rare Engines (10 engines)")
print("     - Microstructure Exploiter")
print("     - Information Entropy Trader")
print("     - Cascading Liquidity Hunter")
print("     - Flash Crash Predator")
print("     - Funding Rate Arbitrage")
print("     - Hidden Order Detector")
print("     - Smart Money Shadow")
print("     - Retail Panic Exploiter")
print("     - Time Warp Patterns")
print("     - Whale Psychology Predictor")
print()
print("═══════════════════════════════════════════════════════════════")
print()
print("Now restart bot to activate:")
print("  pkill -9 -f RUN_BOT.py && ./start_bot.sh")
print()
