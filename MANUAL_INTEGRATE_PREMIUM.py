#!/usr/bin/env python3
"""
MANUAL PREMIUM INTEGRATION - Simple and Safe
No complex regex, just careful manual edits
"""

print("╔══════════════════════════════════════════════════════════════╗")
print("║  MANUAL PREMIUM INTEGRATION (Safe & Simple)                 ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()

# ==============================================================================
# PART 1: ADAPTIVE CONFIDENCE - Manual Integration
# ==============================================================================

print("1️⃣  Integrating Adaptive Confidence...")

with open('EXECUTION_ORCHESTRATOR.py', 'r') as f:
    lines = f.readlines()

# Backup
with open('EXECUTION_ORCHESTRATOR.py.backup_manual', 'w') as f:
    f.writelines(lines)

# Step 1: Add import at top (after existing imports)
import_added = False
for i, line in enumerate(lines):
    if 'from datetime import datetime' in line and not import_added:
        # Add after this line
        lines.insert(i + 1, '\n# 🧠 Premium features\ntry:\n    from ADAPTIVE_CONFIDENCE_ENGINE import get_adaptive_confidence_engine\n    ADAPTIVE_AVAILABLE = True\nexcept:\n    ADAPTIVE_AVAILABLE = False\n')
        import_added = True
        print("   ✅ Added import (safe)")
        break

# Step 2: Add to __init__ (after self.min_confidence = 0.80)
init_added = False
for i, line in enumerate(lines):
    if 'self.min_confidence = 0.80' in line and not init_added:
        # Add after this line
        indent = '        '  # 8 spaces for class method
        adaptive_init = f'''
{indent}# 🧠 Adaptive Confidence Engine (optional premium feature)
{indent}self.use_adaptive = False
{indent}self.adaptive_engine = None
{indent}if ADAPTIVE_AVAILABLE:
{indent}    try:
{indent}        self.adaptive_engine = get_adaptive_confidence_engine()
{indent}        self.use_adaptive = True
{indent}        logger.info("🧠 Adaptive Confidence Engine: ENABLED")
{indent}    except Exception as e:
{indent}        logger.warning(f"Adaptive Confidence unavailable: {{e}}")
'''
        lines.insert(i + 1, adaptive_init)
        init_added = True
        print("   ✅ Added to __init__ (safe)")
        break

# Step 3: Use adaptive threshold (find confidence check in process_signal or process_decision)
threshold_updated = False
for i, line in enumerate(lines):
    if 'if confidence < self.min_confidence:' in line and not threshold_updated:
        # Replace this line with adaptive version
        indent = ' ' * (len(line) - len(line.lstrip()))
        adaptive_check = f'''{indent}# Get threshold (adaptive or static)
{indent}threshold = self.min_confidence
{indent}if self.use_adaptive and self.adaptive_engine:
{indent}    try:
{indent}        threshold = self.adaptive_engine.get_adaptive_threshold(
{indent}            symbol=symbol,
{indent}            confidence=confidence,
{indent}            market_regime='neutral'
{indent}        )
{indent}    except:
{indent}        pass  # Fall back to static
{indent}
{indent}if confidence < threshold:
'''
        lines[i] = adaptive_check
        threshold_updated = True
        print("   ✅ Using adaptive thresholds (safe)")
        break

# Write back
with open('EXECUTION_ORCHESTRATOR.py', 'w') as f:
    f.writelines(lines)

# Test import
import sys
sys.path.insert(0, '.')

try:
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator
    print("   ✅ EXECUTION_ORCHESTRATOR imports OK!")
    print()
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Restoring backup...")
    with open('EXECUTION_ORCHESTRATOR.py.backup_manual', 'r') as f:
        with open('EXECUTION_ORCHESTRATOR.py', 'w') as out:
            out.write(f.read())
    print()
    sys.exit(1)

# ==============================================================================
# PART 2: ULTRA RARE ENGINES - Manual Integration  
# ==============================================================================

print("2️⃣  Integrating Ultra Rare Engines...")

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
    lines = f.readlines()

# Backup
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup_manual', 'w') as f:
    f.writelines(lines)

# Step 1: Add import (after Dynamic Pair Discovery import)
import_added = False
for i, line in enumerate(lines):
    if 'from DYNAMIC_PAIR_DISCOVERY import' in line and not import_added:
        # Add after this line
        lines.insert(i + 1, '\n# Ultra Rare Engines (premium)\ntry:\n    from ULTRA_RARE_ENGINES import UltraRareEnginesOrchestrator\n    ULTRA_RARE_AVAILABLE = True\nexcept:\n    ULTRA_RARE_AVAILABLE = False\n')
        import_added = True
        print("   ✅ Added import (safe)")
        break

# Step 2: Initialize in advanced systems (find where "ALL 8 ADVANCED SYSTEMS" is)
ultra_added = False
for i, line in enumerate(lines):
    if "ALL 8 ADVANCED SYSTEMS INITIALIZED" in line and not ultra_added:
        # Add BEFORE this line
        indent = ' ' * (len(line) - len(line.lstrip()))
        ultra_init = f'''
{indent}# 9. ULTRA RARE ENGINES (10 advanced profit engines)
{indent}if ULTRA_RARE_AVAILABLE:
{indent}    try:
{indent}        self.advanced_systems['ultra_rare'] = UltraRareEnginesOrchestrator()
{indent}        logger.info('⚡ Ultra Rare Engines initialized (10 engines!)')
{indent}    except Exception as e:
{indent}        logger.warning(f'Ultra Rare Engines unavailable: {{e}}')
{indent}
'''
        lines.insert(i, ultra_init)
        # Update the message
        lines[i + len(ultra_init.split('\n'))] = line.replace('ALL 8', 'ALL 9')
        ultra_added = True
        print("   ✅ Added to initialization (safe)")
        break

# Write back
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as f:
    f.writelines(lines)

# Test import
try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("   ✅ COMPLETE_ULTIMATE_ORCHESTRATOR imports OK!")
    print()
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Restoring backup...")
    with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup_manual', 'r') as f:
        with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as out:
            out.write(f.read())
    print()
    sys.exit(1)

# ==============================================================================
# SUCCESS!
# ==============================================================================

print("═══════════════════════════════════════════════════════════════")
print("✅ ALL PREMIUM FEATURES INTEGRATED!")
print("═══════════════════════════════════════════════════════════════")
print()
print("Integrated:")
print("  ✅ Adaptive Confidence Engine (smart thresholds)")
print("  ✅ Ultra Rare Engines (10 profit engines)")
print()
print("Restart bot to activate!")
print()
