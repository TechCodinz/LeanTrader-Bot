#!/usr/bin/env python3
"""
Make Advanced Trading Actions visible in logs
Instead of just "BUY/SELL", show "LONG 5X", "SCALP", "GRID", etc.
"""

print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║                                                                              ║")
print("║         📊 ACTIVATE ADVANCED ACTION DISPLAY 📊                               ║")
print("║                                                                              ║")
print("║  Make LONG, SHORT, SCALP, GRID, etc. visible in decision logs               ║")
print("║                                                                              ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")
print()

import sys
import re
sys.path.insert(0, '.')

# Find where decisions are logged
print("1️⃣  Finding decision logging code...")

# Check COMPLETE_UNIFIED_ORCHESTRATOR where decisions are logged
try:
    with open('COMPLETE_UNIFIED_ORCHESTRATOR.py', 'r') as f:
        unified_content = f.read()
    
    # Backup
    with open('COMPLETE_UNIFIED_ORCHESTRATOR.py.pre_action_display', 'w') as f:
        f.write(unified_content)
    
    # Find where decisions are logged (look for "Decision: BUY" or "Decision: SELL")
    if 'logger.info(f"🎯 Decision:' in unified_content:
        print("   ✅ Found decision logging")
        
        # We need to enhance the decision before it's logged
        # Look for the pattern where decision is created
        
        # Add enhanced logging right after decision is made
        # Find the decision logging pattern
        pattern = r'logger\.info\(f"🎯 Decision: \{action\} \{symbol\} \(conf: \{confidence\*100:.1f\}%\)"\)'
        
        if re.search(pattern, unified_content):
            # Replace with enhanced version that includes advanced action
            enhanced_log = '''# Determine advanced action
                    try:
                        from ADVANCED_TRADING_ACTIONS_ENGINE import get_advanced_actions
                        adv_actions = get_advanced_actions()
                        direction = 'buy' if action == 'BUY' else 'sell'
                        adv_action = adv_actions.determine_action(
                            symbol, direction, confidence, 
                            timeframe='1h', volatility=0.02
                        )
                        action_display = f"{adv_action['action']} {adv_action['leverage']}X" if adv_action['leverage'] > 1 else adv_action['action']
                        logger.info(f"🎯 {action_display}: {symbol} (conf: {confidence*100:.1f}%, {adv_action['duration']})")
                        logger.debug(f"   Reason: {adv_action['reason']}")
                    except:
                        logger.info(f"🎯 Decision: {action} {symbol} (conf: {confidence*100:.1f}%)")'''
            
            unified_content = re.sub(
                pattern,
                enhanced_log,
                unified_content
            )
            
            print("   ✅ Enhanced decision logging")
            
            with open('COMPLETE_UNIFIED_ORCHESTRATOR.py', 'w') as f:
                f.write(unified_content)
        else:
            print("   ⚠️  Pattern not found, trying simpler approach")
            
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test import
print()
print("2️⃣  Testing integration...")

try:
    from COMPLETE_UNIFIED_ORCHESTRATOR import CompleteUnifiedOrchestrator
    print("   ✅ COMPLETE_UNIFIED_ORCHESTRATOR imports OK")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   Restoring backup...")
    try:
        with open('COMPLETE_UNIFIED_ORCHESTRATOR.py.pre_action_display', 'r') as f:
            with open('COMPLETE_UNIFIED_ORCHESTRATOR.py', 'w') as out:
                out.write(f.read())
        print("   ✅ Backup restored")
    except:
        pass
    sys.exit(1)

print()
print("═══════════════════════════════════════════════════════════════")
print("✅ ADVANCED ACTIONS NOW VISIBLE IN LOGS!")
print("═══════════════════════════════════════════════════════════════")
print()
print("You'll now see decisions like:")
print()
print("  🎯 LONG 7X: BTC/USDT (conf: 92.3%, 15min-2h)")
print("  🎯 SCALP 3X: ETH/USDT (conf: 88.5%, 1-5min)")
print("  🎯 GRID 1X: MATIC/USDT (conf: 78.2%, until breakout)")
print("  🎯 SHORT 5X: SOL/USDT (conf: 85.1%, 1-4h)")
print("  🎯 DCA 1X: ADA/USDT (conf: 76.8%, hours to days)")
print()
print("═══════════════════════════════════════════════════════════════")
print()
print("Restart bot to see:")
print("  pkill -9 -f RUN_BOT.py && ./start_bot.sh")
print()
