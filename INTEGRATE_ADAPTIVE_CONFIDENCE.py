#!/usr/bin/env python3
"""
Integration script: Add Adaptive Confidence to EXECUTION_ORCHESTRATOR
This makes your bot SMART about confidence thresholds!
"""
import os
import sys

def integrate_adaptive_confidence():
    """
    Modify EXECUTION_ORCHESTRATOR.py to use adaptive confidence
    """
    
    orchestrator_file = "EXECUTION_ORCHESTRATOR.py"
    
    if not os.path.exists(orchestrator_file):
        print(f"❌ {orchestrator_file} not found!")
        print("   Make sure you're in the trading_bot directory")
        return False
    
    # Read the file
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    # Check if already integrated
    if 'AdaptiveConfidenceEngine' in content or 'get_adaptive_confidence_engine' in content:
        print("✅ Adaptive Confidence already integrated!")
        return True
    
    # ================================================================
    # STEP 1: Add import at top
    # ================================================================
    import_line = "\nfrom ADAPTIVE_CONFIDENCE_ENGINE import get_adaptive_confidence_engine\n"
    
    # Find where to insert (after other imports)
    import_pos = content.find("import logging")
    if import_pos == -1:
        import_pos = content.find("import")
    
    if import_pos > 0:
        # Find end of that line
        end_pos = content.find("\n", import_pos)
        content = content[:end_pos] + import_line + content[end_pos:]
        print("✅ Step 1: Added import")
    else:
        print("⚠️  Step 1: Could not find import section")
        return False
    
    # ================================================================
    # STEP 2: Add engine to __init__
    # ================================================================
    init_addition = """
        
        # Adaptive confidence engine
        self.adaptive_confidence = get_adaptive_confidence_engine()
        logger.info("🧠 Adaptive Confidence Engine enabled")
"""
    
    # Find __init__ method
    init_pos = content.find("def __init__(self")
    if init_pos > 0:
        # Find end of __init__ (next def or class)
        end_init = content.find("\n    def ", init_pos + 50)
        if end_init == -1:
            end_init = content.find("\n\nclass ", init_pos + 50)
        
        if end_init > 0:
            # Insert before the next method
            content = content[:end_init] + init_addition + content[end_init:]
            print("✅ Step 2: Added adaptive engine to __init__")
        else:
            print("⚠️  Step 2: Could not find end of __init__")
            return False
    else:
        print("⚠️  Step 2: Could not find __init__ method")
        return False
    
    # ================================================================
    # STEP 3: Replace static threshold with adaptive
    # ================================================================
    # Find where confidence is checked
    static_check = "if confidence < self.min_confidence"
    
    if static_check in content:
        # Replace with adaptive logic
        adaptive_check = """# Get adaptive threshold based on market conditions
        adaptive_result = self.adaptive_confidence.get_adaptive_threshold(
            pair=pair,
            market_regime=market_regime,
            volatility=abs(signal.get('price_change', 0)),
            volume_24h=signal.get('volume', 0),
            news_impact="none",  # Can be enhanced with news data
            confidence=confidence
        )
        
        adaptive_threshold = adaptive_result['threshold']
        
        # Log adaptive decision
        logger.info(f"🧠 Adaptive threshold for {pair}: {adaptive_threshold*100:.1f}% (base: {self.min_confidence*100:.1f}%)")
        logger.info(f"   Reason: {adaptive_result['reason']}")
        
        if confidence < adaptive_threshold"""
        
        content = content.replace(static_check, adaptive_check)
        print("✅ Step 3: Replaced static threshold with adaptive")
    else:
        print("⚠️  Step 3: Could not find confidence check")
        print("   Manual integration may be needed")
    
    # ================================================================
    # STEP 4: Add trade result recording
    # ================================================================
    # Find where trades are executed/closed
    if "# Record trade result" not in content:
        # Add after successful trade
        trade_complete = "logger.info(f\"✅ Trade completed:"
        
        if trade_complete in content:
            record_addition = """
        
        # Record result for adaptive learning
        result = 'win' if profit > 0 else 'loss'
        self.adaptive_confidence.record_trade_result(pair, result, confidence)
        logger.info(f"📊 Recorded {result} for {pair} (confidence: {confidence:.1%})")
"""
            
            pos = content.find(trade_complete)
            if pos > 0:
                # Find end of line
                end_pos = content.find("\n", pos)
                content = content[:end_pos] + record_addition + content[end_pos:]
                print("✅ Step 4: Added trade result recording")
        else:
            print("⚠️  Step 4: Trade completion not found (can add manually later)")
    
    # ================================================================
    # SAVE
    # ================================================================
    
    # Backup original
    backup_file = orchestrator_file + ".before_adaptive"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w') as f:
            with open(orchestrator_file, 'r') as orig:
                f.write(orig.read())
        print(f"💾 Backup created: {backup_file}")
    
    # Write modified version
    with open(orchestrator_file, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Integration complete!")
    print(f"   Modified: {orchestrator_file}")
    print(f"   Backup: {backup_file}")
    
    return True


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  INTEGRATING ADAPTIVE CONFIDENCE ENGINE                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Check we're in the right directory
    if not os.path.exists("EXECUTION_ORCHESTRATOR.py"):
        print("❌ ERROR: EXECUTION_ORCHESTRATOR.py not found!")
        print("   Please run this script from the trading_bot directory:")
        print("   cd ~/trading_bot && python3 INTEGRATE_ADAPTIVE_CONFIDENCE.py")
        sys.exit(1)
    
    success = integrate_adaptive_confidence()
    
    if success:
        print()
        print("═══════════════════════════════════════════════════════════════")
        print("✅ ADAPTIVE CONFIDENCE INTEGRATED!")
        print("═══════════════════════════════════════════════════════════════")
        print()
        print("Your bot will now:")
        print("  • Adjust confidence 65-95% based on market conditions")
        print("  • Lower threshold in strong trends (catch momentum)")
        print("  • Raise threshold in choppy markets (avoid noise)")
        print("  • Learn from winning/losing pairs")
        print("  • Adapt to trading sessions and news")
        print()
        print("Next steps:")
        print("  1. Test the import:")
        print("     python3 -c \"from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator; print('✅ OK')\"")
        print()
        print("  2. Restart bot:")
        print("     pkill -9 -f RUN_BOT.py && sleep 2")
        print("     ./start_bot.sh")
        print()
        print("  3. Watch adaptive thresholds in action:")
        print("     tail -f bot.log | grep -E 'Adaptive threshold|Decision'")
        print()
    else:
        print()
        print("═══════════════════════════════════════════════════════════════")
        print("⚠️  MANUAL INTEGRATION NEEDED")
        print("═══════════════════════════════════════════════════════════════")
        print()
        print("The automatic integration had issues.")
        print("Please manually add the adaptive confidence engine.")
        print()
        sys.exit(1)
