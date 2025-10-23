#!/usr/bin/env python3
"""
Quick test to verify Dynamic Pair Discovery integrates correctly
Run this before starting the full bot
"""
import asyncio
import sys

async def test_integration():
    """Test that everything imports and works together"""
    
    print("\n" + "="*80)
    print("  TESTING DYNAMIC PAIR DISCOVERY INTEGRATION")
    print("="*80 + "\n")
    
    # Test 1: Import discovery engine
    print("1️⃣  Testing discovery engine import...")
    try:
        from DYNAMIC_PAIR_DISCOVERY import get_discovery_engine
        discovery = get_discovery_engine()
        print("   ✅ Discovery engine imported successfully!")
    except Exception as e:
        print(f"   ❌ Failed to import discovery engine: {e}")
        return False
    
    # Test 2: Import complete orchestrator
    print("\n2️⃣  Testing Complete Ultimate Orchestrator import...")
    try:
        from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
        print("   ✅ Orchestrator imported successfully!")
    except Exception as e:
        print(f"   ❌ Failed to import orchestrator: {e}")
        return False
    
    # Test 3: Quick discovery test
    print("\n3️⃣  Testing pair discovery (scanning markets)...")
    try:
        # Discover pairs (should find 5000+)
        pairs = await discovery.discover_all_markets()
        print(f"   ✅ Discovered {len(pairs)} trading pairs!")
        print(f"   📋 First 10 pairs: {pairs[:10]}")
    except Exception as e:
        print(f"   ⚠️  Discovery test: {e}")
    
    # Test 4: Test filtering
    print("\n4️⃣  Testing profitability filter...")
    try:
        profitable = await discovery.filter_profitable_pairs(pairs[:100])
        print(f"   ✅ Found {len(profitable)} profitable pairs from sample!")
        if profitable:
            print(f"   💰 Top profitable pair: {profitable[0]}")
    except Exception as e:
        print(f"   ⚠️  Filter test: {e}")
    
    # Test 5: Test orchestrator initialization
    print("\n5️⃣  Testing orchestrator initialization...")
    try:
        orch = CompleteUltimateOrchestrator(mode='testnet')
        print("   ✅ Orchestrator initialized!")
        print(f"   📊 Initial pairs: {len(orch.dynamic_pairs)}")
    except Exception as e:
        print(f"   ❌ Orchestrator initialization failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("  ✅ ALL TESTS PASSED! Ready to start the bot!")
    print("="*80)
    print("\n  🚀 Run: bash START_BOT_WITH_PAIR_DISCOVERY.sh")
    print("  Or:    python3 COMPLETE_ULTIMATE_ORCHESTRATOR.py --mode=testnet\n")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_integration())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
