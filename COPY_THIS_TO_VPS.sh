#!/bin/bash
#
# INTEGRATION HELPER - Run this on your VPS
# This shows you what needs to be updated
#

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     DYNAMIC PAIR DISCOVERY - VPS INTEGRATION HELPER          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot || exit 1

echo "1️⃣  Checking current files..."
echo ""

# Check if files exist
if [ -f "DYNAMIC_PAIR_DISCOVERY.py" ]; then
    echo "✅ DYNAMIC_PAIR_DISCOVERY.py exists (already created)"
else
    echo "❌ DYNAMIC_PAIR_DISCOVERY.py NOT FOUND!"
    echo "   You need to create it first (see previous agent messages)"
    exit 1
fi

if [ -f "COMPLETE_ULTIMATE_ORCHESTRATOR.py" ]; then
    echo "✅ COMPLETE_ULTIMATE_ORCHESTRATOR.py exists"
else
    echo "❌ COMPLETE_ULTIMATE_ORCHESTRATOR.py NOT FOUND!"
    echo "   File needs to be created or downloaded"
    exit 1
fi

echo ""
echo "2️⃣  Creating test file..."
cat > SIMPLE_PAIR_DISCOVERY_TEST.py << 'TESTEOF'
#!/usr/bin/env python3
"""Simple test - just verify pair discovery works"""
import asyncio

async def test():
    print("\n" + "="*80)
    print("  TESTING PAIR DISCOVERY (SIMPLIFIED)")
    print("="*80 + "\n")
    
    print("1️⃣  Importing discovery engine...")
    from DYNAMIC_PAIR_DISCOVERY import get_discovery_engine
    print("   ✅ Import successful!")
    
    print("\n2️⃣  Discovering markets across ALL exchanges...")
    engine = get_discovery_engine()
    pairs = await engine.discover_all_markets()
    print(f"   ✅ Found {len(pairs)} total pairs!")
    
    print("\n3️⃣  Filtering for profitable pairs...")
    profitable = await engine.filter_profitable_pairs(pairs[:200])
    print(f"   ✅ Found {len(profitable)} profitable pairs!")
    
    if profitable:
        print(f"\n   💰 Top 5 profitable pairs:")
        for i, pair in enumerate(profitable[:5], 1):
            print(f"      {i}. {pair}")
    
    print("\n" + "="*80)
    print("  ✅ PAIR DISCOVERY WORKING PERFECTLY!")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    asyncio.run(test())
TESTEOF

chmod +x SIMPLE_PAIR_DISCOVERY_TEST.py
echo "✅ Test file created: SIMPLE_PAIR_DISCOVERY_TEST.py"

echo ""
echo "3️⃣  Running quick test..."
echo ""
python3 SIMPLE_PAIR_DISCOVERY_TEST.py

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  NEXT STEP: Update your orchestrator"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "The COMPLETE_ULTIMATE_ORCHESTRATOR.py on your VPS needs to be"
echo "updated with the pair discovery integration."
echo ""
echo "You have 2 options:"
echo ""
echo "OPTION 1 (Recommended): Copy the updated file from workspace"
echo "  - The file is in /workspace/COMPLETE_ULTIMATE_ORCHESTRATOR.py"
echo "  - Download it and upload to your VPS"
echo "  - Replace the existing file"
echo ""
echo "OPTION 2: Manually add the integration"
echo "  - See INTEGRATION_COMPLETE_README.md for code changes"
echo "  - Add the 3 key sections marked in the guide"
echo ""
echo "After updating, run:"
echo "  python3 COMPLETE_ULTIMATE_ORCHESTRATOR.py --mode=testnet"
echo ""
echo "═══════════════════════════════════════════════════════════════"
