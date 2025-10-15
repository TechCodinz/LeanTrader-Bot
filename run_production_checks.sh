#!/bin/bash

echo "🔍 PRODUCTION READINESS CHECK"
echo "=============================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
if [[ "$python_version" > "3.8" ]]; then
    echo -e "${GREEN}✅ Python $python_version${NC}"
else
    echo -e "${RED}❌ Python 3.8+ required${NC}"
    exit 1
fi

# Check critical files exist
echo ""
echo "2. Checking critical files..."
critical_files=(
    "COMPLETE_ULTIMATE_ORCHESTRATOR.py"
    "DEX_ORCHESTRATOR.py"
    "DEX_SWAP_ENGINE.py"
    "EXECUTION_ORCHESTRATOR.py"
    ".env.example"
    "requirements.txt"
)

for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file missing${NC}"
    fi
done

# Check .env file
echo ""
echo "3. Checking environment configuration..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo "   Run: cp .env.example .env"
fi

# Test imports
echo ""
echo "4. Testing critical imports..."
python3 << 'PYEOF'
import sys
try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("✅ COMPLETE_ULTIMATE_ORCHESTRATOR")
except Exception as e:
    print(f"❌ COMPLETE_ULTIMATE_ORCHESTRATOR: {e}")
    sys.exit(1)

try:
    from DEX_SWAP_ENGINE import DEXSwapEngine
    print("✅ DEX_SWAP_ENGINE")
except Exception as e:
    print(f"❌ DEX_SWAP_ENGINE: {e}")
    sys.exit(1)

try:
    from DEX_ORCHESTRATOR import DEXOrchestrator
    print("✅ DEX_ORCHESTRATOR")
except Exception as e:
    print(f"❌ DEX_ORCHESTRATOR: {e}")
    sys.exit(1)

print("\n✅ All critical imports successful!")
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================="
    echo "✅ PRODUCTION CHECKS PASSED"
    echo "==============================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.example to .env"
    echo "2. Add your API keys to .env"
    echo "3. Test on testnet:"
    echo "   python3 COMPLETE_ULTIMATE_ORCHESTRATOR.py --mode testnet"
    echo ""
else
    echo ""
    echo -e "${RED}=============================="
    echo "❌ PRODUCTION CHECKS FAILED"
    echo "==============================${NC}"
    exit 1
fi
