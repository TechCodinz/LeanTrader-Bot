#!/bin/bash
# Quick Start Script for Unified Trading System

echo "╔══════════════════════════════════════════════════════╗"
echo "║     UNIFIED TRADING SYSTEM - QUICK START            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python3 --version

# Run integration test
echo ""
echo "🧪 Running integration tests..."
python3 test_unified_system.py

echo ""
echo "✅ Integration complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Install dependencies: pip install -r complete_requirements.txt"
echo "  2. Configure .env file with your API keys"
echo "  3. Test: TRADING_MODE=paper python3 unified_trading_system.py"
echo "  4. Monitor: tail -f unified_trading_system.log"
echo ""
echo "📚 Documentation:"
echo "  - INTEGRATION_ARCHITECTURE.md - System design"
echo "  - DEPLOYMENT_GUIDE.md - Full deployment guide"
echo "  - INTEGRATION_COMPLETE.md - Integration summary"
echo ""
