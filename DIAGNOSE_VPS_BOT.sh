#!/bin/bash
# VPS Bot Diagnostic Script
# Run this on your VPS at root@vmi2817884:~/trading_bot

echo "=========================================="
echo "🔍 TRADING BOT DIAGNOSTIC"
echo "=========================================="
echo ""

echo "1️⃣  Current Location:"
pwd
echo ""

echo "2️⃣  Files in current directory:"
ls -lh *.py | head -10
echo ""

echo "3️⃣  Check COMPLETE_ULTIMATE_ORCHESTRATOR.py line 401:"
if [ -f COMPLETE_ULTIMATE_ORCHESTRATOR.py ]; then
    echo "   File exists, checking syntax..."
    python3 -m py_compile COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>&1
    echo ""
    echo "   Lines 395-410:"
    sed -n '395,410p' COMPLETE_ULTIMATE_ORCHESTRATOR.py | cat -n
else
    echo "   ❌ File not found in current directory"
fi
echo ""

echo "4️⃣  Check for hidden characters around line 401:"
if [ -f COMPLETE_ULTIMATE_ORCHESTRATOR.py ]; then
    sed -n '401p' COMPLETE_ULTIMATE_ORCHESTRATOR.py | od -c | head -3
fi
echo ""

echo "5️⃣  Test Python import:"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("✅ Import successful!")
except SyntaxError as e:
    print(f"❌ Syntax Error:")
    print(f"   File: {e.filename}")
    print(f"   Line: {e.lineno}")
    print(f"   Column: {e.offset}")
    print(f"   Message: {e.msg}")
    if e.text:
        print(f"   Problem text: {repr(e.text)}")
except Exception as e:
    print(f"❌ Other error: {type(e).__name__}: {e}")
PYEOF
echo ""

echo "6️⃣  Check for duplicate/conflicting files:"
find . -name "COMPLETE_ULTIMATE_ORCHESTRATOR.py" 2>/dev/null
echo ""

echo "7️⃣  Check Python version:"
python3 --version
echo ""

echo "8️⃣  File encoding check:"
if [ -f COMPLETE_ULTIMATE_ORCHESTRATOR.py ]; then
    file COMPLETE_ULTIMATE_ORCHESTRATOR.py
fi
echo ""

echo "=========================================="
echo "✅ Diagnostic Complete"
echo "=========================================="
echo ""
echo "📋 Please copy ALL output above and send it back"
