#!/bin/bash
# Fix syntax error on VPS at line 401
# Run this in ~/trading_bot directory

echo "🔧 Fixing COMPLETE_ULTIMATE_ORCHESTRATOR.py syntax error..."
echo ""

# Backup the file first
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# Fix the broken string literal at line 401
# The problem is: logger.info(' split across two lines
# Should be: logger.info('🚀 Initializing Ultra Systems...')

python3 << 'PYEOF'
import re

# Read the file
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix pattern: logger.info('NEWLINE🚀 text...')
# Replace with: logger.info('🚀 text...')
content = re.sub(
    r"logger\.info\('\s*\n\s*🚀 Initializing Ultra Systems\.\.\.'\)",
    "logger.info('🚀 Initializing Ultra Systems...')",
    content
)

# Also fix any similar broken strings
content = re.sub(
    r"logger\.info\('\s*\n",
    "logger.info('",
    content
)

# Write back
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed broken string literals")
PYEOF

echo ""
echo "🧪 Testing the fix..."
python3 -m py_compile COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ✅ ✅ SYNTAX ERROR FIXED! ✅ ✅ ✅"
    echo ""
    echo "Now try starting the bot:"
    echo "  python3 RUN_BOT.py --testnet --auto-confirm"
else
    echo ""
    echo "❌ Still has errors. Checking line 401 again:"
    sed -n '398,405p' COMPLETE_ULTIMATE_ORCHESTRATOR.py | cat -n
    echo ""
    echo "You can restore backup with:"
    echo "  cp COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup.* COMPLETE_ULTIMATE_ORCHESTRATOR.py"
fi
