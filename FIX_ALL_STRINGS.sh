#!/bin/bash
# Fix ALL broken string literals in COMPLETE_ULTIMATE_ORCHESTRATOR.py
# Run this in ~/trading_bot directory

echo "🔧 Comprehensive fix for all broken strings..."
echo ""

# Backup
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup.allstrings.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# Use Python to fix all broken strings
python3 << 'PYEOF'
import re

print("📝 Reading file...")
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"   Found {len(lines)} lines")

fixed_lines = []
i = 0
fixes_made = 0

while i < len(lines):
    line = lines[i]
    
    # Check if line ends with an unterminated string (ends with ' or " or f' or f")
    # Pattern: ends with logger.info(' or logger.info(f' or logger.info(" without closing
    if re.search(r"logger\.(info|warning|error|debug)\s*\(\s*[f]?['\"]$", line.strip()):
        # This line has an unclosed string
        # Next line should be the continuation
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Merge the lines
            merged = line.rstrip() + next_line
            fixed_lines.append(merged + '\n')
            fixes_made += 1
            print(f"   Fixed broken string at line {i+1}")
            i += 2  # Skip next line since we merged it
            continue
    
    fixed_lines.append(line)
    i += 1

print(f"✅ Made {fixes_made} fixes")

# Write back
with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ File updated")
PYEOF

echo ""
echo "🧪 Testing syntax..."
python3 -m py_compile COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 🎉 🎉 ALL SYNTAX ERRORS FIXED! 🎉 🎉 🎉"
    echo ""
    echo "Ready to start bot!"
else
    echo ""
    echo "⚠️  Checking for remaining errors..."
    python3 << 'PYEOF'
try:
    compile(open('COMPLETE_ULTIMATE_ORCHESTRATOR.py').read(), 'COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'exec')
    print("✅ No syntax errors!")
except SyntaxError as e:
    print(f"❌ Error at line {e.lineno}: {e.msg}")
    with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
        lines = f.readlines()
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 2)
        print(f"\nContext around line {e.lineno}:")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"{marker}{i+1:4d} | {lines[i]}", end='')
PYEOF
fi
