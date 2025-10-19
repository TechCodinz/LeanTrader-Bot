#!/bin/bash
echo "🔧 Fixing line 518 async syntax error..."
echo ""

# Show the problem area
echo "📋 Current problem (lines 515-525):"
sed -n '515,525p' COMPLETE_ULTIMATE_ORCHESTRATOR.py | cat -n
echo ""

# Backup
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup.async.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"
echo ""

# Fix the async line
python3 << 'PYEOF'
print("🔧 Fixing async statement...")

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # If line is just "async" or "async " with whitespace, merge with next line
    if line.strip() == 'async' or line.strip() == 'async ':
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Get the indentation from the async line
            indent = line[:len(line) - len(line.lstrip())]
            # Merge: async + next line content
            merged = indent + 'async ' + next_line.lstrip()
            fixed_lines.append(merged)
            print(f"   Fixed line {i+1}: merged 'async' with next line")
            i += 2
            continue
    
    fixed_lines.append(line)
    i += 1

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ File updated")
PYEOF

echo ""
echo "📋 After fix (lines 515-525):"
sed -n '515,525p' COMPLETE_ULTIMATE_ORCHESTRATOR.py | cat -n
echo ""

echo "🧪 Testing syntax..."
python3 -m py_compile COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 🎉 🎉 SUCCESS! ALL SYNTAX ERRORS FIXED! 🎉 🎉 🎉"
    echo ""
    echo "✅ Ready to start the bot:"
    echo "   pkill -f RUN_BOT.py && sleep 2 && nohup python3 RUN_BOT.py --testnet --auto-confirm > bot.log 2>&1 &"
else
    echo ""
    echo "⚠️  Still has errors. Checking..."
    python3 << 'PYEOF'
try:
    compile(open('COMPLETE_ULTIMATE_ORCHESTRATOR.py').read(), 'COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'exec')
except SyntaxError as e:
    print(f"❌ Error at line {e.lineno}: {e.msg}")
    with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
        lines = f.readlines()
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 3)
        print(f"\nContext:")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"{marker}{i+1:4d} | {lines[i]}", end='')
PYEOF
fi
