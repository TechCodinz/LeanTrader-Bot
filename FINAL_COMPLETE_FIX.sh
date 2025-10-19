#!/bin/bash
################################################################################
# COMPLETE BOT FIX - ONE SCRIPT TO FIX EVERYTHING
# Professional systematic solution - Run this ONCE on your VPS
################################################################################

set -e  # Exit on any error

echo "================================================================================"
echo "🚀 COMPLETE BOT FIX - PROFESSIONAL SOLUTION"
echo "================================================================================"
echo ""

cd ~/trading_bot

# Create backup
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r *.py "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backup created: $BACKUP_DIR"
echo ""

################################################################################
# FIX 1: Add typing imports to ALL files that need them
################################################################################
echo "📋 FIX 1: Adding typing imports to all Ultra systems..."

for file in tools/market_data.py tools/ultra_trainer.py ultra_ml_pipeline.py \
            ultra_arbitrage_engine.py ultra_scalping_engine.py \
            ultra_quantum_intelligence.py ultra_swarm_consciousness.py \
            ultra_fluid_mechanics.py; do
    
    if [ -f "$file" ]; then
        # Check if file needs typing but doesn't have it
        if grep -q "Optional\|Dict\[" "$file" && ! grep -q "^from typing import" "$file"; then
            # Add typing import after first import line
            sed -i '0,/^import\|^from/a from typing import Dict, List, Tuple, Optional, Any, Union' "$file"
            echo "   ✅ Added typing to $file"
        elif grep -q "^from typing import" "$file" && ! grep -q "Optional.*Dict" "$file"; then
            # Update existing import
            sed -i 's/^from typing import.*/from typing import Dict, List, Tuple, Optional, Any, Union/' "$file"
            echo "   ✅ Updated typing in $file"
        fi
    fi
done

echo ""

################################################################################
# FIX 2: Wrap ALL Ultra system initializations in COMPLETE_ULTIMATE_ORCHESTRATOR
################################################################################
echo "📋 FIX 2: Fixing system initializations with error handling..."

python3 << 'PYEOF'
import re

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this is a Ultra system initialization line
    if re.match(r'(\s+)self\.(ultra_\w+|trade_planner|models_450|trader_core|unified_trading) = \w+\(\)\s*$', line):
        indent = len(line) - len(line.lstrip())
        var_name = re.search(r'self\.(\w+)', line).group(1)
        class_name = re.search(r'= (\w+)\(\)', line).group(1)
        
        # Wrap in try-except
        new_lines.append(' ' * indent + 'try:\n')
        new_lines.append(line)
        new_lines.append(' ' * (indent + 4) + f"self.advanced_orchestrators['{var_name}'] = self.{var_name}\n")
        new_lines.append(' ' * (indent + 4) + f"logger.info('✅ {class_name} initialized')\n")
        new_lines.append(' ' * indent + 'except Exception as e:\n')
        new_lines.append(' ' * (indent + 4) + f"logger.warning(f'⚠️  {class_name} failed: {{e}}')\n")
        new_lines.append(' ' * (indent + 4) + f"self.{var_name} = None\n")
        i += 1
        continue
    
    new_lines.append(line)
    i += 1

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Added error handling to all system initializations")
PYEOF

echo ""

################################################################################
# FIX 3: Test syntax
################################################################################
echo "📋 FIX 3: Testing syntax..."

python3 -m py_compile COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Syntax check passed!"
else
    echo "❌ Syntax error - restoring backup"
    cp "$BACKUP_DIR/COMPLETE_ULTIMATE_ORCHESTRATOR.py" .
    exit 1
fi

echo ""

################################################################################
# FIX 4: Restart bot
################################################################################
echo "📋 FIX 4: Restarting bot..."

pkill -f RUN_BOT.py || true
sleep 3

nohup python3 RUN_BOT.py --testnet --auto-confirm > bot.log 2>&1 &

echo "⏳ Waiting 20 seconds for initialization..."
sleep 20

################################################################################
# CHECK STATUS
################################################################################
echo ""
echo "================================================================================"
echo "📊 BOT STATUS"
echo "================================================================================"

if ps aux | grep "RUN_BOT.py" | grep -v grep > /dev/null; then
    echo "✅ Bot is RUNNING"
    echo ""
    ps aux | grep RUN_BOT.py | grep -v grep | awk '{print "PID: "$2" | CPU: "$3"% | RAM: "$4"%"}'
    echo ""
    echo "📋 Systems Status:"
    tail -300 bot.log | grep -E "✅.*available|✅.*initialized|⚠️.*not available|ALL.*SYSTEMS|WIRING|RUNNING" | tail -30
    echo ""
    echo "================================================================================"
    echo "✅ COMPLETE! Bot is running with all available systems."
    echo "================================================================================"
    echo ""
    echo "📝 Monitor with: tail -f bot.log"
    echo "🛑 Stop with: pkill -f RUN_BOT.py"
else
    echo "❌ Bot failed to start"
    echo ""
    echo "Error details:"
    tail -50 bot.log
    exit 1
fi
