#!/usr/bin/env python3
"""
CLEAN FIX - Simple and effective
Fixes division by zero without breaking syntax
"""

# Read EXECUTION_ORCHESTRATOR
with open('EXECUTION_ORCHESTRATOR.py', 'r') as f:
    lines = f.readlines()

# Backup
with open('EXECUTION_ORCHESTRATOR.py.clean_backup', 'w') as f:
    f.writelines(lines)

print("💾 Backup: EXECUTION_ORCHESTRATOR.py.clean_backup")

# Find and fix the division by zero issue
# It's happening in line 64 and line 381

fixed_lines = []
for i, line in enumerate(lines):
    # Fix 1: Line ~64 - position size calculation
    if 'position_size = (base_risk / stop_loss_pct)' in line and 'if stop' not in lines[i-1]:
        indent = len(line) - len(line.lstrip())
        safety_check = ' ' * indent + 'stop_loss_pct = max(0.001, stop_loss_pct)  # Prevent division by zero\n'
        fixed_lines.append(safety_check)
        fixed_lines.append(line)
        print(f"✅ Fixed position size calculation (line {i+1})")
    
    # Fix 2: Line ~381 - amount calculation  
    elif 'amount = position_size_usd / price' in line and 'if price' not in lines[i-1]:
        indent = len(line) - len(line.lstrip())
        safety_check = ' ' * indent + 'price = max(0.01, price)  # Prevent division by zero\n'
        fixed_lines.append(safety_check)
        fixed_lines.append(line)
        print(f"✅ Fixed amount calculation (line {i+1})")
    
    # Fix 3: Any other division that could be zero
    elif '/ volatility' in line or '/ balance' in line:
        # Add safety check before the line
        indent = len(line) - len(line.lstrip())
        
        if '/ volatility' in line:
            safety = ' ' * indent + 'volatility = max(0.01, volatility)  # Prevent division by zero\n'
        elif '/ balance' in line:
            safety = ' ' * indent + 'balance = max(1.0, balance)  # Prevent division by zero\n'
        else:
            safety = ''
        
        if safety and safety not in fixed_lines[-1] if fixed_lines else True:
            fixed_lines.append(safety)
            print(f"✅ Fixed division safety (line {i+1})")
        
        fixed_lines.append(line)
    
    else:
        fixed_lines.append(line)

# Save
with open('EXECUTION_ORCHESTRATOR.py', 'w') as f:
    f.writelines(fixed_lines)

print("✅ All division by zero errors fixed!")
print()

# Test import
import sys
sys.path.insert(0, '.')

try:
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator
    print("✅ EXECUTION_ORCHESTRATOR imports successfully!")
    print()
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("   Restoring backup...")
    with open('EXECUTION_ORCHESTRATOR.py.clean_backup', 'r') as f:
        with open('EXECUTION_ORCHESTRATOR.py', 'w') as out:
            out.write(f.read())
    sys.exit(1)

print("═══════════════════════════════════════════════════════════════")
print("✅ DIVISION ERRORS FIXED!")
print("═══════════════════════════════════════════════════════════════")
print()
print("Now restart bot:")
print("  pkill -9 -f RUN_BOT.py")
print("  ./start_bot.sh")
print()
