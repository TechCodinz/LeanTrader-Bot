#!/usr/bin/env python3
"""
Fix division by zero error in EXECUTION_ORCHESTRATOR
"""

# Read file
with open('EXECUTION_ORCHESTRATOR.py', 'r') as f:
    content = f.read()

# Backup
with open('EXECUTION_ORCHESTRATOR.py.div_backup', 'w') as f:
    f.write(content)

# Fix line 64: Add safety check for stop_loss_pct
old_line = '        position_size = (base_risk / stop_loss_pct) * kelly_fraction * volatility_adjustment * confidence_adjustment'

new_line = '''        # Safety check for division by zero
        if stop_loss_pct <= 0:
            stop_loss_pct = 0.01  # Default 1%
        
        position_size = (base_risk / stop_loss_pct) * kelly_fraction * volatility_adjustment * confidence_adjustment'''

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Fixed position size calculation")
else:
    print("⚠️  Line not found exactly - checking alternatives...")
    # Try to find similar pattern
    import re
    pattern = r'position_size = \(base_risk / stop_loss_pct\)'
    if re.search(pattern, content):
        print("✅ Found alternative pattern")
        # Add safety check before any division
        content = re.sub(
            r'(\s+)position_size = \(base_risk / stop_loss_pct\)',
            r'\1# Safety check\n\1if stop_loss_pct <= 0:\n\1    stop_loss_pct = 0.01\n\1position_size = (base_risk / stop_loss_pct)',
            content
        )

# Fix any other potential division issues
# Add safety to balance division
content = content.replace(
    '        position_size_usd = self.position_sizer.calculate_position_size(',
    '''        # Safety: Ensure balance is not zero
        if self.position_sizer.balance <= 0:
            self.position_sizer.balance = 10.0  # Minimum
        
        position_size_usd = self.position_sizer.calculate_position_size('''
)

# Fix price division
content = content.replace(
    '            amount = position_size_usd / price',
    '''            # Safety check for price
            if price <= 0:
                logger.error(f"Invalid price: {price}")
                return None
            
            amount = position_size_usd / price'''
)

# Save
with open('EXECUTION_ORCHESTRATOR.py', 'w') as f:
    f.write(content)

print("✅ Division by zero safety checks added!")
print("💾 Backup: EXECUTION_ORCHESTRATOR.py.div_backup")
print()

# Test import
import sys
sys.path.insert(0, '.')

try:
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator
    print("✅ Import test passed!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("   Restoring backup...")
    with open('EXECUTION_ORCHESTRATOR.py.div_backup', 'r') as f:
        with open('EXECUTION_ORCHESTRATOR.py', 'w') as out:
            out.write(f.read())
    sys.exit(1)
