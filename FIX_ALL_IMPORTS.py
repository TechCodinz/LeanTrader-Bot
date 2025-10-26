#!/usr/bin/env python3
"""Make all imports optional to handle missing classes"""

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r') as f:
    lines = f.readlines()

output = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if it's a problematic import line
    if line.startswith('from ') and 'import' in line:
        # Check if already wrapped in try/except
        if i > 0 and 'try:' in lines[i-1]:
            output.append(line)
            i += 1
            continue
        
        # Skip if it's a safe import
        safe_imports = ['typing', 'datetime', 'pathlib', 'asyncio', 'logging', 'collections']
        if any(safe in line for safe in safe_imports):
            output.append(line)
            i += 1
            continue
        
        # Check if it's in the ultra/deep systems section (lines 150-310 roughly)
        if 150 <= i <= 310 and ('ultra_' in line or 'router' in line.lower() or 'nobel' in line.lower() or '_ENGINE' in line):
            # Make it optional
            module_name = line.split('from ')[1].split(' import')[0].strip()
            import_parts = line.split(' import ')[1].strip()
            
            output.append(f'try:\n')
            output.append(line)
            output.append(f'except (ImportError, ModuleNotFoundError, AttributeError):\n')
            
            # Create None assignments
            if ',' in import_parts:
                classes = [c.strip().split(' as ')[0] for c in import_parts.split(',')]
                for cls in classes:
                    output.append(f'    {cls} = None\n')
            else:
                cls = import_parts.split(' as ')[0].strip()
                output.append(f'    {cls} = None\n')
            output.append('\n')
            i += 1
            continue
    
    output.append(line)
    i += 1

with open('COMPLETE_ULTIMATE_ORCHESTRATOR_fixed.py', 'w') as f:
    f.writelines(output)

print("✅ Created COMPLETE_ULTIMATE_ORCHESTRATOR_fixed.py with optional imports")
print("Replacing original...")

import shutil
shutil.move('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup_before_optional')
shutil.move('COMPLETE_ULTIMATE_ORCHESTRATOR_fixed.py', 'COMPLETE_ULTIMATE_ORCHESTRATOR.py')

print("✅ Done!")

