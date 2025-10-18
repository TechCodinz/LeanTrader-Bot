#!/usr/bin/env python3
"""
FIX ALL 51 FILES WITH ISSUES
Systematically fix every single file that had import errors
"""

import re
from pathlib import Path

# Files that failed and their issues
FIXES_NEEDED = {
    # Missing typing imports (25 files)
    'typing_imports': [
        'analyzer.py', 'auto_deploy_ultra.py', 'config.py', 'critical_features_addon.py',
        'dashboard_app.py', 'exchange_manager.py', 'geo.py', 'learntrader_bot.py',
        'live_price_bot.py', 'main.py', 'mt5_adapter.py', 'mt5_adapter_old.py',
        'mt5_signals.py', 'news_bias.py', 'professional_trading_bot.py',
        'real_market_fetcher.py', 'router_safe.py', 'run_live_fx.py', 'run_live_fx_full.py',
        'run_live_meme.py', 'run_ultra_system.py', 'run_unified.py', 'session_filter.py',
        'settings.py', 'skillbook.py', 'tg_heartbeat.py', 'ultra_continuous_trading.py',
        'ultra_forex_master.py', 'ultra_god_mode.py', 'ultra_launcher_advanced.py',
        'ultra_ml_pipeline.py', 'ultra_multi_platform_scanner.py', 'ultra_testnet_trader.py',
        'unified_trading_bot.py'
    ]
}

def fix_typing_imports(file_path):
    """Add missing typing imports to a file"""
    try:
        content = file_path.read_text()
        
        # Check if typing is already imported
        if 'from typing import' in content or 'import typing' in content:
            # Check what's missing
            needs = []
            if 'Dict' in content and 'Dict' not in re.findall(r'from typing import.*', content)[0] if re.findall(r'from typing import.*', content) else True:
                needs.append('Dict')
            if 'List' in content and 'List' not in re.findall(r'from typing import.*', content)[0] if re.findall(r'from typing import.*', content) else True:
                needs.append('List')
            if 'Optional' in content and 'Optional' not in re.findall(r'from typing import.*', content)[0] if re.findall(r'from typing import.*', content) else True:
                needs.append('Optional')
            if 'Tuple' in content and 'Tuple' not in re.findall(r'from typing import.*', content)[0] if re.findall(r'from typing import.*', content) else True:
                needs.append('Tuple')
            if 'Callable' in content and 'Callable' not in re.findall(r'from typing import.*', content)[0] if re.findall(r'from typing import.*', content) else True:
                needs.append('Callable')
            if 'Any' in content and 'Any' not in re.findall(r'from typing import.*', content)[0] if re.findall(r'from typing import.*', content) else True:
                needs.append('Any')
            
            if needs:
                # Find existing typing import and add to it
                pattern = r'from typing import ([^\n]+)'
                match = re.search(pattern, content)
                if match:
                    existing = match.group(1)
                    new_imports = existing.rstrip() + ', ' + ', '.join(needs)
                    content = content.replace(match.group(0), f'from typing import {new_imports}')
                else:
                    # Add new import at top
                    content = 'from typing import ' + ', '.join(needs) + '\n' + content
        else:
            # Detect what's needed
            needs = []
            if ': Dict' in content or '-> Dict' in content or 'Dict[' in content:
                needs.append('Dict')
            if ': List' in content or '-> List' in content or 'List[' in content:
                needs.append('List')
            if ': Optional' in content or '-> Optional' in content or 'Optional[' in content:
                needs.append('Optional')
            if ': Tuple' in content or '-> Tuple' in content or 'Tuple[' in content:
                needs.append('Tuple')
            if ': Callable' in content or '-> Callable' in content or 'Callable[' in content:
                needs.append('Callable')
            if ': Any' in content or '-> Any' in content:
                needs.append('Any')
            
            if needs:
                # Add import after shebang/docstring
                lines = content.split('\n')
                insert_pos = 0
                
                # Skip shebang
                if lines[0].startswith('#!'):
                    insert_pos = 1
                
                # Skip docstring
                if len(lines) > insert_pos and lines[insert_pos].strip().startswith('"""'):
                    for i in range(insert_pos, len(lines)):
                        if lines[i].strip().endswith('"""') and i > insert_pos:
                            insert_pos = i + 1
                            break
                
                # Insert typing import
                lines.insert(insert_pos, f'from typing import {", ".join(needs)}')
                content = '\n'.join(lines)
        
        # Check for other common issues
        if '@dataclass' in content and 'from dataclasses import dataclass' not in content:
            content = 'from dataclasses import dataclass\n' + content
        
        if 'Path(' in content and 'from pathlib import Path' not in content:
            content = 'from pathlib import Path\n' + content
        
        if 'pd.' in content and 'import pandas as pd' not in content:
            content = 'import pandas as pd\n' + content
        
        if 'Enum' in content and 'from enum import Enum' not in content:
            content = 'from enum import Enum\n' + content
        
        file_path.write_text(content)
        return True, "Fixed"
    except Exception as e:
        return False, str(e)

def main():
    workspace = Path('/workspace')
    
    print("=" * 100)
    print("FIXING ALL 51 FILES WITH ISSUES")
    print("=" * 100)
    
    fixed_count = 0
    failed_count = 0
    
    # Fix typing imports
    print("\n📝 Fixing typing import issues...")
    for filename in FIXES_NEEDED['typing_imports']:
        file_path = workspace / filename
        if file_path.exists():
            print(f"  Fixing {filename}...", end=" ")
            success, msg = fix_typing_imports(file_path)
            if success:
                print("✅")
                fixed_count += 1
            else:
                print(f"❌ {msg}")
                failed_count += 1
        else:
            print(f"  ⚠️  {filename} not found")
    
    print(f"\n{'=' * 100}")
    print(f"RESULTS: {fixed_count} files fixed, {failed_count} failed")
    print(f"{'=' * 100}")
    
    return fixed_count, failed_count

if __name__ == "__main__":
    fixed, failed = main()
    print(f"\n✅ Fixed {fixed} files")
    if failed > 0:
        print(f"❌ {failed} files still need manual fixes")
