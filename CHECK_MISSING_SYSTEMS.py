#!/usr/bin/env python3
"""
Check for missing critical systems like online learner, news, etc.
"""

import os
from pathlib import Path

workspace = Path('/workspace')

# Critical systems to check
critical_systems = {
    'online_learner.py': 'Online learning (SGD, real-time adaptation)',
    'news_service.py': 'News harvesting & sentiment analysis',
    'news_adapter.py': 'News adapter for trading',
    'news_harvest.py': 'News collection system',
    'ADAPTIVE_CONFIDENCE_ENGINE.py': 'Adaptive confidence system',
    'alpha_engines.py': 'Alpha generation engines',
    'awareness.py': 'Market awareness system',
}

print("=" * 80)
print("CHECKING FOR CRITICAL MISSING SYSTEMS")
print("=" * 80)
print()

# Check which exist
existing = []
for filename, description in critical_systems.items():
    filepath = workspace / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        existing.append((filename, description, size_kb))
        print(f"✅ {filename:<40} ({size_kb:.1f} KB)")
        print(f"   {description}")
    else:
        print(f"❌ {filename:<40} NOT FOUND")

print()
print("=" * 80)

# Check orchestrator
orch_file = workspace / 'COMPLETE_ULTIMATE_ORCHESTRATOR.py'
with open(orch_file, 'r') as f:
    orch_content = f.read()

print("INTEGRATION STATUS IN ORCHESTRATOR:")
print("=" * 80)
print()

for filename, description, size_kb in existing:
    module_name = filename.replace('.py', '')
    is_imported = f"from {module_name} import" in orch_content or f"import {module_name}" in orch_content
    
    if is_imported:
        print(f"✅ {module_name:<40} IMPORTED & INTEGRATED")
    else:
        print(f"❌ {module_name:<40} EXISTS BUT NOT INTEGRATED!")

print()
print("=" * 80)
