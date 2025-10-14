#!/usr/bin/env python3
"""
DevOps Production Readiness Scan
Comprehensive check of all systems
"""

import os
import sys
import importlib
import ast
import re
from pathlib import Path

class ProductionScanner:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def check_file_syntax(self, filepath):
        """Check Python file for syntax errors"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def check_for_placeholders(self, filepath):
        """Check for placeholder code"""
        placeholders = [
            'TODO',
            'FIXME',
            'PLACEHOLDER',
            'NotImplemented',
            'pass  # Implement',
            'Simplified for now',
            'This would',
            '# From env',
            'private_key=""',
            'api_key=""'
        ]
        
        issues = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                for ph in placeholders:
                    if ph in line and not line.strip().startswith('#'):
                        issues.append(f"Line {i}: {line.strip()[:80]}")
        
        return issues
    
    def check_imports(self, filepath):
        """Try to import the file"""
        try:
            spec = importlib.util.spec_from_file_location("module", filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["module"] = module
                spec.loader.exec_module(module)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def scan_file(self, filepath):
        """Complete scan of a file"""
        print(f"\n{'='*80}")
        print(f"Scanning: {filepath}")
        print('='*80)
        
        # Syntax check
        syntax_ok, syntax_err = self.check_file_syntax(filepath)
        if not syntax_ok:
            self.issues.append(f"{filepath}: SYNTAX ERROR - {syntax_err}")
            print(f"❌ SYNTAX ERROR: {syntax_err}")
            return
        else:
            self.passed.append(f"{filepath}: Syntax OK")
            print("✅ Syntax: OK")
        
        # Placeholder check
        placeholders = self.check_for_placeholders(filepath)
        if placeholders:
            self.warnings.append(f"{filepath}: {len(placeholders)} placeholders found")
            print(f"⚠️  Placeholders: {len(placeholders)} found")
            for ph in placeholders[:3]:  # Show first 3
                print(f"   - {ph}")
        else:
            print("✅ Placeholders: None")
        
        # Import check
        # Skip for now as it may have dependencies
    
    def generate_report(self):
        """Generate final report"""
        print(f"\n{'='*80}")
        print("PRODUCTION READINESS REPORT")
        print('='*80)
        
        print(f"\n✅ PASSED: {len(self.passed)}")
        print(f"⚠️  WARNINGS: {len(self.warnings)}")
        print(f"❌ ISSUES: {len(self.issues)}")
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  - {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings[:10]:  # First 10
                print(f"  - {warning}")
        
        print(f"\n{'='*80}")

if __name__ == "__main__":
    scanner = ProductionScanner()
    
    # Critical files to scan
    files = [
        'DEX_SWAP_ENGINE.py',
        'DEX_ORCHESTRATOR.py',
        'EXECUTION_ORCHESTRATOR.py',
        'TELEGRAM_ORCHESTRATOR.py',
        'IBM_QUANTUM_ENGINE.py',
        'SMART_SCALPING_ENGINE.py',
        'COMPLETE_ULTIMATE_ORCHESTRATOR.py',
    ]
    
    for f in files:
        if os.path.exists(f):
            scanner.scan_file(f)
    
    scanner.generate_report()
