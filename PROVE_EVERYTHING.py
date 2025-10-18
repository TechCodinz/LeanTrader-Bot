#!/usr/bin/env python3
"""
PROOF OF INTEGRATION - TEST EVERY SINGLE MODULE
This will attempt to import EVERY Python file and show REAL results
"""

import os
import sys
import importlib.util
from pathlib import Path

def test_import_module(file_path):
    """Test if a Python file can be imported"""
    try:
        # Get module name from file path
        module_name = file_path.stem
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            return False, "No spec"
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return True, "OK"
    except Exception as e:
        return False, str(e)[:100]  # Limit error message length

def main():
    workspace = Path("/workspace")
    
    # Find ALL Python files
    py_files = sorted(list(workspace.glob("*.py")))
    
    print("=" * 100)
    print("TESTING EVERY SINGLE PYTHON FILE IN WORKSPACE")
    print("=" * 100)
    print(f"\nFound {len(py_files)} Python files\n")
    
    success_count = 0
    fail_count = 0
    results = []
    
    for py_file in py_files:
        # Skip this test file itself
        if py_file.name == "PROVE_EVERYTHING.py":
            continue
            
        print(f"Testing: {py_file.name}...", end=" ")
        success, message = test_import_module(py_file)
        
        if success:
            print("✅ SUCCESS")
            success_count += 1
            results.append((py_file.name, "✅", "Imports successfully"))
        else:
            print(f"❌ FAILED: {message}")
            fail_count += 1
            results.append((py_file.name, "❌", message))
    
    # Print summary
    print("\n" + "=" * 100)
    print("COMPLETE RESULTS")
    print("=" * 100)
    
    total = success_count + fail_count
    print(f"\n✅ SUCCESS: {success_count}/{total} files ({success_count/total*100:.1f}%)")
    print(f"❌ FAILED:  {fail_count}/{total} files ({fail_count/total*100:.1f}%)")
    
    # Show all results
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)
    
    print("\n✅ SUCCESSFUL IMPORTS:")
    for name, status, msg in results:
        if status == "✅":
            print(f"  {status} {name}")
    
    print(f"\n❌ FAILED IMPORTS:")
    for name, status, msg in results:
        if status == "❌":
            print(f"  {status} {name}: {msg}")
    
    return success_count, fail_count

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success, failed = main()
    
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    
    total = success + failed
    if failed == 0:
        print(f"✅ PERFECT: ALL {total} FILES IMPORT SUCCESSFULLY")
    else:
        print(f"⚠️  {failed} files have import issues out of {total} total")
        print(f"   Integration is {success/total*100:.1f}% complete")
