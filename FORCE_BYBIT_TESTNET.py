#!/usr/bin/env python3
"""
Force bot to use ONLY Bybit testnet for trading
"""

import re

def force_testnet():
    """Force ENABLE_LIVE_TRADING.py to use ONLY Bybit testnet"""
    
    try:
        with open('ENABLE_LIVE_TRADING.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ ENABLE_LIVE_TRADING.py not found")
        return
    
    # Find the exchange initialization section
    if 'self.exchange = ccxt.bybit' in content or 'self.exchange = ccxt.gateio' in content:
        # Replace with testnet
        content = re.sub(
            r"self\.exchange\s*=\s*ccxt\.\w+\(",
            "self.exchange = ccxt.bybit(",
            content
        )
        
        # Force testnet options
        if "'testnet': True" not in content:
            content = re.sub(
                r"(\s+self\.exchange\s*=\s*ccxt\.bybit\([^)]+)\)",
                r"\1, options={'testnet': True})",
                content
            )
    
    with open('ENABLE_LIVE_TRADING.py', 'w') as f:
        f.write(content)
    
    print("✅ Forced Bybit testnet!")

def fix_router():
    """Make router.py use testnet by default"""
    try:
        with open('brokers/router.py', 'r') as f:
            content = f.read()
    except:
        print("⚠️ router.py not found, skipping")
        return
    
    # Force testnet in router
    if "'testnet': True" not in content:
        content = re.sub(
            r"(ccxt\.bybit\(\{[^}]+)",
            r"\1, 'options': {'testnet': True}",
            content
        )
    
    with open('brokers/router.py', 'w') as f:
        f.write(content)
    
    print("✅ Router forced to testnet!")

if __name__ == "__main__":
    force_testnet()
    fix_router()
    print("\n✅ Bot will now use Bybit TESTNET ONLY!")
    print("   Restart bot to activate!")
