#!/usr/bin/env python3
"""
Show if bot is ACTUALLY placing orders or just generating signals
"""
import asyncio
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
import logging

# Monitor execution at DEBUG level
logging.basicConfig(level=logging.DEBUG)

async def test():
    print("\n🔍 Checking if bot EXECUTES trades...")
    orch = CompleteUltimateOrchestrator()
    
    # Check execution orchestrator
    if hasattr(orch, 'orchestrator') and orch.orchestrator:
        print(f"\n✅ Has execution orchestrator")
        print(f"   Type: {type(orch.orchestrator).__name__}")
        
        # Check for execute_trade method
        if hasattr(orch.orchestrator, 'execute_trade'):
            print(f"✅ Has execute_trade method")
        
        # Check data hub
        if hasattr(orch, 'data_hub'):
            print(f"✅ Has data_hub for signals")
    
    # Start and monitor
    print("\n🚀 Running for 30 seconds - watching for order execution...\n")
    
    task = asyncio.create_task(orch.start())
    await asyncio.sleep(30)
    task.cancel()
    
    print("\n✅ Done")

asyncio.run(test())

