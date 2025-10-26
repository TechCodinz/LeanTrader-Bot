#!/usr/bin/env python3
"""
Check if decisions are actually going to alert_queue for execution
"""
import asyncio
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

async def main():
    print("\n🔍 Checking alert_queue for execution...")
    
    orch = CompleteUltimateOrchestrator()
    
    # Check if alert_queue exists
    if hasattr(orch, 'data_hub') and orch.data_hub:
        if hasattr(orch.data_hub, 'alert_queue'):
            print(f"✅ alert_queue exists")
            print(f"   Type: {type(orch.data_hub.alert_queue)}")
            print(f"   Size: {orch.data_hub.alert_queue.qsize()}")
        else:
            print(f"❌ NO alert_queue in data_hub!")
    
    # Check execution orchestrator
    if 'execution' in orch.advanced_orchestrators:
        print(f"✅ Execution orchestrator exists")
        exec_orch = orch.advanced_orchestrators['execution']
        print(f"   Min confidence: {exec_orch.min_confidence}")
        print(f"   Execution enabled: {exec_orch.execution_enabled}")
    else:
        print(f"❌ NO execution orchestrator!")
    
    # Start and monitor for 30 seconds
    print(f"\n🚀 Starting bot for 30s...")
    task = asyncio.create_task(orch.start())
    
    for i in range(6):
        await asyncio.sleep(5)
        if hasattr(orch, 'data_hub') and hasattr(orch.data_hub, 'alert_queue'):
            size = orch.data_hub.alert_queue.qsize()
            print(f"   [{(i+1)*5}s] alert_queue size: {size}")
    
    task.cancel()
    print("\n✅ Done\n")

asyncio.run(main())

