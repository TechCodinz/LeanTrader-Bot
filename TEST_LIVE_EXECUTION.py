#!/usr/bin/env python3
"""
Test if bot actually EXECUTES trades or just generates signals
"""
import asyncio
import logging
from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("\n" + "="*80)
    print("🔥 TESTING LIVE TRADE EXECUTION")
    print("="*80 + "\n")
    
    orch = CompleteUltimateOrchestrator()
    
    # Check execution components
    print("Checking execution systems:")
    print(f"  - Has data_hub: {hasattr(orch, 'data_hub')}")
    print(f"  - Has orchestrator: {hasattr(orch, 'orchestrator')}")
    print(f"  - Has execution_engine: {hasattr(orch, 'execution_engine')}")
    print(f"  - Has exchange_manager: {hasattr(orch, 'exchange_manager')}")
    
    # Check if execution methods exist
    if hasattr(orch, 'orchestrator') and orch.orchestrator:
        methods = [m for m in dir(orch.orchestrator) if 'execute' in m.lower()]
        print(f"\n  Orchestrator execution methods: {methods[:5]}")
    
    # Start and watch for actual order creation
    print("\n" + "="*80)
    print("STARTING BOT - WATCHING FOR ACTUAL TRADES...")
    print("="*80)
    
    # Run for 60 seconds
    try:
        task = asyncio.create_task(orch.start())
        await asyncio.sleep(60)
        task.cancel()
    except Exception as e:
        logger.error(f"Error: {e}")
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    asyncio.run(main())

