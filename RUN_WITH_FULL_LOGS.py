#!/usr/bin/env python3
"""
Run bot with MAXIMUM logging to see EVERYTHING
"""
import asyncio
import os
import logging

# Force live mode
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

# MAXIMUM logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

async def main():
    print("\n" + "="*80)
    print("🔥 RUNNING WITH FULL DEBUG LOGGING")
    print("="*80 + "\n")
    
    orch = CompleteUltimateOrchestrator()
    
    print("\n🚀 Starting bot for 60 seconds...")
    print("   Watching for ACTUAL order execution...\n")
    
    task = asyncio.create_task(orch.start())
    await asyncio.sleep(60)
    task.cancel()
    
    print("\n✅ Done\n")

asyncio.run(main())

