#!/usr/bin/env python3
"""
Watch ACTUAL trade execution - orders placed, positions opened/closed, profits made
"""
import asyncio
import os
import logging

os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'
os.environ['BYBIT_TESTNET'] = 'false'

# Set logging to capture EVERYTHING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)

from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

async def main():
    print("\n" + "="*80)
    print("💰 WATCHING FOR ACTUAL TRADES - ORDERS, EXECUTIONS, PROFITS")
    print("="*80)
    print("\n✅ LIVE MODE ACTIVE")
    print("✅ ENABLE_LIVE=true")
    print("✅ ALLOW_LIVE=true")
    print("\n⏰ Running for 3 MINUTES - watching for:")
    print("   📊 Decisions made")
    print("   📈 Orders placed")
    print("   💰 Positions opened")
    print("   ✅ Positions closed")
    print("   💵 Profits realized")
    print("\n" + "="*80 + "\n")
    
    orch = CompleteUltimateOrchestrator()
    
    # Check what's loaded
    print(f"\n✅ Systems loaded:")
    if hasattr(orch, 'real_profit_bot') and orch.real_profit_bot:
        print(f"   ✅ REAL_PROFIT_BOT (Gate.io)")
    if hasattr(orch, 'orchestrator'):
        print(f"   ✅ EXECUTION_ORCHESTRATOR")
    if hasattr(orch, 'micro_wallet_grower') and orch.micro_wallet_grower:
        print(f"   ✅ MICRO_WALLET_GROWER")
    
    print("\n🚀 Starting trading...\n")
    
    # Run for 3 minutes (180 seconds)
    task = asyncio.create_task(orch.start())
    
    # Print status every 30 seconds
    for i in range(6):
        await asyncio.sleep(30)
        print(f"\n⏰ [{(i+1)*30}s] Still monitoring...")
    
    task.cancel()
    
    print("\n" + "="*80)
    print("✅ 3-MINUTE TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

