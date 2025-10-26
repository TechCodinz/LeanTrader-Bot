#!/usr/bin/env python3
"""
FINAL TEST - Bot should actually EXECUTE trades now
"""
import asyncio
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

async def main():
    print("\n" + "="*80)
    print("🔥 FINAL LIVE TRADING TEST - WITH REAL_PROFIT_BOT FIX")
    print("="*80 + "\n")
    
    orch = CompleteUltimateOrchestrator()
    
    print(f"✅ Orchestrator created")
    print(f"✅ Mode: {orch.mode}")
    
    # Check if REAL_PROFIT_BOT loaded
    if hasattr(orch, 'real_profit_bot') and orch.real_profit_bot:
        print(f"✅ REAL_PROFIT_BOT loaded!")
        print(f"   Exchange: {orch.real_profit_bot.gate.name}")
        print(f"   Pairs: {len(orch.real_profit_bot.crypto_pairs)}")
    else:
        print(f"⚠️  REAL_PROFIT_BOT not loaded")
    
    print("\n🚀 Running for 90 seconds - watching for ACTUAL trades...")
    print("   Looking for: REAL PROFIT BUY/SELL messages\n")
    
    task = asyncio.create_task(orch.start())
    await asyncio.sleep(90)
    task.cancel()
    
    print("\n✅ Test complete\n")

asyncio.run(main())

