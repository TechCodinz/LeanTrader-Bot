#!/usr/bin/env python3
"""
TEST EXECUTION IN WORKSPACE
Shows actual execution flow with all discovered pairs
"""
import asyncio
import os
import logging

# Set to testnet for safe testing
os.environ['TRADING_MODE'] = 'testnet'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s'
)

from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator

async def main():
    print("\n" + "="*80)
    print("🔍 TESTING EXECUTION IN WORKSPACE")
    print("="*80 + "\n")
    
    orch = CompleteUltimateOrchestrator(mode='testnet')
    
    print("✅ Orchestrator created\n")
    
    # Check execution orchestrator
    print("🔍 Checking execution systems...")
    await orch.initialize_all_systems()
    
    if 'execution' in orch.advanced_orchestrators:
        exec_orch = orch.advanced_orchestrators['execution']
        print(f"✅ Execution orchestrator found")
        print(f"   Min confidence: {exec_orch.min_confidence}")
        print(f"   Mode: {exec_orch.mode}")
    else:
        print("❌ NO execution orchestrator!")
    
    # Check dynamic discovery
    if hasattr(orch, 'market_scanner'):
        print(f"✅ Market scanner found")
    if hasattr(orch, 'dynamic_pair_discovery'):
        print(f"✅ Dynamic pair discovery found")
    
    # Check REAL_PROFIT_BOT
    if hasattr(orch, 'real_profit_bot') and orch.real_profit_bot:
        print(f"✅ REAL_PROFIT_BOT found")
        print(f"   Initial pairs: {len(orch.real_profit_bot.crypto_pairs)}")
        print(f"   Pairs: {orch.real_profit_bot.crypto_pairs[:10]}...")  # First 10
    
    print(f"\n🚀 Starting bot (will run for 60 seconds)...")
    print("="*80 + "\n")
    
    # Start bot with timeout
    task = asyncio.create_task(orch.start())
    
    # Monitor for 60 seconds
    for i in range(12):
        await asyncio.sleep(5)
        print(f"[{(i+1)*5}s] Running...")
        
        # Check execution stats if available
        if hasattr(orch, 'advanced_orchestrators') and 'execution' in orch.advanced_orchestrators:
            exec_orch = orch.advanced_orchestrators['execution']
            if hasattr(exec_orch, 'total_trades'):
                print(f"      Trades: {exec_orch.total_trades}, Profit: ${exec_orch.total_profit:.2f}")
    
    task.cancel()
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE - Check logs above for:")
    print("   1. 'EXECUTION LOOP STARTED' message")
    print("   2. Dynamic pair discovery messages")
    print("   3. 'ORDER' or 'EXECUTE' messages")
    print("   4. Number of active pairs expanding")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

