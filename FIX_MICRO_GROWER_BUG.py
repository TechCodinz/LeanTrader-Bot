#!/usr/bin/env python3
"""
🔧 FIX MICRO GROWER BUG - Position Closing Issue
==================================================
BUG: Only taking $1.04 profit but leaving full stake in position
FIX: Close FULL position, compound ALL profit into next trade
"""

import os
import sys

def fix_micro_grower_bug():
    """Fix the position closing logic in MICRO wallet grower"""
    
    print("🔧 Fixing MICRO Wallet Grower bug...")
    print("="*70)
    
    filepath = "COMPLETE_ULTIMATE_ORCHESTRATOR.py"
    
    # Backup
    backup_path = f"{filepath}.backup_micro_fix"
    with open(filepath, 'r') as f:
        with open(backup_path, 'w') as b:
            b.write(f.read())
    print(f"✅ Backup: {backup_path}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find and replace the buggy position closing logic
    old_code = '''                        # AUTO-CLOSE profitable positions to free up capital
                        try:
                            positions = self.micro_wallet_grower.gate.fetch_balance()
                            for coin, amt in positions['total'].items():
                                if coin != 'USDT' and amt > 0:
                                    symbol = f"{coin}/USDT"
                                    if symbol in self.micro_wallet_grower.gate.markets:
                                        # Check current price
                                        ticker = self.micro_wallet_grower.gate.fetch_ticker(symbol)
                                        current_price = ticker['last']
                                        position_value = amt * current_price
                                        
                                        # Sell if position > $3 and frees up capital
                                        if position_value >= 3.0:
                                            logger.info(f"🔄 Closing position: {symbol} - {amt:.4f} tokens worth ${position_value:.2f}")
                                            self.micro_wallet_grower.gate.create_market_sell_order(symbol, amt)
                                            logger.info(f"   ✅ Freed up ${position_value:.2f} USDT!")
                        except Exception as e:
                            logger.debug(f"Position close: {e}")'''
    
    new_code = '''                        # AUTO-CLOSE ALL positions to COMPOUND profit
                        try:
                            positions = self.micro_wallet_grower.gate.fetch_balance()
                            total_freed = 0.0
                            
                            for coin, amt in positions['total'].items():
                                if coin != 'USDT' and amt > 0:
                                    symbol = f"{coin}/USDT"
                                    if symbol in self.micro_wallet_grower.gate.markets:
                                        # Get available (not locked) amount
                                        available_amt = positions['free'].get(coin, 0)
                                        
                                        if available_amt > 0:
                                            # Check current price
                                            ticker = self.micro_wallet_grower.gate.fetch_ticker(symbol)
                                            current_price = ticker['last']
                                            position_value = available_amt * current_price
                                            
                                            # Close ANY position > $1 to free up capital for next trade
                                            if position_value >= 1.0:
                                                logger.info(f"🔄 CLOSING FULL POSITION: {symbol}")
                                                logger.info(f"   Amount: {available_amt:.8f} {coin}")
                                                logger.info(f"   Value: ${position_value:.2f}")
                                                
                                                # Create market sell for FULL available amount
                                                order = self.micro_wallet_grower.gate.create_market_sell_order(
                                                    symbol, 
                                                    available_amt  # FULL AMOUNT
                                                )
                                                
                                                logger.info(f"   ✅ CLOSED! Order: {order['id']}")
                                                logger.info(f"   💰 Freed ${position_value:.2f} → Ready for next trade!")
                                                total_freed += position_value
                                            else:
                                                logger.debug(f"   ⏭️  {symbol} too small: ${position_value:.2f}")
                            
                            if total_freed > 0:
                                logger.info(f"💰 TOTAL FREED THIS CYCLE: ${total_freed:.2f}")
                                
                        except Exception as e:
                            logger.debug(f"Position close: {e}")'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ FIXED position closing logic!")
        print("\n🔧 Changes made:")
        print("   ❌ OLD: Only closed positions >= $3")
        print("   ❌ OLD: Didn't use 'free' balance (could have locked tokens)")
        print("   ❌ OLD: Left capital stuck in positions")
        print("")
        print("   ✅ NEW: Closes ANY position >= $1")
        print("   ✅ NEW: Uses 'free' (available) balance only")
        print("   ✅ NEW: Closes FULL available amount")
        print("   ✅ NEW: Frees up capital immediately for compounding")
        print("")
        print("✅ This will COMPOUND profit properly!")
        return True
    else:
        print("⚠️  Could not find exact code to replace")
        print("   The code may have changed - manual review needed")
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║       🔧 FIX MICRO GROWER BUG - Position Closing Issue 🔧      ║
    ║                                                                ║
    ║  BUG: Only taking $1.04 profit, leaving full stake stuck      ║
    ║  FIX: Close FULL position, compound ALL profit                 ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    if fix_micro_grower_bug():
        print("\n" + "="*70)
        print("✅ BUG FIXED!")
        print("="*70)
        print("\n📊 What changed:")
        print("   - Now closes positions >= $1 (was $3)")
        print("   - Uses 'free' balance to avoid locked tokens")
        print("   - Closes FULL available amount")
        print("   - Properly compounds profit into next trades")
        print("")
        print("🚀 Ready to restart bot!")
        return 0
    else:
        print("\n❌ Fix failed - manual review needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
