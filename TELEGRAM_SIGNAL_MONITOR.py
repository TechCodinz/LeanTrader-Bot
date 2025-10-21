#!/usr/bin/env python3
"""
Telegram Signal Monitor - Routes signals to Telegram channels
DIRECT INTEGRATION with ULTIMATE_TELEGRAM_VIP_BOT
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def monitor_signals_for_telegram(orchestrator):
    """
    Monitor signals and send to Telegram VIP/FREE channels
    - VIP: 70%+ confidence
    - FREE: 75%+ confidence  
    """
    
    logger.info("📱 Starting Telegram Signal Monitor...")
    
    # DIRECT IMPORT - Try multiple file names
    telegram_bot = None
    try:
        from ultra_telegram_master import UltraTelegramMaster as TelegramBot
        telegram_bot = TelegramBot()
        logger.info("✅ Telegram Bot loaded (ultra_telegram_master)!")
    except:
        try:
            from PREMIUM_VIP_TELEGRAM_SYSTEM import PremiumVIPTelegramBot as TelegramBot
            telegram_bot = TelegramBot()
            logger.info("✅ Telegram Bot loaded (PREMIUM_VIP)!")
        except Exception as e:
            logger.error(f"❌ Failed to load any Telegram bot: {e}")
            return
    
    if not telegram_bot:
        logger.error("❌ No Telegram bot available")
        return
    
    sent_signals = set()  # Track sent signals to avoid duplicates
    
    while orchestrator.is_running:
        try:
            # Check if there are signals in the queue
            if not orchestrator.data_hub.signal_queue.empty():
                signal = orchestrator.data_hub.signal_queue.get_nowait()
                
                # Extract signal details
                symbol = signal.get('symbol', 'UNKNOWN')
                side = signal.get('side', signal.get('action', '').lower())
                confidence = signal.get('confidence', 0.0)
                
                # Create unique signal ID
                signal_id = f"{symbol}_{side}_{confidence:.2f}"
                
                # Skip if already sent
                if signal_id in sent_signals:
                    continue
                
                # Send to VIP channel (70%+ confidence)
                if confidence >= 0.70:
                    try:
                        await telegram_bot.send_ultra_signal(
                            symbol=symbol,
                            side=side,
                            confidence=confidence,
                            analytics=signal.get('data', {}),
                            is_vip=True
                        )
                        logger.info(f"✅ VIP signal sent: {side} {symbol} ({confidence:.1%})")
                        sent_signals.add(signal_id)
                        
                        # Cleanup old entries
                        if len(sent_signals) > 1000:
                            sent_signals.clear()
                    except Exception as e:
                        logger.error(f"❌ VIP signal error: {e}")
                
                # Send to FREE channel (75%+ confidence)
                if confidence >= 0.75:
                    try:
                        await telegram_bot.send_ultra_signal(
                            symbol=symbol,
                            side=side,
                            confidence=confidence,
                            analytics=signal.get('data', {}),
                            is_vip=False
                        )
                        logger.info(f"✅ FREE signal sent: {side} {symbol} ({confidence:.1%})")
                    except Exception as e:
                        logger.error(f"❌ FREE signal error: {e}")
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)
