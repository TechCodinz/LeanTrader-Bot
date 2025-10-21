#!/usr/bin/env python3
"""
TELEGRAM SIGNAL MONITOR - SIMPLE AND WORKING
Monitors signal queue and sends to Telegram channels
"""
import asyncio
import logging
from WORKING_TELEGRAM_BOT import get_telegram_bot

logger = logging.getLogger(__name__)


async def monitor_signals_for_telegram(orchestrator):
    """
    Monitor signals and send to Telegram
    VIP: 70%+ confidence
    FREE: 75%+ confidence
    """
    
    logger.info("📱 Starting Telegram Signal Monitor...")
    
    try:
        telegram = get_telegram_bot()
        logger.info("✅ Telegram bot ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Telegram: {e}")
        return
    
    sent_count_vip = 0
    sent_count_free = 0
    
    loop_count = 0
    last_processed_count = 0
    
    while orchestrator.is_running:
        try:
            loop_count += 1
            if loop_count % 30 == 0:  # Log every 30 seconds
                total_signals = len(orchestrator.data_hub.recent_signals)
                logger.info(f"📊 Monitor: {total_signals} total signals, {sent_count_vip} VIP sent, {sent_count_free} FREE sent")
            
            # Get signals from RECENT_SIGNALS (queue is consumed by decision engine)
            total_signals = len(orchestrator.data_hub.recent_signals)
            if total_signals > last_processed_count:
                # Process new signals
                for i in range(last_processed_count, total_signals):
                    if i >= len(orchestrator.data_hub.recent_signals):
                        break
                    signal = orchestrator.data_hub.recent_signals[i]
                    last_processed_count = i + 1
                
                symbol = signal.get('symbol', 'UNKNOWN')
                side = signal.get('side', signal.get('action', 'HOLD')).upper()
                confidence = signal.get('confidence', 0.0)
                
                # Skip low confidence
                if confidence < 0.70:
                    continue
                
                # Calculate simple levels
                price = signal.get('data', {}).get('price', 100.0)
                if side == 'BUY':
                    entry = price
                    tp1 = price * 1.01
                    tp2 = price * 1.02
                    tp3 = price * 1.03
                    sl = price * 0.99
                else:
                    entry = price
                    tp1 = price * 0.99
                    tp2 = price * 0.98
                    tp3 = price * 0.97
                    sl = price * 1.01
                
                # Send to VIP (70%+)
                if confidence >= 0.70:
                    success = await telegram.send_vip_signal(
                        symbol, side, confidence, entry, tp1, tp2, tp3, sl
                    )
                    if success:
                        sent_count_vip += 1
                        logger.info(f"✅ VIP #{sent_count_vip}: {side} {symbol} ({confidence:.1%})")
                
                # Send to FREE (75%+)
                if confidence >= 0.75:
                    success = await telegram.send_free_signal(symbol, side, confidence)
                    if success:
                        sent_count_free += 1
                        logger.info(f"✅ FREE #{sent_count_free}: {side} {symbol} ({confidence:.1%})")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)
