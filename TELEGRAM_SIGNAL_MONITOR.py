#!/usr/bin/env python3
import asyncio
import logging
from SIMPLE_TELEGRAM_BOT import SimpleTelegramBot

logger = logging.getLogger(__name__)

async def monitor_signals_for_telegram(orchestrator):
    logger.info("📱 Starting Telegram Monitor...")
    bot = SimpleTelegramBot()
    
    while orchestrator.is_running:
        try:
            if not orchestrator.data_hub.signal_queue.empty():
                signal = orchestrator.data_hub.signal_queue.get_nowait()
                symbol = signal.get('symbol', 'UNKNOWN')
                side = signal.get('side', signal.get('action', 'HOLD'))
                conf = signal.get('confidence', 0.0)
                
                if conf >= 0.70:
                    await bot.send_signal(symbol, side, conf, is_vip=True)
                    logger.info(f"✅ VIP: {side} {symbol} ({conf:.1%})")
                
                if conf >= 0.75:
                    await bot.send_signal(symbol, side, conf, is_vip=False)
                    logger.info(f"✅ FREE: {side} {symbol} ({conf:.1%})")
            
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)
