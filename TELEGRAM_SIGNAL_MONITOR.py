#!/usr/bin/env python3
"""
Telegram Signal Monitor - Routes signals to appropriate Telegram channels
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def monitor_signals_for_telegram(orchestrator):
    """
    Monitor signals and route to appropriate Telegram channels
    High confidence → VIP channel with buttons
    Medium confidence → Free channel basic
    All trades → Admin notifications
    """
    
    logger.info("📱 Starting Telegram signal monitor...")
    
    telegram = orchestrator.advanced_orchestrators.get('telegram')
    if not telegram or not telegram.enabled:
        logger.info("📱 Telegram not enabled - skipping monitor")
        return
    
    while orchestrator.is_running:
        try:
            # Monitor signal queue
            if not orchestrator.data_hub.signal_queue.empty():
                signal = await orchestrator.data_hub.signal_queue.get()
                
                signal_data = signal.get('data', {})
                confidence = signal_data.get('confidence', 0)
                
                # Skip if confidence is 0 (avoid division by zero)
                if confidence == 0:
                    continue
                
                # High confidence signals → VIP channel
                if confidence >= 0.80:
                    await telegram.send_signal_to_vip(signal_data)
                
                # Medium confidence → Free channel
                elif confidence >= 0.65:
                    await telegram.send_signal_to_free(signal_data)
            
            # Monitor trade queue for admin notifications  
            if not orchestrator.data_hub.trade_data_queue.empty():
                trade = await orchestrator.data_hub.trade_data_queue.get()
                
                # Notify admin of all trades (if methods exist)
                if hasattr(telegram, 'notify_trade_executed') and trade.get('status') == 'open':
                    await telegram.notify_trade_executed(trade)
                elif hasattr(telegram, 'notify_trade_closed') and trade.get('status') == 'closed':
                    await telegram.notify_trade_closed(trade)
            
            await asyncio.sleep(1)  # Check every second
            
        except ZeroDivisionError as e:
            logger.debug(f"Skipping signal with zero confidence")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Telegram monitor error: {e}")
            await asyncio.sleep(5)
