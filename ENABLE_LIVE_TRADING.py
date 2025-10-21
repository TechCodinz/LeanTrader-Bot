#!/usr/bin/env python3
"""
ENABLE LIVE TRADING - Connect executor to decision engine
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


async def enable_live_trading(orchestrator):
    """
    Monitor decisions and execute real trades
    Only trades high-confidence signals (80%+)
    """
    
    logger.info("💰 Starting LIVE TRADING executor...")
    
    try:
        from LIVE_TRADE_EXECUTOR import get_executor
        executor = get_executor()
        logger.info("✅ Live executor ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize executor: {e}")
        logger.info("⚠️ Trading in SIMULATION mode only")
        return
    
    executed_count = 0
    
    while orchestrator.is_running:
        try:
            # Monitor signal queue for high-confidence signals
            if not orchestrator.data_hub.signal_queue.empty():
                signal = orchestrator.data_hub.signal_queue.get_nowait()
                
                symbol = signal.get('symbol', '')
                side = signal.get('side', signal.get('action', '')).upper()
                confidence = signal.get('confidence', 0.0)
                
                # Only execute 80%+ confidence
                if confidence >= 0.80 and symbol and side in ['BUY', 'SELL']:
                    # Simple price estimate
                    price = signal.get('data', {}).get('price', 100.0)
                    
                    # Calculate TP and SL
                    if side == 'BUY':
                        tp = price * 1.02  # 2% profit target
                        sl = price * 0.99  # 1% stop loss
                    else:
                        tp = price * 0.98
                        sl = price * 1.01
                    
                    # Execute trade
                    success, trade_id, msg = await executor.execute_signal(
                        symbol, side, confidence, price, tp, sl
                    )
                    
                    if success:
                        executed_count += 1
                        logger.info(f"💰 TRADE #{executed_count}: {msg}")
                        
                        # Log to data hub
                        trade_data = {
                            'trade_id': trade_id,
                            'symbol': symbol,
                            'side': side,
                            'confidence': confidence,
                            'entry': price,
                            'tp': tp,
                            'sl': sl,
                            'status': 'open'
                        }
                        orchestrator.data_hub.trade_data_queue.put_nowait(trade_data)
                    else:
                        logger.warning(f"⚠️ Trade rejected: {msg}")
            
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Live trading error: {e}")
            await asyncio.sleep(5)
