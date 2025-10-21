#!/usr/bin/env python3
"""
WORKING TELEGRAM BOT - Simple, reliable Telegram signal sender
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Telegram configuration from environment
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TG_VIP_CHAT_ID = os.getenv('TG_VIP_CHAT_ID')
TG_FREE_CHAT_ID = os.getenv('TG_FREE_CHAT_ID')

# Try importing telegram library
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("⚠️  python-telegram-bot not installed. Install: pip install python-telegram-bot")


class WorkingTelegramBot:
    """Simple, working Telegram bot for sending trading signals"""
    
    def __init__(self):
        self.bot = None
        self.vip_chat_id = TG_VIP_CHAT_ID
        self.free_chat_id = TG_FREE_CHAT_ID
        
        if not TELEGRAM_AVAILABLE:
            logger.error("❌ Telegram library not available!")
            return
        
        if not TELEGRAM_BOT_TOKEN:
            logger.error("❌ TELEGRAM_BOT_TOKEN not set in .env!")
            return
        
        try:
            self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
            logger.info("✅ Telegram bot initialized successfully!")
            
            if self.vip_chat_id:
                logger.info(f"✅ VIP channel configured: {self.vip_chat_id}")
            else:
                logger.warning("⚠️  TG_VIP_CHAT_ID not set in .env")
            
            if self.free_chat_id:
                logger.info(f"✅ FREE channel configured: {self.free_chat_id}")
            else:
                logger.warning("⚠️  TG_FREE_CHAT_ID not set in .env")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram bot: {e}")
    
    async def send_message(self, chat_id: str, text: str, parse_mode: str = 'HTML') -> bool:
        """Send a message to a Telegram chat"""
        if not self.bot:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram send error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    async def send_vip_signal(
        self,
        symbol: str,
        side: str,
        confidence: float,
        entry: float,
        tp1: float,
        tp2: float,
        tp3: float,
        sl: float
    ) -> bool:
        """Send VIP signal with full details"""
        
        if not self.vip_chat_id:
            return False
        
        # Format signal message
        message = f"""
🔥 <b>VIP SIGNAL #{datetime.now().strftime('%H%M')}</b> 🔥

📊 <b>Pair:</b> {symbol}
🎯 <b>Action:</b> {side}
💯 <b>Confidence:</b> {confidence:.1%}

📍 <b>Entry:</b> ${entry:.6f}

🎯 <b>Take Profits:</b>
   TP1: ${tp1:.6f} (+{((tp1/entry-1)*100):.1f}%)
   TP2: ${tp2:.6f} (+{((tp2/entry-1)*100):.1f}%)
   TP3: ${tp3:.6f} (+{((tp3/entry-1)*100):.1f}%)

🛡 <b>Stop Loss:</b> ${sl:.6f} ({((sl/entry-1)*100):.1f}%)

⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>💎 VIP Premium Signal - Trade Carefully!</i>
"""
        
        return await self.send_message(self.vip_chat_id, message)
    
    async def send_free_signal(
        self,
        symbol: str,
        side: str,
        confidence: float
    ) -> bool:
        """Send FREE signal with basic info"""
        
        if not self.free_chat_id:
            return False
        
        # Format free signal message
        message = f"""
📢 <b>FREE SIGNAL</b>

📊 {symbol}
🎯 {side}
💯 {confidence:.1%} Confidence

⏰ {datetime.now().strftime('%H:%M:%S')}

<i>🆓 Join VIP for detailed signals!</i>
"""
        
        return await self.send_message(self.free_chat_id, message)
    
    async def send_trade_notification(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        status: str = "EXECUTED"
    ) -> bool:
        """Send trade execution notification to VIP"""
        
        if not self.vip_chat_id:
            return False
        
        message = f"""
⚡ <b>TRADE {status}</b> ⚡

📊 {symbol}
🎯 {side}
💰 Price: ${price:.6f}
📦 Amount: {amount:.6f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return await self.send_message(self.vip_chat_id, message)


# Global bot instance
_telegram_bot = None


def get_telegram_bot() -> WorkingTelegramBot:
    """Get or create the global Telegram bot instance"""
    global _telegram_bot
    
    if _telegram_bot is None:
        _telegram_bot = WorkingTelegramBot()
    
    return _telegram_bot


# For testing
if __name__ == "__main__":
    async def test():
        print("Testing Telegram bot...")
        bot = get_telegram_bot()
        
        if bot.bot:
            print("✅ Bot initialized!")
            print(f"VIP Chat: {bot.vip_chat_id}")
            print(f"FREE Chat: {bot.free_chat_id}")
            
            # Test VIP signal
            success = await bot.send_vip_signal(
                symbol="BTC/USDT",
                side="BUY",
                confidence=0.92,
                entry=50000,
                tp1=50500,
                tp2=51000,
                tp3=51500,
                sl=49500
            )
            print(f"VIP signal sent: {success}")
            
            # Test FREE signal
            success = await bot.send_free_signal(
                symbol="ETH/USDT",
                side="SELL",
                confidence=0.85
            )
            print(f"FREE signal sent: {success}")
        else:
            print("❌ Bot not initialized!")
    
    asyncio.run(test())
