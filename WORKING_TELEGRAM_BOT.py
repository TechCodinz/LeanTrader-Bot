#!/usr/bin/env python3
"""
SIMPLE WORKING TELEGRAM BOT - NO COMPLEXITY, JUST WORKS
"""
import os
import asyncio
import logging
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class WorkingTelegramBot:
    """Simple, reliable Telegram bot that actually sends messages"""
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.vip_channel = os.getenv("VIP_CHANNEL_ID", "-1002983007302")
        self.free_channel = os.getenv("FREE_CHANNEL_ID", "-1002930953007")
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
        
        self.bot = Bot(token=self.token)
        logger.info(f"✅ Telegram Bot initialized")
        logger.info(f"   VIP: {self.vip_channel}")
        logger.info(f"   FREE: {self.free_channel}")
    
    async def send_vip_signal(self, symbol: str, side: str, confidence: float, 
                              entry: float, tp1: float, tp2: float, tp3: float, sl: float):
        """Send VIP signal with full details"""
        
        side_emoji = "🟢 BUY" if side.upper() == "BUY" else "🔴 SELL"
        
        message = f"""⭐ VIP SIGNAL ⭐

{side_emoji} {symbol}
Confidence: {confidence:.1%}

📍 Entry: ${entry:.4f}
🎯 TP1: ${tp1:.4f}
🎯 TP2: ${tp2:.4f}
🎯 TP3: ${tp3:.4f}
🛡️ SL: ${sl:.4f}

⏰ {asyncio.get_event_loop().time()}"""
        
        try:
            await self.bot.send_message(
                chat_id=self.vip_channel,
                text=message,
                parse_mode='HTML'
            )
            return True
        except TelegramError as e:
            logger.error(f"VIP send failed: {e}")
            return False
    
    async def send_free_signal(self, symbol: str, side: str, confidence: float):
        """Send FREE signal - basic info only"""
        
        side_emoji = "🟢" if side.upper() == "BUY" else "🔴"
        
        message = f"""🌟 FREE SIGNAL

{side_emoji} {side.upper()} {symbol}
Confidence: {confidence:.1%}

💎 Upgrade to VIP for entry/TP/SL levels!"""
        
        try:
            await self.bot.send_message(
                chat_id=self.free_channel,
                text=message
            )
            return True
        except TelegramError as e:
            logger.error(f"FREE send failed: {e}")
            return False


# Global instance
_telegram_bot = None

def get_telegram_bot():
    """Get or create singleton Telegram bot"""
    global _telegram_bot
    if _telegram_bot is None:
        _telegram_bot = WorkingTelegramBot()
    return _telegram_bot
