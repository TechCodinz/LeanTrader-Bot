#!/usr/bin/env python3
import os
import asyncio
from telegram import Bot

class SimpleTelegramBot:
    def __init__(self):
        self.bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        self.vip_channel = os.getenv("VIP_CHANNEL_ID", "-1002983007302")
        self.free_channel = os.getenv("FREE_CHANNEL_ID", "-1002930953007")
    
    async def send_signal(self, symbol, side, confidence, is_vip=True):
        channel = self.vip_channel if is_vip else self.free_channel
        msg = f"{'⭐ VIP' if is_vip else '🌟 FREE'} | {side.upper()} {symbol}\nConfidence: {confidence:.1%}"
        try:
            await self.bot.send_message(chat_id=channel, text=msg)
            return True
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
