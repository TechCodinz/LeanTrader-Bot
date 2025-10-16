#!/usr/bin/env python3
"""
Direct test of Telegram channels to see EXACTLY what's wrong
"""
import asyncio
import os
from pathlib import Path
from telegram import Bot
import telegram

# Load .env
env_path = Path('/root/trading_bot/.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

async def test_channels():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    free_id = os.getenv('TELEGRAM_FREE_CHANNEL')
    vip_id = os.getenv('TELEGRAM_VIP_CHANNEL')
    admin_id = os.getenv('TELEGRAM_ADMIN_CHAT_ID')
    
    print("═" * 70)
    print("📱 TELEGRAM CHANNEL DIAGNOSTIC TEST")
    print("═" * 70)
    print()
    
    if not token:
        print("❌ No TELEGRAM_BOT_TOKEN in .env")
        return
    
    bot = Bot(token=token)
    
    # Get bot info
    try:
        me = await bot.get_me()
        print(f"✅ Bot connected: @{me.username}")
        print(f"   Bot ID: {me.id}")
        print()
    except Exception as e:
        print(f"❌ Bot connection failed: {e}")
        return
    
    # Test admin chat
    print("Testing ADMIN chat...")
    if admin_id:
        try:
            result = await bot.send_message(
                chat_id=admin_id,
                text='🧪 Admin test - If you see this, admin chat works!'
            )
            print(f"✅ ADMIN: Message sent successfully (msg_id: {result.message_id})")
        except Exception as e:
            print(f"❌ ADMIN: {e}")
    else:
        print("⚠️  ADMIN: No chat ID set")
    
    print()
    
    # Test FREE channel
    print("Testing FREE channel...")
    print(f"Channel ID: {free_id}")
    if free_id and free_id != '@your_free_channel':
        try:
            result = await bot.send_message(
                chat_id=free_id,
                text='📢 FREE channel test - If you see this, FREE channel works!'
            )
            print(f"✅ FREE: Message sent successfully (msg_id: {result.message_id})")
        except telegram.error.Forbidden as e:
            print(f"❌ FREE: Bot not in channel or no permission!")
            print(f"   Error: {e}")
            print(f"   FIX: Add bot @{me.username} to channel {free_id} as ADMIN")
            print(f"        Enable 'Post Messages' permission")
        except telegram.error.BadRequest as e:
            print(f"❌ FREE: Invalid channel ID!")
            print(f"   Error: {e}")
            print(f"   Channel ID: {free_id}")
            print(f"   FIX: Channel ID should be numeric like -1001234567890")
        except Exception as e:
            print(f"❌ FREE: {type(e).__name__}: {e}")
    else:
        print("⚠️  FREE: No channel ID set")
    
    print()
    
    # Test VIP channel
    print("Testing VIP channel...")
    print(f"Channel ID: {vip_id}")
    if vip_id and vip_id != '@your_vip_channel':
        try:
            result = await bot.send_message(
                chat_id=vip_id,
                text='🌟 VIP channel test - If you see this, VIP channel works!'
            )
            print(f"✅ VIP: Message sent successfully (msg_id: {result.message_id})")
        except telegram.error.Forbidden as e:
            print(f"❌ VIP: Bot not in channel or no permission!")
            print(f"   Error: {e}")
            print(f"   FIX: Add bot @{me.username} to channel {vip_id} as ADMIN")
            print(f"        Enable 'Post Messages' permission")
        except telegram.error.BadRequest as e:
            print(f"❌ VIP: Invalid channel ID!")
            print(f"   Error: {e}")
            print(f"   Channel ID: {vip_id}")
            print(f"   FIX: Channel ID should be numeric like -1001234567890")
        except Exception as e:
            print(f"❌ VIP: {type(e).__name__}: {e}")
    else:
        print("⚠️  VIP: No channel ID set")
    
    print()
    print("═" * 70)
    print("✅ TEST COMPLETE")
    print("═" * 70)
    print()
    print("If you saw ✅ messages above, check your Telegram channels!")
    print("If you saw ❌ errors, follow the FIX instructions above.")
    print()

if __name__ == '__main__':
    asyncio.run(test_channels())
