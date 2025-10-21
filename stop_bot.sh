#!/bin/bash
echo "⏹️  Stopping bot..."
pkill -9 -f RUN_BOT.py
screen -S trading_bot -X quit 2>/dev/null
echo "✅ Bot stopped"
