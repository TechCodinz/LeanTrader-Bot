#!/bin/bash
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          🔍 CHECKING ALL ACTIVE SYSTEMS                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd ~/trading_bot

echo "1️⃣  Core Systems:"
tail -500 bot.log | grep -E "26 SYSTEMS|SYSTEMS INITIALIZED" | tail -3
echo ""

echo "2️⃣  Advanced Systems:"
tail -500 bot.log | grep -E "7 NEW SYSTEMS|8 ADVANCED|9 ADVANCED" | tail -3
echo ""

echo "3️⃣  Moon Spotter:"
tail -500 bot.log | grep -i "moon" | tail -5
echo ""

echo "4️⃣  Ultra Rare Engines:"
tail -500 bot.log | grep -i "ultra rare" | tail -5
echo ""

echo "5️⃣  Adaptive Confidence:"
tail -500 bot.log | grep -i "adaptive" | tail -5
echo ""

echo "6️⃣  Omniscient Engine:"
tail -500 bot.log | grep -i "omniscient" | tail -5
echo ""

echo "7️⃣  Dynamic Pair Discovery:"
tail -500 bot.log | grep -E "DISCOVERED.*pairs|5[0-9][0-9][0-9] pairs" | tail -3
echo ""

echo "8️⃣  Recent signals (Moon/Scalp/Arb):"
tail -200 bot.log | grep -E "Moon|Scalp|Arbitrage|signal" | tail -10
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Systems Check Complete!"
echo "═══════════════════════════════════════════════════════════════"
