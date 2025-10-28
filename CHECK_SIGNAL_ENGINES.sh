#!/bin/bash
echo "🔍 CHECKING IF SIGNAL ENGINES ARE ACTUALLY WORKING"
echo "="*70

cd ~/bot

echo ""
echo "1️⃣  Check which engines started:"
grep "ACTIVE" bot.log 2>/dev/null | tail -30

echo ""
echo "2️⃣  Check if engines are publishing signals:"
grep -E "publish_signal|→.*:|SIGNAL:" bot.log 2>/dev/null | tail -20

echo ""  
echo "3️⃣  Check MICRO status:"
grep -E "MICRO.*using|crypto_pairs|MICRO GROWTH" bot.log 2>/dev/null | tail -10

echo ""
echo "4️⃣  Check for actual trades:"
grep -E "execute_trade|ORDER PLACED|Execution:" bot.log 2>/dev/null | tail -10

echo ""
echo "5️⃣  Check market scanner discoveries:"
grep -E "DISCOVERED|profitable_pairs|active_pairs" bot.log 2>/dev/null | tail -10

echo ""
echo "6️⃣  Recent decision symbols:"
tail -100 bot.log 2>/dev/null | grep "Decision:" | awk '{print $5}' | sort | uniq -c | sort -rn | head -15

echo ""
echo "="*70
echo "ANALYSIS:"
echo ""
echo "If you see:"
echo "  ✅ Many 'ACTIVE' engines = engines started"
echo "  ✅ 'publish_signal' messages = engines working"
echo "  ✅ 'MICRO using X pairs' = dynamic pairs loaded"
echo "  ✅ 'execute_trade' = actual trades happening"
echo "  ✅ Varied symbols = using dynamic discovery"
echo "  ❌ Same 3-5 symbols = hardcoded/fallback pairs"
