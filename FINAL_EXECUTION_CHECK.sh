#!/bin/bash
echo "🔍 Searching for actual order execution code..."
echo ""

echo "Files with exchange.create_order:"
grep -l "exchange\.create_order" *.py 2>/dev/null | head -10

echo ""
echo "Files with place_order/execute_trade:"
grep -l "def execute_trade\|def place_order" *.py 2>/dev/null | head -10

echo ""
echo "Checking if orders are being logged..."
tail -50 force_live_output.log 2>/dev/null | grep -i "order\|executed\|placed\|filled"

