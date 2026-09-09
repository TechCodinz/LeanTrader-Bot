#!/usr/bin/env python3
import ccxt
import os
import time
import requests
from datetime import datetime

class REAL_PROFIT_BOT:
    def __init__(self, universe=None):
        # TELEGRAM CONFIGURATION
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.admin_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.vip_chat_id = os.getenv("TELEGRAM_VIP_CHAT_ID", "")
        self.free_chat_id = os.getenv("TELEGRAM_FREE_CHAT_ID", "")

        # GATE.IO API CONFIGURATION (REAL TRADING)
        self.exchange_id = os.getenv("LEGACY_PROFIT_EXCHANGE", "gateio").strip().lower()

        self.gate_config = {
            "enableRateLimit": True,
        }

        exchange_name = (
            "gateio"
            if self.exchange_id in {"gate", "gateio"}
            else self.exchange_id
        )

        exchange_class = getattr(
            ccxt,
            exchange_name,
        )

        self.gate = exchange_class(
            self.gate_config
        )

        # REAL PROFIT POSITION SIZES - CALCULATED FOR MEANINGFUL INCOME
        # These are designed to generate $50-200 daily profits to cover bills
        self.position_sizes = {
            'BTC/USDT': 0.01,  # ~$430 (major profit potential)
            'ETH/USDT': 0.05,  # ~$125 (major profit potential)
            'BNB/USDT': 0.5,  # ~$75 (major profit potential)
            'SOL/USDT': 5.0,  # ~$50 (major profit potential)
            'ADA/USDT': 1000.0,  # ~$240 (major profit potential)
            'XRP/USDT': 500.0,  # ~$240 (major profit potential)
            'DOGE/USDT': 10000.0,  # ~$1200 (major profit potential)
            'SHIB/USDT': 50000000.0,  # ~$120 (major profit potential)
            'PEPE/USDT': 100000000.0,  # ~$240 (major profit potential)
        }

        # Profit tracking
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.running = True

        # USE ALL DISCOVERED PAIRS - NO LIMITS!
        # Start with universe if provided (will be replaced by dynamic discovery)
        if universe and len(universe) > 0:
            # Filter to crypto/forex only, NO LIMITS
            self.crypto_pairs = [p for p in universe if '/USDT' in p or '/USD' in p or '/' in p]
        else:
            # Minimal fallback - will be replaced by dynamic discovery
            self.crypto_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
        
        # This list will be continuously updated by:
        # 1. DYNAMIC_PAIR_DISCOVERY (discovers all pairs from all exchanges)
        # 2. DYNAMIC_MARKET_SCANNER (filters dead pairs, adds trending ones)
        # 3. Orchestrator injects discovered pairs every scan cycle
        
        print(f"🚀 REAL PROFIT BOT INITIALIZED with {len(self.crypto_pairs)} pairs!")
        print("💰 TRADING EXCHANGE: Gate.io (REAL INCOME GENERATION)")
        print(f"📊 {len(self.crypto_pairs)} Initial Pairs (expanding with dynamic discovery - NO LIMITS!)")
        print("🎯 TARGET: $50-200 DAILY PROFITS FOR BILLS!")
        print("🔄 Pairs auto-update every hour with fresh discoveries!")

    def send_telegram(self, message, chat_id=None):
        if chat_id is None:
            chat_id = self.admin_chat_id

        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
            requests.post(url, data=data, timeout=10)
            print("✅ Telegram message sent")
            return True
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False

    def check_gate_balance(self):
        from src.leantrader.execution.router import (
            route_balance,
        )

        try:
            balance = (
                route_balance(
                    exchange_id=(
                        self.exchange_id
                    )
                )
                or {}
            )

            free = (
                balance.get("free")
                or {}
            )

            value = (
                free.get("USDT")
                if isinstance(
                    free,
                    dict,
                )
                else None
            )

            if value is None:
                coin = (
                    balance.get("USDT")
                    or {}
                )

                if isinstance(
                    coin,
                    dict,
                ):
                    value = coin.get(
                        "free"
                    )

            return float(
                value or 0.0
            )

        except Exception as exc:
            print(
                "Balance unavailable:",
                type(exc).__name__,
            )
            return 0.0

    def get_gate_ticker(
        self,
        symbol,
    ):
        from src.leantrader.execution.router import (
            route_ticker,
        )

        try:
            ticker = (
                route_ticker(
                    symbol,
                    exchange_id=(
                        self.exchange_id
                    ),
                )
                or {}
            )

            price = float(
                ticker.get("last")
                or ticker.get("close")
                or 0.0
            )

            if price <= 0.0:
                return None

            return {
                "price": price,
                "change": float(
                    ticker.get(
                        "percentage"
                    )
                    or 0.0
                ),
                "volume": float(
                    ticker.get(
                        "quoteVolume"
                    )
                    or 0.0
                ),
            }

        except Exception as exc:
            print(
                "Ticker unavailable:",
                type(exc).__name__,
            )
            return None

    def analyze_market(self, symbol):
        """Advanced market analysis for REAL PROFITS"""
        try:
            ticker_data = self.get_gate_ticker(symbol)
            if not ticker_data:
                return "HOLD", 0, 0, 0, 0

            price = ticker_data['price']
            change = ticker_data['change']
            volume = ticker_data['volume']

            # AGGRESSIVE PROFIT STRATEGIES
            signals = []
            confidences = []

            # Strategy 1: High momentum trading (for big profits)
            if abs(change) > 3.0 and volume > 1000000:
                signals.append("BUY" if change > 0 else "SELL")
                confidences.append(95)

            # Strategy 2: Volume explosion (for big profits)
            if volume > 5000000 and abs(change) > 2.0:
                signals.append("BUY" if change > 0 else "SELL")
                confidences.append(90)

            # Strategy 3: Meme coin moonshot (for massive profits)
            if (
                symbol in ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
                and change > 5.0
                and volume > 2000000
            ):
                signals.append("BUY")
                confidences.append(95)

            # Strategy 4: Major crypto breakout (for consistent profits)
            if symbol in ['BTC/USDT', 'ETH/USDT'] and abs(change) > 1.5 and volume > 10000000:
                signals.append("BUY" if change > 0 else "SELL")
                confidences.append(85)

            # Strategy 5: Altcoin pump (for quick profits)
            if (
                symbol in ['ADA/USDT', 'XRP/USDT', 'SOL/USDT', 'BNB/USDT']
                and change > 4.0
                and volume > 2000000
            ):
                signals.append("BUY")
                confidences.append(90)

            # Get best signal
            if signals and confidences:
                best_idx = confidences.index(max(confidences))
                return signals[best_idx], max(confidences), price, change, volume

            return "HOLD", 50, price, change, volume

        except Exception as e:
            print(f"❌ Market analysis error for {symbol}: {e}")
            return "HOLD", 0, 0, 0, 0

    def execute_trade(
        self,
        symbol,
        signal,
        price,
    ):
        from src.leantrader.execution.router import (
            route_order,
        )

        side = str(
            signal or ""
        ).lower()

        if side not in {
            "buy",
            "sell",
        }:
            return None

        amount = float(
            self.position_sizes.get(
                symbol,
                0.01,
            )
        )

        return route_order(
            {
                "symbol": symbol,
                "side": side,
                "qty": amount,
                "price": float(
                    price or 0.0
                ),
                "order_type": "market",
                "exchange_id": (
                    self.exchange_id
                ),
                "backend": "ccxt",
            }
        )

    def run_real_profit_trading(self):
        """
        Preserve historical signal generation
        while requiring reconciled closes before
        realized PnL is counted.
        """
        balance = (
            self.check_gate_balance()
        )

        self.send_telegram(
            (
                "REAL PROFIT STRATEGY ACTIVE\n"
                f"Balance: {balance:.2f}\n"
                f"Pairs: {len(self.crypto_pairs)}\n"
                "Execution environment is selected "
                "by the universal router."
            )
        )

        trade_count = 0

        while self.running:
            try:
                for symbol in (
                    self.crypto_pairs
                ):
                    (
                        signal,
                        confidence,
                        price,
                        change,
                        volume,
                    ) = self.analyze_market(
                        symbol
                    )

                    if (
                        confidence < 85
                        or signal
                        == "HOLD"
                    ):
                        continue

                    result = (
                        self.execute_trade(
                            symbol,
                            signal,
                            price,
                        )
                    )

                    if not (
                        isinstance(
                            result,
                            dict,
                        )
                        and result.get(
                            "ok"
                        )
                    ):
                        continue

                    trade_count += 1
                    self.total_trades += 1

                    print(
                        "REAL PROFIT ENTRY",
                        symbol,
                        signal,
                        "confidence=",
                        confidence,
                        "mode=",
                        result.get(
                            "execution_mode"
                        ),
                        "exchange=",
                        result.get(
                            "exchange"
                        ),
                        "realized_pnl=PENDING_CLOSE",
                    )

                    time.sleep(45)

                print(
                    "Real-profit cycle complete",
                    "entries=",
                    trade_count,
                    "confirmed_realized_pnl=",
                    self.total_profit,
                )

                time.sleep(15)

            except Exception as exc:
                print(
                    "Real-profit cycle error:",
                    type(exc).__name__,
                )
                time.sleep(30)

    def run(self):
        try:
            self.run_real_profit_trading()
        except KeyboardInterrupt:
            print("🛑 Real profit bot stopped")
            self.running = False
        except Exception as e:
            print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    bot = REAL_PROFIT_BOT()
    bot.run()

# Alias for compatibility
RealProfitBot = REAL_PROFIT_BOT
