#!/usr/bin/env python3
import ccxt
import os
import time
import requests
from datetime import datetime

class MICRO_GATE_BOT:
    def __init__(self):
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

        # MICRO POSITION SIZES - VERY SMALL TO ENSURE SUFFICIENT BALANCE
        # These are extremely conservative to avoid "BALANCE_NOT_ENOUGH" errors
        self.position_sizes = {
            'BTC/USDT': 0.0001,  # ~$4.30 (well below 3 USDT minimum, but let's test)
            'ETH/USDT': 0.001,  # ~$2.50 (well below 3 USDT minimum)
            'BNB/USDT': 0.01,  # ~$1.50 (well below 3 USDT minimum)
            'SOL/USDT': 0.1,  # ~$1.00 (well below 3 USDT minimum)
            'ADA/USDT': 10.0,  # ~$2.40 (well below 1 ADA minimum)
            'XRP/USDT': 5.0,  # ~$2.40 (well below 1 XRP minimum)
            'DOGE/USDT': 50.0,  # ~$6.00 (well below 10 DOGE minimum)
            'SHIB/USDT': 100000.0,  # ~$2.40 (well below 100000 SHIB minimum)
            'PEPE/USDT': 200000.0,  # ~$2.40 (well below 1000000 PEPE minimum)
        }

        # Profit tracking
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.running = True

        # Only trade pairs that definitely meet minimums
        # self.crypto_pairs = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT']  # DISABLED - Using dynamic discovery
        self.crypto_pairs = []  # Will be populated by scanner
        
        # SAFETY FEATURES
        self.starting_balance = self.check_gate_balance()
        self.max_daily_loss = self.starting_balance * 0.20  # Max 20% loss per day
        self.daily_loss = 0.0
        self.max_trades_per_day = 50  # Limit trades  # Multiple micro pairs

        print("🚀 MICRO GATE.IO BOT INITIALIZED!")
        print("💰 TRADING EXCHANGE: Gate.io (MICRO POSITIONS)")
        print(f"📊 {len(self.crypto_pairs)} Crypto Pairs")
        print("🎯 READY FOR MICRO PROFITS!")

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
        """Simple market analysis"""
        try:
            ticker_data = self.get_gate_ticker(symbol)
            if not ticker_data:
                return "HOLD", 0, 0, 0, 0

            price = ticker_data['price']
            change = ticker_data['change']
            volume = ticker_data['volume']

            # Simple strategy - buy on positive momentum
            if change > 0.5 and volume > 10000:
                return "BUY", 80, price, change, volume
            elif change < -0.5 and volume > 10000:
                return "SELL", 80, price, change, volume
            else:
                return "HOLD", 50, price, change, volume

        except Exception as e:
            print(f"❌ Market analysis error for {symbol}: {e}")
            return "HOLD", 0, 0, 0, 0

    def execute_trade(
        self,
        symbol,
        signal,
        price,
        quantity=None,
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
            quantity
            if quantity is not None
            else self.position_sizes.get(
                symbol,
                0.001,
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

    def run_micro_trading(self):
        """
        Preserve the historical micro strategy.

        Entry fills are not counted as profit.
        Realized PnL requires a reconciled close.
        """
        balance = (
            self.check_gate_balance()
        )

        self.send_telegram(
            (
                "MICRO STRATEGY ACTIVE\n"
                f"Balance: {balance:.2f}\n"
                f"Pairs: {len(self.crypto_pairs)}\n"
                "Execution mode is selected "
                "by LeanTrader's universal router."
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
                        confidence < 80
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
                        "MICRO ENTRY",
                        symbol,
                        signal,
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

                    time.sleep(60)

                print(
                    "Micro cycle complete",
                    "entries=",
                    trade_count,
                    "confirmed_realized_pnl=",
                    self.total_profit,
                )

                time.sleep(30)

            except Exception as exc:
                print(
                    "Micro cycle error:",
                    type(exc).__name__,
                )
                time.sleep(60)

    def run(self):
        try:
            self.run_micro_trading()
        except KeyboardInterrupt:
            print("🛑 Micro bot stopped")
            self.running = False
        except Exception as e:
            print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    bot = MICRO_GATE_BOT()
    bot.run()
