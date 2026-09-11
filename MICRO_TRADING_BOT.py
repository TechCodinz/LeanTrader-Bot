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

        from ccxt_exchange_compat import (
            resolve_exchange_class,
        )

        exchange_class = (
            resolve_exchange_class(
                ccxt,
                self.exchange_id,
            )
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

        # Pairs come from the canonical market registry, not from a list
        # kept here. The comment that used to sit on this line said "will be
        # populated by scanner" -- nothing ever did, so the trading loop
        # iterated an empty list forever and this engine never produced a
        # single candidate. set_universe() is how the orchestrator fills it;
        # refresh_universe() is the fallback for a standalone run.
        self.crypto_pairs = []
        self.universe_source = "unset"

        # SAFETY FEATURES
        self.starting_balance = self.check_gate_balance()

        # One balance read, reused for the first universe pull.
        self.refresh_universe(balance=self.starting_balance)
        self.max_daily_loss = self.starting_balance * 0.20  # Max 20% loss per day
        self.daily_loss = 0.0
        self.max_trades_per_day = 50  # Limit trades  # Multiple micro pairs

        print("🚀 MICRO TRADING BOT INITIALIZED!")
        # The execution venue is whichever one the universal router is
        # authenticated against. self.exchange_id names this engine's
        # historical market-data client and says nothing about where orders go.
        print(f"📈 Market data client: {self.exchange_id}")
        print("💱 Execution venue: selected by the universal router")
        print(f"📊 {len(self.crypto_pairs)} pairs ({self.universe_source})")
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

    def set_universe(self, symbols):
        """Accept the ranked universe from the orchestrator.

        Symbols are normalized and filtered to what the authenticated
        execution venue currently lists, because this engine sends orders --
        studying a market it cannot trade would just waste its cycles.
        """
        from src.leantrader.execution.preflight import normalize_symbol
        from src.leantrader.universe.registry import universe

        accepted = []
        for symbol in symbols or []:
            normalized = normalize_symbol(symbol)
            if not normalized:
                continue
            market = universe.get(normalized)
            if market is not None and market.execution_eligible is False:
                continue
            accepted.append(normalized)

        if accepted:
            self.crypto_pairs = accepted
            self.universe_source = "orchestrator"

        return len(self.crypto_pairs)

    def refresh_universe(self, limit=None, balance=None):
        """Pull micro-account candidates straight from the registry.

        Used when this engine runs on its own. The ranking is the registry's:
        markets whose venue minimum this balance can actually fund, with
        headroom left for fees and an exit. Nominal unit price is not what
        puts a market on this list.
        """
        try:
            from src.leantrader.universe.registry import universe
        except Exception:
            return 0

        if limit is None:
            try:
                limit = int(os.getenv("MICRO_UNIVERSE_LIMIT", "40"))
            except ValueError:
                limit = 40

        if balance is None:
            try:
                balance = float(self.check_gate_balance() or 0.0)
            except Exception:
                balance = 0.0
        balance = float(balance or 0.0)

        candidates = universe.micro_candidates(
            capital_quote=balance,
            limit=limit,
        )

        symbols = [market.symbol for market in candidates]
        if symbols:
            self.crypto_pairs = symbols
            self.universe_source = f"registry(capital={balance:.4f})"

        return len(self.crypto_pairs)

    def execute_trade(
        self,
        symbol,
        signal,
        price,
        quantity=None,
    ):
        from src.leantrader.execution import (
            preflight,
        )
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

        intent = {
            "symbol": symbol,
            "side": side,
            "price": float(
                price or 0.0
            ),
            "order_type": "market",
        }

        # self.exchange_id names this engine's historical Gate.io market-data
        # client. Passing it as the order venue overrode whichever exchange
        # the runtime is authenticated against, addressing orders to an
        # account that does not exist. The router resolves the venue from the
        # runtime configuration unless an operator sets an explicit override.
        override = os.getenv(
            "EXECUTION_EXCHANGE_OVERRIDE",
            "",
        ).strip().lower()

        if override:
            intent["exchange_id"] = override

        if quantity is not None:
            # An explicit size from the caller is respected as given.
            intent["qty"] = amount
            intent["backend"] = "ccxt"
            return route_order(intent)

        # Otherwise size against the account's real free balance and the
        # venue's own minimums, rather than the fixed self.position_sizes
        # table a small wallet can never fund.
        prepared, blocked = (
            preflight.prepare_order(
                intent
            )
        )

        if prepared is None:
            return blocked.as_dict()

        receipt = route_order(
            prepared.to_payload()
        )

        blocker = (
            preflight
            .classify_receipt(
                receipt
            )
        )

        if blocker is None:
            preflight.record_event(
                "acknowledged"
            )
            preflight.invalidate_balance_cache()
        else:
            preflight.record_blocker(
                blocker,
                str(
                    receipt.get(
                        "error",
                        "",
                    )
                )[:200],
            )

        return receipt

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
