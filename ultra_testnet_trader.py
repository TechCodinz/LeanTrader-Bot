from dataclasses import dataclass
from typing import Dict, List, Any

"""
ULTRA TESTNET TRADING SYSTEM
Continuous Testnet Trading with Swarm Intelligence Integration

This system:
- Trades continuously on testnet across all timeframes
- Integrates with swarm consciousness for collective intelligence
- Tests strategies before live deployment
- Provides real-time performance feedback to the swarm
"""

import asyncio
import os
import time
import logging
from collections import deque
import numpy as np

# Core imports
from ultra_core import UltraCore
from risk_engine import RiskEngine
from ultra_swarm_consciousness import SwarmConsciousness, SwarmSignal

@dataclass
class TestnetTrade:
    """Testnet trade record"""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: float
    agent_id: str
    confidence: float
    reasoning: str
    status: str  # 'open', 'closed', 'cancelled'
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_timestamp: float = 0.0

class TestnetTradingEngine:
    """Main testnet trading engine"""

    def __init__(
        self,
        ultra_core: UltraCore,
        risk_engine: RiskEngine,
        swarm_consciousness: SwarmConsciousness,
    ):
        self.ultra_core = ultra_core
        self.risk_engine = risk_engine
        self.swarm = swarm_consciousness
        self.logger = logging.getLogger("testnet_trader")

        # Trading configuration
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.max_concurrent_trades = 10
        self.min_confidence = 0.7

        # Trade management
        self.active_trades: Dict[str, TestnetTrade] = {}
        self.trade_history: List[TestnetTrade] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
        }

        # Market simulation
        self.market_prices = {}
        self.price_history = {}
        self.volatility_data = {}

        # Initialize market data
        self._initialize_market_data()

    def _initialize_market_data(self):
        """Initialize empty state; prices are populated from real public feeds."""
        for symbol in self.symbols:
            self.market_prices[
                symbol
            ] = 0.0

            self.price_history[
                symbol
            ] = deque(
                maxlen=1000
            )

            self.volatility_data[
                symbol
            ] = 0.0

    async def start_testnet_trading(self):
        """Start the testnet trading system"""
        self.logger.info("🚀 Starting Ultra Testnet Trading System...")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._market_simulation_loop()),
            asyncio.create_task(self._signal_processing_loop()),
            asyncio.create_task(self._trade_management_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._swarm_feedback_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in testnet trading: {e}")

    async def _market_simulation_loop(self):
        """Simulate market price movements"""
        while True:
            try:
                await self._update_market_prices()
                await asyncio.sleep(0.1)  # 100ms update cycle

            except Exception as e:
                self.logger.error(f"Error in market simulation: {e}")
                await asyncio.sleep(1)

    async def _update_market_prices(self):
        """Refresh real market prices from UltraCore."""
        current_time = time.time()

        for symbol in self.symbols:
            try:
                payload = await self.ultra_core.get_market_data(
                    symbol,
                    "1m",
                )

                if not isinstance(
                    payload,
                    dict,
                ):
                    continue

                price = float(
                    payload.get(
                        "close",
                        0.0,
                    )
                    or 0.0
                )

                if price <= 0.0:
                    continue

                rows = (
                    payload.get(
                        "ohlcv",
                    )
                    or []
                )

                volume = 0.0

                if (
                    rows
                    and isinstance(
                        rows[-1],
                        (list, tuple),
                    )
                    and len(rows[-1]) >= 6
                ):
                    volume = float(
                        rows[-1][5]
                        or 0.0
                    )

                self.market_prices[
                    symbol
                ] = price

                self.price_history[
                    symbol
                ].append(
                    {
                        "timestamp": (
                            current_time
                        ),
                        "price": price,
                        "volume": volume,
                    }
                )

                history = list(
                    self.price_history[
                        symbol
                    ]
                )

                if len(history) >= 20:
                    prices = np.asarray(
                        [
                            row["price"]
                            for row in history[-20:]
                            if row["price"] > 0
                        ],
                        dtype=float,
                    )

                    if len(prices) >= 2:
                        returns = np.diff(
                            np.log(prices)
                        )

                        self.volatility_data[
                            symbol
                        ] = float(
                            np.std(
                                returns
                            )
                        )

            except Exception as exc:
                self.logger.error(
                    "Real market update failed "
                    "for %s: %s",
                    symbol,
                    exc,
                )

    async def _signal_processing_loop(self):
        """Process signals from swarm consciousness"""
        while True:
            try:
                # Get signals from swarm
                signals = await self._get_swarm_signals()

                # Process each signal
                for signal in signals:
                    await self._process_signal(signal)

                await asyncio.sleep(0.5)  # 500ms processing cycle

            except Exception as e:
                self.logger.error(f"Error in signal processing: {e}")
                await asyncio.sleep(1)

    async def _get_swarm_signals(self):
        """Consume recent real-data signals produced by the swarm."""
        current_time = time.time()

        recent = list(
            getattr(
                self.swarm,
                "recent_signals",
                [],
            )
        )

        selected = []
        seen = set()

        for signal in reversed(
            recent
        ):
            try:
                if (
                    current_time
                    - float(
                        signal.timestamp
                    )
                    > 3.0
                ):
                    continue

                if (
                    float(
                        signal.confidence
                    )
                    < self.min_confidence
                ):
                    continue

                key = (
                    signal.agent_id,
                    signal.symbol,
                    signal.timeframe,
                    signal.signal_type,
                    signal.timestamp,
                )

                if key in seen:
                    continue

                seen.add(key)
                selected.append(
                    signal
                )

                if len(selected) >= 50:
                    break

            except Exception:
                continue

        return list(
            reversed(selected)
        )

    async def _process_signal(self, signal: SwarmSignal):
        """Process a single signal and potentially execute a trade"""
        try:
            # Check if we should execute this signal
            if not await self._should_execute_signal(signal):
                return

            # Check if we have capacity for new trades
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.logger.warning(
                    f"Max concurrent trades reached, skipping signal for {signal.symbol}"
                )
                return

            # Execute testnet trade
            await self._execute_testnet_trade(signal)

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")

    async def _should_execute_signal(self, signal: SwarmSignal) -> bool:
        """Determine if we should execute a signal"""
        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            return False

        # Check if we already have a trade for this symbol
        for trade in self.active_trades.values():
            if trade.symbol == signal.symbol and trade.status == 'open':
                return False

        # Check risk parameters
        if not await self.risk_engine.check_scalp_risk(signal):
            return False

        return True

    async def _execute_testnet_trade(
        self,
        signal: SwarmSignal,
    ):
        """Submit an authenticated exchange Testnet order."""
        try:
            from src.leantrader.execution.router import (
                route_order,
            )

            price = float(
                signal.price
                or self.market_prices.get(
                    signal.symbol,
                    0.0,
                )
                or 0.0
            )

            if price <= 0.0:
                return

            order_usd = float(
                os.getenv(
                    "TESTNET_ORDER_USD",
                    "5.0",
                )
            )

            quantity = (
                order_usd / price
            )

            result = await asyncio.to_thread(
                route_order,
                {
                    "symbol": (
                        signal.symbol
                    ),
                    "side": (
                        signal.signal_type
                    ),
                    "qty": quantity,
                    "price": price,
                },
                "ccxt",
            )

            if (
                not result.get("ok")
                or not result.get(
                    "executed"
                )
                or result.get(
                    "simulated"
                )
            ):
                self.logger.warning(
                    "Testnet order rejected: "
                    "%s",
                    result.get(
                        "error",
                        result,
                    ),
                )
                return

            order = (
                result.get("order")
                or {}
            )

            fill_price = float(
                order.get("average")
                or order.get("price")
                or price
            )

            trade_id = str(
                order.get("id")
                or (
                    f"testnet_"
                    f"{int(time.time()*1000)}"
                )
            )

            trade = TestnetTrade(
                trade_id=trade_id,
                symbol=signal.symbol,
                side=signal.signal_type,
                price=fill_price,
                quantity=quantity,
                timestamp=time.time(),
                agent_id=signal.agent_id,
                confidence=(
                    signal.confidence
                ),
                reasoning=(
                    signal.reasoning
                ),
                status="open",
            )

            self.active_trades[
                trade_id
            ] = trade

            self.logger.info(
                "AUTH TESTNET OPEN %s %s "
                "qty=%.8f price=%.8f "
                "order=%s",
                trade.side,
                trade.symbol,
                trade.quantity,
                trade.price,
                trade.trade_id,
            )

        except Exception as exc:
            self.logger.error(
                "Authenticated Testnet "
                "execution error: %s",
                exc,
            )

    async def _trade_management_loop(self):
        """Manage open trades and check for exit conditions"""
        while True:
            try:
                await self._manage_open_trades()
                await asyncio.sleep(1)  # 1 second management cycle

            except Exception as e:
                self.logger.error(f"Error in trade management: {e}")
                await asyncio.sleep(5)

    async def _manage_open_trades(self):
        """Manage all open trades"""
        current_time = time.time()
        trades_to_close = []

        for trade_id, trade in self.active_trades.items():
            if trade.status != 'open':
                continue

            # Check if trade should be closed
            should_close, exit_price, exit_reason = await self._check_exit_conditions(trade)

            if should_close:
                trades_to_close.append((trade_id, exit_price, exit_reason))

        # Close trades
        for trade_id, exit_price, exit_reason in trades_to_close:
            await self._close_trade(trade_id, exit_price, exit_reason)

    async def _check_exit_conditions(self, trade: TestnetTrade) -> tuple[bool, float, str]:
        """Check if a trade should be closed"""
        current_price = self.market_prices[trade.symbol]
        current_time = time.time()

        # Calculate P&L
        if trade.side == 'buy':
            pnl = (current_price - trade.price) / trade.price
        else:
            pnl = (trade.price - current_price) / trade.price

        # Check time-based exit (max 1 hour)
        if current_time - trade.timestamp > 3600:
            return True, current_price, "Time-based exit"

        # Check profit target (2%)
        if pnl > 0.02:
            return True, current_price, "Profit target reached"

        # Check stop loss (1%)
        if pnl < -0.01:
            return True, current_price, "Stop loss triggered"

        # Check confidence-based exit (if confidence drops significantly)
        if current_time - trade.timestamp > 300:  # After 5 minutes
            # Simulate confidence decay
            time_decay = (current_time - trade.timestamp) / 3600  # Decay over 1 hour
            effective_confidence = trade.confidence * (1 - time_decay * 0.5)

            if effective_confidence < 0.5:
                return True, current_price, "Confidence decay"

        return False, 0.0, ""

    async def _close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ):
        """Close with an authenticated opposite Testnet order."""
        try:
            from src.leantrader.execution.router import (
                route_order,
            )

            trade = self.active_trades[
                trade_id
            ]

            close_side = (
                "sell"
                if trade.side == "buy"
                else "buy"
            )

            result = await asyncio.to_thread(
                route_order,
                {
                    "symbol": trade.symbol,
                    "side": close_side,
                    "qty": trade.quantity,
                    "price": exit_price,
                },
                "ccxt",
            )

            if (
                not result.get("ok")
                or not result.get(
                    "executed"
                )
                or result.get(
                    "simulated"
                )
            ):
                self.logger.warning(
                    "Testnet close rejected "
                    "for %s: %s",
                    trade_id,
                    result.get(
                        "error",
                        result,
                    ),
                )
                return

            order = (
                result.get("order")
                or {}
            )

            actual_exit = float(
                order.get("average")
                or order.get("price")
                or exit_price
            )

            if trade.side == "buy":
                pnl = (
                    actual_exit
                    - trade.price
                ) / trade.price
            else:
                pnl = (
                    trade.price
                    - actual_exit
                ) / trade.price

            trade.status = "closed"
            trade.exit_price = (
                actual_exit
            )
            trade.exit_timestamp = (
                time.time()
            )
            trade.pnl = pnl

            self.trade_history.append(
                trade
            )

            del self.active_trades[
                trade_id
            ]

            self._update_performance_metrics(
                trade
            )

            self.logger.info(
                "AUTH TESTNET CLOSE %s "
                "%s pnl=%.6f reason=%s",
                trade.symbol,
                trade.side,
                pnl,
                exit_reason,
            )

        except Exception as exc:
            self.logger.error(
                "Authenticated Testnet "
                "close error %s: %s",
                trade_id,
                exc,
            )

    def _update_performance_metrics(self, trade: TestnetTrade):
        """Update performance metrics with new trade"""
        self.performance_metrics['total_trades'] += 1

        if trade.pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1

        self.performance_metrics['total_pnl'] += trade.pnl

        # Calculate win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades']
                / self.performance_metrics['total_trades']
            )

        # Calculate average win/loss
        if self.performance_metrics['winning_trades'] > 0:
            wins = [t.pnl for t in self.trade_history if t.pnl > 0]
            self.performance_metrics['avg_win'] = np.mean(wins) if wins else 0

        if self.performance_metrics['losing_trades'] > 0:
            losses = [t.pnl for t in self.trade_history if t.pnl < 0]
            self.performance_metrics['avg_loss'] = np.mean(losses) if losses else 0

    async def _performance_monitoring_loop(self):
        """Monitor and report performance metrics"""
        while True:
            try:
                await self._log_performance_metrics()
                await asyncio.sleep(60)  # Log every minute

            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

    async def _log_performance_metrics(self):
        """Log current performance metrics"""
        metrics = self.performance_metrics

        self.logger.info(
            f"Testnet Performance: "
            f"Trades: {metrics['total_trades']}, "
            f"Win Rate: {metrics['win_rate']:.1%}, "
            f"Total P&L: {metrics['total_pnl']:.2%}, "
            f"Active Trades: {len(self.active_trades)}"
        )

    async def _swarm_feedback_loop(self):
        """Provide feedback to swarm consciousness"""
        while True:
            try:
                await self._send_performance_feedback()
                await asyncio.sleep(30)  # Send feedback every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in swarm feedback: {e}")
                await asyncio.sleep(30)

    async def _send_performance_feedback(self):
        """Send performance feedback to swarm consciousness"""
        # This would send performance data back to the swarm
        # for collective learning and improvement

        feedback = {
            'timestamp': time.time(),
            'total_trades': self.performance_metrics['total_trades'],
            'win_rate': self.performance_metrics['win_rate'],
            'total_pnl': self.performance_metrics['total_pnl'],
            'active_trades': len(self.active_trades),
            'market_prices': self.market_prices,
        }

        # In a real implementation, this would be sent to the swarm consciousness
        # For now, we'll just log it
        self.logger.debug(f"Swarm feedback: {feedback}")

    def get_testnet_status(self) -> Dict[str, Any]:
        """Get current testnet trading status"""
        return {
            'timestamp': time.time(),
            'active_trades': len(self.active_trades),
            'total_trades': len(self.trade_history),
            'performance_metrics': self.performance_metrics,
            'market_prices': self.market_prices,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
        }

# Integration function
def integrate_testnet_trader(
    ultra_core: UltraCore, risk_engine: RiskEngine, swarm_consciousness: SwarmConsciousness
) -> TestnetTradingEngine:
    """Integrate Testnet Trading Engine with core system"""
    return TestnetTradingEngine(ultra_core, risk_engine, swarm_consciousness)

# Main execution function
async def main():
    """Main function to run testnet trading"""
    # Initialize core components
    risk_engine = RiskEngine()
    ultra_core = UltraCore(
        mode="paper", symbols=["BTC/USDT", "ETH/USDT"], logger=logging.getLogger()
    )

    # Create swarm consciousness
    from ultra_swarm_consciousness import integrate_swarm_consciousness

    swarm = integrate_swarm_consciousness(ultra_core, risk_engine)

    # Create testnet trader
    testnet_trader = integrate_testnet_trader(ultra_core, risk_engine, swarm)

    # Start testnet trading
    await testnet_trader.start_testnet_trading()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
