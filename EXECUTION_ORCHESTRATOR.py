#!/usr/bin/env python3
"""
EXECUTION ORCHESTRATOR - THE MISSING PIECE
Smart trade execution with risk management, position sizing, and profit optimization
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import time

from src.leantrader.execution import intent as execution_intent
from src.leantrader.execution import inventory as inventory_reconciler
from src.leantrader.execution import preflight
from src.leantrader.execution.router import route_order

logger = logging.getLogger(__name__)


class SmartPositionSizer:
    """Smart position sizing based on account balance and risk"""

    def __init__(self, initial_balance: float = 1000.0):
        self.balance = initial_balance
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.max_position_pct = 0.10  # 10% max position size
        self.min_position_usd = 10.0  # $10 minimum

        # Dynamic sizing based on confidence
        self.use_dynamic_sizing = True
        self.aggressive_mode = True  # Grow account faster

    def calculate_position_size(self,
                               confidence: float,
                               volatility: float = 0.02,
                               stop_loss_pct: float = 0.01) -> float:
        """
        Calculate optimal position size using Kelly Criterion + Risk Management

        Args:
            confidence: AI confidence (0.0 to 1.0)
            volatility: Market volatility estimate
            stop_loss_pct: Stop loss as percentage

        Returns:
            Position size in USD
        """
        # Base risk amount (2% of balance)
        base_risk = self.balance * self.max_risk_per_trade

        # Kelly Criterion adjustment
        # Kelly = (p * b - q) / b where p=win_prob, q=loss_prob, b=win/loss ratio
        win_prob = confidence
        loss_prob = 1 - confidence
        win_loss_ratio = 2.0  # Assume 2:1 reward/risk

        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% Kelly

        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - volatility)

        # Adjust for confidence (higher confidence = larger size)
        confidence_adjustment = 0.5 + (confidence * 0.5)  # 50% to 100%

        # Calculate final position size
        position_size = (base_risk / stop_loss_pct) * kelly_fraction * volatility_adjustment * confidence_adjustment

        # AGGRESSIVE MODE: Grow account faster with high confidence trades
        if self.aggressive_mode and confidence >= 0.85:
            # Increase size by up to 50% for very high confidence
            confidence_boost = 1.0 + ((confidence - 0.80) * 3.0)  # More aggressive for 80%+ confidence  # 85% conf = 1.0x, 100% conf = 1.3x
            position_size *= confidence_boost
            logger.info(f"🚀 Aggressive sizing: confidence {confidence:.1%} → {confidence_boost:.2f}x boost")

        # Apply limits
        max_position = self.balance * self.max_position_pct
        position_size = min(position_size, max_position)
        position_size = max(position_size, self.min_position_usd)

        # Balance-aware: As balance grows, increase position sizes proportionally
        if self.balance > 1000:
            growth_multiplier = (self.balance / 1000) ** 0.5  # Square root scaling
            position_size *= growth_multiplier
            logger.debug(f"💰 Balance-aware sizing: ${self.balance:.0f} → {growth_multiplier:.2f}x multiplier")

        return position_size

    def update_balance(self, new_balance: float):
        """Update balance after trades"""
        self.balance = new_balance


class SmartRiskManager:
    """Smart risk management and validation"""

    def __init__(self):
        self.max_open_positions = 999999  # ♾️ INFINITE TRADING!
        self.max_daily_trades = 999999    # ♾️ UNLIMITED TRADES!
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_correlated_positions = 999999  # No correlation limits!

        self.open_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()

    def can_open_position(self, symbol: str, signal_type: str) -> tuple[bool, str]:
        """Check if we can open a new position"""

        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2%}"

        # Check max positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max positions reached: {len(self.open_positions)}/{self.max_open_positions}"

        # Check if already in this position
        if symbol in self.open_positions:
            return False, f"Already in position: {symbol}"

        # Check correlation (BTC/ETH, etc)
        correlated_count = self._count_correlated_positions(symbol)
        if correlated_count >= self.max_correlated_positions:
            return False, f"Max correlated positions: {correlated_count}/{self.max_correlated_positions}"

        return True, "OK"

    def _count_correlated_positions(self, symbol: str) -> int:
        """Count positions in correlated assets"""
        # Simplified correlation groups
        correlation_groups = {
            'BTC': ['BTC/USDT', 'BTC/USD'],
            'ETH': ['ETH/USDT', 'ETH/USD'],
            'MAJOR': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        }

        count = 0
        for group, symbols in correlation_groups.items():
            if symbol in symbols:
                # Count positions in same group
                for open_symbol in self.open_positions:
                    if open_symbol in symbols:
                        count += 1

        return count

    def record_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Record opened position"""
        self.open_positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'timestamp': datetime.now()
        }

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position and calculate P&L"""
        if symbol not in self.open_positions:
            return 0.0

        position = self.open_positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']

        # Calculate P&L
        if side == 'buy':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        # Update daily stats
        self.daily_pnl += pnl
        self.daily_trades += 1

        # Remove position
        del self.open_positions[symbol]

        return pnl

    def reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today


class ExecutionOrchestrator:
    """
    SMART EXECUTION ORCHESTRATOR
    The critical missing piece that actually executes trades

    Features:
    - Smart position sizing (Kelly Criterion)
    - Risk management and validation
    - Stop loss and take profit automation
    - Multi-exchange support
    - Performance tracking
    - Trade recovery and retry logic
    """

    def __init__(self, data_hub, trading_engines, risk_engine, ledger, mode: str = "testnet"):
        self.data_hub = data_hub
        self.engines = trading_engines
        self.risk_engine = risk_engine
        self.ledger = ledger
        self.mode = mode

        # Smart components
        self.position_sizer = SmartPositionSizer(initial_balance=1000.0)
        self.risk_manager = SmartRiskManager()

        # ADVANCED COMPONENTS (wired externally)
        self.action_decider = None  # AdvancedActionDecider
        self.trailing_stop = None  # TrailingStopManager
        self.compound_engine = None  # CompoundEngine
        self.partial_tp = None  # PartialTPManager

        # Execution settings
        self.min_confidence = 0.80  # 80% minimum confidence to execute
        self.execution_enabled = True

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.execution_times = deque(maxlen=100)

        logger.info("⚡ Execution Orchestrator initialized")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Min Confidence: {self.min_confidence}")
        logger.info("   Sizing: adaptive, from the account's real free balance")
        logger.info("   Placement: universal execution router (route_order)")

    async def run_execution_loop(self):
        """
        Main execution loop - monitors decisions and executes high-confidence trades
        """
        logger.info("⚡ Starting SMART execution loop...")

        while self.execution_enabled:
            try:
                # Reset daily stats if new day
                self.risk_manager.reset_daily_stats()

                # Check for decisions in alert queue
                if not self.data_hub.alert_queue.empty():
                    decision = await self.data_hub.alert_queue.get()

                    preflight.record_event('decisions_consumed')
                    await self.process_decision(decision)

                # Monitor open positions
                await self.monitor_positions()

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(5)

    async def process_decision(self, decision: Dict[str, Any]):
        """Process AI decision using ADVANCED ACTION LOGIC"""

        try:
            # Extract decision details
            signal = decision.get('signal', {})
            action = decision.get('action', 'hold')
            confidence = decision.get('confidence', 0.0)

            symbol = signal.get('symbol') or signal.get('data', {}).get('symbol')

            # One identity for this attempt, carried to wherever it stops.
            # Without it the funnel could say how many attempts happened but
            # not which engine produced them or where they died.
            intent = execution_intent.intent_from_decision(
                decision,
                source_engine=str(
                    signal.get('source')
                    or signal.get('data', {}).get('source')
                    or 'execution_orchestrator'
                ),
                environment=self.mode,
            )
            intent.advance(execution_intent.DECISION)

            if not symbol:
                preflight.record_blocker(preflight.INVALID_INTENT, 'no_symbol')
                intent.stop(
                    execution_intent.DECISION,
                    preflight.INVALID_INTENT,
                    'decision carried no symbol',
                )
                logger.debug("No symbol in decision")
                return

            # Validate confidence. This threshold is not lowered to produce
            # activity; attempts that fail it are counted so a quiet run can
            # be attributed rather than guessed at.
            if confidence < self.min_confidence:
                # A structured rejection, not a silent end. Most live
                # decisions sit below the threshold, and that is an
                # acceptable answer -- but it has to be distinguishable from
                # a broken handoff.
                preflight.record_blocker(
                    preflight.CONFIDENCE_BELOW_THRESHOLD,
                    f"{symbol} {confidence:.4f}<{self.min_confidence:.4f}",
                )
                intent.stop(
                    execution_intent.THRESHOLD,
                    execution_intent.DECISION_REJECTED_CONFIDENCE,
                    f"confidence {confidence:.4f} < threshold "
                    f"{self.min_confidence:.4f}",
                )
                logger.debug(f"Low confidence: {confidence:.2%} < {self.min_confidence:.2%}")
                return

            # USE ADVANCED ACTION DECIDER if available
            if self.action_decider:
                # Real recent closes. This used to pass [price] * 50 -- a flat
                # synthetic series -- which made every volatility and regime
                # reading the decider produced meaningless. If real history is
                # unavailable the decider is skipped rather than fed invented
                # bars.
                price_history = await self._recent_closes(symbol)

                if not price_history:
                    preflight.record_blocker(
                        preflight.MARKET_METADATA_UNAVAILABLE,
                        f"{symbol}:no_price_history",
                    )
                    logger.info(
                        f"   ⏸️  {symbol}: no real price history for the "
                        "action decider"
                    )
                    return

                # Get all current opportunities (for portfolio balancing)
                opportunities = []  # Would get from data_hub

                # DECIDE SOPHISTICATED ACTION!
                advanced_decision = await self.action_decider.decide_action(
                    symbol=symbol,
                    signal=signal,
                    confidence=confidence,
                    price_history=price_history,
                    opportunities=opportunities
                )

                action = advanced_decision['action']
                size = advanced_decision.get('size', 0)
                reason = advanced_decision.get('reason', '')

                logger.info(f"🧠 ADVANCED DECISION: {action.upper()} {symbol}")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   Confidence: {confidence:.1%}")
                logger.info(f"   Market Regime: {advanced_decision.get('regime', 'unknown')}")

                # Handle different actions
                if action == 'avoid' or action == 'hold':
                    preflight.record_blocker(
                        preflight.STRATEGY_REJECT, f"{symbol}:{action}"
                    )
                    intent.stop(
                        execution_intent.CANDIDATE, preflight.STRATEGY_REJECT, f"decider chose {action}"
                    )
                    logger.info(f"   ⏸️  No action taken")
                    return

                elif action == 'scale_in':
                    logger.info(f"   📈 Scaling into position (DCA)")

                elif action == 'scale_out':
                    logger.info(f"   📉 Scaling out {size:.0%} of position")
                    # Would execute partial close
                    return

            else:
                # Fallback to simple logic
                if action not in ['buy', 'sell']:
                    preflight.record_blocker(
                        preflight.STRATEGY_REJECT, f"{symbol}:{action}"
                    )
                    return

            # Check risk management
            can_trade, reason = self.risk_manager.can_open_position(symbol, action)
            if not can_trade:
                preflight.record_blocker(
                    preflight.RISK_REJECT, f"{symbol}:{reason}"
                )
                intent.stop(
                    execution_intent.CANDIDATE, preflight.RISK_REJECT, reason
                )
                logger.info(f"⚠️ Trade blocked: {reason}")
                return

            # EXECUTE THE TRADE!
            logger.info(f"⚡ EXECUTING: {action.upper()} {symbol} (confidence: {confidence:.1%})")

            preflight.record_event('candidates_execution_eligible')
            intent.candidate_id = execution_intent.new_candidate_id()
            intent.advance(execution_intent.CANDIDATE)

            result = await self.execute_trade(
                symbol=symbol,
                side=action,
                confidence=confidence,
                signal=signal,
                intent=intent,
            )

            # execute_trade returns a receipt on acknowledgement and a
            # classified blocker otherwise. Both are dicts, so the truthiness
            # of the return value says nothing -- read ok.
            if isinstance(result, dict) and result.get('ok'):
                order = result.get('order') or {}
                logger.info(
                    f"✅ Order acknowledged: {symbol} id={order.get('id')}"
                )
            elif isinstance(result, dict):
                logger.warning(
                    f"❌ No order placed for {symbol}: "
                    f"{result.get('blocker', 'UNCLASSIFIED')}"
                )
            else:
                preflight.record_blocker(preflight.ROUTER_REFUSED, symbol)
                logger.warning(f"❌ No order placed for {symbol}: no receipt")

        except Exception as e:
            logger.error(f"Decision processing error: {e}")

    async def execute_trade(self,
                           symbol: str,
                           side: str,
                           confidence: float,
                           signal: Dict[str, Any],
                           intent=None) -> Optional[Dict[str, Any]]:
        """Submit one order through the universal execution router.

        This used to size from a fixed $1000 assumption, hand the order to
        REAL_PROFIT_BOT, and -- if that produced anything at all, including a
        refusal receipt -- record a position, publish a trade record and log
        "TRADE EXECUTED". A refusal dict is truthy, so refusals were being
        counted as fills and fed to the learning path as trade outcomes. There
        was also a fallback branch that built a trade record out of nothing and
        marked it simulated.

        Both are gone. The order is sized against the account's real free
        balance and the venue's own limits, submitted through route_order, and
        only treated as executed when the router reports execution and the
        exchange returns an order id. Anything else is classified, counted and
        returned as a blocker.
        """
        start_time = time.time()

        if intent is None:
            intent = execution_intent.ExecutionIntent(
                symbol=symbol,
                side=side,
                confidence=confidence,
                environment=self.mode,
                source_engine='execution_orchestrator',
            )

        order_intent = {
            'symbol': symbol,
            'side': side,
            'confidence': confidence,
            'order_type': 'market',
        }

        # What a new buy may actually spend: free cash, not the balance the
        # account had before it bought anything. Capital already in inventory
        # is committed, and proposing orders as though it were still cash is
        # what produced dozens of identical sub-minimum attempts.
        order_intent['risk_budget'] = await self._spendable_budget()

        price = signal.get('data', {}).get('price') or signal.get('price')
        if price:
            order_intent['price'] = price
        else:
            fetched = await self.get_current_price(symbol)
            if fetched:
                order_intent['price'] = fetched

        override = os.getenv('EXECUTION_EXCHANGE_OVERRIDE', '').strip().lower()
        if override:
            order_intent['exchange_id'] = override

        # Preflight touches the network (markets, balance, ticker) and
        # route_order blocks on the exchange call, so both run off the event
        # loop. Holding the loop here stalls every other engine.
        intent.advance(execution_intent.PREFLIGHT)
        prepared, blocked = await asyncio.to_thread(
            preflight.prepare_order, order_intent
        )

        if prepared is None:
            intent.stop(
                execution_intent.PREFLIGHT, blocked.blocker, blocked.detail
            )
            logger.info(
                f"⛔ {symbol} {side.upper()} not submitted: "
                f"{blocked.blocker} ({blocked.detail})"
            )
            payload = blocked.as_dict()
            payload['intent_id'] = intent.intent_id
            payload['correlation_id'] = intent.correlation_id
            return payload

        logger.info(
            f"⚡ SUBMITTING {side.upper()} {prepared.symbol} "
            f"amount={prepared.amount} notional={prepared.notional:.4f} "
            f"{prepared.quote_currency} "
            f"[{prepared.execution_mode}@{prepared.exchange_id}] "
            f"sizing: {prepared.sizing_reason}"
        )

        preflight.record_event('submitted')
        intent.advance(execution_intent.ROUTE_ORDER)

        # Identity travels with the order, so the receipt can be tied back to
        # the signal that produced it.
        payload = prepared.to_payload()
        payload.setdefault('params', {}).update(intent.to_payload())

        receipt = await asyncio.to_thread(route_order, payload)

        blocker = preflight.classify_receipt(receipt)
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        preflight.record_stage_latency('submit_to_receipt', execution_time)

        if blocker is not None:
            detail = str((receipt or {}).get('error', ''))[:200]
            preflight.record_blocker(blocker, f"{symbol}:{detail}")
            intent.stop(execution_intent.EXCHANGE_ORDER, blocker, detail)
            logger.warning(
                f"❌ {symbol} {side.upper()} not acknowledged: {blocker} {detail}"
            )
            return {
                'ok': False,
                'prepared': True,
                'blocker': blocker,
                'detail': detail,
                'receipt': receipt,
                'intent_id': intent.intent_id,
                'correlation_id': intent.correlation_id,
            }

        order = receipt.get('order') or {}
        order_id = order.get('id')
        filled = order.get('filled')
        avg_price = order.get('average') or order.get('price') or prepared.price
        status = order.get('status')

        preflight.record_event('acknowledged')
        intent.advance(execution_intent.EXCHANGE_ORDER, str(order_id))
        if filled:
            preflight.record_event('fills')
            intent.succeed(execution_intent.FILL, str(filled))
        preflight.record_event('positions_opened')
        intent.succeed(execution_intent.POSITION)
        preflight.invalidate_balance_cache()

        self.risk_manager.record_position(
            symbol=prepared.symbol,
            side=prepared.side,
            size=prepared.amount,
            entry_price=avg_price
        )

        # Only facts the exchange returned, plus the intent that produced
        # them. No stop/target is invented here: nothing has placed one.
        trade_record = {
            'symbol': prepared.symbol,
            'side': prepared.side,
            'order_id': order_id,
            'amount': prepared.amount,
            'filled': filled,
            'entry_price': avg_price,
            'notional': prepared.notional,
            'quote_currency': prepared.quote_currency,
            'confidence': confidence,
            'exchange': receipt.get('exchange'),
            'execution_mode': receipt.get('execution_mode'),
            'authority': receipt.get('authority'),
            'order_status': status,
            'timestamp': datetime.now(),
            'status': 'open',
            'realized_pnl': None,
            'intent_id': intent.intent_id,
            'correlation_id': intent.correlation_id,
            'source_engine': intent.source_engine,
        }

        await self.data_hub.publish_trade(trade_record)

        self.total_trades += 1

        logger.info("⚡ ORDER ACKNOWLEDGED:")
        logger.info(f"   Symbol: {prepared.symbol}")
        logger.info(f"   Side: {prepared.side.upper()}")
        logger.info(f"   Order ID: {order_id}")
        logger.info(f"   Amount: {prepared.amount}")
        logger.info(f"   Price: {avg_price}")
        logger.info(f"   Filled: {filled}")
        logger.info(f"   Status: {status}")
        logger.info(f"   Venue: {receipt.get('exchange')} "
                    f"({receipt.get('execution_mode')})")
        logger.info(f"   Latency: {execution_time:.2f}s")

        return receipt

    async def _spendable_budget(self) -> float:
        """Free cash, after reconciling what is already held.

        Returns 0.0 when the account cannot be read, which preflight treats
        as no budget rather than as unlimited.
        """
        try:
            def _read():
                broker = preflight.shared_broker()
                report = inventory_reconciler.reconcile_from_broker(
                    broker,
                    known_positions={
                        symbol: {**position, "owner_alive": True}
                        for symbol, position
                        in self.risk_manager.open_positions.items()
                    },
                )
                if not report.get("available"):
                    return 0.0
                return float(report["capital"]["spendable_quote"])

            return await asyncio.to_thread(_read)
        except Exception as exc:
            logger.debug(f"Spendable budget unavailable: {type(exc).__name__}")
            return 0.0

    async def _recent_closes(self, symbol: str, limit: int = 50) -> List[float]:
        """Real recent closes for ``symbol``, or [] if they cannot be read.

        Returns closes only -- never a padded or interpolated series. An empty
        list means the caller must not run analysis that assumes history.
        """
        normalized = preflight.normalize_symbol(symbol) or symbol

        def _fetch() -> List[float]:
            broker = preflight.shared_broker()

            # A venue that does not list the market has no history for it.
            from src.leantrader.universe.routing import may_call_venue

            allowed, _classification, _detail = may_call_venue(
                broker.exchange_id,
                normalized,
                environment=broker.resolve_mode(),
            )
            if not allowed:
                return []

            candles = broker.fetch_ohlcv(normalized, timeframe="1m", limit=limit)
            closes: List[float] = []
            for candle in candles or []:
                try:
                    close = float(candle[4])
                except (IndexError, TypeError, ValueError):
                    continue
                if close > 0.0:
                    closes.append(close)
            return closes

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:
            logger.debug(f"OHLCV unavailable for {symbol}: {type(exc).__name__}")
            return []

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Last price for ``symbol`` on the venue this orchestrator trades.

        The previous implementation built a fresh ccxt.async_support client on
        every call and only closed it on the success path, so each failed fetch
        leaked an aiohttp session -- from a loop that runs once per second per
        open position. It also fell back to Binance, marking positions held on
        one venue against another venue's book, which on testnet is a
        different market entirely.

        Both are gone: the price comes from the shared authenticated client,
        through a short-lived cache, off the event loop.
        """
        normalized = preflight.normalize_symbol(symbol) or symbol

        # A live engine client is still preferred when one is already open on
        # this venue -- it is warm and costs nothing extra.
        if self.engines:
            for engine_name, engine in self.engines.items():
                exchange = getattr(engine, 'exchange', None)
                if not exchange:
                    continue
                try:
                    ticker = await exchange.fetch_ticker(normalized)
                except Exception as e:
                    logger.debug(f"{engine_name} ticker failed: {type(e).__name__}")
                    continue
                last = (ticker or {}).get('last')
                if last:
                    return float(last)

        try:
            return await asyncio.to_thread(
                lambda: preflight.fetch_last_price_cached(
                    preflight.shared_broker(), normalized
                )
            )
        except Exception as e:
            logger.debug(f"Ticker unavailable for {symbol}: {type(e).__name__}")
            return None

    async def monitor_positions(self):
        """Monitor open positions with ADVANCED FEATURES (Trailing Stop, Partial TP)"""

        for symbol, position in list(self.risk_manager.open_positions.items()):
            try:
                # Get current price
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    continue

                entry_price = position['entry_price']
                side = position['side']

                # Calculate current P&L
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # USE TRAILING STOP if available
                if self.trailing_stop:
                    new_stop = self.trailing_stop.update(
                        symbol=symbol,
                        current_price=current_price,
                        entry_price=entry_price,
                        initial_stop=entry_price * 0.99  # 1% initial stop
                    )

                    # Check if trailing stop hit
                    if current_price <= new_stop:
                        logger.info(f"🛑 Trailing stop triggered: {symbol} ({pnl_pct:.2%})")
                        await self.close_position(symbol, current_price, 'trailing_stop')
                        continue

                # USE PARTIAL TP if available
                if self.partial_tp:
                    tp_orders = await self.partial_tp.check_tp_levels(symbol, current_price)
                    if tp_orders:
                        for order in tp_orders:
                            logger.info(f"🎯 {order['tp_level']} hit: {symbol}")
                            # Would execute partial close here

                # Fallback to basic stop loss and take profit
                else:
                    # Check stop loss (1% loss)
                    if pnl_pct < -0.01:
                        logger.warning(f"🛑 Stop loss triggered: {symbol} ({pnl_pct:.2%})")
                        await self.close_position(symbol, current_price, 'stop_loss')

                    # Check take profit (2% profit)
                    elif pnl_pct > 0.02:
                        logger.info(f"🎯 Take profit triggered: {symbol} ({pnl_pct:.2%})")
                        await self.close_position(symbol, current_price, 'take_profit')

            except Exception as e:
                logger.debug(f"Position monitoring error for {symbol}: {e}")

    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Submit the closing order, then record realized PnL from the fill.

        This used to compute a PnL from the in-memory entry price and a fetched
        ticker, publish a "closed" trade record carrying that number, and log a
        new balance -- without ever sending a closing order. The position stayed
        open on the exchange while the bot reported it closed and fed the
        invented PnL to the learning path.

        Now the opposite-side order goes through the universal router first. If
        it is not acknowledged the position stays open and the attempt is
        counted; nothing is published. Realized PnL is computed from the price
        the exchange actually filled at and the fees it actually charged.
        """
        position = self.risk_manager.open_positions.get(symbol)
        if not position:
            return None

        entry_price = float(position['entry_price'])
        size = float(position['size'])
        side = str(position['side']).lower()
        closing_side = 'sell' if side == 'buy' else 'buy'

        override = os.getenv('EXECUTION_EXCHANGE_OVERRIDE', '').strip().lower()
        payload = {
            'symbol': preflight.normalize_symbol(symbol) or symbol,
            'side': closing_side,
            'qty': size,
            'order_type': 'market',
            'reference_price': exit_price,
            'backend': 'ccxt',
        }
        if override:
            payload['exchange_id'] = override

        preflight.record_event('submitted')
        preflight.record_event('close_orders_submitted')
        started = time.time()
        try:
            receipt = await asyncio.to_thread(route_order, payload)
        except Exception as exc:
            preflight.record_blocker(
                preflight.EXCHANGE_REJECT, f"close:{symbol}:{type(exc).__name__}"
            )
            logger.error(f"Close order failed for {symbol}: {type(exc).__name__}")
            return None

        preflight.record_stage_latency('close_submit', time.time() - started)

        blocker = preflight.classify_receipt(receipt)
        if blocker is not None:
            detail = str((receipt or {}).get('error', ''))[:200]
            preflight.record_blocker(blocker, f"close:{symbol}:{detail}")
            logger.warning(
                f"🛑 {symbol} close NOT placed ({blocker}): position remains open"
            )
            return None

        order = receipt.get('order') or {}
        fill_price = order.get('average') or order.get('price')
        try:
            fill_price = float(fill_price)
        except (TypeError, ValueError):
            fill_price = 0.0

        if fill_price <= 0.0:
            # Acknowledged but no price to settle against. Leave the position
            # recorded rather than book a PnL we cannot substantiate.
            preflight.record_blocker(
                preflight.NO_ORDER_ID, f"close:{symbol}:no_fill_price"
            )
            logger.warning(
                f"🛑 {symbol} close acknowledged without a fill price; "
                "PnL not booked"
            )
            return None

        entry_fee, exit_fee = self._order_fees(symbol, order, fill_price, size)

        gross_pnl = (
            (fill_price - entry_price) * size
            if side == 'buy'
            else (entry_price - fill_price) * size
        )
        net_pnl = gross_pnl - entry_fee - exit_fee

        # Settle the in-memory book at the real fill, not at the ticker.
        self.risk_manager.close_position(symbol, fill_price)
        preflight.record_event('acknowledged')
        preflight.record_event('closes')
        preflight.record_event('close_orders_filled')
        preflight.record_event('reconciled_cycles')
        preflight.invalidate_balance_cache()

        if net_pnl > 0:
            self.winning_trades += 1
        self.total_profit += net_pnl

        close_record = {
            'symbol': symbol,
            'order_id': order.get('id'),
            'side': closing_side,
            'size': size,
            'entry_price': entry_price,
            'exit_price': fill_price,
            'gross_pnl': gross_pnl,
            'fees': entry_fee + exit_fee,
            'realized_pnl': net_pnl,
            'reason': reason,
            'exchange': receipt.get('exchange'),
            'execution_mode': receipt.get('execution_mode'),
            'timestamp': datetime.now(),
            'status': 'closed',
        }

        await self.data_hub.publish_trade(close_record)

        logger.info(f"💰 POSITION CLOSED:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Order ID: {order.get('id')}")
        logger.info(f"   Entry: {entry_price}")
        logger.info(f"   Exit fill: {fill_price}")
        logger.info(f"   Gross P&L: {gross_pnl:.8f}")
        logger.info(f"   Fees: {entry_fee + exit_fee:.8f}")
        logger.info(f"   Realized net P&L: {net_pnl:.8f}")
        logger.info(f"   Reason: {reason}")

        return close_record

    def _order_fees(self, symbol, order, fill_price, size):
        """Fees for the round trip, from the exchange where it reports them.

        ccxt puts the charged fee on the order as ``fee``/``fees``. When the
        venue does not return one, the market's taker rate is applied to both
        legs -- an estimate, and labelled as one on the record it feeds.
        """
        def _fee_of(payload):
            if isinstance(payload, dict):
                cost = payload.get('cost')
                if cost is not None:
                    try:
                        return abs(float(cost))
                    except (TypeError, ValueError):
                        return None
            return None

        exit_fee = _fee_of(order.get('fee'))
        if exit_fee is None:
            for entry in order.get('fees') or []:
                candidate = _fee_of(entry)
                if candidate is not None:
                    exit_fee = (exit_fee or 0.0) + candidate

        rate = 0.001
        try:
            broker = preflight.shared_broker()
            market = preflight.load_markets_cached(broker).get(
                preflight.normalize_symbol(symbol) or symbol
            )
            if isinstance(market, dict) and market.get('taker') is not None:
                rate = float(market['taker'])
        except Exception:
            pass

        notional = fill_price * size
        if exit_fee is None:
            exit_fee = notional * rate

        # The entry fee belongs to the opening order, which is not in hand
        # here; it is estimated at the same rate against the entry notional.
        entry_fee = notional * rate

        return entry_fee, exit_fee

    def get_win_rate(self) -> float:
        """Get current win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def get_stats(self) -> Dict[str, Any]:
        """Execution statistics.

        Counts here are acknowledged orders and settled closes only. The
        blocker breakdown says where every other attempt stopped, so a run
        with no trades is attributable. ``current_balance`` is not reported:
        this object does not hold one, and the placeholder it used to print
        was a fixed $1000 that no account ever had.
        """
        telemetry = preflight.telemetry_snapshot()
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.get_win_rate(),
            'realized_net_profit': self.total_profit,
            'open_positions': len(self.risk_manager.open_positions),
            'daily_pnl': self.risk_manager.daily_pnl,
            'daily_trades': self.risk_manager.daily_trades,
            'avg_execution_time': sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0,
            'attempts': telemetry.get('attempts', 0),
            'fills': telemetry.get('fills', 0),
            'closes': telemetry.get('closes', 0),
            'prepared': telemetry.get('prepared', 0),
            'submitted': telemetry.get('submitted', 0),
            'acknowledged': telemetry.get('acknowledged', 0),
            'blockers': telemetry.get('blockers', {}),
        }
