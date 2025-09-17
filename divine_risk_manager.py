"""
Divine Risk Management System
Ensures safe and profitable trading with multiple safety layers
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


class DivineRiskManager:
    """
    Comprehensive risk management system with multiple safety layers.
    Protects capital while allowing for growth.
    """
    
    def __init__(self, config_path: str = "divine_config.yml"):
        """Initialize risk manager with configuration."""
        self.config = self._load_config(config_path)
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        self.peak_balance = self.config['trading']['initial_capital']
        self.current_balance = self.config['trading']['initial_capital']
        self.trade_history = []
        self.is_paused = False
        self.pause_until = None
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Risk metrics
        self.risk_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'recovery_factor': 0.0
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default safe config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default safe configuration."""
        return {
            'trading': {'initial_capital': 100.0},
            'risk_management': {
                'risk_per_trade_percent': 1.0,
                'max_daily_loss_percent': 5.0,
                'max_consecutive_losses': 3,
                'default_stop_loss_percent': 2.0,
                'default_take_profit_percent': 4.0
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging system."""
        logger = logging.getLogger('DivineRiskManager')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler('risk_management.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def check_pre_trade(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Comprehensive pre-trade risk check.
        Returns (can_trade, reason)
        """
        
        # Check if trading is paused
        if self.is_paused:
            if self.pause_until and datetime.now() < self.pause_until:
                return False, f"Trading paused until {self.pause_until}"
            else:
                self.is_paused = False
                self.pause_until = None
        
        # Check daily trade limit
        max_daily = self.config['safety']['max_daily_trades']
        if self.trades_today >= max_daily:
            return False, f"Daily trade limit reached ({max_daily})"
        
        # Check daily loss limit
        max_daily_loss = self.config['risk_management']['max_daily_loss_percent']
        daily_loss_pct = (self.daily_pnl / self.current_balance) * 100
        if daily_loss_pct <= -max_daily_loss:
            self._pause_trading(hours=24)
            return False, f"Daily loss limit reached ({daily_loss_pct:.2f}%)"
        
        # Check consecutive losses
        max_consecutive = self.config['risk_management']['max_consecutive_losses']
        if self.consecutive_losses >= max_consecutive:
            self._pause_trading(hours=1)
            return False, f"Max consecutive losses reached ({self.consecutive_losses})"
        
        # Check drawdown
        max_dd = self.config['risk_management']['max_drawdown_percent']
        if self.current_drawdown >= max_dd:
            self._pause_trading(hours=48)
            return False, f"Maximum drawdown reached ({self.current_drawdown:.2f}%)"
        
        # Check signal confidence
        min_confidence = self.config['divine_intelligence']['min_signal_confidence']
        if signal.get('confidence', 0) < min_confidence:
            return False, f"Signal confidence too low ({signal.get('confidence', 0):.2f})"
        
        # Check divine confirmations
        divine_confirms = self._count_divine_confirmations(signal)
        required = self.config['divine_intelligence']['divine_confirmation_required']
        if divine_confirms < required:
            return False, f"Insufficient divine confirmations ({divine_confirms}/{required})"
        
        # Check position sizing
        position_size = self.calculate_position_size(signal)
        if position_size <= 0:
            return False, "Position size too small"
        
        # Check correlation with existing positions
        if not self._check_correlation(signal):
            return False, "Position too correlated with existing trades"
        
        # All checks passed
        self.logger.info(f"Pre-trade checks passed for {signal.get('symbol')}")
        return True, "All risk checks passed"
    
    def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate safe position size using multiple methods.
        """
        
        # Get base position size from config
        risk_pct = self.config['risk_management']['risk_per_trade_percent']
        
        # Adjust for current drawdown (reduce size in drawdown)
        if self.current_drawdown > 10:
            risk_pct *= 0.5  # Half size in drawdown
        elif self.current_drawdown > 5:
            risk_pct *= 0.75  # 75% size in mild drawdown
        
        # Adjust for consecutive losses
        if self.consecutive_losses > 0:
            risk_pct *= (1 - self.consecutive_losses * 0.2)  # Reduce 20% per loss
        
        # Calculate dollar risk
        dollar_risk = self.current_balance * (risk_pct / 100)
        
        # Adjust for signal confidence
        confidence = signal.get('confidence', 0.5)
        dollar_risk *= confidence
        
        # Adjust for divine intelligence scores
        if 'divine_intelligence' in signal:
            divine = signal['divine_intelligence']
            if divine.get('manifestation_probability', 0) > 0.8:
                dollar_risk *= 1.5  # Increase for high manifestation
            elif divine.get('manifestation_probability', 0) < 0.3:
                dollar_risk *= 0.5  # Decrease for low manifestation
        
        # Apply min/max constraints
        min_trade = self.config['trading']['min_trade_amount']
        max_trade = self.config['trading']['max_trade_amount']
        
        position_size = np.clip(dollar_risk, min_trade, max_trade)
        
        # Final check: don't risk more than 10% of account on single trade
        max_position = self.current_balance * 0.1
        position_size = min(position_size, max_position)
        
        self.logger.info(f"Calculated position size: ${position_size:.2f}")
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, signal: Dict[str, Any]) -> float:
        """
        Calculate intelligent stop loss based on multiple factors.
        """
        
        # Base stop loss from config
        base_stop_pct = self.config['risk_management']['default_stop_loss_percent']
        
        # Adjust for volatility
        if 'volatility' in signal:
            vol = signal['volatility']
            if vol > 0.03:  # High volatility
                base_stop_pct *= 1.5  # Wider stop
            elif vol < 0.01:  # Low volatility
                base_stop_pct *= 0.75  # Tighter stop
        
        # Use sacred levels if available
        if 'divine_intelligence' in signal:
            sacred = signal['divine_intelligence'].get('sacred_levels', [])
            if sacred and signal.get('side') == 'BUY':
                # Find nearest support below entry
                supports = [s for s in sacred if s < entry_price]
                if supports:
                    sacred_stop = max(supports)
                    sacred_stop_pct = ((entry_price - sacred_stop) / entry_price) * 100
                    base_stop_pct = min(base_stop_pct, sacred_stop_pct)
        
        # Calculate stop loss price
        if signal.get('side') == 'BUY':
            stop_loss = entry_price * (1 - base_stop_pct / 100)
        else:
            stop_loss = entry_price * (1 + base_stop_pct / 100)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            signal: Dict[str, Any]) -> List[float]:
        """
        Calculate multiple take profit levels.
        """
        
        # Calculate risk
        risk = abs(entry_price - stop_loss)
        
        # Base risk-reward ratio
        base_rr = self.config['risk_management']['default_take_profit_percent'] / \
                 self.config['risk_management']['default_stop_loss_percent']
        
        # Adjust for divine signals
        if 'divine_intelligence' in signal:
            divine = signal['divine_intelligence']
            if divine.get('optimal_timeline', {}).get('profit_potential', 0) > 0.05:
                base_rr *= 1.5  # Increase targets for high potential
        
        # Calculate multiple targets
        targets = []
        
        if signal.get('side') == 'BUY':
            # First target: 1:1 RR
            targets.append(entry_price + risk)
            # Second target: 2:1 RR
            targets.append(entry_price + risk * 2)
            # Third target: Based on divine intelligence
            targets.append(entry_price + risk * base_rr)
        else:
            # First target: 1:1 RR
            targets.append(entry_price - risk)
            # Second target: 2:1 RR
            targets.append(entry_price - risk * 2)
            # Third target: Based on divine intelligence
            targets.append(entry_price - risk * base_rr)
        
        return targets
    
    def update_trade_result(self, trade: Dict[str, Any]):
        """
        Update risk metrics after trade completion.
        """
        
        # Update trade history
        self.trade_history.append(trade)
        
        # Update daily PnL
        self.daily_pnl += trade.get('pnl', 0)
        
        # Update consecutive losses
        if trade.get('pnl', 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update balance
        self.current_balance += trade.get('pnl', 0)
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = ((self.peak_balance - self.current_balance) / 
                                    self.peak_balance) * 100
        
        # Update metrics
        self._update_risk_metrics()
        
        # Log trade
        self.logger.info(f"Trade completed: {trade.get('symbol')} "
                        f"PnL: ${trade.get('pnl', 0):.2f} "
                        f"Balance: ${self.current_balance:.2f}")
        
        # Check if we need to pause
        if self.consecutive_losses >= self.config['risk_management']['max_consecutive_losses']:
            self._pause_trading(hours=1)
    
    def _update_risk_metrics(self):
        """Update performance metrics."""
        
        if len(self.trade_history) < 2:
            return
        
        # Win rate
        wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        self.risk_metrics['win_rate'] = wins / len(self.trade_history)
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0))
        if gross_loss > 0:
            self.risk_metrics['profit_factor'] = gross_profit / gross_loss
        
        # Sharpe ratio (simplified)
        returns = [t.get('pnl', 0) / self.current_balance for t in self.trade_history]
        if len(returns) > 1:
            self.risk_metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-10)
        
        # Max drawdown
        self.risk_metrics['max_drawdown'] = max(self.current_drawdown, 
                                               self.risk_metrics['max_drawdown'])
    
    def _count_divine_confirmations(self, signal: Dict[str, Any]) -> int:
        """Count how many divine systems confirm the trade."""
        
        confirmations = 0
        
        if 'divine_intelligence' not in signal:
            return 0
        
        divine = signal['divine_intelligence']
        
        # Check each divine system
        if divine.get('consciousness_level', 0) > 500:
            confirmations += 1
        
        if divine.get('akashic_confidence', 0) > 0.6:
            confirmations += 1
        
        if divine.get('temporal_arbitrage', {}).get('exists', False):
            confirmations += 1
        
        if divine.get('manifestation_probability', 0) > 0.6:
            confirmations += 1
        
        if divine.get('karmic_direction') == signal.get('side', '').lower():
            confirmations += 1
        
        return confirmations
    
    def _check_correlation(self, signal: Dict[str, Any]) -> bool:
        """Check if new position is too correlated with existing."""
        
        # Simplified correlation check
        # In production, would check actual price correlation
        
        # For now, just limit same symbol trades
        symbol = signal.get('symbol')
        open_symbols = [t['symbol'] for t in self.trade_history 
                       if t.get('status') == 'open']
        
        if symbol in open_symbols:
            return False
        
        return True
    
    def _pause_trading(self, hours: int):
        """Pause trading for specified hours."""
        
        self.is_paused = True
        self.pause_until = datetime.now() + timedelta(hours=hours)
        
        self.logger.warning(f"Trading paused until {self.pause_until}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        
        return {
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'trades_today': self.trades_today,
            'is_paused': self.is_paused,
            'pause_until': str(self.pause_until) if self.pause_until else None,
            'risk_metrics': self.risk_metrics,
            'trade_count': len(self.trade_history),
            'status': 'PAUSED' if self.is_paused else 'ACTIVE'
        }
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of new day)."""
        
        self.trades_today = 0
        self.daily_pnl = 0.0
        
        self.logger.info("Daily counters reset")
    
    def emergency_stop(self):
        """Emergency stop - pause all trading immediately."""
        
        self.is_paused = True
        self.pause_until = datetime.now() + timedelta(hours=24)
        
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        
        return "Emergency stop activated - trading paused for 24 hours"


class PositionManager:
    """
    Manages open positions with trailing stops and partial profits.
    """
    
    def __init__(self, risk_manager: DivineRiskManager):
        self.risk_manager = risk_manager
        self.open_positions = {}
        self.closed_positions = []
        
    def open_position(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Open a new position with proper risk management."""
        
        # Check if we can trade
        can_trade, reason = self.risk_manager.check_pre_trade(signal)
        if not can_trade:
            return {'success': False, 'reason': reason}
        
        # Calculate position details
        position_size = self.risk_manager.calculate_position_size(signal)
        entry_price = signal.get('entry_price', 0)
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, signal)
        take_profits = self.risk_manager.calculate_take_profit(entry_price, stop_loss, signal)
        
        # Create position
        position = {
            'id': f"{signal['symbol']}_{datetime.now().timestamp()}",
            'symbol': signal['symbol'],
            'side': signal['side'],
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'current_tp_index': 0,
            'partial_closed': 0,
            'status': 'open',
            'open_time': datetime.now(),
            'divine_signal': signal.get('divine_intelligence', {})
        }
        
        # Store position
        self.open_positions[position['id']] = position
        
        # Update risk manager
        self.risk_manager.trades_today += 1
        
        return {
            'success': True,
            'position': position
        }
    
    def update_position(self, position_id: str, current_price: float) -> Dict[str, Any]:
        """Update position with trailing stop and partial profits."""
        
        if position_id not in self.open_positions:
            return {'success': False, 'reason': 'Position not found'}
        
        position = self.open_positions[position_id]
        updates = {}
        
        # Check for partial profit taking
        if position['current_tp_index'] < len(position['take_profits']):
            tp = position['take_profits'][position['current_tp_index']]
            
            if position['side'] == 'BUY' and current_price >= tp:
                # Take partial profit
                updates['partial_close'] = self._take_partial_profit(position, current_price)
                position['current_tp_index'] += 1
                
            elif position['side'] == 'SELL' and current_price <= tp:
                # Take partial profit
                updates['partial_close'] = self._take_partial_profit(position, current_price)
                position['current_tp_index'] += 1
        
        # Update trailing stop
        new_stop = self._calculate_trailing_stop(position, current_price)
        if new_stop:
            position['stop_loss'] = new_stop
            updates['new_stop'] = new_stop
        
        # Check stop loss
        if self._check_stop_loss(position, current_price):
            updates['close'] = self._close_position(position, current_price, 'stop_loss')
        
        return {
            'success': True,
            'updates': updates
        }
    
    def _take_partial_profit(self, position: Dict, current_price: float) -> Dict:
        """Take partial profit on position."""
        
        # Calculate partial close amount (33% each time)
        partial_amount = position['position_size'] * 0.33
        
        # Calculate profit
        if position['side'] == 'BUY':
            profit = (current_price - position['entry_price']) * partial_amount
        else:
            profit = (position['entry_price'] - current_price) * partial_amount
        
        # Update position
        position['partial_closed'] += partial_amount
        position['position_size'] -= partial_amount
        
        return {
            'amount': partial_amount,
            'profit': profit,
            'remaining': position['position_size']
        }
    
    def _calculate_trailing_stop(self, position: Dict, current_price: float) -> Optional[float]:
        """Calculate trailing stop for position."""
        
        trail_pct = self.risk_manager.config['risk_management']['trailing_stop_percent']
        
        if position['side'] == 'BUY':
            # For long positions
            trail_stop = current_price * (1 - trail_pct / 100)
            if trail_stop > position['stop_loss']:
                return trail_stop
                
        else:
            # For short positions
            trail_stop = current_price * (1 + trail_pct / 100)
            if trail_stop < position['stop_loss']:
                return trail_stop
        
        return None
    
    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss is hit."""
        
        if position['side'] == 'BUY':
            return current_price <= position['stop_loss']
        else:
            return current_price >= position['stop_loss']
    
    def _close_position(self, position: Dict, current_price: float, reason: str) -> Dict:
        """Close position and calculate PnL."""
        
        # Calculate final PnL
        if position['side'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['position_size']
        else:
            pnl = (position['entry_price'] - current_price) * position['position_size']
        
        # Add partial profits
        pnl += position.get('partial_profits', 0)
        
        # Update position
        position['status'] = 'closed'
        position['close_price'] = current_price
        position['close_time'] = datetime.now()
        position['close_reason'] = reason
        position['pnl'] = pnl
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[position['id']]
        
        # Update risk manager
        self.risk_manager.update_trade_result({
            'symbol': position['symbol'],
            'pnl': pnl,
            'side': position['side']
        })
        
        return {
            'pnl': pnl,
            'reason': reason,
            'duration': (position['close_time'] - position['open_time']).total_seconds()
        }
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return list(self.open_positions.values())
    
    def close_all_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Close all open positions (emergency or end of day)."""
        
        results = []
        positions = list(self.open_positions.values())
        
        for position in positions:
            symbol = position['symbol']
            current_price = current_prices.get(symbol, position['entry_price'])
            
            result = self._close_position(position, current_price, 'force_close')
            results.append({
                'symbol': symbol,
                'pnl': result['pnl']
            })
        
        return results


# Integration function
def integrate_risk_management(pipeline):
    """Integrate Divine Risk Management into trading pipeline."""
    
    # Initialize risk manager
    risk_manager = DivineRiskManager()
    position_manager = PositionManager(risk_manager)
    
    # Add to pipeline
    pipeline.risk_manager = risk_manager
    pipeline.position_manager = position_manager
    
    print("✅ Divine Risk Management System Activated")
    print("  🛡️ Multi-layer protection enabled")
    print("  📊 Position sizing optimized")
    print("  🎯 Stop loss & take profit automated")
    print("  ⚠️ Emergency stops ready")
    
    return pipeline


if __name__ == "__main__":
    # Test risk management system
    risk_manager = DivineRiskManager()
    
    print("Divine Risk Management System Test")
    print("="*50)
    
    # Test signal
    test_signal = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'entry_price': 50000,
        'confidence': 0.85,
        'volatility': 0.02,
        'divine_intelligence': {
            'consciousness_level': 600,
            'akashic_confidence': 0.75,
            'manifestation_probability': 0.8,
            'karmic_direction': 'up'
        }
    }
    
    # Check pre-trade
    can_trade, reason = risk_manager.check_pre_trade(test_signal)
    print(f"Can trade: {can_trade}")
    print(f"Reason: {reason}")
    
    # Calculate position size
    position_size = risk_manager.calculate_position_size(test_signal)
    print(f"Position size: ${position_size:.2f}")
    
    # Calculate stops and targets
    stop_loss = risk_manager.calculate_stop_loss(50000, test_signal)
    take_profits = risk_manager.calculate_take_profit(50000, stop_loss, test_signal)
    
    print(f"Stop loss: ${stop_loss:.2f}")
    print(f"Take profits: {[f'${tp:.2f}' for tp in take_profits]}")
    
    # Get risk report
    report = risk_manager.get_risk_report()
    print(f"\nRisk Report:")
    print(json.dumps(report, indent=2))