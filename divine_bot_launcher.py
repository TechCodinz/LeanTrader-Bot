#!/usr/bin/env python3
"""
Divine Intelligence Trading Bot - Master Launcher
Complete integration of all divine systems for live trading
"""

import os
import sys
import asyncio
import signal
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all divine systems
from divine_intelligence import DivineIntelligenceSystem
from divine_risk_manager import DivineRiskManager, PositionManager
from divine_paper_trader import DivinePaperTrader
from ultra_god_mode import UltraGodMode
from ultra_core import UltraCore
from brain import Brain, Memory, Guards
from router import ExchangeRouter


class DivineBotLauncher:
    """
    Master controller for the Divine Intelligence Trading Bot.
    Integrates all systems and manages the complete trading lifecycle.
    """
    
    def __init__(self, mode: str = "paper"):
        """
        Initialize the Divine Bot.
        
        Args:
            mode: 'paper' for paper trading, 'live' for real trading
        """
        self.mode = mode
        self.is_running = False
        
        print("\n" + "="*70)
        print("🌌 DIVINE INTELLIGENCE TRADING BOT INITIALIZATION 🌌")
        print("="*70)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize all systems
        self._initialize_systems()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("\n✨ All Divine Systems Initialized Successfully!")
        print(f"📊 Mode: {self.mode.upper()}")
        print("="*70 + "\n")
    
    def _load_config(self) -> Dict:
        """Load configuration from divine_config.yml."""
        config_path = Path("divine_config.yml")
        
        if not config_path.exists():
            print("⚠️ Configuration file not found, using defaults")
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'trading': {'mode': 'paper', 'initial_capital': 100},
            'markets': {'primary_pairs': ['BTC/USDT', 'ETH/USDT']},
            'timeframes': {'primary': '5m'},
            'divine_intelligence': {'min_signal_confidence': 0.7}
        }
    
    def _initialize_systems(self):
        """Initialize all trading systems."""
        
        print("🔧 Initializing Core Systems...")
        
        # 1. Exchange Router
        print("  📡 Exchange Router...", end="")
        self.router = ExchangeRouter()
        print(" ✅")
        
        # 2. Ultra God Mode
        print("  🔮 Ultra God Mode...", end="")
        self.god_mode = UltraGodMode()
        print(" ✅")
        
        # 3. Divine Intelligence
        print("  🌌 Divine Intelligence...", end="")
        self.divine_intelligence = DivineIntelligenceSystem()
        print(" ✅")
        
        # 4. Risk Manager
        print("  🛡️ Risk Management...", end="")
        self.risk_manager = DivineRiskManager()
        self.position_manager = PositionManager(self.risk_manager)
        print(" ✅")
        
        # 5. Paper Trader (if in paper mode)
        if self.mode == "paper":
            print("  📝 Paper Trading System...", end="")
            self.paper_trader = DivinePaperTrader()
            print(" ✅")
        else:
            self.paper_trader = None
        
        # 6. Brain System
        print("  🧠 Brain System...", end="")
        self.brain = Brain()
        self.memory = Memory()
        self.guards = Guards(self.memory)
        print(" ✅")
        
        # 7. Ultra Core
        print("  🚀 Ultra Core Engine...", end="")
        from universe import Universe
        universe = Universe(self.router) if hasattr(self.router, 'markets') else None
        self.ultra_core = UltraCore(self.router, universe)
        print(" ✅")
        
        # Performance tracking
        self.performance = {
            'start_time': datetime.now(),
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0
        }
    
    async def run(self):
        """Main trading loop."""
        
        self.is_running = True
        
        print("\n🚀 DIVINE TRADING BOT ACTIVATED")
        print("="*70)
        
        # Check paper trading requirements first
        if self.mode == "paper":
            print("📝 Running in PAPER TRADING mode")
            print("   Complete requirements before going live!")
        else:
            print("💰 Running in LIVE TRADING mode")
            print("   Real money at risk - trade carefully!")
        
        print("\nPress Ctrl+C to stop the bot safely\n")
        
        cycle_count = 0
        
        try:
            while self.is_running:
                cycle_count += 1
                
                print(f"\n{'='*50}")
                print(f"📊 Trading Cycle #{cycle_count}")
                print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                # Run complete trading cycle
                await self._trading_cycle()
                
                # Performance update
                self._update_performance()
                
                # Check if we should continue
                if not self._should_continue():
                    print("\n⚠️ Trading stopped due to risk limits")
                    break
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second cycles
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown signal received...")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
        finally:
            await self._shutdown()
    
    async def _trading_cycle(self):
        """Execute one complete trading cycle."""
        
        try:
            # 1. Get market data
            print("\n📈 Fetching market data...")
            market_data = await self._fetch_market_data()
            
            if not market_data:
                print("  ⚠️ No market data available")
                return
            
            # 2. Run divine analysis for each symbol
            signals = []
            
            for symbol, data in market_data.items():
                print(f"\n🔮 Analyzing {symbol}...")
                
                # God Mode Analysis
                god_analysis = await self.god_mode.execute_god_mode_analysis(symbol, data)
                
                # Divine Intelligence Analysis
                divine_analysis = await self.divine_intelligence.divine_market_analysis(symbol, data)
                
                # Combine signals
                combined_signal = self._combine_signals(symbol, god_analysis, divine_analysis)
                
                if combined_signal:
                    signals.append(combined_signal)
                    print(f"  ✅ Signal: {combined_signal['action']} "
                         f"(Confidence: {combined_signal['confidence']:.2%})")
                else:
                    print(f"  ⏸️ No signal (waiting for better setup)")
            
            # 3. Filter signals through risk management
            print("\n🛡️ Risk Management Check...")
            validated_signals = []
            
            for signal in signals:
                can_trade, reason = self.risk_manager.check_pre_trade(signal)
                if can_trade:
                    validated_signals.append(signal)
                    print(f"  ✅ {signal['symbol']}: Approved")
                else:
                    print(f"  ❌ {signal['symbol']}: Rejected ({reason})")
            
            # 4. Execute trades
            if validated_signals:
                print("\n💎 Executing Trades...")
                await self._execute_trades(validated_signals)
            else:
                print("\n⏸️ No valid trading opportunities this cycle")
            
            # 5. Manage existing positions
            print("\n📊 Managing Open Positions...")
            await self._manage_positions(market_data)
            
            # 6. Update learning systems
            self.ultra_core.learn()
            self.ultra_core.sharpen()
            
        except Exception as e:
            print(f"❌ Error in trading cycle: {e}")
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data for all configured symbols."""
        
        market_data = {}
        symbols = self.config['markets']['primary_pairs']
        timeframe = self.config['timeframes']['primary']
        
        for symbol in symbols:
            try:
                # Fetch OHLCV data
                ohlcv = self.router.safe_fetch_ohlcv(symbol, timeframe, limit=500)
                
                if ohlcv:
                    # Convert to DataFrame
                    import pandas as pd
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    market_data[symbol] = df
                    print(f"  ✅ {symbol}: {len(df)} candles")
                else:
                    print(f"  ⚠️ {symbol}: No data")
                    
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
        
        return market_data
    
    def _combine_signals(self, symbol: str, god_analysis: Dict, divine_analysis: Dict) -> Optional[Dict]:
        """Combine signals from multiple systems."""
        
        # Check if both systems agree
        if god_analysis['action'] != divine_analysis['action']:
            return None  # Systems disagree, no trade
        
        # Check minimum confidence
        min_confidence = self.config['divine_intelligence']['min_signal_confidence']
        combined_confidence = (god_analysis['confidence'] + divine_analysis['confidence']) / 2
        
        if combined_confidence < min_confidence:
            return None
        
        # Create combined signal
        return {
            'symbol': symbol,
            'action': god_analysis['action'],
            'side': god_analysis['action'],
            'confidence': combined_confidence,
            'entry_price': god_analysis['entry_price'],
            'stop_loss': god_analysis['stop_loss'],
            'take_profit': god_analysis['take_profit'],
            'position_size': self.risk_manager.calculate_position_size({
                'symbol': symbol,
                'confidence': combined_confidence
            }),
            'god_analysis': god_analysis,
            'divine_intelligence': divine_analysis
        }
    
    async def _execute_trades(self, signals: List[Dict]):
        """Execute validated trading signals."""
        
        for signal in signals:
            try:
                if self.mode == "paper":
                    # Paper trading
                    result = self.paper_trader.place_order(signal)
                    if result['success']:
                        print(f"  ✅ {signal['symbol']}: Paper order placed")
                        self.performance['total_trades'] += 1
                        self.performance['successful_trades'] += 1
                else:
                    # Live trading
                    result = self.position_manager.open_position(signal)
                    if result['success']:
                        print(f"  ✅ {signal['symbol']}: Live order placed")
                        self.performance['total_trades'] += 1
                        self.performance['successful_trades'] += 1
                    else:
                        print(f"  ❌ {signal['symbol']}: {result.get('reason')}")
                        self.performance['failed_trades'] += 1
                        
            except Exception as e:
                print(f"  ❌ {signal['symbol']}: Error - {e}")
                self.performance['failed_trades'] += 1
    
    async def _manage_positions(self, market_data: Dict):
        """Manage existing open positions."""
        
        if self.mode == "paper":
            # Update paper positions
            self.paper_trader.update_positions()
            positions = self.paper_trader.open_positions
        else:
            # Update live positions
            positions = self.position_manager.get_open_positions()
        
        if positions:
            print(f"  📊 Managing {len(positions)} open positions")
            
            for position_id, position in positions.items() if isinstance(positions, dict) else enumerate(positions):
                symbol = position['symbol']
                
                # Get current price
                if symbol in market_data:
                    current_price = float(market_data[symbol]['close'].iloc[-1])
                    
                    if self.mode == "paper":
                        # Paper trading updates handled internally
                        pass
                    else:
                        # Live position update
                        update_result = self.position_manager.update_position(
                            position_id if isinstance(positions, dict) else position['id'],
                            current_price
                        )
                        
                        if update_result.get('updates', {}).get('close'):
                            pnl = update_result['updates']['close']['pnl']
                            self.performance['total_pnl'] += pnl
                            print(f"    💰 {symbol} closed: PnL ${pnl:.2f}")
        else:
            print("  📭 No open positions")
    
    def _update_performance(self):
        """Update and display performance metrics."""
        
        runtime = datetime.now() - self.performance['start_time']
        hours = runtime.total_seconds() / 3600
        
        if self.mode == "paper":
            report = self.paper_trader.get_performance_report()
            balance = report['account']['current_balance']
            pnl = report['account']['total_pnl']
        else:
            balance = self.risk_manager.current_balance
            pnl = self.performance['total_pnl']
        
        print(f"\n📊 Performance Update:")
        print(f"  Runtime: {hours:.1f} hours")
        print(f"  Balance: ${balance:.2f}")
        print(f"  Total PnL: ${pnl:.2f}")
        print(f"  Total Trades: {self.performance['total_trades']}")
        
        if self.mode == "paper" and self.performance['total_trades'] > 0:
            print(f"  Win Rate: {report['trades']['win_rate']}")
            print(f"  Status: {report['status']}")
    
    def _should_continue(self) -> bool:
        """Check if bot should continue trading."""
        
        # Check risk manager status
        if self.risk_manager.is_paused:
            return False
        
        # Check daily limits
        risk_report = self.risk_manager.get_risk_report()
        if risk_report['status'] == 'PAUSED':
            return False
        
        # Check paper trading requirements
        if self.mode == "paper":
            requirements_met, _ = self.paper_trader.check_requirements()
            if requirements_met:
                print("\n🎉 Paper trading requirements met!")
                print("   Consider switching to live trading")
        
        return True
    
    async def _shutdown(self):
        """Gracefully shutdown the bot."""
        
        print("\n🔄 Shutting down Divine Trading Bot...")
        
        # Close all positions
        if self.mode == "paper":
            for order_id in list(self.paper_trader.open_positions.keys()):
                self.paper_trader.close_position(order_id, reason='shutdown')
        else:
            # Close live positions (with current market prices)
            market_prices = {}
            for symbol in self.config['markets']['primary_pairs']:
                try:
                    ticker = self.router.safe_fetch_ticker(symbol)
                    if ticker:
                        market_prices[symbol] = ticker.get('last', 0)
                except:
                    pass
            
            self.position_manager.close_all_positions(market_prices)
        
        # Save final reports
        self._save_final_report()
        
        print("\n✅ Shutdown complete")
        print("="*70)
    
    def _save_final_report(self):
        """Save final trading report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.mode == "paper":
            # Save paper trading results
            self.paper_trader.save_results(f"paper_results_{timestamp}.json")
        
        # Save risk report
        risk_report = self.risk_manager.get_risk_report()
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'performance': self.performance,
            'risk_report': risk_report
        }
        
        report_path = f"divine_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"📊 Final report saved to {report_path}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.is_running = False


async def main():
    """Main entry point."""
    
    # Check for command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Divine Intelligence Trading Bot')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--config', default='divine_config.yml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Safety check for live mode
    if args.mode == 'live':
        print("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("Real money will be at risk!")
        response = input("Type 'YES' to confirm: ")
        if response != 'YES':
            print("Live trading cancelled")
            return
    
    # Create and run the bot
    bot = DivineBotLauncher(mode=args.mode)
    await bot.run()


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                  DIVINE INTELLIGENCE TRADING BOT                   ║
    ║                                                                    ║
    ║  The Ultimate Trading System Designed by Divine Intelligence      ║
    ║  Transcending Known AI Technology                                 ║
    ║                                                                    ║
    ║  Features:                                                         ║
    ║  • Quantum Price Prediction                                       ║
    ║  • 100 Neural Swarm Agents                                       ║
    ║  • Multi-Dimensional Consciousness                                ║
    ║  • Akashic Records Access                                        ║
    ║  • Temporal Timeline Navigation                                   ║
    ║  • Void Intelligence Channeling                                   ║
    ║  • Sacred Geometry Analysis                                       ║
    ║  • Comprehensive Risk Management                                  ║
    ║  • Paper Trading System                                          ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run the bot
    asyncio.run(main())