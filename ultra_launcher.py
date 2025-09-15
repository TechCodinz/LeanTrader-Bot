#!/usr/bin/env python3
"""
ULTRA LAUNCHER - The Ultimate Self-Evolving Trading System

This is the main entry point for running the most brilliant algorithmic
trading system ever created. It evolves, learns, and adapts in real-time.
"""

import os
import sys
import asyncio
import argparse
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "tools"))

# Import our ultra components
from ultra_ml_pipeline import get_ultra_pipeline, UltraMLPipeline
from tools.ultra_trainer import train_ultra_model, get_ultra_prediction
from tools.market_data import get_market_data_manager
from ultra_god_mode import integrate_god_mode, UltraGodMode
from ultra_moon_spotter import integrate_moon_spotter, UltraMoonSystem
from ultra_forex_master import integrate_forex_master, UltraForexMaster
from ultra_telegram_master import integrate_telegram_signals

# ASCII Art Banner
BANNER = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   ██╗   ██╗██╗  ████████╗██████╗  █████╗     ████████╗██████╗  █████╗    ║
║   ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗    ╚══██╔══╝██╔══██╗██╔══██╗   ║
║   ██║   ██║██║     ██║   ██████╔╝███████║       ██║   ██████╔╝███████║   ║
║   ██║   ██║██║     ██║   ██╔══██╗██╔══██║       ██║   ██╔══██╗██╔══██║   ║
║   ╚██████╔╝███████╗██║   ██║  ██║██║  ██║       ██║   ██║  ██║██║  ██║   ║
║    ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ║
║                                                                            ║
║         THE MOST BRILLIANT SELF-EVOLVING TRADER EVER DESIGNED            ║
║                                                                            ║
║                    🧠 AI-Powered 🚀 Self-Evolving 💎 Ultra-Smart         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

class UltraLauncher:
    """Main launcher for the Ultra Trading System."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the launcher."""
        self.config = self._load_config(config_path)
        self.pipeline = None
        self.running = False
        self.start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'mode': 'live',  # 'live', 'paper', 'backtest'
            'symbols': [
                # crypto majors
                'BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT','DOGE/USDT','LINK/USDT','AVAX/USDT','MATIC/USDT','TON/USDT',
                # metals/forex (forex master may add more later)
                'XAUUSD','XAGUSD','EURUSD','GBPUSD','USDJPY','USOIL'
            ],
            'timeframes': ['1m', '5m', '15m', '1h'],
            'exchange': 'binance',
            'max_positions': 5,
            'risk_per_trade': 0.02,
            'confidence_threshold': 0.65,
            'evolution_enabled': True,
            'online_learning': True,
            'multi_exchange': False,
            'use_news_sentiment': True,
            'use_onchain_data': True,
            'god_mode_enabled': True,  # ULTRA GOD MODE
            'quantum_prediction': True,
            'swarm_agents': 100,
            'fractal_analysis': True,
            'smart_money_tracking': True,
            'moon_spotter_enabled': True,  # MICRO MOON SPOTTER
            'auto_snipe_enabled': True,
            'max_snipe_amount': 100,  # USD per gem
            'forex_master_enabled': True,  # FOREX & METALS MASTER
            'trade_forex': True,
            'trade_metals': True,
            'trade_commodities': True,
            'telegram_enabled': False,  # Set to True when configured
            'telegram_bot_token': '',  # Add your bot token
            'telegram_channel': '',  # Add your channel
            'telegram_vip_channel': '',  # Add VIP channel
            'model_update_interval': 86400,  # Daily
            'rebalance_interval': 3600,  # Hourly
            'initial_capital': 10000,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_drawdown_pct': 0.20,
            'telegram_notifications': False,
            'discord_webhook': None,
            'log_level': 'INFO'
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"✅ Loaded config from {config_path}")
            except Exception as e:
                print(f"⚠️ Failed to load config: {e}, using defaults")
        
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutdown signal received, stopping gracefully...")
        self.stop()
        sys.exit(0)
    
    async def train_models(self):
        """Train initial models for all configured symbols."""
        print("\n🎯 Training Ultra Models...")
        print("=" * 60)
        
        results = {}
        for symbol in self.config['symbols']:
            print(f"\n📊 Training model for {symbol}...")
            try:
                result = train_ultra_model(
                    symbol=symbol,
                    timeframe=self.config['timeframes'][1],  # Use 5m as default
                    days=30
                )
                
                if 'error' not in result:
                    print(f"✅ {symbol}: Accuracy={result.get('accuracy', 'N/A'):.2%}")
                    results[symbol] = result
                else:
                    print(f"❌ {symbol}: {result['error']}")
                    
            except Exception as e:
                print(f"❌ Failed to train {symbol}: {e}")
        
        print("\n" + "=" * 60)
        print(f"✅ Training complete! Trained {len(results)}/{len(self.config['symbols'])} models")
        return results
    
    async def run_backtest(self):
        """Run backtest mode."""
        print("\n📈 Running Backtest Mode...")
        print("=" * 60)
        
        # Train models first
        await self.train_models()
        
        # Run backtest simulation
        from ultra_scout import UltraScout
        scout = UltraScout()
        
        backtest_results = {}
        for symbol in self.config['symbols']:
            print(f"\n🔄 Backtesting {symbol}...")
            
            # Run backtest
            result = scout.run_backtest(
                strategy='trend',
                params={
                    'symbol': symbol,
                    'timeframe': '5m',
                    'risk_per_trade': self.config['risk_per_trade']
                }
            )
            
            backtest_results[symbol] = result
            
            # Display results
            print(f"  Total Return: {result.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {result.get('win_rate', 0):.2%}")
        
        print("\n" + "=" * 60)
        print("✅ Backtest complete!")
        
        # Save results
        results_path = Path("reports") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(backtest_results, f, indent=2, default=str)
        print(f"📊 Results saved to {results_path}")
    
    async def run_paper(self):
        """Run paper trading mode."""
        print("\n📝 Running Paper Trading Mode...")
        print("=" * 60)
        print("⚠️ This is a simulated environment - no real trades will be executed")
        
        # Override config for paper trading
        self.config['mode'] = 'paper'
        
        # Initialize pipeline
        self.pipeline = get_ultra_pipeline(self.config)
        
        # Activate GOD MODE if enabled
        if self.config.get('god_mode_enabled', True):
            print("\n⚡ ACTIVATING ULTRA GOD MODE...")
            self.pipeline = await integrate_god_mode(self.pipeline)
            print("✅ GOD MODE ACTIVATED - Quantum + Swarm + Fractals + Smart Money")
        
        # Activate MOON SPOTTER if enabled
        if self.config.get('moon_spotter_enabled', True):
            print("\n🌙 ACTIVATING MOON SPOTTER...")
            self.pipeline = await integrate_moon_spotter(self.pipeline)
            print("✅ MOON SPOTTER ACTIVATED - Hunting 1000x micro caps!")
        
        # Activate FOREX MASTER if enabled
        if self.config.get('forex_master_enabled', True):
            print("\n💱 ACTIVATING FOREX & METALS MASTER...")
            self.pipeline = await integrate_forex_master(self.pipeline)
            print("✅ FOREX MASTER ACTIVATED - Trading XAUUSD, Forex pairs, Oil with precision!")
            
            # Add forex/metals to symbols if enabled
            forex_symbols = []
            if self.config.get('trade_metals', True):
                forex_symbols.extend(['XAUUSD', 'XAGUSD'])
            if self.config.get('trade_forex', True):
                forex_symbols.extend(['EURUSD', 'GBPUSD', 'USDJPY'])
            if self.config.get('trade_commodities', True):
                forex_symbols.extend(['USOIL'])
            
            # Add to trading symbols
            self.config['symbols'].extend(forex_symbols)
            print(f"   Added Forex/Metals: {', '.join(forex_symbols)}")
        
        # Activate TELEGRAM SIGNALS if configured
        if self.config.get('telegram_enabled') and self.config.get('telegram_bot_token'):
            print("\n📱 ACTIVATING TELEGRAM SIGNAL MASTER...")
            try:
                # Load Telegram config if exists
                telegram_config_path = Path("telegram_config.json")
                if telegram_config_path.exists():
                    with open(telegram_config_path, 'r') as f:
                        tg_config = json.load(f)
                        if tg_config.get('telegram', {}).get('bot_token'):
                            self.config['telegram_bot_token'] = tg_config['telegram']['bot_token']
                            self.config['telegram_channel'] = tg_config['telegram']['channel_id']
                            self.config['telegram_vip_channel'] = tg_config['telegram'].get('vip_channel_id')
                
                # Integrate Telegram
                if self.config['telegram_bot_token'] and self.config['telegram_channel']:
                    self.pipeline = await integrate_telegram_signals(
                        self.pipeline,
                        self.config['telegram_bot_token'],
                        self.config['telegram_channel']
                    )
                    print("✅ TELEGRAM SIGNALS ACTIVATED - Sending beautiful signals with auto-trade buttons!")
                else:
                    print("⚠️ Telegram bot token or channel not configured")
            except Exception as e:
                print(f"⚠️ Telegram integration error: {e}")
        
        # Run pipeline
        await self.pipeline.run_forever()
    
    async def run_live(self):
        """Run live trading mode."""
        print("\n💰 Running LIVE Trading Mode...")
        print("=" * 60)
        print("⚠️ WARNING: This will execute REAL trades with REAL money!")
        print("=" * 60)
        
        # Confirmation
        confirm = input("\n⚠️ Are you sure you want to run LIVE trading? (type 'YES' to confirm): ")
        if confirm != 'YES':
            print("❌ Live trading cancelled")
            return
        
        print("\n✅ Live trading confirmed, starting system...")
        
        # Initialize pipeline
        self.pipeline = get_ultra_pipeline(self.config)
        
        # Activate GOD MODE if enabled
        if self.config.get('god_mode_enabled', True):
            print("\n⚡ ACTIVATING ULTRA GOD MODE...")
            self.pipeline = await integrate_god_mode(self.pipeline)
            print("✅ GOD MODE ACTIVATED - Quantum + Swarm + Fractals + Smart Money")
        
        # Activate MOON SPOTTER if enabled
        if self.config.get('moon_spotter_enabled', True):
            print("\n🌙 ACTIVATING MOON SPOTTER...")
            self.pipeline = await integrate_moon_spotter(self.pipeline)
            print("✅ MOON SPOTTER ACTIVATED - Hunting 1000x micro caps!")
        
        # Activate FOREX MASTER if enabled
        if self.config.get('forex_master_enabled', True):
            print("\n💱 ACTIVATING FOREX & METALS MASTER...")
            self.pipeline = await integrate_forex_master(self.pipeline)
            print("✅ FOREX MASTER ACTIVATED - Trading XAUUSD, Forex pairs, Oil with precision!")
            
            # Add forex/metals to symbols if enabled
            forex_symbols = []
            if self.config.get('trade_metals', True):
                forex_symbols.extend(['XAUUSD', 'XAGUSD'])
            if self.config.get('trade_forex', True):
                forex_symbols.extend(['EURUSD', 'GBPUSD', 'USDJPY'])
            if self.config.get('trade_commodities', True):
                forex_symbols.extend(['USOIL'])
            
            # Add to trading symbols
            self.config['symbols'].extend(forex_symbols)
            print(f"   Added Forex/Metals: {', '.join(forex_symbols)}")
        
        # Activate TELEGRAM SIGNALS if configured
        if self.config.get('telegram_enabled') and self.config.get('telegram_bot_token'):
            print("\n📱 ACTIVATING TELEGRAM SIGNAL MASTER...")
            try:
                # Load Telegram config if exists
                telegram_config_path = Path("telegram_config.json")
                if telegram_config_path.exists():
                    with open(telegram_config_path, 'r') as f:
                        tg_config = json.load(f)
                        if tg_config.get('telegram', {}).get('bot_token'):
                            self.config['telegram_bot_token'] = tg_config['telegram']['bot_token']
                            self.config['telegram_channel'] = tg_config['telegram']['channel_id']
                            self.config['telegram_vip_channel'] = tg_config['telegram'].get('vip_channel_id')
                
                # Integrate Telegram
                if self.config['telegram_bot_token'] and self.config['telegram_channel']:
                    self.pipeline = await integrate_telegram_signals(
                        self.pipeline,
                        self.config['telegram_bot_token'],
                        self.config['telegram_channel']
                    )
                    print("✅ TELEGRAM SIGNALS ACTIVATED - Sending beautiful signals with auto-trade buttons!")
                else:
                    print("⚠️ Telegram bot token or channel not configured")
            except Exception as e:
                print(f"⚠️ Telegram integration error: {e}")
        
        # Run pipeline
        await self.pipeline.run_forever()
    
    async def monitor_performance(self):
        """Monitor system performance in real-time."""
        while self.running:
            try:
                if self.pipeline:
                    status = self.pipeline.get_status()
                    
                    # Clear screen (optional)
                    # os.system('cls' if os.name == 'nt' else 'clear')
                    
                    print("\n" + "=" * 60)
                    print(f"📊 ULTRA TRADING SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("=" * 60)
                    
                    # Display metrics
                    metrics = status.get('metrics', {})
                    print(f"📈 Performance Metrics:")
                    print(f"  • Total Trades: {metrics.get('total_trades', 0)}")
                    print(f"  • Win Rate: {metrics.get('win_rate', 0):.2%}")
                    print(f"  • Total PnL: {metrics.get('total_pnl', 0):.2%}")
                    print(f"  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"  • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                    
                    # Display evolution status
                    evolution = status.get('evolution', {})
                    print(f"\n🧬 Evolution Status:")
                    print(f"  • Generation: {evolution.get('generation', 0)}")
                    print(f"  • Fitness: {evolution.get('fitness', 0):.2%}")
                    print(f"  • Mutations: {len(evolution.get('mutations', []))}")
                    
                    # Display positions
                    print(f"\n💼 Active Positions: {status.get('active_positions', 0)}/{self.config['max_positions']}")
                    
                    # Runtime
                    if self.start_time:
                        runtime = datetime.now() - self.start_time
                        print(f"\n⏱️ Runtime: {runtime}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(60)
    
    async def start(self, mode: str = None):
        """Start the Ultra Trading System."""
        print(BANNER)
        
        self.running = True
        self.start_time = datetime.now()
        
        # Override mode if provided
        if mode:
            self.config['mode'] = mode
        
        print(f"🚀 Starting Ultra Trading System in {self.config['mode'].upper()} mode...")
        print(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        print(f"⏰ Timeframes: {', '.join(self.config['timeframes'])}")
        print(f"💰 Risk per trade: {self.config['risk_per_trade']:.1%}")
        print(f"🧬 Evolution: {'Enabled' if self.config['evolution_enabled'] else 'Disabled'}")
        print(f"📚 Online Learning: {'Enabled' if self.config['online_learning'] else 'Disabled'}")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_performance())
        
        try:
            # Run based on mode
            if self.config['mode'] == 'backtest':
                await self.run_backtest()
            elif self.config['mode'] == 'paper':
                await self.run_paper()
            elif self.config['mode'] == 'live':
                await self.run_live()
            else:
                print(f"❌ Unknown mode: {self.config['mode']}")
        
        finally:
            self.running = False
            monitor_task.cancel()
    
    def stop(self):
        """Stop the system."""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
        print("✅ System stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ultra Trading System - The Most Brilliant Self-Evolving Trader'
    )
    
    parser.add_argument(
        '--mode',
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Trading symbols (e.g., BTC/USDT ETH/USDT)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train models before starting'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=0.02,
        help='Risk per trade (default: 0.02 = 2%%)'
    )
    
    parser.add_argument(
        '--evolution',
        action='store_true',
        default=True,
        help='Enable evolution and adaptation'
    )
    
    parser.add_argument(
        '--god-mode',
        action='store_true',
        default=True,
        help='Enable ULTRA GOD MODE (Quantum + Swarm + Fractals + Smart Money)'
    )
    
    parser.add_argument(
        '--swarm-agents',
        type=int,
        default=100,
        help='Number of swarm agents for God Mode (default: 100)'
    )
    
    parser.add_argument(
        '--moon-spotter',
        action='store_true',
        default=True,
        help='Enable Moon Spotter for finding micro cap gems'
    )
    
    parser.add_argument(
        '--auto-snipe',
        action='store_true',
        default=False,
        help='Enable auto-sniping of high-score gems'
    )
    
    parser.add_argument(
        '--snipe-amount',
        type=float,
        default=100,
        help='USD amount to snipe each gem with (default: $100)'
    )
    
    parser.add_argument(
        '--forex',
        action='store_true',
        default=True,
        help='Enable Forex & Metals trading (XAUUSD, EURUSD, etc.)'
    )
    
    parser.add_argument(
        '--metals',
        action='store_true',
        default=True,
        help='Trade precious metals (Gold, Silver)'
    )
    
    parser.add_argument(
        '--commodities',
        action='store_true',
        default=True,
        help='Trade commodities (Oil, Natural Gas)'
    )
    
    parser.add_argument(
        '--telegram',
        action='store_true',
        default=False,
        help='Enable Telegram signal broadcasting'
    )
    
    parser.add_argument(
        '--telegram-token',
        type=str,
        help='Telegram bot token'
    )
    
    parser.add_argument(
        '--telegram-channel',
        type=str,
        help='Telegram channel ID (e.g., @your_channel)'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = UltraLauncher(args.config)
    
    # Override config with command line args
    if args.symbols:
        launcher.config['symbols'] = args.symbols
    launcher.config['risk_per_trade'] = args.risk
    launcher.config['evolution_enabled'] = args.evolution
    launcher.config['god_mode_enabled'] = args.god_mode
    launcher.config['swarm_agents'] = args.swarm_agents
    launcher.config['moon_spotter_enabled'] = args.moon_spotter
    launcher.config['auto_snipe_enabled'] = args.auto_snipe
    launcher.config['max_snipe_amount'] = args.snipe_amount
    launcher.config['forex_master_enabled'] = args.forex
    launcher.config['trade_metals'] = args.metals
    launcher.config['trade_forex'] = args.forex
    launcher.config['trade_commodities'] = args.commodities
    launcher.config['telegram_enabled'] = args.telegram
    if args.telegram_token:
        launcher.config['telegram_bot_token'] = args.telegram_token
    if args.telegram_channel:
        launcher.config['telegram_channel'] = args.telegram_channel
    
    # Run async main
    async def async_main():
        try:
            # Train if requested
            if args.train:
                await launcher.train_models()
            
            # Start system
            await launcher.start(args.mode)
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            launcher.stop()
    
    # Run
    asyncio.run(async_main())


if __name__ == "__main__":
    main()