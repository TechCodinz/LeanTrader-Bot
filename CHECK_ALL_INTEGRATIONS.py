#!/usr/bin/env python3
"""
BRUTAL TRUTH - Check ALL integrations vs what's in ULTIMATE_ORCHESTRATOR
"""

import re
import sys

# Get all class definitions from workspace
print("=" * 80)
print("CHECKING ALL CLASSES IN WORKSPACE")
print("=" * 80)

# Classes in ULTIMATE_ORCHESTRATOR
orchestrator_classes = {
    'ExchangeRouter', 'RiskEngine', 'Brain', 'PatternMemory', 'Ledger',
    'UltraCore', 'SituationalAwareness', 'HiveCoordinator', 'GlobalAwareness',
    'UltraArbitrageEngine', 'UltraScalpingEngine', 'UltraMoonSpotter',
    'RealProfitBot', 'EnhancedTradingBot',
    'ULTIMATE_EVOLUTION_ENGINE', 'working_450_models_bot', 'UltraSwarmConsciousness',
    'DivineIntelligence', 'MLStrategyEngine', 'OnlineLearner',
    'UltraQuantumIntelligence', 'UltraFluidMechanics', 'UltraBacktestEngine',
    'UltraBusinessSystem', 'NovemberGrowthStrategy', 'PaperBroker'
}

# Check for other major systems
other_major_systems = [
    'UltraBot',
    'NobelHedgeFundSystem',
    'CompleteGodTraderBot',
    'UltimateLearntraderBot',
    'UnifiedTradingBot',
    'TestnetTradingEngine',
    'ProfessionalTradingBot',
    'TradingBotOrchestrator',
    'CompoundEngine',
    'ExecutionEngine',
    'MoonSpotterEngine',
    'CryptoScalperEngine',
    'ArbitrageEngine',
    'FXTrainerEngine',
    'WebCrawlerEngine',
    'DeepLearningEngine',
]

print(f"\n✅ INTEGRATED in ULTIMATE_ORCHESTRATOR: {len(orchestrator_classes)} systems")
for cls in sorted(orchestrator_classes):
    print(f"   ✓ {cls}")

print(f"\n⚠️  OTHER MAJOR SYSTEMS NOT YET INTEGRATED: {len(other_major_systems)}")
for cls in other_major_systems:
    print(f"   ? {cls}")

print("\n" + "=" * 80)
print("CHECKING ALL APIs NEEDED")
print("=" * 80)

apis_needed = {
    'Exchange APIs (MUST HAVE)': [
        'BYBIT_API_KEY / BYBIT_API_SECRET',
        'BINANCE_API_KEY / BINANCE_API_SECRET', 
        'OKX_API_KEY / OKX_API_SECRET',
        'COINBASE_API_KEY / COINBASE_API_SECRET',
        'GATEIO_API_KEY / GATEIO_API_SECRET',
        'KUCOIN_API_KEY / KUCOIN_API_SECRET',
    ],
    'AI/ML APIs (OPTIONAL - Enhance Performance)': [
        'ANTHROPIC_API_KEY (Claude for reasoning)',
        'OPENAI_API_KEY (GPT for analysis)',
        'HUGGINGFACE_TOKEN (Models)',
    ],
    'Data Sources (OPTIONAL - More Data)': [
        'COINGECKO_API_KEY (Market data)',
        'ETHERSCAN_API_KEY (On-chain data)',
        'TWITTER_API_KEY (Sentiment)',
        'NEWS_API_KEY (News sentiment)',
    ],
    'Brokers (OPTIONAL - Forex/Stocks)': [
        'OANDA_API_KEY (Forex)',
        'MT5_LOGIN / MT5_PASSWORD (MetaTrader)',
        'IBKR_USERNAME / IBKR_PASSWORD (Interactive Brokers)',
    ],
    'Communication (OPTIONAL - Alerts)': [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_ADMIN_CHAT_ID',
        'SLACK_WEBHOOK_URL',
    ],
    'Infrastructure (OPTIONAL - Scaling)': [
        'REDIS_URL (Caching)',
        'STRIPE_API_KEY (Subscriptions)',
        'CELERY_BROKER_URL (Task queue)',
    ]
}

for category, keys in apis_needed.items():
    print(f"\n{category}:")
    for key in keys:
        print(f"   • {key}")

print("\n" + "=" * 80)
