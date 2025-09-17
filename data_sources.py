# data_sources.py
from __future__ import annotations

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd


# Mock data source classes for testing
class EconomicCalendarSource:
    """Mock economic calendar data source."""
    
    def __init__(self):
        self.events = []
    
    def fetch_events(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch economic calendar events."""
        return self.events
    
    def get_high_impact_events(self) -> List[Dict[str, Any]]:
        """Get high impact economic events."""
        return [e for e in self.events if e.get('impact') == 'high']


class FundingRateSource:
    """Mock funding rate data source."""
    
    def __init__(self):
        self.rates = {}
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch funding rate for a symbol."""
        return self.rates.get(symbol, 0.0001)
    
    def fetch_historical_rates(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch historical funding rates."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='8h')
        rates = [self.rates.get(symbol, 0.0001) for _ in range(len(dates))]
        return pd.DataFrame({'timestamp': dates, 'rate': rates})


class NewsSource:
    """Mock news data source."""
    
    def __init__(self):
        self.articles = []
    
    def fetch_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch latest news articles."""
        return self.articles[:limit]
    
    def fetch_news_by_symbol(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch news for a specific symbol."""
        return [a for a in self.articles if symbol in a.get('symbols', [])][:limit]


class OnChainSource:
    """Mock on-chain data source."""
    
    def __init__(self):
        self.metrics = {}
    
    def fetch_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch on-chain metrics for a symbol."""
        return self.metrics.get(symbol, {
            'tvl': 0,
            'volume_24h': 0,
            'active_addresses': 0,
            'hash_rate': 0
        })
    
    def fetch_whale_movements(self, symbol: str, threshold: float = 1000000) -> List[Dict[str, Any]]:
        """Fetch whale movements above threshold."""
        return []


class SentimentSource:
    """Mock sentiment data source."""
    
    def __init__(self):
        self.sentiment_data = {}
    
    def fetch_sentiment(self, symbol: str) -> Dict[str, float]:
        """Fetch sentiment scores for a symbol."""
        return self.sentiment_data.get(symbol, {
            'twitter': 0.5,
            'reddit': 0.5,
            'news': 0.5,
            'overall': 0.5
        })
    
    def fetch_fear_greed_index(self) -> float:
        """Fetch fear and greed index (0-100)."""
        return 50.0


# Aliases for compatibility
NewsSentimentSource = SentimentSource
OnchainMetricSource = OnChainSource

# Helper function for merging external data
def merge_externals(*sources) -> Dict[str, Any]:
    """Merge data from multiple external sources."""
    merged = {}
    for source in sources:
        if isinstance(source, dict):
            merged.update(source)
    return merged
