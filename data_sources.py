"""Data sources module"""
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

class EconomicCalendarSource:
    def __init__(self):
        self.events = []
    def fetch_events(self, start_date, end_date):
        return self.events
    def get_high_impact_events(self):
        return []

class FundingRateSource:
    def __init__(self):
        self.rates = {}
    def fetch_funding_rate(self, symbol):
        return 0.0001
    def fetch_historical_rates(self, symbol, days=7):
        return pd.DataFrame()

class NewsSource:
    def __init__(self):
        self.articles = []
    def fetch_latest_news(self, limit=10):
        return []
    def fetch_news_by_symbol(self, symbol, limit=5):
        return []

class OnChainSource:
    def __init__(self):
        self.metrics = {}
    def fetch_metrics(self, symbol):
        return {'tvl': 0, 'volume_24h': 0}
    def fetch_whale_movements(self, symbol, threshold=1000000):
        return []

class SentimentSource:
    def __init__(self):
        self.sentiment_data = {}
    def fetch_sentiment(self, symbol):
        return {'overall': 0.5}
    def fetch_fear_greed_index(self):
        return 50.0

NewsSentimentSource = SentimentSource
OnchainMetricSource = OnChainSource

def merge_externals(*sources):
    merged = {}
    for source in sources:
        if isinstance(source, dict):
            merged.update(source)
    return merged
