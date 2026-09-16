import unittest

from leantrader.production.market_feed import MarketFeed


class MarketFeedTimeframeCacheTest(unittest.TestCase):
    def test_market_feed_cache_uses_historical_timeframe_conversion(self):
        self.assertEqual(MarketFeed._timeframe_cache_seconds("1m"), 15.0)
        self.assertEqual(MarketFeed._timeframe_cache_seconds("5m"), 75.0)
        self.assertEqual(MarketFeed._timeframe_cache_seconds("15m"), 225.0)
        self.assertEqual(MarketFeed._timeframe_cache_seconds("1h"), 900.0)


if __name__ == "__main__":
    unittest.main()
