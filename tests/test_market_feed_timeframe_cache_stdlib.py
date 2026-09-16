import unittest

from leantrader.production.market_feed import MarketFeed


class MarketFeedTimeframeCacheTest(unittest.TestCase):

    def test_known_timeframes_have_expected_cache_ttl(self):
        self.assertEqual(
            MarketFeed._timeframe_cache_seconds("1m"),
            15.0,
        )
        self.assertEqual(
            MarketFeed._timeframe_cache_seconds("5m"),
            75.0,
        )
        self.assertEqual(
            MarketFeed._timeframe_cache_seconds("15m"),
            225.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
