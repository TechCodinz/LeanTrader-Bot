import json
import os
import unittest

from leantrader.execution.broker_ccxt import BrokerCCXT


class BybitRecvWindowTests(unittest.TestCase):
    def setUp(self):
        self.original = os.environ.get("CCXT_OPTIONS_JSON")
        os.environ.pop("CCXT_OPTIONS_JSON", None)

    def tearDown(self):
        if self.original is None:
            os.environ.pop("CCXT_OPTIONS_JSON", None)
        else:
            os.environ["CCXT_OPTIONS_JSON"] = self.original

    def test_bybit_defaults_to_10_second_recv_window(self):
        broker = BrokerCCXT(
            execution_mode="paper",
            exchange_id="bybit",
        )

        options = broker._base_options()

        self.assertEqual(
            options.get("recvWindow"),
            10000,
        )

    def test_operator_can_override_bybit_recv_window(self):
        os.environ["CCXT_OPTIONS_JSON"] = json.dumps({
            "recvWindow": 15000,
        })

        broker = BrokerCCXT(
            execution_mode="paper",
            exchange_id="bybit",
        )

        options = broker._base_options()

        self.assertEqual(
            options.get("recvWindow"),
            15000,
        )

    def test_other_exchanges_do_not_inherit_bybit_recv_window(self):
        broker = BrokerCCXT(
            execution_mode="paper",
            exchange_id="binance",
        )

        options = broker._base_options()

        self.assertNotIn(
            "recvWindow",
            options,
        )


if __name__ == "__main__":
    unittest.main()
