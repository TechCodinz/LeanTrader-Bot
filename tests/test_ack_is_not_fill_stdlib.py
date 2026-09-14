import unittest

from src.leantrader.execution.broker_ccxt import BrokerCCXT
from src.leantrader.execution import preflight


class OpenZeroFillExchange:

    def create_order(
        self,
        symbol,
        order_type,
        side,
        qty,
        price,
        params,
    ):
        return {
            "id": "ack-only-123",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "status": "open",
            "filled": 0.0,
            "remaining": qty,
            "average": None,
            "price": price,
        }

    def fetch_order(
        self,
        order_id,
        symbol,
    ):
        return {
            "id": order_id,
            "symbol": symbol,
            "status": "open",
            "filled": 0.0,
            "remaining": 1.0,
            "average": None,
        }


class PartialFillExchange:

    def create_order(
        self,
        symbol,
        order_type,
        side,
        qty,
        price,
        params,
    ):
        return {
            "id": "partial-123",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "status": "open",
            "filled": 0.25,
            "remaining": 0.75,
            "average": 100.0,
            "price": price,
        }

    def fetch_order(
        self,
        order_id,
        symbol,
    ):
        return {
            "id": order_id,
            "symbol": symbol,
            "status": "open",
            "filled": 0.25,
            "remaining": 0.75,
            "average": 100.0,
        }


class BrokerHarness(BrokerCCXT):

    def __init__(self, exchange):
        self.exchange_id = "bybit"
        self.market_mode = "spot"
        self._test_exchange = exchange

    def resolve_mode(self):
        return "testnet"

    @property
    def authority(self):
        return "testnet"

    def _make_exchange(
        self,
        environment,
        authenticated,
    ):
        return self._test_exchange


class AckIsNotFillTest(unittest.TestCase):

    def test_zero_fill_ack_is_not_executed(self):
        broker = BrokerHarness(
            OpenZeroFillExchange()
        )

        receipt = broker.order(
            symbol="BTC/USDT",
            order_type="market",
            side="buy",
            qty=1.0,
        )

        self.assertTrue(
            receipt.get("ok"),
            "Exchange acknowledgement itself remains factual.",
        )

        self.assertFalse(
            receipt.get("executed"),
            (
                "ACK-only order must not be called executed "
                "when authenticated filled quantity is zero."
            ),
        )

    def test_classifier_rejects_zero_fill_ack(self):
        receipt = {
            "ok": True,
            "executed": True,
            "execution_mode": "testnet",
            "authority": "testnet",
            "order": {
                "id": "ack-only-123",
                "status": "open",
                "filled": 0.0,
                "remaining": 1.0,
            },
        }

        blocker = preflight.classify_receipt(
            receipt
        )

        self.assertIsNotNone(
            blocker,
            (
                "Receipt classification must not accept "
                "an order id as proof of a fill."
            ),
        )

    def test_real_partial_fill_is_execution_evidence(self):
        broker = BrokerHarness(
            PartialFillExchange()
        )

        receipt = broker.order(
            symbol="BTC/USDT",
            order_type="market",
            side="buy",
            qty=1.0,
        )

        self.assertTrue(
            receipt.get("executed"),
            (
                "Positive authenticated filled quantity "
                "is real execution evidence."
            ),
        )

        self.assertIsNone(
            preflight.classify_receipt(
                receipt
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
