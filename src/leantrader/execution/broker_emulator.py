import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EmuOrder:
    id: str
    symbol: str
    side: str
    qty: float
    price: float
    filled: float = 0.0
    status: str = "submitted"


class BrokerEmulator:
    """
    Paper execution emulator.

    Market orders simulate fills.

    Limit/stop/conditional orders remain
    OPEN/PENDING rather than being falsely
    represented as immediate fills.
    """

    def __init__(
        self,
        slippage_bps: float = 2.0,
        mean_latency_ms: int = 150,
    ):
        self.slip = slippage_bps
        self.lat = (
            mean_latency_ms
            / 1000.0
        )

        self.orders: Dict[
            str,
            EmuOrder,
        ] = {}

    def _oid(
        self,
    ) -> str:
        return (
            f"emu-{int(time.time()*1000)}-"
            f"{random.randint(100,999)}"
        )

    def market(
        self,
        symbol: str,
        side: str,
        qty: float,
        ref_price: float,
    ) -> Dict[str, Any]:
        if float(
            ref_price or 0.0
        ) <= 0.0:
            return {
                "id": None,
                "symbol": symbol,
                "side": side,
                "filled": 0.0,
                "avg_px": 0.0,
                "status": "rejected",
                "error": (
                    "paper_reference_price_"
                    "unavailable"
                ),
            }

        oid = self._oid()

        time.sleep(
            max(
                0.0,
                random.gauss(
                    self.lat,
                    self.lat * 0.2,
                ),
            )
        )

        slip = (
            self.slip
            / 10000.0
            * ref_price
            * (
                1
                if side.lower()
                == "buy"
                else -1
            )
        )

        avg_fill_px = (
            ref_price
            + slip
        )

        filled1 = (
            qty
            * random.uniform(
                0.4,
                0.7,
            )
        )

        time.sleep(
            0.05
        )

        filled2 = (
            qty
            - filled1
        )

        self.orders[
            oid
        ] = EmuOrder(
            id=oid,
            symbol=symbol,
            side=side,
            qty=qty,
            price=avg_fill_px,
            filled=qty,
            status="filled",
        )

        return {
            "id": oid,
            "symbol": symbol,
            "side": side,
            "filled": qty,
            "avg_px": avg_fill_px,
            "status": "filled",
            "partials": [
                filled1,
                filled2,
            ],
        }

    def submit_pending(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[
            float
        ] = None,
        params: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        oid = self._oid()

        px = float(
            price or 0.0
        )

        self.orders[
            oid
        ] = EmuOrder(
            id=oid,
            symbol=symbol,
            side=side,
            qty=qty,
            price=px,
            filled=0.0,
            status="open",
        )

        return {
            "id": oid,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": qty,
            "price": (
                px
                if px > 0.0
                else None
            ),
            "filled": 0.0,
            "status": "open",
            "pending": True,
            "params": dict(
                params or {}
            ),
        }
