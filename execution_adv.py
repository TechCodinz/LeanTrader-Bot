from typing import Any, Dict

from order_utils import (
    place_market,
    safe_create_order as universal_safe_create_order,
)


def safe_create_order(
    ex,
    type_: str,
    symbol: str,
    side: str,
    amount: float,
    price: float = None,
    params: Dict[
        str,
        Any,
    ] | None = None,
):
    return (
        universal_safe_create_order(
            ex,
            type_,
            symbol,
            side,
            amount,
            price,
            params,
        )
    )


class LimitMakerExecutor:
    """
    Historical maker executor.

    It preserves post-only intent while
    routing all order creation through
    the universal execution fabric.
    """

    def __init__(
        self,
        ex,
        logger,
        fee_frac: float = 0.001,
    ):
        self.ex = ex
        self.log = logger
        self.fee_frac = fee_frac

    def _maker(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
    ) -> Dict[str, Any]:
        result = (
            universal_safe_create_order(
                self.ex,
                "limit",
                symbol,
                side,
                amount,
                price,
                {
                    "postOnly": True
                },
            )
        )

        if (
            isinstance(
                result,
                dict,
            )
            and result.get(
                "ok",
                True,
            )
        ):
            return result

        self.log.warning(
            "postOnly unavailable; "
            "retrying normal limit"
        )

        result = (
            universal_safe_create_order(
                self.ex,
                "limit",
                symbol,
                side,
                amount,
                price,
                {},
            )
        )

        if (
            isinstance(
                result,
                dict,
            )
            and result.get(
                "ok",
                True,
            )
        ):
            return result

        return place_market(
            self.ex,
            symbol,
            side,
            amount,
        )

    def limit_maker_buy(
        self,
        symbol: str,
        price: float,
        amount: float,
    ) -> Dict[str, Any]:
        result = self._maker(
            symbol,
            "buy",
            price,
            amount,
        )

        self.log.info(
            "BUY intent %s px=%s amt=%s",
            symbol,
            price,
            amount,
        )

        return result

    def limit_maker_sell(
        self,
        symbol: str,
        price: float,
        amount: float,
    ) -> Dict[str, Any]:
        result = self._maker(
            symbol,
            "sell",
            price,
            amount,
        )

        self.log.info(
            "SELL intent %s px=%s amt=%s",
            symbol,
            price,
            amount,
        )

        return result

    def safe_cancel(
        self,
        order_id: str,
        symbol: str,
    ) -> None:
        """
        Preserve historical cancellation
        compatibility for now.
        """
        try:
            if hasattr(
                self.ex,
                "safe_cancel_order",
            ):
                self.ex.safe_cancel_order(
                    order_id,
                    symbol,
                )

            elif hasattr(
                self.ex,
                "cancel_order",
            ):
                self.ex.cancel_order(
                    order_id,
                    symbol,
                )

            self.log.info(
                "CANCEL %s order=%s",
                symbol,
                order_id,
            )

        except Exception as exc:
            self.log.error(
                "Cancel failed %s: %s",
                symbol,
                type(exc).__name__,
            )

    def get_order_status(
        self,
        order_id: str,
        symbol: str,
    ) -> Dict[str, Any]:
        try:
            if hasattr(
                self.ex,
                "get_order_status",
            ):
                return (
                    self.ex
                    .get_order_status(
                        order_id,
                        symbol,
                    )
                    or {}
                )

            if hasattr(
                self.ex,
                "safe_fetch_order",
            ):
                return (
                    self.ex
                    .safe_fetch_order(
                        order_id,
                        symbol,
                    )
                    or {}
                )

            if hasattr(
                self.ex,
                "fetch_order",
            ):
                return (
                    self.ex
                    .fetch_order(
                        order_id,
                        symbol,
                    )
                    or {}
                )

            return {
                "error": (
                    "order_status_"
                    "unavailable"
                )
            }

        except Exception as exc:
            return {
                "error": str(exc)
            }
