from __future__ import annotations

import time

from leantrader.production.testnet_execution_first_candidates_v1619 import (
    _ExecutionFirstCandidateProxy,
)
from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
)
from tests.test_production_testnet_exit_recycle_v1608 import (
    hyper_lane,
)
from tests.test_testnet_execution import (
    engine,
)


class SelectiveBybit(
    PriceGuardBybit
):
    def __init__(self):
        super().__init__()

        for symbol in (
            "EXPENSIVE/USDT",
            "BAD/USDT",
            "GOOD/USDT",
        ):
            self.markets[symbol] = {
                "symbol": symbol,
                "spot": True,
                "active": True,
                "quote": "USDT",
            }

        self.balance_total[
            "USDT"
        ] = 1.8

        self.balance_free[
            "USDT"
        ] = 1.8

    def market(
        self,
        symbol,
    ):
        row = super().market(
            symbol
        )

        row["id"] = (
            symbol.replace(
                "/",
                "",
            )
        )

        minimum = (
            5.0
            if symbol
            == "EXPENSIVE/USDT"
            else 0.10
        )

        row["limits"] = {
            "cost": {
                "min": minimum,
            },
            "amount": {
                "min": 0.001,
            },
        }

        return row

    def public_get_v5_market_price_limit(
        self,
        params,
    ):
        self.calls.append(
            (
                "price_limit",
                dict(params),
            )
        )

        native = str(
            params["symbol"]
        ).upper()

        sell = (
            105.0
            if native
            == "BADUSDT"
            else 90.0
        )

        return {
            "retCode": 0,
            "result": {
                "symbol": native,
                "buyLmt": "200",
                "sellLmt": str(
                    sell
                ),
            },
        }


class RankedService:
    def collective_candidates(
        self,
        limit=8,
    ):
        return [
            "EXPENSIVE/USDT",
            "BAD/USDT",
            "GOOD/USDT",
        ][:limit]


def _runtime(
    tmp_path,
):
    fake = SelectiveBybit()

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )

    instance.start()

    lane, _, _ = hyper_lane(
        tmp_path,
        testnet=instance,
        order_usd=1.0,
    )

    return (
        lane,
        instance,
        fake,
    )


def test_execution_first_proxy_substitutes_deeper_affordable_executable_candidate(
    tmp_path,
):
    (
        lane,
        instance,
        fake,
    ) = _runtime(
        tmp_path
    )

    proxy = (
        _ExecutionFirstCandidateProxy(
            RankedService(),
            lane,
            time.time(),
        )
    )

    selected = (
        proxy.collective_candidates(
            limit=8
        )
    )

    assert selected == [
        "GOOD/USDT"
    ]

    # EXPENSIVE is rejected from
    # market metadata without network
    # execution probing.
    reasons = (
        lane.state[
            "v1619_candidate_block_reasons"
        ]
    )

    assert (
        reasons[
            "minimum_cost_exceeds_free_quote"
        ]
        >= 1
    )

    # BAD reaches the authoritative
    # round-trip probe and fails its
    # prospective sell boundary.
    assert (
        reasons[
            "prospective_exit_price_limit_unexecutable"
        ]
        >= 1
    )

    assert (
        lane.state[
            "v1619_candidate_probe_passes"
        ]
        == 1
    )

    assert (
        lane.state[
            "v1619_candidate_probe_checks"
        ]
        == 2
    )

    # Candidate probes never create
    # exchange orders.
    assert fake.created == []

    assert (
        instance.state.get(
            "orders"
        )
        or {}
    ) == {}

    selection = (
        lane.state[
            "v1619_last_candidate_selection"
        ]
    )

    assert (
        selection[
            "arbitrary_market_injection"
        ]
        is False
    )

    assert (
        selection[
            "strategy_rank_order_preserved"
        ]
        is True
    )


def test_execution_first_pass_cache_reuses_success_for_two_minutes(
    tmp_path,
):
    (
        lane,
        _instance,
        fake,
    ) = _runtime(
        tmp_path
    )

    now = time.time()

    first = (
        _ExecutionFirstCandidateProxy(
            RankedService(),
            lane,
            now,
        ).collective_candidates(
            limit=8
        )
    )

    assert first == [
        "GOOD/USDT"
    ]

    price_limit_before = len(
        [
            row
            for row in fake.calls
            if (
                row[0]
                == "price_limit"
                and str(
                    row[1].get("symbol")
                    or ""
                ).upper()
                == "GOODUSDT"
            )
        ]
    )

    second = (
        _ExecutionFirstCandidateProxy(
            RankedService(),
            lane,
            now + 120.0,
        ).collective_candidates(
            limit=8
        )
    )

    assert second == [
        "GOOD/USDT"
    ]

    price_limit_after = len(
        [
            row
            for row in fake.calls
            if (
                row[0]
                == "price_limit"
                and str(
                    row[1].get("symbol")
                    or ""
                ).upper()
                == "GOODUSDT"
            )
        ]
    )

    assert (
        price_limit_after
        == price_limit_before
    )

    assert (
        lane.state[
            "v1619_probe_cache_hits"
        ]
        >= 1
    )

    assert fake.created == []

    health = lane.health()

    guard = health[
        "execution_first_candidate_substitution"
    ]

    assert (
        guard[
            "executor_order_created_during_probe"
        ]
        is False
    )

    assert (
        guard[
            "v1613_executor_recheck_preserved"
        ]
        is True
    )

    assert (
        health["live_authority"]
        is False
    )
