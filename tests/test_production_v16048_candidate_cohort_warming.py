from __future__ import annotations

import threading

from leantrader.production.testnet_execution_first_candidates_v1619 import (
    MAX_EMPTY_SELECTION_NETWORK_PROBES_PER_CALL,
    MAX_NETWORK_PROBES_PER_CALL,
    SIGNAL_REFRESH_PIN_SECONDS,
    _ExecutionFirstCandidateProxy,
)


class Lane:
    def __init__(self):
        self._lock = threading.RLock()
        self.state = {}


class Service:
    def __init__(self):
        self._precision_micro_capacity = 4
        self.pin_calls = []

    def pin_execution_candidate_symbols(
        self,
        symbols,
        *,
        ttl_seconds,
    ):
        self.pin_calls.append(
            (list(symbols), float(ttl_seconds))
        )


def test_warms_one_adaptive_candidate_cohort():
    lane = Lane()
    service = Service()

    proxy = _ExecutionFirstCandidateProxy(
        service,
        lane,
        123.0,
    )

    warmed = proxy._warm_execution_candidate_cohort(
        [
            "SUI/USDT",
            "XRP/USDT",
            "ARB/USDT",
            "ETH/USDT",
            "DOGE/USDT",
            "SUI/USDT",
        ]
    )

    assert warmed == [
        "SUI/USDT",
        "XRP/USDT",
        "ARB/USDT",
        "ETH/USDT",
    ]

    # The service stores pins in insertion order and the microstream
    # consumes them newest-first, so insertion is intentionally reversed.
    assert service.pin_calls == [
        (
            list(reversed(warmed)),
            float(SIGNAL_REFRESH_PIN_SECONDS),
        )
    ]

    row = lane.state[
        "v1648_last_candidate_warm_cohort"
    ]

    assert row["count"] == 4
    assert row["adaptive_capacity"] == 4
    assert row["execution_authority"] is False
    assert row["testnet_order_created"] is False
    assert (
        row["normal_execution_preflight_still_required"]
        is True
    )
    assert row["network_probe_budget_changed"] is False
    assert row["live_authority"] is False


class StaleService(Service):
    def collective_signal(self, symbol):
        return {
            "fresh": False,
            "micro_velocity": {
                "fresh": False,
                "age_seconds": 1_000_000.0,
            },
        }


def test_stale_check_can_defer_individual_pin_to_batch():
    lane = Lane()
    service = StaleService()

    proxy = _ExecutionFirstCandidateProxy(
        service,
        lane,
        456.0,
    )

    assert (
        proxy._signal_ready(
            "SUI/USDT",
            pin_on_miss=False,
        )
        is False
    )

    assert service.pin_calls == []

    row = lane.state[
        "v1625_last_signal_refresh"
    ]

    assert row["candidate_returned"] is False
    assert row["microstream_pinned"] is False
    assert row["execution_preflight_bypassed"] is False


def test_existing_probe_budgets_remain_unchanged():
    assert MAX_NETWORK_PROBES_PER_CALL == 2
    assert (
        MAX_EMPTY_SELECTION_NETWORK_PROBES_PER_CALL
        == 4
    )


def test_warming_honors_single_slot_adaptive_capacity():
    lane = Lane()
    service = Service()
    service._precision_micro_capacity = 1

    proxy = _ExecutionFirstCandidateProxy(
        service,
        lane,
        789.0,
    )

    warmed = proxy._warm_execution_candidate_cohort(
        [
            "SUI/USDT",
            "XRP/USDT",
            "ARB/USDT",
        ]
    )

    assert warmed == ["SUI/USDT"]
    assert service.pin_calls == [
        (
            ["SUI/USDT"],
            float(SIGNAL_REFRESH_PIN_SECONDS),
        )
    ]
