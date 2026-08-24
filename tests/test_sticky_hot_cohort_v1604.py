from __future__ import annotations

import threading

from leantrader.agents.swarm_service import (
    ReadOnlySwarmService,
)


def schedule(
    *,
    now=100.0,
    current_hot=None,
    hot_until=0.0,
    current_explorer=None,
    explorer_until=0.0,
    velocity=None,
    cursor=0,
):
    return (
        ReadOnlySwarmService
        ._build_sticky_precision_queue(
            scout_symbols=[
                "A/USDT",
                "B/USDT",
                "C/USDT",
                "D/USDT",
                "E/USDT",
                "F/USDT",
                "G/USDT",
                "H/USDT",
            ],
            sticky_symbols=[],
            due_symbols=[
                "R/USDT",
            ],
            velocity_symbols=(
                velocity or []
            ),
            capacity=5,
            cursor=cursor,
            now=now,
            current_hot=(
                current_hot or []
            ),
            hot_until=hot_until,
            current_explorer=(
                current_explorer
            ),
            explorer_until=(
                explorer_until
            ),
            hot_hold_seconds=4.0,
            explorer_hold_seconds=3.0,
        )
    )


def test_hot_cohort_stays_continuous_inside_hold_window():
    queue1, cursor1, state1 = schedule(
        now=100.0,
    )

    queue2, _, state2 = schedule(
        now=101.0,
        current_hot=state1[
            "hot_symbols"
        ],
        hot_until=state1[
            "hot_until"
        ],
        current_explorer=state1[
            "explorer_symbol"
        ],
        explorer_until=state1[
            "explorer_until"
        ],
        cursor=cursor1,
    )

    assert (
        state2["hot_symbols"]
        == state1["hot_symbols"]
    )

    assert (
        state2["explorer_symbol"]
        == state1["explorer_symbol"]
    )

    assert queue1[:3] == queue2[:3]


def test_explorer_rotates_only_after_hold_expires():
    _, cursor1, state1 = schedule(
        now=100.0,
    )

    _, cursor2, state2 = schedule(
        now=104.0,
        current_hot=state1[
            "hot_symbols"
        ],
        hot_until=state1[
            "hot_until"
        ],
        current_explorer=state1[
            "explorer_symbol"
        ],
        explorer_until=state1[
            "explorer_until"
        ],
        cursor=cursor1,
    )

    assert (
        state2["explorer_symbol"]
        != state1["explorer_symbol"]
    )

    assert cursor2 != cursor1


def test_fresh_velocity_market_promotes_into_hot_cohort():
    _, cursor1, state1 = schedule(
        now=100.0,
    )

    assert (
        "H/USDT"
        not in state1["hot_symbols"]
    )

    _, _, state2 = schedule(
        now=101.0,
        current_hot=state1[
            "hot_symbols"
        ],
        hot_until=state1[
            "hot_until"
        ],
        current_explorer=state1[
            "explorer_symbol"
        ],
        explorer_until=state1[
            "explorer_until"
        ],
        cursor=cursor1,
        velocity=[
            "H/USDT",
        ],
    )

    assert (
        "H/USDT"
        in state2["hot_symbols"]
    )

    assert (
        "H/USDT"
        in state2["promoted_symbols"]
    )


def test_wide_router_is_not_six_symbol_bound():
    service = object.__new__(
        ReadOnlySwarmService
    )

    service._lock = threading.RLock()

    service._precision_scout_symbols = [
        f"S{i}/USDT"
        for i in range(12)
    ]

    service._microstream_symbols = []

    service.last_step = {
        "ranked": [
            {
                "symbol": f"R{i}/USDT"
            }
            for i in range(40)
        ]
    }

    service.micro_velocity_candidates = (
        lambda *args, **kwargs: []
    )

    result = service.collective_candidates(
        limit=48
    )

    assert len(result) > 6
    assert len(result) <= 48
