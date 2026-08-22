from __future__ import annotations

import pandas as pd

from leantrader.agents.timeframe_mind import MultiTimeframeMind


def _frame(start: float, moves: list[float], repeat: int = 4) -> pd.DataFrame:
    prices = [start]
    sequence = (moves * repeat)[:47]
    for move in sequence:
        prices.append(prices[-1] * (1.0 + move))
    return pd.DataFrame({"close": prices})


def test_timeframe_mind_independently_qualifies_strong_long_move() -> None:
    mind = MultiTimeframeMind()
    assessment = mind.assess(
        symbol="FAST/USDT",
        timeframe="5m",
        candles=_frame(1.0, [0.01] * 12),
    )
    assert assessment.direction == "long"
    assert assessment.independently_qualified is True
    assert assessment.expected_edge_bps > assessment.modeled_round_trip_cost_bps
    assert assessment.confidence >= mind.minimum_confidence
    assert mind.agrees_with_position(assessment, side="long") is True
    assert mind.agrees_with_position(assessment, side="short") is False


def test_timeframe_mind_can_qualify_short_without_inheriting_scalp_side() -> None:
    mind = MultiTimeframeMind()
    assessment = mind.assess(
        symbol="FAST/USDT",
        timeframe="15m",
        candles=_frame(10.0, [-0.01] * 12),
    )
    assert assessment.direction == "short"
    assert assessment.independently_qualified is True
    assert mind.agrees_with_position(assessment, side="long") is False
    assert mind.agrees_with_position(assessment, side="short") is True


def test_timeframe_mind_rejects_move_that_does_not_clear_cost() -> None:
    mind = MultiTimeframeMind()
    assessment = mind.assess(
        symbol="SLOW/USDT",
        timeframe="1h",
        candles=_frame(100.0, [0.0005] * 12),
    )
    assert assessment.direction == "long"
    assert assessment.independently_qualified is False
    assert assessment.reason == "timeframe_edge_does_not_clear_modeled_cost"
    assert assessment.modeled_round_trip_cost_bps >= 30.0


def test_timeframe_mind_rejects_choppy_directionless_path() -> None:
    mind = MultiTimeframeMind()
    assessment = mind.assess(
        symbol="CHOP/USDT",
        timeframe="5m",
        candles=_frame(1.0, [0.01, -0.01] * 6),
    )
    assert assessment.independently_qualified is False
    assert assessment.confidence < mind.minimum_confidence or assessment.expected_edge_bps <= assessment.modeled_round_trip_cost_bps


def test_each_timeframe_is_measured_independently() -> None:
    mind = MultiTimeframeMind()
    assessments = mind.assess_many(
        symbol="MIXED/USDT",
        frames={
            "5m": _frame(1.0, [0.01] * 12),
            "15m": _frame(1.0, [0.0005] * 12),
            "1h": _frame(1.0, [-0.01] * 12),
        },
    )
    assert assessments["5m"].independently_qualified is True
    assert assessments["15m"].independently_qualified is False
    assert assessments["1h"].independently_qualified is True
    assert assessments["5m"].direction == "long"
    assert assessments["1h"].direction == "short"


def test_timeframe_mind_never_grants_execution_or_promotion_authority() -> None:
    health = MultiTimeframeMind().health()
    assert health["independent_timeframe_qualification"] is True
    assert health["research_hypothesis"] is True
    assert health["predictive_profit_claim"] is False
    assert health["automatic_promotion"] is False
    assert health["execution_authority"] is False
    assert health["testnet_authority"] is False
    assert health["live_authority"] is False
