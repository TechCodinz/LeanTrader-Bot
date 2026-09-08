from leantrader.production.legacy_engine_bridge import (
    merge_restored_signal,
)


def test_real_cost_positive_fast_legacy_becomes_execution_intent():
    result = merge_restored_signal(
        {
            "timeframe_assessments": {},
            "ranked_opportunity": {},
        },
        {
            "score": 0.8,
            "confidence": 0.85,
            "contributors": ["ultra_scalping.micro_momentum"],
            "contributions": [
                {
                    "source": "ultra_scalping.micro_momentum",
                    "timeframe": "1m",
                    "direction": "long",
                    "confidence": 0.85,
                    "expected_edge_bps": 20.0,
                    "metadata": {
                        "gross_edge_bps": 50.0,
                        "modeled_round_trip_cost_bps": 30.0,
                        "minimum_positive_net_edge_bps": 5.0,
                        "conservative_net_edge_bps": 20.0,
                        "economically_positive": True,
                    },
                }
            ],
        },
    )

    rows = list(
        result["timeframe_assessments"].values()
    )

    assert len(rows) == 1
    assert rows[0]["independently_qualified"] is True
    assert (
        rows[0]["legacy_execution_intent_qualified"]
        is True
    )
    assert result["legacy_execution_intent_active"] is True
    assert (
        result["legacy_execution_intent_delegate"]
        == "authenticated_testnet_fast_lane"
    )
    assert (
        result["legacy_direct_execution_authority"]
        is False
    )
    assert result["live_authority"] is False


def test_non_fast_or_non_economic_legacy_stays_observation_only():
    result = merge_restored_signal(
        {
            "timeframe_assessments": {},
            "ranked_opportunity": {},
        },
        {
            "score": 0.7,
            "confidence": 0.8,
            "contributions": [
                {
                    "source": "continuous_mean_reversion",
                    "timeframe": "15m",
                    "direction": "long",
                    "confidence": 0.8,
                    "expected_edge_bps": 20.0,
                    "metadata": {
                        "gross_edge_bps": 50.0,
                        "modeled_round_trip_cost_bps": 30.0,
                        "minimum_positive_net_edge_bps": 5.0,
                        "conservative_net_edge_bps": 20.0,
                        "economically_positive": True,
                    },
                }
            ],
        },
    )

    rows = list(
        result["timeframe_assessments"].values()
    )

    assert rows[0]["independently_qualified"] is False
    assert (
        rows[0]["legacy_execution_intent_qualified"]
        is False
    )
    assert result["legacy_execution_intent_active"] is False
