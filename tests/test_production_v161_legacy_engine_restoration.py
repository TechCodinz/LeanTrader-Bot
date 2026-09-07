from __future__ import annotations

from leantrader.production.legacy_engine_bridge import (
    LegacyEngineBridge,
    merge_candidate_symbols,
    merge_restored_signal,
)


def test_restored_candidate_ranking_prefers_positive_legacy_signal():
    merged = merge_candidate_symbols(
        ["BTC/USDT", "ETH/USDT"],
        [
            {
                "symbol": "SOL/USDT",
                "score": 0.8,
                "confidence": 0.7,
            },
            {
                "symbol": "XRP/USDT",
                "score": 0.2,
                "confidence": 0.9,
            },
        ],
        limit=3,
    )
    assert merged == [
        "SOL/USDT",
        "XRP/USDT",
        "BTC/USDT",
    ]


def test_restored_signal_adds_legacy_evidence_without_execution_authority():
    base = {
        "ranked_opportunity": {
            "quality_multiplier": 0.10,
        },
        "timeframe_assessments": {},
        "live_authority": False,
    }
    legacy = {
        "score": 0.75,
        "confidence": 0.80,
        "contributors": ["ultra_scalping.micro_momentum"],
        "contributions": [
            {
                "source": "ultra_scalping.micro_momentum",
                "timeframe": "1m",
                "direction": "long",
                "confidence": 0.82,
                "expected_edge_bps": 8.0,
            }
        ],
    }

    merged = merge_restored_signal(base, legacy)

    assert merged["legacy_restoration_active"] is True
    assert (
        merged["ranked_opportunity"][
            "legacy_restoration_score"
        ]
        == 0.75
    )
    assert (
        merged["ranked_opportunity"]["quality_multiplier"]
        > 0.10
    )

    rows = list(
        merged["timeframe_assessments"].values()
    )
    assert len(rows) == 1
    assert rows[0]["direction"] == "long"
    assert rows[0]["confidence"] == 0.82
    assert rows[0]["independently_qualified"] is False
    assert rows[0]["execution_authority"] is False
    assert rows[0]["live_authority"] is False


def test_restored_legacy_edge_is_net_of_modeled_cost_before_handoff():
    bridge = object.__new__(
        LegacyEngineBridge
    )
    bridge.minimum_round_trip_cost_bps = 30.0
    bridge.minimum_positive_net_edge_bps = 5.0

    rows = []

    bridge._append(
        rows,
        source="legacy.subcost",
        timeframe="1m",
        direction="long",
        confidence=0.90,
        expected_edge_bps=2.0,
    )

    assert len(rows) == 1
    assert rows[0]["expected_edge_bps"] == 0.0
    assert (
        rows[0]["metadata"][
            "economically_positive"
        ]
        is False
    )

    base = {
        "ranked_opportunity": {
            "quality_multiplier": 0.10,
        },
        "timeframe_assessments": {},
        "live_authority": False,
    }

    subcost = merge_restored_signal(
        base,
        {
            "score": 0.95,
            "confidence": 0.95,
            "contributors": [
                "legacy.subcost"
            ],
            "contributions": rows,
        },
    )

    assert (
        subcost["ranked_opportunity"][
            "quality_multiplier"
        ]
        == 0.10
    )
    assert (
        subcost["ranked_opportunity"][
            "legacy_restoration_economic_support"
        ]
        is False
    )
    assert (
        subcost["timeframe_assessments"]
        == {}
    )

    bridge._append(
        rows,
        source="legacy.viable",
        timeframe="1m",
        direction="long",
        confidence=0.90,
        expected_edge_bps=50.0,
    )

    assert rows[1]["expected_edge_bps"] == 20.0
    assert (
        rows[1]["metadata"][
            "economically_positive"
        ]
        is True
    )

    viable = merge_restored_signal(
        base,
        {
            "score": 0.80,
            "confidence": 0.90,
            "contributors": [
                "legacy.viable"
            ],
            "contributions": [rows[1]],
        },
    )

    assessments = list(
        viable[
            "timeframe_assessments"
        ].values()
    )

    assert len(assessments) == 1
    assert (
        assessments[0]["source"]
        == "legacy.viable"
    )
    assert (
        assessments[0][
            "expected_edge_bps"
        ]
        == 20.0
    )
    assert (
        assessments[0][
            "legacy_economically_positive"
        ]
        is True
    )
    assert (
        assessments[0][
            "independently_qualified"
        ]
        is False
    )
    assert (
        assessments[0][
            "live_authority"
        ]
        is False
    )
