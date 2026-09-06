from __future__ import annotations

from leantrader.production.legacy_engine_bridge import (
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
