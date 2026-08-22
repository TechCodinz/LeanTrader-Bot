import time
import pandas as pd

from leantrader.agents.microstructure_sniper import MicroAgentFoundry, UltraMicrostructureSniper


def candles():
    return pd.DataFrame({"close": [100 + i * 0.2 for i in range(40)]})


def book(bid_amount=1000.0, ask_amount=100.0):
    return {
        "bids": [[100.00 - i * 0.01, bid_amount / (i + 1)] for i in range(10)],
        "asks": [[100.02 + i * 0.01, ask_amount / (i + 1)] for i in range(10)],
    }


def test_cost_floor_and_no_execution_authority():
    engine = UltraMicrostructureSniper()
    f = engine.extract(
        symbol="BTC/USDT",
        order_book=book(),
        trades=[{"timestamp": time.time() * 1000, "price": 100.02, "amount": 2, "side": "buy"}],
        candles=candles(),
    )
    rows = engine.assess(f, modeled_round_trip_cost_bps=30.0)
    assert {r.horizon_seconds for r in rows} == {5, 15, 30, 60}
    assert all(r.modeled_round_trip_cost_bps >= 30 for r in rows)
    assert all(not r.execution_authority and not r.testnet_authority and not r.live_authority for r in rows)


def test_foundry_is_bounded_and_shadow_only():
    engine = UltraMicrostructureSniper(minimum_confidence=0.50, minimum_depth_usd=1.0)
    f = engine.extract(
        symbol="FAST/USDT",
        order_book=book(5000.0, 10.0),
        trades=[
            {"timestamp": time.time() * 1000 - 200, "price": 100.02, "amount": 100, "side": "buy"},
            {"timestamp": time.time() * 1000 - 100, "price": 100.02, "amount": 100, "side": "buy"},
        ],
        candles=candles(),
    )
    proposals = MicroAgentFoundry().propose(engine.assess(f, modeled_round_trip_cost_bps=30.0))
    assert len(proposals) <= 2
    assert all(p["independently_qualified"] for p in proposals)
    assert all(not p["automatic_promotion"] and not p["execution_authority"] for p in proposals)

def test_foundry_requires_positive_prospective_evidence():
    engine = UltraMicrostructureSniper(
        minimum_confidence=0.0,
        minimum_depth_usd=1.0,
    )
    f = engine.extract(
        symbol="FAST/USDT",
        order_book=book(5000.0, 10.0),
        trades=[
            {
                "timestamp": time.time() * 1000 - 200,
                "price": 100.02,
                "amount": 100,
                "side": "buy",
            },
            {
                "timestamp": time.time() * 1000 - 100,
                "price": 100.02,
                "amount": 100,
                "side": "buy",
            },
        ],
        candles=candles(),
    )

    assessments = engine.assess(
        f,
        modeled_round_trip_cost_bps=30.0,
    )

    # No evidence means no proposal.
    assert MicroAgentFoundry().propose(
        assessments,
        evidence_rankings={},
    ) == []

    target = assessments[0]
    key = (
        f"{target.specialist}|"
        f"{target.horizon_seconds}|"
        f"{target.regime}"
    )

    evidence = {
        key: {
            "samples": 40,
            "directional_accuracy": 0.70,
            "average_net_after_cost_bps": 4.0,
            "conservative_net_after_cost_bps": 1.5,
            "evidence_qualified": True,
        }
    }

    proposals = MicroAgentFoundry().propose(
        assessments,
        evidence_rankings=evidence,
    )

    assert len(proposals) == 1
    proposal = proposals[0]

    assert proposal["evidence_qualified"] is True
    assert proposal["evidence_samples"] == 40
    assert proposal["conservative_net_edge_bps"] == 1.5
    assert proposal["expected_edge_bps"] > proposal["modeled_round_trip_cost_bps"]
    assert proposal["execution_authority"] is False
    assert proposal["testnet_authority"] is False
    assert proposal["live_authority"] is False


def test_foundry_rejects_negative_or_under_sampled_evidence():
    engine = UltraMicrostructureSniper(
        minimum_confidence=0.0,
        minimum_depth_usd=1.0,
    )
    f = engine.extract(
        symbol="FAST/USDT",
        order_book=book(5000.0, 10.0),
        trades=[
            {
                "timestamp": time.time() * 1000,
                "price": 100.02,
                "amount": 100,
                "side": "buy",
            }
        ],
        candles=candles(),
    )

    assessments = engine.assess(
        f,
        modeled_round_trip_cost_bps=30.0,
    )
    target = assessments[0]

    key = (
        f"{target.specialist}|"
        f"{target.horizon_seconds}|"
        f"{target.regime}"
    )

    foundry = MicroAgentFoundry()

    assert foundry.propose(
        assessments,
        evidence_rankings={
            key: {
                "samples": 100,
                "directional_accuracy": 0.8,
                "average_net_after_cost_bps": -1.0,
                "conservative_net_after_cost_bps": -2.0,
                "evidence_qualified": False,
            }
        },
    ) == []

    assert foundry.propose(
        assessments,
        evidence_rankings={
            key: {
                "samples": 10,
                "directional_accuracy": 0.8,
                "average_net_after_cost_bps": 5.0,
                "conservative_net_after_cost_bps": 3.0,
                "evidence_qualified": False,
            }
        },
    ) == []

def test_generalized_calibration_journal_enforces_owned_horizons(tmp_path):
    from leantrader.agents.micro_calibration import MicroCalibrationJournal

    slow = MicroCalibrationJournal(
        tmp_path / "slow.json",
        accepted_horizons=(120, 300, 900),
    )

    rows = [
        {
            "horizon_seconds": 60,
            "direction": "long",
            "confidence": 0.5,
            "pressure_score": 0.5,
            "expected_edge_bps": 40.0,
            "modeled_round_trip_cost_bps": 30.0,
            "independently_qualified": True,
            "reason": "qualified",
            "specialist": "wrong_horizon",
            "regime": "long",
        },
        {
            "horizon_seconds": 300,
            "direction": "long",
            "confidence": 0.6,
            "pressure_score": 0.7,
            "expected_edge_bps": 50.0,
            "modeled_round_trip_cost_bps": 30.0,
            "independently_qualified": True,
            "reason": "qualified",
            "specialist": "timeframe_mind_5m",
            "regime": "long",
        },
    ]

    added = slow.register(
        symbol="BTC/USDT",
        midpoint=100.0,
        assessments=rows,
        observed_at=1000.0,
    )

    assert added == 1

    health = slow.health()
    assert health["accepted_horizons_seconds"] == [120, 300, 900]
    assert health["pending_labels"] == 1

def test_v150_temporal_history_warms_before_qualification():
    engine = UltraMicrostructureSniper(
        minimum_confidence=0.0,
        minimum_depth_usd=1.0,
    )

    for index in range(2):
        features = engine.extract(
            symbol="FAST/USDT",
            order_book=book(6000.0 + index * 500, 20.0),
            trades=[{
                "timestamp": (1000.0 + index * 5.0) * 1000,
                "price": 100.02,
                "amount": 200,
                "side": "buy",
            }],
            candles=candles(),
            now=1000.0 + index * 5.0,
        )

        rows = engine.assess(
            features,
            modeled_round_trip_cost_bps=30.0,
        )

        assert all(
            row.reason == "micro_temporal_history_warming"
            for row in rows
        )

    assert features.temporal_samples == 2


def test_v150_temporal_features_measure_change():
    engine = UltraMicrostructureSniper(
        minimum_confidence=0.0,
        minimum_depth_usd=1.0,
    )

    engine.extract(
        symbol="MOVE/USDT",
        order_book=book(1000.0, 1000.0),
        trades=[{
            "timestamp": 1000.0 * 1000,
            "price": 100.02,
            "amount": 20,
            "side": "sell",
        }],
        candles=candles(),
        now=1000.0,
    )

    second = engine.extract(
        symbol="MOVE/USDT",
        order_book=book(7000.0, 100.0),
        trades=[{
            "timestamp": 1005.0 * 1000,
            "price": 100.02,
            "amount": 300,
            "side": "buy",
        }],
        candles=candles(),
        now=1005.0,
    )

    assert second.temporal_samples == 2
    assert second.depth_imbalance_velocity > 0
    assert second.trade_imbalance_velocity > 0


def test_v150_preserves_cost_floor_and_execution_safety():
    engine = UltraMicrostructureSniper(
        minimum_confidence=0.0,
        minimum_depth_usd=1.0,
    )

    features = None

    for index in range(4):
        features = engine.extract(
            symbol="RARE/USDT",
            order_book=book(
                3000.0 + index * 2500.0,
                max(10.0, 700.0 - index * 200.0),
            ),
            trades=[{
                "timestamp": (1000.0 + index * 5.0) * 1000,
                "price": 100.02,
                "amount": 100 + index * 150,
                "side": "buy",
            }],
            candles=candles(),
            now=1000.0 + index * 5.0,
        )

    rows = engine.assess(
        features,
        modeled_round_trip_cost_bps=1.0,
    )

    assert all(
        row.modeled_round_trip_cost_bps >= 30.0
        for row in rows
    )
    assert all(not row.execution_authority for row in rows)
    assert all(not row.testnet_authority for row in rows)
    assert all(not row.live_authority for row in rows)
