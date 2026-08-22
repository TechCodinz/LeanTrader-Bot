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
