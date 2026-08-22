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
