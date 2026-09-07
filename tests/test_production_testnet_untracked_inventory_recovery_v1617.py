from leantrader.production.testnet_untracked_inventory_recovery_v1617 import (
    VERSION,
    _full_balance_maps,
    _rotating_batch,
    _tracked_assets,
)


def test_full_balance_maps_keeps_all_nonzero_assets():
    total, free, used = _full_balance_maps(
        {
            "total": {
                "USDT": 1.8,
                "ATOM": 5.0,
                "ADA": 0.0,
            },
            "free": {
                "USDT": 1.8,
                "ATOM": 5.0,
                "ADA": 0.0,
            },
            "used": {
                "USDT": 0.0,
                "ATOM": 0.0,
                "ADA": 0.0,
            },
        }
    )

    assert total["USDT"] == 1.8
    assert total["ATOM"] == 5.0
    assert free["ATOM"] == 5.0
    assert "ADA" not in total
    assert used["ATOM"] == 0.0


def test_tracked_assets_include_positions_and_dust():
    state = {
        "positions": {
            "CSPR/USDT": 10.0,
        },
        "non_tradeable_dust": {
            "CHIP/USDT": {},
        },
    }

    assets = _tracked_assets(state)

    assert "USDT" in assets
    assert "CSPR" in assets
    assert "CHIP" in assets


def test_rotation_reaches_assets_beyond_first_six():
    candidates = [
        "LUNC/USDT",
        "ETH/USDT",
        "ADA/USDT",
        "ETC/USDT",
        "BTC/USDT",
        "SOL/USDT",
        "USDC/USDT",
        "TRX/USDT",
        "BNB/USDT",
        "GALA/USDT",
        "ATOM/USDT",
    ]

    first, start1, next1 = _rotating_batch(
        candidates,
        cursor=0,
        limit=6,
    )

    second, start2, next2 = _rotating_batch(
        candidates,
        cursor=next1,
        limit=6,
    )

    assert start1 == 0
    assert first == candidates[:6]

    assert start2 == 6
    assert "ATOM/USDT" in second

    assert next2 == 1


def test_version():
    assert VERSION == "1.61.8"
