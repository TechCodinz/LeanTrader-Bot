from __future__ import annotations

from typing import Any


def install_testnet_exit_recycle_compat_v1608() -> None:
    """Keep legacy Testnet adapters observable without weakening execution gates.

    The supported runtime's real Bybit executor already exposes safe_snapshot().
    Older deterministic runner adapters used by compatibility tests only expose
    health(). Give those adapters the equivalent read-only snapshot surface.
    Sell preparation and all execution safety checks remain unchanged.
    """

    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_testnet_exit_recycle_compat_v1608_installed",
        False,
    ):
        return

    original_init = HyperSpeedCollectiveTestnetLane.__init__

    def init_with_snapshot_compat(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        snapshot = getattr(self.testnet, "safe_snapshot", None)
        health = getattr(self.testnet, "health", None)
        if not callable(snapshot) and callable(health):
            # Read-only compatibility only. Do not synthesize prepare_sell(),
            # reconciliation, balances, execution authority or order methods.
            self.testnet.safe_snapshot = health

    HyperSpeedCollectiveTestnetLane.__init__ = init_with_snapshot_compat
    HyperSpeedCollectiveTestnetLane._testnet_exit_recycle_compat_v1608_installed = True
