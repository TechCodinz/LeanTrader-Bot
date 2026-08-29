"""v1.60.33 realized-dust accounting integrity.

Bybit Testnet only.

This module corrects one accounting distinction:

* finalized residual dust from a proven filled BUY -> filled SELL cycle is a
  realized-cycle cost and may reduce realized Testnet net;
* wallet-held inventory that is merely below the exchange's current executable
  sell minimum is still owned inventory, not realized loss;
* exit-impaired executor positions remain governed by v1.60.30 quarantine.

No exchange order is submitted here. No position, wallet balance, fill,
closed-position count, or global exchange-realized PnL is fabricated or
modified.
"""

from __future__ import annotations

import copy
from typing import Any

from .testnet_residual_dust_cycle_v1627 import _n


VERSION = "1.60.33"

MAX_ACCOUNTED_CYCLE_KEYS = 5000


def _eligible_finalized_cycle(
    row: Any,
) -> bool:
    return bool(
        isinstance(row, dict)
        and row.get("completed_executable_cycle") is True
        and row.get("counted_as_executed_close") is True
        and str(row.get("cycle_key") or "")
    )


def _sync_finalized_dust_ledger(
    engine: Any,
) -> dict[str, Any]:
    """Persist a cumulative finalized-residual ledger idempotently.

    The canonical v1.60.27 row list is bounded. We therefore migrate its
    currently available completed cycles into a durable cumulative amount and
    cycle-key ledger. Future completed cycles are added exactly once.
    """

    with engine._io_lock:
        state = engine.state

        existing_keys = [
            str(key)
            for key in (
                state.get(
                    "v1633_finalized_residual_dust_cycle_keys"
                )
                or []
            )
            if str(key or "")
        ]

        key_set = set(existing_keys)

        stored_total = state.get(
            "v1633_finalized_residual_dust_cost_basis_usd"
        )

        initialized = stored_total is not None

        total = max(
            0.0,
            _n(stored_total),
        )

        added = 0
        added_cost = 0.0

        rows = (
            state.get(
                "v1627_completed_executable_cycles"
            )
            or []
        )

        if not initialized:
            total = 0.0
            existing_keys = []
            key_set = set()

        for row in rows:
            if not _eligible_finalized_cycle(row):
                continue

            key = str(
                row.get("cycle_key") or ""
            )

            if key in key_set:
                continue

            cost = max(
                0.0,
                _n(
                    row.get(
                        "residual_dust_cost_basis_usd"
                    )
                ),
            )

            total += cost
            added_cost += cost
            added += 1

            existing_keys.append(key)
            key_set.add(key)

        changed = bool(
            not initialized
            or added > 0
        )

        if changed:
            state[
                "v1633_finalized_residual_dust_cost_basis_usd"
            ] = total

            state[
                "v1633_finalized_residual_dust_cycle_keys"
            ] = existing_keys[
                -MAX_ACCOUNTED_CYCLE_KEYS:
            ]

            state[
                "v1633_finalized_residual_dust_cycles"
            ] = len(existing_keys)

            state[
                "v1633_last_ledger_sync"
            ] = {
                "version": VERSION,
                "cycles_added": added,
                "cost_basis_added_usd": added_cost,
                "finalized_residual_dust_cost_basis_usd": (
                    total
                ),
                "global_realized_pnl_mutated": False,
                "positions_mutated": False,
                "order_submitted": False,
                "testnet_only": True,
                "live_authority": False,
            }

            engine._save_state()

        return {
            "finalized_residual_dust_cost_basis_usd": (
                total
            ),
            "finalized_cycle_count": len(
                existing_keys
            ),
            "cycles_added_this_sync": added,
            "cost_basis_added_this_sync_usd": (
                added_cost
            ),
            "global_realized_pnl_mutated": False,
            "positions_mutated": False,
            "order_submitted": False,
            "testnet_only": True,
            "live_authority": False,
        }


def realized_dust_accounting(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Return corrected realized-vs-held-inventory accounting."""

    realized = _n(
        state.get("realized_pnl_usd")
    )

    legacy_dust = max(
        0.0,
        _n(
            state.get(
                "dust_cost_basis_usd_total"
            )
        ),
    )

    stored_finalized = state.get(
        "v1633_finalized_residual_dust_cost_basis_usd"
    )

    if stored_finalized is None:
        seen: set[str] = set()
        finalized = 0.0

        for row in (
            state.get(
                "v1627_completed_executable_cycles"
            )
            or []
        ):
            if not _eligible_finalized_cycle(row):
                continue

            key = str(
                row.get("cycle_key") or ""
            )

            if key in seen:
                continue

            seen.add(key)

            finalized += max(
                0.0,
                _n(
                    row.get(
                        "residual_dust_cost_basis_usd"
                    )
                ),
            )

        finalized_cycle_count = len(seen)

    else:
        finalized = max(
            0.0,
            _n(stored_finalized),
        )

        finalized_cycle_count = len(
            {
                str(key)
                for key in (
                    state.get(
                        "v1633_finalized_residual_dust_cycle_keys"
                    )
                    or []
                )
                if str(key or "")
            }
        )

    free = (
        (
            state.get("account_balance")
            or {}
        ).get("free")
        or {}
    )

    wallet_rows: list[
        dict[str, Any]
    ] = []

    wallet_held_cost = 0.0
    wallet_held_estimated = 0.0

    for symbol, row in (
        state.get(
            "non_tradeable_dust"
        )
        or {}
    ).items():
        if not isinstance(row, dict):
            continue

        if (
            row.get(
                "counted_as_executed_close"
            )
            is True
        ):
            continue

        normalized = str(
            symbol or ""
        ).upper()

        if not normalized:
            continue

        base = normalized.split(
            "/",
            1,
        )[0]

        quantity = max(
            0.0,
            _n(row.get("quantity")),
        )

        wallet_free = max(
            0.0,
            _n(free.get(base)),
        )

        cost = max(
            0.0,
            _n(
                row.get(
                    "cost_basis_usd"
                )
            ),
        )

        estimated = max(
            0.0,
            _n(
                row.get(
                    "estimated_value_usd"
                )
            ),
        )

        if (
            quantity <= 0.0
            and wallet_free <= 0.0
        ):
            continue

        wallet_held_cost += cost
        wallet_held_estimated += (
            estimated
        )

        wallet_rows.append(
            {
                "symbol": normalized,
                "quantity": quantity,
                "wallet_free_quantity": (
                    wallet_free
                ),
                "cost_basis_usd": cost,
                "estimated_value_usd": (
                    estimated
                ),
                "reason": row.get(
                    "reason"
                ),
                "counted_as_executed_close": (
                    False
                ),
                "realized_loss": False,
                "liquid_quote_capital": False,
                "testnet_only": True,
                "live_authority": False,
            }
        )

    corrected = (
        realized
        - finalized
    )

    legacy_net = (
        realized
        - legacy_dust
    )

    return {
        "version": VERSION,
        "exchange_realized_pnl_usd": (
            realized
        ),
        "legacy_cumulative_dust_cost_basis_usd": (
            legacy_dust
        ),
        "finalized_residual_dust_cost_basis_usd": (
            finalized
        ),
        "finalized_cycle_count": (
            finalized_cycle_count
        ),
        "wallet_held_nonfinalized_cost_basis_usd": (
            wallet_held_cost
        ),
        "wallet_held_nonfinalized_estimated_value_usd": (
            wallet_held_estimated
        ),
        "wallet_held_nonfinalized_inventory": (
            wallet_rows
        ),
        "realized_net_after_finalized_dust_usd": (
            corrected
        ),
        "legacy_realized_net_after_all_dust_usd": (
            legacy_net
        ),
        "wallet_held_inventory_is_not_realized_loss": (
            True
        ),
        "wallet_held_inventory_is_not_liquid_quote": (
            True
        ),
        "global_realized_pnl_mutated": False,
        "positions_mutated": False,
        "order_submitted": False,
        "testnet_only": True,
        "live_authority": False,
    }


def install_testnet_realized_dust_integrity_v1633(
) -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .testnet_execution import (
        BybitTestnetExecutionEngine,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        BybitTestnetExecutionEngine,
        "_v1633_realized_dust_integrity_installed",
        False,
    ):
        return

    original_engine_health = (
        BybitTestnetExecutionEngine.health
    )

    original_prepare_sell = (
        BybitTestnetExecutionEngine.prepare_sell
    )

    original_start = (
        BybitTestnetExecutionEngine.start
    )

    original_lane_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def start(
        self: Any,
    ) -> None:
        original_start(self)

        # Metadata-only migration. No exchange order, position mutation,
        # fill fabrication, or realized-PnL mutation.
        _sync_finalized_dust_ledger(
            self
        )

    def prepare_sell(
        self: Any,
        symbol: str,
        requested_quantity: float,
        reference_price: float,
    ) -> dict[str, Any]:
        result = original_prepare_sell(
            self,
            symbol,
            requested_quantity,
            reference_price,
        )

        # v1.60.29 may have finalized a real BUY->SELL residual cycle
        # inside the wrapped preparation. Pick up that new durable cycle.
        if (
            str(
                (result or {}).get(
                    "status"
                )
                or ""
            )
            == "dust"
        ):
            _sync_finalized_dust_ledger(
                self
            )

        return result

    def engine_health(
        self: Any,
    ) -> dict[str, Any]:
        _sync_finalized_dust_ledger(
            self
        )

        payload = (
            original_engine_health(
                self
            )
        )

        with self._io_lock:
            state_copy = {
                "realized_pnl_usd": (
                    self.state.get(
                        "realized_pnl_usd"
                    )
                ),
                "dust_cost_basis_usd_total": (
                    self.state.get(
                        "dust_cost_basis_usd_total"
                    )
                ),
                "v1633_finalized_residual_dust_cost_basis_usd": (
                    self.state.get(
                        "v1633_finalized_residual_dust_cost_basis_usd"
                    )
                ),
                "v1633_finalized_residual_dust_cycle_keys": (
                    list(
                        self.state.get(
                            "v1633_finalized_residual_dust_cycle_keys"
                        )
                        or []
                    )
                ),
                "v1627_completed_executable_cycles": (
                    copy.deepcopy(
                        self.state.get(
                            "v1627_completed_executable_cycles"
                        )
                        or []
                    )
                ),
                "non_tradeable_dust": (
                    copy.deepcopy(
                        self.state.get(
                            "non_tradeable_dust"
                        )
                        or {}
                    )
                ),
                "account_balance": (
                    copy.deepcopy(
                        self.state.get(
                            "account_balance"
                        )
                        or {}
                    )
                ),
            }

        accounting = (
            realized_dust_accounting(
                state_copy
            )
        )

        performance = dict(
            payload.get(
                "performance"
            )
            or {}
        )

        realized = _n(
            performance.get(
                "realized_pnl_usd"
            ),
            accounting[
                "exchange_realized_pnl_usd"
            ],
        )

        legacy = max(
            0.0,
            _n(
                performance.get(
                    "non_tradeable_dust_cost_basis_usd"
                ),
                accounting[
                    "legacy_cumulative_dust_cost_basis_usd"
                ],
            ),
        )

        finalized = (
            accounting[
                "finalized_residual_dust_cost_basis_usd"
            ]
        )

        # This is the key compatibility correction. v1.60.8's existing
        # compounding gate reads this field dynamically through safe_snapshot().
        # It must represent only finalized residual dust, not unsold wallet
        # inventory.
        performance[
            "legacy_non_tradeable_dust_cost_basis_usd"
        ] = legacy

        performance[
            "non_tradeable_dust_cost_basis_usd"
        ] = finalized

        performance[
            "finalized_residual_dust_cost_basis_usd"
        ] = finalized

        performance[
            "wallet_held_nonfinalized_cost_basis_usd"
        ] = accounting[
            "wallet_held_nonfinalized_cost_basis_usd"
        ]

        performance[
            "wallet_held_nonfinalized_estimated_value_usd"
        ] = accounting[
            "wallet_held_nonfinalized_estimated_value_usd"
        ]

        performance[
            "realized_net_after_dust_usd"
        ] = (
            realized
            - finalized
        )

        performance[
            "legacy_realized_net_after_dust_usd"
        ] = (
            realized
            - legacy
        )

        performance[
            "wallet_held_inventory_is_not_realized_loss"
        ] = True

        payload[
            "performance"
        ] = performance

        payload[
            "realized_dust_accounting_integrity"
        ] = {
            **accounting,
            "compounding_input_dust_basis": (
                "finalized_residual_dust_only"
            ),
            "legacy_counter_preserved_for_audit": (
                True
            ),
            "legacy_counter_used_for_compounding": (
                False
            ),
        }

        payload[
            "live_authority"
        ] = False

        return payload

    def lane_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_lane_health(
                self
            )
        )

        testnet = getattr(
            self,
            "testnet",
            None,
        )

        if testnet is None:
            return payload

        _sync_finalized_dust_ledger(
            testnet
        )

        with testnet._io_lock:
            accounting = (
                realized_dust_accounting(
                    {
                        "realized_pnl_usd": (
                            testnet.state.get(
                                "realized_pnl_usd"
                            )
                        ),
                        "dust_cost_basis_usd_total": (
                            testnet.state.get(
                                "dust_cost_basis_usd_total"
                            )
                        ),
                        "v1633_finalized_residual_dust_cost_basis_usd": (
                            testnet.state.get(
                                "v1633_finalized_residual_dust_cost_basis_usd"
                            )
                        ),
                        "v1633_finalized_residual_dust_cycle_keys": (
                            list(
                                testnet.state.get(
                                    "v1633_finalized_residual_dust_cycle_keys"
                                )
                                or []
                            )
                        ),
                        "v1627_completed_executable_cycles": (
                            copy.deepcopy(
                                testnet.state.get(
                                    "v1627_completed_executable_cycles"
                                )
                                or []
                            )
                        ),
                        "non_tradeable_dust": (
                            copy.deepcopy(
                                testnet.state.get(
                                    "non_tradeable_dust"
                                )
                                or {}
                            )
                        ),
                        "account_balance": (
                            copy.deepcopy(
                                testnet.state.get(
                                    "account_balance"
                                )
                                or {}
                            )
                        ),
                    }
                )
            )

        with self._lock:
            sizing = copy.deepcopy(
                self.state.get(
                    "last_sizing"
                )
                or {}
            )

        quarantine = dict(
            payload.get(
                "exit_impaired_quarantine"
            )
            or {}
        )

        quarantine[
            "actual_realized_net_usd"
        ] = accounting[
            "realized_net_after_finalized_dust_usd"
        ]

        quarantine[
            "legacy_actual_realized_net_usd"
        ] = accounting[
            "legacy_realized_net_after_all_dust_usd"
        ]

        quarantine[
            "wallet_held_nonfinalized_cost_basis_usd"
        ] = accounting[
            "wallet_held_nonfinalized_cost_basis_usd"
        ]

        quarantine[
            "wallet_held_inventory_is_not_realized_loss"
        ] = True

        quarantine[
            "compounding_active"
        ] = bool(
            sizing.get(
                "compounding"
            )
        )

        quarantine[
            "compounding_reason"
        ] = (
            sizing.get(
                "compounding_gate"
            )
            or "awaiting_actual_testnet_gate"
        )

        quarantine[
            "actual_testnet_net_after_modeled_cost_usd"
        ] = sizing.get(
            "actual_testnet_net_after_modeled_cost_usd"
        )

        quarantine[
            "live_authority"
        ] = False

        payload[
            "exit_impaired_quarantine"
        ] = quarantine

        payload[
            "realized_dust_accounting_integrity"
        ] = {
            **accounting,
            "modeled_round_trip_cost_floor_unchanged": (
                True
            ),
            "compounding_decided_by_existing_v1608_gate": (
                True
            ),
        }

        payload[
            "version"
        ] = VERSION

        payload[
            "live_authority"
        ] = False

        return payload

    BybitTestnetExecutionEngine.start = (
        start
    )

    BybitTestnetExecutionEngine.prepare_sell = (
        prepare_sell
    )

    BybitTestnetExecutionEngine.health = (
        engine_health
    )

    HyperSpeedCollectiveTestnetLane.health = (
        lane_health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        VERSION
    )

    VelocitySniperTestnetLane.VERSION = (
        VERSION
    )

    BybitTestnetExecutionEngine._v1633_realized_dust_integrity_installed = (
        True
    )
