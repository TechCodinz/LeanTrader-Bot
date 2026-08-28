from __future__ import annotations

import pytest

from leantrader.production.testnet_exit_impaired_quarantine_v1630 import (
    REASON_FREE_BALANCE,
    REASON_PRICE_LIMIT,
    STATE_EXECUTABILITY_UNPROVABLE,
    STATE_HEALTHY,
    STATE_NOT_EXECUTABLE_RESIDUAL,
    STATE_TRUE_DUST,
    STATE_UNRESOLVED_ORDER,
    actual_realized_net_usd,
    classify_symbol_exit_state,
    exit_impaired_snapshot,
)
from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
    seed,
)
from tests.test_production_testnet_exit_recycle_v1608 import hyper_lane
from tests.test_testnet_execution import engine


class _Exchange:
    """Minimum-resolving exchange double (CCXT limits present)."""

    def __init__(self, min_amount=0.001, min_cost=1.0):
        self._min_amount = min_amount
        self._min_cost = min_cost

    def market(self, symbol):
        return {
            "symbol": symbol,
            "limits": {
                "amount": {"min": self._min_amount},
                "cost": {"min": self._min_cost},
            },
            "info": {},
        }


def _engine_state(**overrides):
    state = {
        "positions": {},
        "position_cost_usd": {},
        "orders": {},
        "account_balance": {"free": {}},
        "non_tradeable_dust": {},
        "realized_pnl_usd": 0.0,
        "dust_cost_basis_usd_total": 0.0,
    }
    state.update(overrides)
    return state


def _cspr_impaired_states():
    """CSPR/JASMY-style live evidence: real positions, boundary not executable."""

    engine_state = _engine_state(
        positions={"CSPR/USDT": 750.9483, "JASMY/USDT": 0.10989},
        position_cost_usd={"CSPR/USDT": 1.239, "JASMY/USDT": 0.000428},
        account_balance={"free": {"CSPR": 750.9483, "JASMY": 0.10989}},
    )
    lane_state = {
        "deferred_exit_recoveries": {
            "CSPR/USDT": {"attempts": 5},
            "JASMY/USDT": {"attempts": 4},
        },
        "v1615_price_limit_watch": {
            "CSPR/USDT": {"executable_boundary": False, "sell_limit": 0.00268},
            "JASMY/USDT": {"executable_boundary": False, "sell_limit": 0.0046},
        },
    }
    return engine_state, lane_state


# ---------------------------------------------------------------------------
# P0 - quarantine classification
# ---------------------------------------------------------------------------


def test_cspr_price_limit_impaired_is_quarantined_and_retained():
    """1: a CSPR-like price-limit-impaired position stays tracked."""

    engine_state, lane_state = _cspr_impaired_states()

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state=lane_state,
        symbol="CSPR/USDT",
        exchange=_Exchange(),
    )

    assert row["impaired"] is True
    assert row["reason"] == REASON_PRICE_LIMIT
    assert row["quantity"] == pytest.approx(750.9483)
    assert row["live_authority"] is False

    # Still a real position: never dust, never deleted.
    assert engine_state["positions"]["CSPR/USDT"] == pytest.approx(750.9483)
    assert "CSPR/USDT" not in engine_state["non_tradeable_dust"]


def test_impaired_snapshot_separates_inventory_classes():
    """2: impaired inventory is counted apart from healthy inventory."""

    engine_state, lane_state = _cspr_impaired_states()
    engine_state["positions"]["BTC/USDT"] = 0.5
    engine_state["position_cost_usd"]["BTC/USDT"] = 50.0
    engine_state["account_balance"]["free"]["BTC"] = 0.5

    snapshot = exit_impaired_snapshot(
        engine_state=engine_state,
        lane_state=lane_state,
        exchange=_Exchange(),
    )

    assert snapshot["exit_impaired_symbols"] == ["CSPR/USDT", "JASMY/USDT"]
    assert snapshot["impaired_inventory_count"] == 2
    assert snapshot["executor_inventory_count"] == 3
    assert snapshot["healthy_inventory_count"] == 1
    assert snapshot["impaired_capital_is_not_liquid"] is True
    assert snapshot["impaired_inventory_notional_usd"] == pytest.approx(
        1.239 + 0.000428
    )


def test_quarantine_lifts_when_position_disappears():
    """3: a real reconciled exit removes quarantine automatically."""

    engine_state, lane_state = _cspr_impaired_states()

    # The exit finally reconciles; the position is gone.
    engine_state["positions"].pop("CSPR/USDT")

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state=lane_state,
        symbol="CSPR/USDT",
        exchange=_Exchange(),
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_HEALTHY
    assert row["reason"] == "no_open_position"

    snapshot = exit_impaired_snapshot(
        engine_state=engine_state,
        lane_state=lane_state,
        exchange=_Exchange(),
    )
    assert "CSPR/USDT" not in snapshot["exit_impaired_symbols"]


def test_quarantine_lifts_when_boundary_becomes_executable():
    """3b: quarantine also lifts when the obstruction genuinely clears."""

    engine_state, lane_state = _cspr_impaired_states()
    lane_state["v1615_price_limit_watch"]["CSPR/USDT"][
        "executable_boundary"
    ] = True
    lane_state["deferred_exit_recoveries"].pop("CSPR/USDT")

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state=lane_state,
        symbol="CSPR/USDT",
        exchange=_Exchange(),
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_HEALTHY


# ---------------------------------------------------------------------------
# P3 - free_balance_not_executable
# ---------------------------------------------------------------------------


def test_free_balance_not_executable_is_quarantined_not_dust():
    """4: executable position + non-executable free balance = impaired."""

    engine_state = _engine_state(
        positions={"AAA/USDT": 5.0},
        position_cost_usd={"AAA/USDT": 10.0},
        account_balance={"free": {"AAA": 0.0005}},
    )

    # 5.0 @ 100.0 = $500 notional, clearing both the amount and cost minimums,
    # while the free base (0.0005) clears neither.
    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state={},
        symbol="AAA/USDT",
        exchange=_Exchange(min_amount=0.001, min_cost=1.0),
        fresh_bid_provider=lambda _symbol: 100.0,
    )

    assert row["impaired"] is True
    assert row["reason"] == REASON_FREE_BALANCE
    assert row["position_executable_in_principle"] is True
    assert row["free_base_quantity"] == pytest.approx(0.0005)
    assert row["position_value_usd"] == pytest.approx(500.0)

    # Never converted to dust, never deleted.
    assert engine_state["positions"]["AAA/USDT"] == pytest.approx(5.0)
    assert engine_state["non_tradeable_dust"] == {}


def test_unresolved_order_fails_closed_and_is_not_quarantined():
    """5: an open order explains the state; never a false quarantine."""

    engine_state = _engine_state(
        positions={"AAA/USDT": 5.0},
        account_balance={"free": {"AAA": 0.0005}},
        orders={
            "o1": {
                "symbol": "AAA/USDT",
                "side": "sell",
                "status": "open",
                "filled": 0.0,
            }
        },
    )

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state={},
        symbol="AAA/USDT",
        exchange=_Exchange(min_amount=0.001),
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_UNRESOLVED_ORDER
    assert row["fail_closed"] is True


def test_true_dust_is_distinguishable_from_impairment():
    """Telemetry must separate true dust from impairment."""

    engine_state = _engine_state(
        positions={},
        non_tradeable_dust={"XRP/USDT": {"quantity": 0.00008, "cost_basis_usd": 0.0001}},
    )

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state={},
        symbol="XRP/USDT",
        exchange=_Exchange(),
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_TRUE_DUST


def test_losing_position_alone_is_never_quarantined():
    """P0: impairment requires live obstruction, not a losing position."""

    engine_state = _engine_state(
        positions={"BTC/USDT": 1.0},
        position_cost_usd={"BTC/USDT": 99999.0},
        account_balance={"free": {"BTC": 1.0}},
    )

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state={"deferred_exit_recoveries": {}, "v1615_price_limit_watch": {}},
        symbol="BTC/USDT",
        exchange=_Exchange(),
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_HEALTHY


def test_single_transient_deferral_is_not_impairment():
    engine_state = _engine_state(
        positions={"BTC/USDT": 1.0},
        account_balance={"free": {"BTC": 1.0}},
    )

    row = classify_symbol_exit_state(
        engine_state=engine_state,
        lane_state={"deferred_exit_recoveries": {"BTC/USDT": {"attempts": 1}}},
        symbol="BTC/USDT",
        exchange=_Exchange(),
    )

    assert row["impaired"] is False


# ---------------------------------------------------------------------------
# P4 - compounding integrity
# ---------------------------------------------------------------------------


def test_negative_actual_realized_net_cannot_enable_compounding():
    """9: negative realized Testnet net blocks compounding."""

    realized = actual_realized_net_usd(
        _engine_state(realized_pnl_usd=-0.0099, dust_cost_basis_usd_total=0.0013)
    )

    assert realized["actual_realized_net_usd"] == pytest.approx(-0.0112)
    assert realized["eligible_for_compounding"] is False


def test_realized_gain_smaller_than_dust_cost_cannot_compound():
    realized = actual_realized_net_usd(
        _engine_state(realized_pnl_usd=0.001, dust_cost_basis_usd_total=0.005)
    )

    assert realized["actual_realized_net_usd"] < 0.0
    assert realized["eligible_for_compounding"] is False


def test_positive_actual_realized_net_enables_compounding():
    """10: only real net after costs and dust is reinvestable."""

    realized = actual_realized_net_usd(
        _engine_state(realized_pnl_usd=2.5, dust_cost_basis_usd_total=0.5)
    )

    assert realized["actual_realized_net_usd"] == pytest.approx(2.0)
    assert realized["eligible_for_compounding"] is True
    assert realized["excludes_unrealized_gains"] is True
    assert realized["excludes_trapped_inventory_value"] is True
    assert realized["excludes_modeled_pnl"] is True


def test_trapped_inventory_value_is_never_realized_profit():
    """6: an impaired position is never counted as free/recovered capital."""

    engine_state, lane_state = _cspr_impaired_states()
    engine_state["realized_pnl_usd"] = 0.0
    engine_state["dust_cost_basis_usd_total"] = 0.0

    snapshot = exit_impaired_snapshot(
        engine_state=engine_state,
        lane_state=lane_state,
        exchange=_Exchange(),
    )
    realized = actual_realized_net_usd(engine_state)

    # Trapped notional exists, but contributes nothing to compoundable profit.
    assert snapshot["impaired_inventory_notional_usd"] > 0.0
    assert realized["actual_realized_net_usd"] == pytest.approx(0.0)
    assert realized["eligible_for_compounding"] is False


# ---------------------------------------------------------------------------
# Lane integration: routing slots, entry block, telemetry
# ---------------------------------------------------------------------------


def _impaired_lane(tmp_path, *, free_usdt=100.0):
    fake = PriceGuardBybit()
    fake.bid = 100.0
    fake.ask = 101.0
    fake.sell_limit = 0.0

    instance, _ = engine(
        tmp_path,
        fake,
        max_order_usd=10.0,
        max_position_usd=20.0,
        max_daily_submitted_usd=50.0,
        max_orders_per_day=20,
    )
    instance.start()
    seed(instance, fake, 0.1, 9.5)
    instance.reconcile_required()

    lane, service, _ = hyper_lane(tmp_path, testnet=instance)

    with instance._io_lock:
        instance.state["positions"]["CSPR/USDT"] = 750.9483
        instance.state["position_cost_usd"]["CSPR/USDT"] = 1.239
        instance.state.setdefault("account_balance", {}).setdefault("free", {})
        instance.state["account_balance"]["free"]["CSPR"] = 750.9483
        instance.state["account_balance"]["free"]["USDT"] = free_usdt
        instance._save_state()

    with lane._lock:
        lane.state.setdefault("v1615_price_limit_watch", {})["CSPR/USDT"] = {
            "executable_boundary": False,
            "sell_limit": 0.00268,
        }
        lane.state.setdefault("deferred_exit_recoveries", {})["CSPR/USDT"] = {
            "attempts": 5,
        }
        lane._save_locked()

    return lane, service, instance, fake


def test_lane_reports_quarantine_telemetry(tmp_path):
    lane, _service, _instance, _fake = _impaired_lane(tmp_path)

    state = lane.exit_impaired_state()
    assert "CSPR/USDT" in state["exit_impaired_symbols"]
    assert state["reasons"]["CSPR/USDT"] == REASON_PRICE_LIMIT

    payload = lane.health()["exit_impaired_quarantine"]
    assert payload["version"] == "1.60.30"
    assert "CSPR/USDT" in payload["exit_impaired_symbols"]
    assert payload["impaired_inventory_count"] == 1
    assert payload["quarantined_positions_never_dust"] is True
    assert payload["quarantined_positions_retained"] is True
    assert payload["impaired_capital_is_not_liquid"] is True
    assert payload["compounding_active"] is False
    assert payload["live_authority"] is False


def test_impaired_symbol_cannot_receive_another_buy(tmp_path):
    """1/11: quarantined symbol is refused at the submission boundary."""

    lane, _service, _instance, fake = _impaired_lane(tmp_path)
    before_created = len(fake.created)

    pending = {
        "kind": "entry",
        "event": {
            "symbol": "CSPR/USDT",
            "side": "buy",
            "price": 0.00165,
            "quantity": 1000.0,
            "reason": "fast_entry",
            "event_id": "e-1",
            "timestamp": "2026-08-28T10:00:00+00:00",
        },
    }

    result = lane._submit_pending(pending, now=1_000.0)

    assert result["reason"].startswith("exit_impaired_quarantine:")
    assert result["details"]["exit_impaired"] is True
    assert result["details"]["order_submitted"] is False
    assert result["details"]["position_retained"] is True
    assert result["details"]["counted_as_dust"] is False
    assert len(fake.created) == before_created


def test_quarantine_release_is_capped_by_free_quote(tmp_path):
    """8: no extra capacity when free USDT cannot fund another ticket."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=0.0)

    capacity = lane._adaptive_position_capacity(
        {},
        lane.testnet.safe_snapshot(),
        candidate_count=5,
        entries_today=0,
    )

    assert capacity["impaired_inventory_count"] == 1
    assert capacity["reconciled_free_quote_usd"] == pytest.approx(0.0)
    assert capacity["quarantine_released_slots"] == 0
    assert capacity["quarantine_release_capped_by_free_quote"] is True
    assert capacity["impaired_capital_is_not_liquid"] is True


def test_quarantine_frees_a_routing_slot_when_capital_exists(tmp_path):
    """2/7: impaired inventory stops occupying a slot, bounded by real money."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=100.0)

    capacity = lane._adaptive_position_capacity(
        {},
        lane.testnet.safe_snapshot(),
        candidate_count=5,
        entries_today=0,
    )

    assert capacity["impaired_inventory_count"] == 1
    assert capacity["reconciled_free_quote_usd"] == pytest.approx(100.0)
    assert capacity["quarantine_released_slots"] == 1
    assert capacity["quarantine_release_capped_by_free_quote"] is False
    assert capacity["healthy_reusable_routing_slots"] >= 1
    # Trapped capital is still never treated as liquid.
    assert capacity["impaired_capital_is_not_liquid"] is True


def test_compounding_decision_is_delegated_to_the_v1608_gate(tmp_path):
    """9/10: v1.60.30 republishes the existing verdict, it never re-decides.

    v1.60.8 already gates compounding on actual Testnet realized PnL net of
    residual dust and the modeled round-trip cost floor. Re-deciding here would
    duplicate that gate and could clamp legitimate compounding.
    """

    lane, _service, _instance, _fake = _impaired_lane(tmp_path)

    sizing = lane._compound_order_notional({}, slots=2)

    assert sizing["compounding_decided_by"] == "v1.60.8_actual_testnet_gate"
    assert sizing["quarantine_never_creates_compoundable_profit"] is True
    assert sizing["live_authority"] is False
    # The wrapper reports a gate verdict without inventing one.
    assert sizing["compounding_gate"] in {
        "actual_realized_net_positive",
        "actual_realized_net_not_positive",
        "no_actual_testnet_verdict_available",
    }


def test_quarantine_does_not_alter_the_compounding_verdict(tmp_path):
    """6: trapped inventory can never turn into compoundable profit."""

    lane, _service, instance, _fake = _impaired_lane(tmp_path)

    impaired = lane.exit_impaired_state()
    assert impaired["impaired_inventory_count"] == 1
    assert impaired["impaired_inventory_notional_usd"] > 0.0

    sizing = lane._compound_order_notional({}, slots=2)

    # No realized profit exists, so quarantined notional must not create any.
    with instance._io_lock:
        realized = actual_realized_net_usd(instance.state)

    assert realized["eligible_for_compounding"] is False
    assert sizing["compounding_gate"] != "actual_realized_net_positive"
    assert sizing["quarantine_never_creates_compoundable_profit"] is True


def test_prospective_exit_price_limit_entry_gate_is_preserved():
    """11: the JUP-style v1.60.13 entry gate is untouched."""

    from leantrader.production import testnet_entry_roundtrip_v1613 as v1613

    source = open(v1613.__file__, encoding="utf-8").read()
    assert '"prospective_exit_price_limit_unexecutable"' in source


def test_lane_without_executor_is_compatible_and_unchanged():
    """A lightweight lane with no executor must behave exactly as before.

    tests/test_adaptive_opportunity_capacity_v1602.py builds such a lane. It has
    no inventory, so it has no impairment and no capacity change.
    """

    from leantrader.production.fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )

    lane = object.__new__(HyperSpeedCollectiveTestnetLane)
    assert not hasattr(lane, "testnet")

    impairment = HyperSpeedCollectiveTestnetLane.exit_impaired_state(lane)

    assert impairment["exit_impaired_symbols"] == []
    assert impairment["impaired_inventory_count"] == 0
    assert impairment["executor_inventory_count"] == 0
    assert impairment["live_authority"] is False


# ---------------------------------------------------------------------------
# P0-B - full-position executability must be proven, not assumed
# ---------------------------------------------------------------------------


def _low_free_state(quantity=5.0):
    return _engine_state(
        positions={"AAA/USDT": quantity},
        position_cost_usd={"AAA/USDT": 10.0},
        account_balance={"free": {"AAA": 0.0005}},
    )


def test_amount_above_minimum_but_value_below_min_cost_is_not_impairment():
    """A genuine min-cost residual must never be called trapped inventory."""

    row = classify_symbol_exit_state(
        engine_state=_low_free_state(quantity=0.002),
        lane_state={},
        symbol="AAA/USDT",
        exchange=_Exchange(min_amount=0.001, min_cost=5.0),
        fresh_bid_provider=lambda _symbol: 100.0,  # 0.002 * 100 = $0.20 < $5
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_NOT_EXECUTABLE_RESIDUAL
    assert row["reason"] == "position_below_minimum_cost"
    assert row["estimated_value_usd"] == pytest.approx(0.2)


def test_unavailable_fresh_bid_fails_closed():
    row = classify_symbol_exit_state(
        engine_state=_low_free_state(),
        lane_state={},
        symbol="AAA/USDT",
        exchange=_Exchange(min_amount=0.001, min_cost=1.0),
        fresh_bid_provider=lambda _symbol: 0.0,
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_EXECUTABILITY_UNPROVABLE
    assert row["reason"] == "fresh_executable_bid_unavailable"
    assert row["fail_closed"] is True


def test_missing_fresh_bid_provider_fails_closed():
    row = classify_symbol_exit_state(
        engine_state=_low_free_state(),
        lane_state={},
        symbol="AAA/USDT",
        exchange=_Exchange(min_amount=0.001, min_cost=1.0),
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_EXECUTABILITY_UNPROVABLE
    assert row["fail_closed"] is True


def test_raising_fresh_bid_provider_fails_closed():
    def _boom(_symbol):
        raise RuntimeError("ticker unavailable")

    row = classify_symbol_exit_state(
        engine_state=_low_free_state(),
        lane_state={},
        symbol="AAA/USDT",
        exchange=_Exchange(min_amount=0.001, min_cost=1.0),
        fresh_bid_provider=_boom,
    )

    assert row["impaired"] is False
    assert row["state"] == STATE_EXECUTABILITY_UNPROVABLE
    assert row["fail_closed"] is True


def test_unresolved_canonical_minimums_fail_closed():
    class _NoMinimums:
        def market(self, symbol):
            return {"symbol": symbol, "limits": {}, "info": {}}

    row = classify_symbol_exit_state(
        engine_state=_low_free_state(),
        lane_state={},
        symbol="AAA/USDT",
        exchange=_NoMinimums(),
        fresh_bid_provider=lambda _symbol: 100.0,
    )

    assert row["impaired"] is False
    assert row["state"] != REASON_FREE_BALANCE


def test_cspr_jasmy_price_limit_behavior_is_unchanged_by_executability_proof():
    """Price-limit impairment is decided before any executability proof."""

    engine_state, lane_state = _cspr_impaired_states()

    for symbol in ("CSPR/USDT", "JASMY/USDT"):
        row = classify_symbol_exit_state(
            engine_state=engine_state,
            lane_state=lane_state,
            symbol=symbol,
            exchange=_Exchange(min_amount=0.001, min_cost=1.0),
            fresh_bid_provider=None,  # never consulted on this path
        )
        assert row["impaired"] is True
        assert row["reason"] == REASON_PRICE_LIMIT


# ---------------------------------------------------------------------------
# P0-A - quarantine may change routing occupancy, never risk authority
# ---------------------------------------------------------------------------


def _growth(*, remaining=15.0, risk=1.0, equity=50.0, allowed=True):
    return {
        "state": "normal",
        "equity": equity,
        "peak_equity": max(50.0, equity),
        "protected_principal": 35.0,
        "locked_profit": 0.0,
        "reinvestable_realized_profit": 0.0,
        "remaining_deployable_notional": remaining,
        "risk_multiplier": risk,
        "new_entries_allowed": allowed,
    }


def _capacity(lane, *, growth=None, candidate_count=8, entries_today=0):
    supervisor = {"capital_growth": growth if growth is not None else _growth()}
    return lane._adaptive_position_capacity(
        supervisor,
        lane.testnet.safe_snapshot(),
        candidate_count=candidate_count,
        entries_today=entries_today,
    )


def _assert_within_envelope(capacity):
    """The universal invariant: never above what healthy occupancy allows."""

    assert capacity["available_slots"] <= capacity[
        "healthy_occupancy_available_slots"
    ]
    assert capacity["risk_authority_unchanged"] is True
    assert capacity["impaired_capital_is_not_liquid"] is True


def test_release_cannot_exceed_candidate_room(tmp_path):
    """5: no candidates means no release, however much capital exists."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane, candidate_count=0)

    assert capacity["impaired_inventory_count"] == 1
    assert capacity["quarantine_released_slots"] == 0
    assert capacity["quarantine_release_ceiling"] == 0
    _assert_within_envelope(capacity)


def test_release_cannot_bypass_zero_risk_multiplier(tmp_path):
    """6: a zero risk multiplier cannot be widened by quarantine."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane, growth=_growth(risk=0.0))

    assert capacity["quarantine_released_slots"] == 0
    _assert_within_envelope(capacity)


def test_release_cannot_bypass_zero_deployable_capital(tmp_path):
    """7: free USDT alone cannot release a slot when capital room is zero."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane, growth=_growth(remaining=0.0))

    assert capacity["reconciled_free_quote_usd"] == pytest.approx(10_000.0)
    assert capacity["quarantine_released_slots"] == 0
    _assert_within_envelope(capacity)


def test_release_cannot_bypass_blocked_new_entries(tmp_path):
    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane, growth=_growth(allowed=False))

    assert capacity["quarantine_released_slots"] == 0
    _assert_within_envelope(capacity)


def test_release_cannot_exceed_daily_entry_room(tmp_path):
    """1: an exhausted daily entry allowance blocks any release."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane, entries_today=10_000)

    assert capacity["quarantine_released_slots"] == 0
    _assert_within_envelope(capacity)


def test_release_is_bounded_by_executor_and_notional_room(tmp_path):
    """2/3/4: every original ceiling is reapplied to healthy occupancy.

    The release ceiling is derived by re-running the original capacity function
    against healthy occupancy, so executor order room, daily submitted-notional
    room and maximum adaptive positions are all enforced by that original code
    rather than re-implemented here.
    """

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane)

    ceiling = capacity["quarantine_release_ceiling"]
    assert capacity["quarantine_released_slots"] <= ceiling
    assert capacity["available_slots"] <= capacity[
        "healthy_occupancy_available_slots"
    ]
    assert capacity["available_slots"] <= capacity["target_positions"] + ceiling
    _assert_within_envelope(capacity)


def test_release_is_bounded_by_real_free_quote(tmp_path):
    """No release when reconciled free quote cannot fund a ticket."""

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=0.0)

    capacity = _capacity(lane)

    assert capacity["reconciled_free_quote_usd"] == pytest.approx(0.0)
    assert capacity["quarantine_released_slots"] == 0
    _assert_within_envelope(capacity)


def test_release_is_zero_when_occupancy_is_not_the_binding_constraint(tmp_path):
    """Quarantine only helps when routing occupancy is what limits capacity.

    available_slots is target - existing, and target is existing +
    risk_adjusted_slots until the adaptive position ceiling binds. So with only
    a few positions held, removing impaired inventory from occupancy changes
    nothing - capacity is limited by risk/capital/candidates instead. Releasing
    a slot here would be exactly the envelope escape this fix removes.
    """

    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=500.0)

    capacity = _capacity(lane, growth=_growth(remaining=200.0, risk=1.0))

    assert capacity["impaired_inventory_count"] == 1
    assert capacity["quarantine_release_ceiling"] == 0
    assert capacity["quarantine_released_slots"] == 0
    _assert_within_envelope(capacity)


def test_releases_one_healthy_slot_when_occupancy_ceiling_binds(tmp_path):
    """8: at the adaptive position ceiling, impaired inventory frees a slot."""

    lane, _service, instance, _fake = _impaired_lane(tmp_path, free_usdt=500.0)

    # Fill the executor to the adaptive position ceiling. One of those holdings
    # (CSPR) is impaired, so healthy occupancy is one below the ceiling.
    ceiling = lane.maximum_adaptive_positions
    with instance._io_lock:
        for index in range(ceiling):
            symbol = f"F{index}/USDT"
            if symbol in instance.state["positions"]:
                continue
            instance.state["positions"][symbol] = 1.0
            instance.state["position_cost_usd"][symbol] = 1.0
            instance.state["account_balance"]["free"][f"F{index}"] = 1.0
            if len(instance.state["positions"]) >= ceiling:
                break
        instance._save_state()

    capacity = _capacity(
        lane,
        growth=_growth(remaining=200.0, risk=1.0),
        candidate_count=20,
    )

    assert capacity["impaired_inventory_count"] == 1
    assert capacity["original_available_slots"] == 0
    assert capacity["quarantine_release_ceiling"] >= 1
    assert capacity["quarantine_released_slots"] == 1
    assert capacity["available_slots"] == 1
    _assert_within_envelope(capacity)


def test_release_never_exceeds_impaired_inventory_count(tmp_path):
    lane, _service, _instance, _fake = _impaired_lane(tmp_path, free_usdt=10_000.0)

    capacity = _capacity(lane, growth=_growth(remaining=5_000.0), candidate_count=50)

    assert capacity["quarantine_released_slots"] <= capacity[
        "impaired_inventory_count"
    ]
    _assert_within_envelope(capacity)
