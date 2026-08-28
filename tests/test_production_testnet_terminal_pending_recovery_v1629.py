from __future__ import annotations

import datetime as dt
import threading
import time

import pytest

from leantrader.agents.swarm_service import ReadOnlySwarmService
from leantrader.production.testnet_terminal_pending_recovery_v1629 import (
    below_canonical_minimum,
    canonical_executable_minimums,
    evaluate_recoverable_cycle,
)
from tests.test_production_testnet_exit_price_guard_v1611 import (
    PriceGuardBybit,
    seed,
)
from tests.test_production_testnet_exit_recycle_v1608 import (
    BalanceBybit,
    hyper_lane,
)
from tests.test_testnet_execution import (
    engine,
)


# ---------------------------------------------------------------------------
# Minimum-resolution doubles
# ---------------------------------------------------------------------------


class _Market:
    """Exchange double exposing controllable CCXT limits and lotSizeFilter."""

    def __init__(
        self,
        *,
        limit_amount_min=None,
        limit_cost_min=None,
        raw_min_qty=None,
        raw_min_amt=None,
        precision_raises=False,
    ):
        self._limit_amount_min = limit_amount_min
        self._limit_cost_min = limit_cost_min
        self._raw_min_qty = raw_min_qty
        self._raw_min_amt = raw_min_amt
        self._precision_raises = precision_raises
        self.precision_calls: list[float] = []

    def market(self, symbol):
        info = {}
        lot: dict[str, object] = {}
        if self._raw_min_qty is not None:
            lot["minOrderQty"] = self._raw_min_qty
        if self._raw_min_amt is not None:
            lot["minOrderAmt"] = self._raw_min_amt
        if lot:
            info["lotSizeFilter"] = lot
        return {
            "symbol": symbol,
            "limits": {
                "amount": {"min": self._limit_amount_min},
                "cost": {"min": self._limit_cost_min},
            },
            "info": info,
        }

    def amount_to_precision(self, symbol, amount):
        self.precision_calls.append(float(amount))
        if self._precision_raises:
            raise ValueError(
                "bybit amount of XRP/USDT must be greater than "
                "minimum amount precision of 0.0001"
            )
        return amount


# ---------------------------------------------------------------------------
# P2 - canonical Bybit executable minimums
# ---------------------------------------------------------------------------


def test_canonical_minimums_prefer_ccxt_limits():
    exchange = _Market(limit_amount_min=0.0001, limit_cost_min=1.0)
    min_amount, min_cost, resolved = canonical_executable_minimums(
        exchange,
        "XRP/USDT",
    )
    assert resolved is True
    assert min_amount == pytest.approx(0.0001)
    assert min_cost == pytest.approx(1.0)


def test_canonical_minimums_fall_back_to_lot_size_filter():
    """CCXT omits the limits; raw Bybit lotSizeFilter must supply them."""

    exchange = _Market(raw_min_qty="0.0001", raw_min_amt="1")
    min_amount, min_cost, resolved = canonical_executable_minimums(
        exchange,
        "XRP/USDT",
    )
    assert resolved is True
    assert min_amount == pytest.approx(0.0001)
    assert min_cost == pytest.approx(1.0)


def test_canonical_minimums_take_strictest_of_both_sources():
    exchange = _Market(
        limit_amount_min=0.00005,
        limit_cost_min=0.5,
        raw_min_qty="0.0001",
        raw_min_amt="1",
    )
    min_amount, min_cost, _resolved = canonical_executable_minimums(
        exchange,
        "XRP/USDT",
    )
    assert min_amount == pytest.approx(0.0001)
    assert min_cost == pytest.approx(1.0)


def test_canonical_minimums_fail_closed_when_unprovable():
    exchange = _Market()
    min_amount, min_cost, resolved = canonical_executable_minimums(
        exchange,
        "XRP/USDT",
    )
    assert resolved is False
    assert min_amount == 0.0
    assert min_cost == 0.0

    assessment = below_canonical_minimum(exchange, "XRP/USDT", 0.00009, 1.41)
    assert assessment["resolved"] is False
    assert assessment["below_minimum"] is False
    assert assessment["live_authority"] is False


def test_raw_residual_below_min_amount_skips_precision_conversion():
    """XRP residual under 0.0001 must not reach amount_to_precision.

    Bybit raises InvalidOrder there, which is the v1.60.28 fast-lane last_error.
    """

    exchange = _Market(
        limit_amount_min=0.0001,
        limit_cost_min=1.0,
        precision_raises=True,
    )

    assessment = below_canonical_minimum(
        exchange,
        "XRP/USDT",
        0.00008899999999989472,
        1.44897,
    )

    assert assessment["below_minimum"] is True
    assert assessment["below_minimum_amount"] is True
    assert assessment["precision_conversion_skipped"] is True
    assert exchange.precision_calls == []


def test_price_limited_but_executable_position_is_not_dust():
    """CSPR/JASMY-style live state passes its minimums and is never dust."""

    exchange = _Market(limit_amount_min=1.0, limit_cost_min=1.0)

    cspr = below_canonical_minimum(exchange, "CSPR/USDT", 750.9483, 0.00165)
    assert cspr["resolved"] is True
    assert cspr["below_minimum"] is False
    assert cspr["estimated_value_usd"] == pytest.approx(1.239, rel=1e-3)

    jasmy_exchange = _Market(limit_amount_min=0.01, limit_cost_min=0.0001)
    jasmy = below_canonical_minimum(
        jasmy_exchange,
        "JASMY/USDT",
        0.10989,
        0.0039,
    )
    assert jasmy["resolved"] is True
    assert jasmy["below_minimum"] is False


# ---------------------------------------------------------------------------
# P0 - terminal pending-event latch
# ---------------------------------------------------------------------------


def _xrp_pending_runtime(tmp_path, *, filled=0.6894, status="closed", remaining=0.0):
    """Reproduce the persisted XRP terminal pending-event latch."""

    fake = PriceGuardBybit()
    fake.bid = 1.44897
    fake.ask = 1.4492
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

    event = {
        "timestamp": "2026-08-27T19:59:17+00:00",
        "symbol": "XRP/USDT",
        "side": "sell",
        "price": 1.44897,
        "quantity": 0.6894,
        "reason": "fast_exit",
        "event_id": "fast57-1787877514637-7112-sell",
    }

    client_id = instance._client_order_id(event)

    # The authoritative balance is what reconciliation trusts.
    fake.balance_total["XRP"] = remaining
    fake.balance_free["XRP"] = remaining

    with instance._io_lock:
        instance.state["orders"][client_id] = {
            "client_order_id": client_id,
            "order_id": "2291277220033069824",
            "symbol": "XRP/USDT",
            "side": "sell",
            "status": status,
            "filled": filled,
            "filled_cost": 0.998919918,
            "average": 1.44897,
            "submitted_at": "2026-08-27T20:00:24+00:00",
        }
        if remaining > 0.0:
            instance.state["positions"]["XRP/USDT"] = remaining
            instance.state["position_cost_usd"]["XRP/USDT"] = remaining * 1.44897
        else:
            instance.state["positions"].pop("XRP/USDT", None)
            instance.state["position_cost_usd"].pop("XRP/USDT", None)
        instance._save_state()

    with lane._lock:
        lane.state["active"]["XRP/USDT"] = {
            "symbol": "XRP/USDT",
            "quantity": 0.6894,
            "initial_quantity": 0.6894,
            "entry_price": 1.410629,
            "entry_notional_usd": 0.9998538352,
            "peak_price": 1.44897,
            "entered_at": 900.0,
            "target_hold_seconds": 30.0,
        }
        lane.state["pending_event"] = {
            "kind": "exit",
            "event": dict(event),
            "assessment": {"exit_reason": "fast_exit"},
        }
        lane._save_locked()

    return lane, service, instance, fake, client_id, event


def test_terminal_closed_pending_sell_is_not_resubmitted(tmp_path):
    lane, _service, instance, fake, client_id, _event = _xrp_pending_runtime(
        tmp_path
    )
    before_created = len(fake.created)

    result = lane._submit_pending(lane.state["pending_event"], now=2_000.0)

    assert result["reason"] == "terminal_pending_reconciled"
    assert result["details"]["order_submitted"] is False
    assert result["details"]["resubmission_suppressed"] is True
    assert result["details"]["client_order_id"] == client_id
    assert result["details"]["fabricated_close"] is False
    assert result["details"]["live_authority"] is False

    # No new exchange order of any kind.
    assert len(fake.created) == before_created


def test_stale_pending_latch_is_cleared(tmp_path):
    lane, _service, _instance, _fake, _client_id, _event = _xrp_pending_runtime(
        tmp_path
    )

    lane._submit_pending(lane.state["pending_event"], now=2_000.0)

    assert lane.state["pending_event"] is None
    assert lane.state["last_error"] is None
    assert lane.state["last_action"]["action"] == "terminal_pending_reconciled"
    assert lane.state["last_action"]["pending_latch_cleared"] is True


def test_authoritative_remaining_quantity_is_retained(tmp_path):
    """A terminal partial fill keeps the executor's remaining quantity."""

    # Remaining is comfortably above the market minimums, so it must be kept
    # as a live position rather than reclassified as dust.
    lane, _service, _instance, fake, _client_id, _event = _xrp_pending_runtime(
        tmp_path,
        filled=0.4,
        remaining=5.0,
    )
    before_created = len(fake.created)

    result = lane._submit_pending(lane.state["pending_event"], now=2_000.0)

    assert result["details"]["residual_recorded_as_dust"] is False
    assert result["details"]["authoritative_remaining_quantity"] == pytest.approx(
        5.0
    )
    assert lane.state["active"]["XRP/USDT"]["quantity"] == pytest.approx(5.0)
    assert lane.state["pending_event"] is None
    assert len(fake.created) == before_created


def test_subminimum_remainder_becomes_dust_without_exchange_order(tmp_path):
    lane, _service, instance, fake, _client_id, _event = _xrp_pending_runtime(
        tmp_path,
        filled=0.6894,
        remaining=0.00008899999999989472,
    )
    before_created = len(fake.created)

    result = lane._submit_pending(lane.state["pending_event"], now=2_000.0)

    assert result["details"]["residual_recorded_as_dust"] is True
    assert result["details"]["authoritative_remaining_quantity"] == 0.0
    assert result["details"]["order_submitted"] is False

    dust = instance.state["non_tradeable_dust"]["XRP/USDT"]
    assert dust["quantity"] == pytest.approx(0.00008899999999989472)
    assert dust["counted_as_executed_close"] is False

    # Slot released, and never via an exchange order.
    assert "XRP/USDT" not in lane.state["active"]
    assert len(fake.created) == before_created


@pytest.mark.parametrize("status", ["open", "submitting"])
def test_unresolved_order_is_fail_closed_and_keeps_latch(tmp_path, status):
    lane, _service, _instance, fake, _client_id, _event = _xrp_pending_runtime(
        tmp_path,
        status=status,
        filled=0.0,
    )
    before_created = len(fake.created)

    result = lane._submit_pending(lane.state["pending_event"], now=2_000.0)

    assert result["reason"] == "terminal_pending_unresolved_order_fail_closed"
    assert result["details"]["order_submitted"] is False
    assert result["details"]["pending_latch_cleared"] is False
    assert result["details"]["position_remains_active"] is True

    # Latch retained, position retained, no order.
    assert lane.state["pending_event"] is not None
    assert "XRP/USDT" in lane.state["active"]
    assert len(fake.created) == before_created


# ---------------------------------------------------------------------------
# P1 - restart-safe completed-cycle recovery
# ---------------------------------------------------------------------------


def _dust_cycle_engine(
    tmp_path,
    *,
    realized=-0.001,
    buy_filled=0.100089,
    buy_cost=10.0,
    sell_filled=0.1,
    sell_cost=None,
):
    """Real closed buy+sell plus recorded dust, with cycle PnL already popped.

    The dust row is written exactly as ``_record_non_tradeable_dust`` leaves it,
    including the destroyed ``position_cycle_pnl_usd`` key that v1.60.27 relied
    on. Residual quantity and cost basis are derived from the orders so the
    strict match conditions hold.
    """

    fake = BalanceBybit()
    instance, _ = engine(tmp_path, fake)
    instance.start()

    if sell_cost is None:
        sell_cost = buy_cost + realized

    dust_quantity = buy_filled - sell_filled
    dust_cost_basis = buy_cost * (dust_quantity / buy_filled)

    with instance._io_lock:
        instance.state["orders"]["cycle-buy"] = {
            "client_order_id": "cycle-buy",
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "closed",
            "submitted_at": "2026-08-27T10:00:00+00:00",
            "filled": buy_filled,
            "filled_cost": buy_cost,
            "average": buy_cost / buy_filled,
            "fee": 0.0,
            "fee_currency": "USDT",
        }
        instance.state["orders"]["cycle-sell"] = {
            "client_order_id": "cycle-sell",
            "symbol": "BTC/USDT",
            "side": "sell",
            "status": "closed",
            "submitted_at": "2026-08-27T10:00:10+00:00",
            "filled": sell_filled,
            "filled_cost": sell_cost,
            "average": sell_cost / sell_filled,
            "fee": 0.0,
            "fee_currency": "USDT",
        }
        instance.state["non_tradeable_dust"]["BTC/USDT"] = {
            "symbol": "BTC/USDT",
            "quantity": dust_quantity,
            "cost_basis_usd": dust_cost_basis,
            "recorded_at": "2026-08-27T10:00:11+00:00",
            "counted_as_executed_close": False,
            "live_authority": False,
        }
        instance.state["position_cycle_pnl_usd"].pop("BTC/USDT", None)
        instance.state["positions"].pop("BTC/USDT", None)
        instance._save_state()

    return instance, fake


def test_restart_safe_cycle_recovery_is_idempotent(tmp_path):
    instance, fake = _dust_cycle_engine(tmp_path)
    before_created = len(fake.created)
    global_pnl_before = instance.state["realized_pnl_usd"]

    first = instance.recover_uncounted_dust_cycles()
    assert first["recovered"] == 1
    assert instance.state["closed_positions"] == 1

    second = instance.recover_uncounted_dust_cycles()
    assert second["recovered"] == 0
    assert instance.state["closed_positions"] == 1

    dust = instance.state["non_tradeable_dust"]["BTC/USDT"]
    assert dust["counted_as_executed_close"] is True

    # Global exchange-realized PnL is never touched by dust accounting.
    assert instance.state["realized_pnl_usd"] == global_pnl_before
    assert first["global_realized_pnl_mutated"] is False
    assert len(fake.created) == before_created


def test_net_after_dust_determines_win_or_loss(tmp_path):
    """A positive realized sell PnL still books a loss once dust is netted.

    Sold portion profits by +0.005, but 0.01 base (cost basis 1.0) is stranded
    and unrecoverable, so the true outcome is a loss.
    """

    instance, _fake = _dust_cycle_engine(
        tmp_path,
        buy_filled=0.11,
        buy_cost=11.0,
        sell_filled=0.1,
        sell_cost=10.005,
    )

    dust_cost = instance.state["non_tradeable_dust"]["BTC/USDT"]["cost_basis_usd"]
    assert dust_cost == pytest.approx(1.0)

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 1
    assert instance.state["closed_positions"] == 1
    assert instance.state.get("winning_positions", 0) == 0

    cycle = outcome["cycles"][0]
    assert cycle["realized_sell_pnl_usd"] > 0.0
    assert cycle["is_win"] is False
    assert cycle["net_realized_after_dust_usd"] == pytest.approx(
        cycle["realized_sell_pnl_usd"] - dust_cost
    )
    assert cycle["net_realized_after_dust_usd"] < 0.0


def test_recovery_counts_a_genuine_win(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path, realized=+0.5)

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 1
    assert instance.state["closed_positions"] == 1
    assert instance.state["winning_positions"] == 1
    assert outcome["cycles"][0]["is_win"] is True


def test_no_close_without_a_real_filled_sell(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)

    with instance._io_lock:
        instance.state["orders"].pop("cycle-sell")
        instance._save_state()

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 0
    assert instance.state.get("closed_positions", 0) == 0
    assert outcome["rejections"][0]["reason"] in {
        "no_real_filled_sell",
        "missing_dust_or_evidence",
        "no_executed_sell_quantity",
    }


def test_later_buy_invalidates_an_older_sell_cycle(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)

    with instance._io_lock:
        instance.state["orders"]["later-buy"] = {
            "client_order_id": "later-buy",
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "closed",
            "submitted_at": "2026-08-27T10:05:00+00:00",
            "filled": 0.05,
            "filled_cost": 5.0,
            "average": 100.0,
        }
        instance._save_state()

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 0
    assert instance.state.get("closed_positions", 0) == 0


def test_dust_quantity_mismatch_fails_closed(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)

    with instance._io_lock:
        instance.state["non_tradeable_dust"]["BTC/USDT"]["quantity"] = 0.05
        instance._save_state()

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 0
    assert instance.state.get("closed_positions", 0) == 0
    assert outcome["rejections"][0]["reason"] == "residual_quantity_mismatch"


def test_dust_cost_basis_mismatch_fails_closed(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)

    with instance._io_lock:
        instance.state["non_tradeable_dust"]["BTC/USDT"]["cost_basis_usd"] = 9.0
        instance._save_state()

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 0
    assert outcome["rejections"][0]["reason"] == "residual_cost_basis_mismatch"


def test_dust_recorded_before_the_sell_fails_closed(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)

    with instance._io_lock:
        instance.state["non_tradeable_dust"]["BTC/USDT"]["recorded_at"] = (
            "2026-08-27T09:59:00+00:00"
        )
        instance._save_state()

    outcome = instance.recover_uncounted_dust_cycles()

    assert outcome["recovered"] == 0
    assert outcome["rejections"][0]["reason"] == "dust_recorded_before_sell"


def test_startup_performs_bounded_recovery(tmp_path):
    """A restart must recover a previously uncounted cycle automatically."""

    # Seed persisted state exactly as a prior run would have left it: a real
    # closed buy/sell pair, recorded dust, and no counted cycle.
    seeded, fake = _dust_cycle_engine(tmp_path)
    assert seeded.state.get("closed_positions", 0) == 0

    # A fresh engine over the same persisted state file is the restart.
    restarted, _ = engine(tmp_path, fake)
    restarted.start()

    recovery = restarted.state["v1629_startup_recovery"]
    assert recovery["ok"] is True
    assert recovery["recovered"] == 1
    assert recovery["bounded_limit"] == 50
    assert recovery["global_realized_pnl_mutated"] is False
    assert restarted.state["closed_positions"] == 1

    # A second start must not double count.
    restarted_again, _ = engine(tmp_path, fake)
    restarted_again.start()
    assert restarted_again.state["closed_positions"] == 1
    assert restarted_again.state["v1629_startup_recovery"]["recovered"] == 0


def test_evaluate_rejects_already_counted_cycle():
    assessment = evaluate_recoverable_cycle(
        exchange=_Market(limit_amount_min=0.0001, limit_cost_min=1.0),
        state={},
        symbol="XRP/USDT",
        dust={"counted_as_executed_close": True},
        evidence={"cycle_key": "abc"},
    )
    assert assessment["eligible"] is False
    assert assessment["reason"] == "already_counted_as_executed_close"


# ---------------------------------------------------------------------------
# P4 - execution candidate wake
# ---------------------------------------------------------------------------


class _PinService:
    """Minimal stand-in exercising the v1.60.29 pin wake contract."""

    def __init__(self):
        self._stop = threading.Event()
        self._execution_pin_event = threading.Event()
        self._lock = threading.RLock()
        self._execution_candidate_pins: dict[str, float] = {}
        self.microstream_pin_wakeups = 0
        # Thread handles stop() walks; none are started in this double.
        self.cadence_seconds = 1.0
        self._thread = None
        self._precision_scout_thread = None
        self._microstream_watchdog_thread = None
        self._microstream_thread = None
        self._calibration_thread = None

    @staticmethod
    def _unique_symbols(symbols):
        seen: list[str] = []
        for symbol in symbols:
            normalized = str(symbol).upper()
            if normalized not in seen:
                seen.append(normalized)
        return seen


def test_pin_wakes_microstream_before_cadence_expiry():
    """A pin must release the cadence sleep well before it would expire."""

    service = _PinService()
    ReadOnlySwarmService.pin_execution_candidate_symbols(service, ["XRP/USDT"])

    assert service._execution_pin_event.is_set() is True
    assert "XRP/USDT" in service._execution_candidate_pins

    # The loop parks on exactly this wait; a 5s cadence must not be waited out.
    cadence_seconds = 5.0
    started = time.monotonic()
    woke = service._execution_pin_event.wait(cadence_seconds)
    waited = time.monotonic() - started

    assert woke is True
    assert waited < 1.0, "pin did not release the cadence sleep"


def test_pin_event_does_not_change_freshness_or_priority_ordering():
    """The wake is scheduling only: TTL floor and ordering are untouched."""

    service = _PinService()
    ReadOnlySwarmService.pin_execution_candidate_symbols(
        service,
        ["AAA/USDT", "BBB/USDT"],
        ttl_seconds=0.1,
    )

    expiries = list(service._execution_candidate_pins.values())
    # The 2.0s TTL floor is preserved; the wake does not shorten freshness.
    assert all(expiry - time.time() >= 1.5 for expiry in expiries)

    # v1.60.28 reinsertion ordering is preserved (newest request last).
    assert list(service._execution_candidate_pins) == ["AAA/USDT", "BBB/USDT"]


def test_pin_with_no_symbols_does_not_wake():
    service = _PinService()
    ReadOnlySwarmService.pin_execution_candidate_symbols(service, [])
    assert service._execution_pin_event.is_set() is False


def test_stop_releases_a_pin_parked_sleep():
    """Shutdown stays as responsive as the previous self._stop.wait()."""

    service = _PinService()
    assert service._execution_pin_event.is_set() is False

    ReadOnlySwarmService.stop(service)

    assert service._stop.is_set() is True
    assert service._execution_pin_event.wait(0.5) is True


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_health_payloads_assert_no_live_authority(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)
    payload = instance.health()

    assert payload["live_authority"] is False
    recovery = payload["terminal_pending_cycle_recovery"]
    assert recovery["version"] == "1.60.29"
    assert recovery["fake_close_allowed"] is False
    assert recovery["global_realized_pnl_mutated"] is False
    assert recovery["win_loss_is_net_of_residual_dust"] is True
    assert recovery["live_authority"] is False


def test_lane_health_reports_terminal_pending_latch(tmp_path):
    lane, _service, _instance, _fake, _client_id, _event = _xrp_pending_runtime(
        tmp_path
    )
    lane._submit_pending(lane.state["pending_event"], now=2_000.0)

    payload = lane.health()
    latch = payload["terminal_pending_latch"]

    assert latch["version"] == "1.60.29"
    assert latch["terminal_closed_pending_event_resubmitted"] is False
    assert latch["unresolved_orders_fail_closed"] is True
    assert latch["deterministic_order_link_id_idempotency"] is True
    assert latch["reconciliations"] >= 1
    assert payload["live_authority"] is False


def test_recovery_never_mutates_global_realized_pnl(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path, realized=+0.5)
    before = instance.state["realized_pnl_usd"]

    instance.recover_uncounted_dust_cycles()

    assert instance.state["realized_pnl_usd"] == before


def test_dust_recorded_at_iso_parsing_is_timezone_safe(tmp_path):
    instance, _fake = _dust_cycle_engine(tmp_path)

    with instance._io_lock:
        naive = dt.datetime(2026, 8, 27, 10, 0, 11).isoformat()
        instance.state["non_tradeable_dust"]["BTC/USDT"]["recorded_at"] = naive
        instance._save_state()

    outcome = instance.recover_uncounted_dust_cycles()
    assert outcome["recovered"] == 1
