from __future__ import annotations

import json
from copy import deepcopy

import pytest

from leantrader.production.unified_control_plane import (
    PaperOrderSimulator,
    UnifiedDecisionControlPlane,
    specialist_correlation_group,
)


def valid_inputs() -> dict:
    return {
        "symbol": "BTC/USDT",
        "current_regime": "trend",
        "correlation_bucket": "crypto_beta",
        "specialist_evidence": [
            {
                "name": "adaptive_trend",
                "correlation_group": "price_trend",
                "timeframe": "15m",
                "regime": "trend",
                "direction": 0.8,
                "confidence": 0.90,
                "calibration_error": 0.10,
                "expected_gross_edge_bps": 85.0,
            },
            {
                "name": "orderbook_imbalance",
                "correlation_group": "microstructure",
                "timeframe": "5m",
                "regime": "trend",
                "direction": 0.7,
                "confidence": 0.85,
                "calibration_error": 0.10,
                "expected_gross_edge_bps": 75.0,
            },
        ],
        "market_data": {
            "age_seconds": 2.0,
            "max_age_seconds": 30.0,
            "quality_score": 0.99,
            "future_leakage_detected": False,
            "survivorship_bias_controlled": True,
            "exchange_anomaly_detected": False,
            "freshness_verified": True,
        },
        "execution": {
            "order_notional_usd": 20.0,
            "liquidity_usd": 1_000_000.0,
            "max_participation_rate": 0.01,
            "rejection_probability": 0.0,
            "spread_bps": 2.0,
            "fee_bps_per_side": 5.0,
            "base_slippage_bps_per_side": 2.0,
            "funding_bps": 0.0,
            "latency_ms": 100.0,
            "volatility_bps_per_second": 1.0,
            "adverse_selection_bps": 1.0,
            "impact_coefficient_bps": 1.0,
        },
        "portfolio": {
            "equity": 1_000.0,
            "peak_equity": 1_000.0,
            "daily_pnl": 0.0,
            "loss_streak": 0,
            "volatility_ratio": 1.0,
            "positions": [],
        },
        "validation": {
            "independent_samples": 120,
            "purged_walk_forward_passed": True,
            "embargo_applied": True,
            "untouched_holdout_passed": True,
            "multiple_testing_controlled": True,
            "prospective_net_positive": True,
            "calibration_reliable": True,
            "drift_stable": True,
            "probability_backtest_overfitting": 0.10,
            "deflated_performance_statistic": 1.2,
            "partitions": {
                "training": {"samples": 500},
                "validation": {"samples": 200},
                "prospective_paper": {"samples": 120},
                "untouched_holdout": {"samples": 100},
            },
        },
        "resources": {
            "load_ratio": 0.20,
            "memory_available_fraction": 0.70,
            "runtime_healthy": True,
        },
        "system": {
            "trading_mode": "paper",
            "enable_live": False,
            "allow_live": False,
            "live_confirm": "NO",
            "testnet_enabled": False,
            "runtime_integrity_ok": True,
            "heartbeat_fresh": True,
        },
        "atr_fraction": 0.01,
    }


def plane(tmp_path, **overrides) -> UnifiedDecisionControlPlane:
    return UnifiedDecisionControlPlane(
        tmp_path / "control-plane.json",
        **overrides,
    )


def test_valid_recommendation_is_still_shadow_only(tmp_path):
    result = plane(tmp_path).evaluate(**valid_inputs())

    assert result["allowed_shadow_recommendation"] is True
    assert result["expected_net_edge_bps"] > 0.0
    assert result["ensemble"]["independent_group_count"] == 2
    assert result["paper_promotion_authority"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False
    assert result["execution_authority"] is False


def test_specialist_family_mapping_is_conservative():
    assert specialist_correlation_group("fluid_liquidity") == "microstructure"
    assert specialist_correlation_group("news_awareness") == "public_context"
    assert specialist_correlation_group("onchain_flow") == "onchain_flow"
    assert specialist_correlation_group("cross_venue_arbitrage") == "cross_venue"
    assert specialist_correlation_group("adaptive_trend") == "price_technical"


def test_correlated_duplicates_do_not_gain_independent_weight(tmp_path):
    inputs = valid_inputs()
    baseline = plane(tmp_path / "a").evaluate(**inputs)

    duplicated = deepcopy(inputs)
    duplicate = deepcopy(duplicated["specialist_evidence"][0])
    duplicate["name"] = "adaptive_trend_clone"
    duplicated["specialist_evidence"].append(duplicate)
    result = plane(tmp_path / "b").evaluate(**duplicated)

    assert result["ensemble"]["raw_specialist_count"] == 3
    assert result["ensemble"]["independent_group_count"] == 2
    assert result["ensemble"]["signed_gross_edge_bps"] == pytest.approx(
        baseline["ensemble"]["signed_gross_edge_bps"]
    )
    assert result["ensemble"]["ensemble_confidence"] == pytest.approx(
        baseline["ensemble"]["ensemble_confidence"]
    )


def test_regime_mismatch_is_rejected_and_fails_independence_gate(tmp_path):
    inputs = valid_inputs()
    inputs["specialist_evidence"][1]["regime"] = "mean_reversion"

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert "insufficient_independent_evidence" in result["reasons"]
    assert result["ensemble"]["rejected"][0]["reason"] == "regime_mismatch"


def test_execution_cost_floor_cannot_be_lowered():
    with pytest.raises(ValueError, match="30-bps"):
        PaperOrderSimulator(minimum_round_trip_cost_bps=29.99)


def test_execution_model_captures_partial_fills_rejection_and_drag():
    result = PaperOrderSimulator().simulate(
        {
            "order_notional_usd": 1_000.0,
            "liquidity_usd": 10_000.0,
            "max_participation_rate": 0.01,
            "rejection_probability": 0.25,
            "spread_bps": 10.0,
            "fee_bps_per_side": 5.0,
            "base_slippage_bps_per_side": 3.0,
            "funding_bps": 2.0,
            "latency_ms": 500.0,
            "volatility_bps_per_second": 4.0,
            "adverse_selection_bps": 5.0,
            "impact_coefficient_bps": 20.0,
        }
    )

    assert result["liquidity_fill_ratio"] == pytest.approx(0.1)
    assert result["expected_fill_ratio"] == pytest.approx(0.075)
    assert result["partial_fill_expected"] is True
    assert result["costs_bps"]["round_trip_total"] >= 30.0
    assert result["costs_bps"]["market_impact_round_trip"] > 0.0
    assert result["costs_bps"]["latency"] > 0.0
    assert result["execution_authority"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("age_seconds", 31.0, "stale_market_data"),
        ("quality_score", 0.80, "market_data_quality_below_floor"),
        ("future_leakage_detected", True, "future_leakage_not_cleared"),
        ("survivorship_bias_controlled", False, "survivorship_bias_not_controlled"),
        ("exchange_anomaly_detected", True, "exchange_anomaly_not_cleared"),
        ("freshness_verified", False, "freshness_not_verified"),
    ],
)
def test_data_quality_gates_fail_closed(tmp_path, field, value, reason):
    inputs = valid_inputs()
    inputs["market_data"][field] = value

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("trading_mode", "live", "trading_mode_not_paper"),
        ("enable_live", True, "enable_live_not_false"),
        ("allow_live", True, "allow_live_not_false"),
        ("live_confirm", "YES", "live_confirm_not_no"),
        ("runtime_integrity_ok", False, "runtime_integrity_not_verified"),
        ("heartbeat_fresh", False, "heartbeat_not_fresh"),
    ],
)
def test_authority_and_system_integrity_gates(tmp_path, field, value, reason):
    inputs = valid_inputs()
    inputs["system"][field] = value

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert reason in result["reasons"]
    assert result["live_authority"] is False


def test_testnet_enabled_is_valid_sandbox_state(tmp_path):
    inputs = valid_inputs()
    inputs["system"]["testnet_enabled"] = True

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is True
    assert "testnet_execution_not_disabled" not in result["reasons"]
    assert "testnet_execution_state_invalid" not in result["reasons"]
    assert result["execution_authority"] is False
    assert result["live_authority"] is False


def test_invalid_testnet_state_fails_closed(tmp_path):
    inputs = valid_inputs()
    inputs["system"]["testnet_enabled"] = "unexpected"

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert "testnet_execution_state_invalid" in result["reasons"]
    assert result["live_authority"] is False


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"equity": 890.0}, "drawdown_circuit_breaker"),
        ({"daily_pnl": -31.0}, "daily_loss_circuit_breaker"),
        ({"loss_streak": 5}, "loss_streak_circuit_breaker"),
        ({"volatility_ratio": 3.0}, "volatility_shock_circuit_breaker"),
    ],
)
def test_portfolio_circuit_breakers(tmp_path, updates, reason):
    inputs = valid_inputs()
    inputs["portfolio"].update(updates)

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert reason in result["reasons"]


def test_correlation_concentration_blocks_new_allocation(tmp_path):
    inputs = valid_inputs()
    inputs["portfolio"]["positions"] = [
        {
            "symbol": "ETH/USDT",
            "notional_usd": 250.0,
            "correlation_bucket": "crypto_beta",
        }
    ]

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert "correlation_bucket_cap" in result["reasons"]
    assert result["portfolio_allocation"]["recommended_shadow_notional_usd"] == 0.0


def test_promotion_requires_independent_evidence_without_threshold_relaxation(tmp_path):
    inputs = valid_inputs()
    inputs["validation"]["independent_samples"] = 99
    inputs["validation"]["purged_walk_forward_passed"] = False
    inputs["validation"]["probability_backtest_overfitting"] = 0.21
    inputs["validation"]["deflated_performance_statistic"] = 0.0

    result = plane(tmp_path).evaluate(**inputs)
    gate = result["promotion_gate"]

    assert gate["eligible_for_human_review"] is False
    assert gate["automatic_promotion"] is False
    assert gate["minimum_independent_samples"] == 100
    assert "insufficient_independent_samples" in gate["reasons"]
    assert "purged_walk_forward_not_passed" in gate["reasons"]
    assert "backtest_overfitting_probability_too_high" in gate["reasons"]
    assert "deflated_performance_not_positive" in gate["reasons"]


def test_strategy_decay_demotes_to_shadow(tmp_path):
    inputs = valid_inputs()
    inputs["validation"]["drift_stable"] = False
    inputs["validation"]["prospective_net_positive"] = False

    result = plane(tmp_path).evaluate(**inputs)

    assert result["promotion_gate"]["demote_to_shadow"] is True
    assert "drift_or_decay_detected" in result["promotion_gate"]["reasons"]


def test_research_yields_resources_to_paper_runtime(tmp_path):
    inputs = valid_inputs()
    inputs["resources"].update(
        {
            "load_ratio": 0.90,
            "memory_available_fraction": 0.10,
        }
    )

    result = plane(tmp_path).evaluate(**inputs)
    budget = result["research_budget"]

    assert budget["research_permitted"] is False
    assert budget["bounded_hypotheses_per_cycle"] == 0
    assert "host_load_budget_exhausted" in budget["reasons"]
    assert "memory_budget_exhausted" in budget["reasons"]


def test_lifecycle_plan_has_stop_target_trailing_time_and_invalidation(tmp_path):
    result = plane(tmp_path).evaluate(**valid_inputs())
    lifecycle = result["trade_lifecycle"]

    assert lifecycle["initial_stop_distance_fraction"] == pytest.approx(0.02)
    assert lifecycle["take_profit_distance_fraction"] == pytest.approx(0.03)
    assert lifecycle["reward_to_risk"] == pytest.approx(1.5)
    assert lifecycle["trailing_activation_r"] == 1.0
    assert lifecycle["time_stop_bars"] > 0
    assert "regime_change" in lifecycle["invalidation"]


def test_lineage_is_reproducible_and_tamper_evident(tmp_path):
    state_path = tmp_path / "control-plane.json"
    first_plane = UnifiedDecisionControlPlane(state_path)
    first = first_plane.evaluate(**valid_inputs())
    repeat = first_plane.evaluate(**valid_inputs())

    assert first["experiment_id"] == repeat["experiment_id"]
    assert first_plane.health()["lineage_integrity_ok"] is True

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["history"][0]["payload"]["expected_net_edge_bps"] = 99_999.0
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = UnifiedDecisionControlPlane(state_path)
    blocked = restarted.evaluate(**valid_inputs())

    assert restarted.health()["lineage_integrity_ok"] is False
    assert blocked["allowed_shadow_recommendation"] is False
    assert "lineage_integrity_failure" in blocked["reasons"]


def test_missing_validation_partitions_prevent_promotion_review(tmp_path):
    inputs = valid_inputs()
    del inputs["validation"]["partitions"]["untouched_holdout"]

    result = plane(tmp_path).evaluate(**inputs)

    assert result["promotion_gate"]["eligible_for_human_review"] is False
    assert "missing_untouched_holdout_partition" in result["promotion_gate"]["reasons"]


def test_non_finite_or_malformed_input_fails_closed(tmp_path):
    inputs = valid_inputs()
    inputs["specialist_evidence"][0]["confidence"] = float("nan")

    result = plane(tmp_path).evaluate(**inputs)

    assert result["allowed_shadow_recommendation"] is False
    assert result["reason"] == "control_plane_input_failure"
    assert result["execution_authority"] is False
