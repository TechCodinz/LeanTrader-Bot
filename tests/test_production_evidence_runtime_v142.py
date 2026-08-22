from __future__ import annotations

import time

from leantrader.production.alpha_tournament import AlphaTournament
from leantrader.production.prospective_validation import ProspectiveValidationLab
from leantrader.production.runtime_evidence_qualification import RuntimeEvidenceQualificationEngine


def ranked_health() -> dict:
    return {
        "evidence_authority": "costed_shadow_episode_v2",
        "round_trip_cost_bps": 30.0,
        "strategies": {
            "engine:trend": {
                "samples": 140,
                "wins": 90,
                "win_rate": 90 / 140,
                "average_return": 0.004,
                "ewma_net_return": 0.003,
                "negative_streak": 0,
            },
            "engine:range": {
                "samples": 130,
                "wins": 80,
                "win_rate": 80 / 130,
                "average_return": 0.003,
                "ewma_net_return": 0.002,
                "negative_streak": 0,
            },
        },
    }


def _manual_v142_experiment(candidate: str = "v142-a") -> dict:
    plan = {
        "protocol": "v1.42_partitioned_evidence_v1",
        "initial_training_samples": 60,
        "walk_forward_folds": 3,
        "validation_samples_per_fold": 20,
        "embargo_samples_per_fold": 1,
        "prospective_paper_samples": 100,
        "untouched_holdout_samples": 100,
    }
    return {
        "candidate_id": candidate,
        "base_strategy": "engine:trend",
        "registered_at": 1_000.0,
        "manifest_sha256": f"manifest-{candidate}",
        "protocol": {
            "prospective_only": True,
            "round_trip_cost_bps": 30.0,
            "automatic_promotion": False,
            "evidence_protocol_version": "1.42",
            "purged_walk_forward_required": True,
            "embargo_required": True,
            "pbo_required": True,
            "deflated_performance_required": True,
            "drift_detection_required": True,
            "untouched_holdout_required": True,
            "freeze_before_outcome_required": True,
            "partition_plan": plan,
        },
        "selection_evidence": {
            "conservative_score": 1.0,
            "selection_frozen_before_partitioned_outcomes": True,
        },
        "outcomes": [],
    }


def test_v142_recorder_rejects_temporally_overlapping_independent_episodes(
    tmp_path,
    monkeypatch,
):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(
        "leantrader.production.prospective_validation_v141.time.time",
        lambda: clock["now"],
    )
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    tournament = AlphaTournament(tmp_path / "alpha.json")
    manifest = tournament.evaluate(strategy_health=ranked_health())["foundry_manifests"][0]
    lab = ProspectiveValidationLab(tmp_path / "prospective.json")
    registered = lab.observe_cycle(
        observatory_authority="costed_shadow_episode_v2",
        observed_round_trip_cost_bps=30.0,
        strategy_episodes=[],
        foundry_manifests=[manifest],
        market_rows={},
    )
    assert registered["experiments_registered"] == [manifest["candidate_id"]]

    strategy = manifest["base_strategy"]
    clock["now"] = 1_100.0
    result = lab.observe_cycle(
        observatory_authority="costed_shadow_episode_v2",
        observed_round_trip_cost_bps=30.0,
        strategy_episodes=[
            {
                "strategy": strategy,
                "symbol": "BTC/USDT",
                "regime": "trend",
                "net_return": 0.01,
                "opened_at": 1_010.0,
                "closed_at": 1_030.0,
                "feature_start": 1_010.0,
                "feature_end": 1_010.0,
                "label_end": 1_030.0,
                "evidence_interval_complete": True,
                "evidence_authority": "costed_shadow_episode_v2",
            },
            {
                "strategy": strategy,
                "symbol": "ETH/USDT",
                "regime": "range",
                "net_return": 0.02,
                "opened_at": 1_020.0,
                "closed_at": 1_040.0,
                "feature_start": 1_020.0,
                "feature_end": 1_020.0,
                "label_end": 1_040.0,
                "evidence_interval_complete": True,
                "evidence_authority": "costed_shadow_episode_v2",
            },
        ],
        foundry_manifests=[manifest],
        market_rows={},
    )
    experiment = lab.state["experiments"][manifest["candidate_id"]]
    assert result["episodes_recorded"] == 1
    assert len(experiment["outcomes"]) == 1
    assert experiment["partition_episode_ordinal"] == 1
    assert experiment["last_independent_label_end"] == 1_030.0
    assert lab.health()["partition_rejections"] >= 1
    assert lab.health()["independence_rule"] == "non_overlapping_label_intervals"


def test_runtime_qualifier_fails_closed_on_overlapping_preexisting_evidence(tmp_path):
    experiment = _manual_v142_experiment()
    experiment["outcomes"] = [
        {
            "episode_ordinal": 0,
            "partition": "training",
            "walk_forward_fold": None,
            "opened_at": 1_010.0,
            "closed_at": 1_030.0,
            "recorded_at": 1_030.0,
            "feature_start": 1_010.0,
            "feature_end": 1_010.0,
            "label_end": 1_030.0,
            "evidence_interval_complete": True,
            "strategy": "engine:trend",
            "symbol": "BTC/USDT",
            "regime": "trend",
            "net_return": 0.01,
            "evidence_authority": "costed_shadow_episode_v2",
        },
        {
            "episode_ordinal": 1,
            "partition": "training",
            "walk_forward_fold": None,
            "opened_at": 1_020.0,
            "closed_at": 1_040.0,
            "recorded_at": 1_040.0,
            "feature_start": 1_020.0,
            "feature_end": 1_020.0,
            "label_end": 1_040.0,
            "evidence_interval_complete": True,
            "strategy": "engine:trend",
            "symbol": "ETH/USDT",
            "regime": "range",
            "net_return": 0.01,
            "evidence_authority": "costed_shadow_episode_v2",
        },
    ]
    engine = RuntimeEvidenceQualificationEngine(tmp_path / "qualification.json")
    result = engine.qualify({"experiments": {"v142-a": experiment}})
    assert result["independent_samples"] == 0
    assert result["purged_walk_forward_passed"] is False
    assert result["untouched_holdout_passed"] is False
    assert result["probability_backtest_overfitting"] == 1.0
    assert result["multiple_testing_controlled"] is False
    assert result["automatic_promotion"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False
    assert any(
        "overlapping_independent_episode" in reason
        for reason in result["qualification"]["reasons"]
    )


def test_non_normality_deflation_penalizes_more_trials():
    values = [0.010, 0.012, 0.009, 0.011, 0.013, 0.008] * 20
    few = RuntimeEvidenceQualificationEngine._non_normality_deflated_performance(
        values,
        number_of_trials=2,
    )
    many = RuntimeEvidenceQualificationEngine._non_normality_deflated_performance(
        values,
        number_of_trials=500,
    )
    assert few["valid"] is True
    assert many["valid"] is True
    assert many["deflated_performance_statistic"] < few["deflated_performance_statistic"]
    assert many["multiple_testing_adjusted_p_value"] >= few["multiple_testing_adjusted_p_value"]
    assert "skewness" in few
    assert "kurtosis" in few


def test_enhanced_drift_tracks_rolling_regime_calibration_and_staleness(
    tmp_path,
    monkeypatch,
):
    clock = {"now": 2_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    engine = RuntimeEvidenceQualificationEngine(tmp_path / "qualification.json")
    rows = [
        {
            "recorded_at": 1_000.0 + index,
            "net_return": 0.010 if index % 2 else 0.008,
            "regime": "trend" if index % 3 else "range",
        }
        for index in range(120)
    ]
    stable = engine._enhanced_drift(
        rows,
        base_validation={"calibration_reliable": True},
    )
    assert stable["valid"] is True
    assert stable["drift_stable"] is True
    assert stable["decay_state"] == "stable"
    assert stable["rolling_expectancy"]
    assert len(stable["regime_statistics"]) == 2
    assert stable["overall"]["lower_95"] > 0.0

    calibration_failed = engine._enhanced_drift(
        rows,
        base_validation={"calibration_reliable": False},
    )
    assert calibration_failed["drift_stable"] is False
    assert calibration_failed["decay_state"] == "calibration_degraded"

    clock["now"] = 1_000.0 + engine.MAXIMUM_EVIDENCE_STALENESS_SECONDS + 10_000.0
    stale = engine._enhanced_drift(
        rows,
        base_validation={"calibration_reliable": True},
    )
    assert stale["drift_stable"] is False
    assert stale["decay_state"] == "stale"


def test_drift_observation_lineage_survives_restart_and_corruption_fails_health(
    tmp_path,
):
    path = tmp_path / "qualification.json"
    engine = RuntimeEvidenceQualificationEngine(path)
    record_hash = engine._append_drift_observation(
        {
            "candidate_id": "v142-a",
            "drift_stable": False,
            "decay_state": "insufficient_evidence",
            "samples": 12,
        }
    )
    assert record_hash
    restarted = RuntimeEvidenceQualificationEngine(path)
    assert restarted.health()["drift_lineage_integrity_ok"] is True
    assert restarted.health()["drift_observations"] == 1

    payload = restarted.state
    payload["drift_observations"][0]["payload"]["samples"] = 999_999
    restarted._save()
    corrupted = RuntimeEvidenceQualificationEngine(path)
    assert corrupted.health()["drift_lineage_integrity_ok"] is False
