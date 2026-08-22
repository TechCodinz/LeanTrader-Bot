from __future__ import annotations

import json
import time

import pytest

from leantrader.production.evidence_qualification import (
    EvidenceQualificationEngine,
    deflated_performance_evidence,
    drift_and_edge_decay,
    probability_of_backtest_overfitting,
    purged_walk_forward_validation,
)
from leantrader.production.unified_control_plane import UnifiedDecisionControlPlane


def observation(start: float, end: float, label_end: float) -> dict:
    return {"feature_start": start, "feature_end": end, "label_end": label_end}


def outcome(index: int, value: float = 0.01, *, recorded_at: float | None = None) -> dict:
    return {
        "recorded_at": float(index) if recorded_at is None else float(recorded_at),
        "strategy": "adaptive_ensemble",
        "symbol": "BTC/USDT",
        "regime": "trend" if index % 2 else "range",
        "net_return": value,
        "evidence_authority": "costed_shadow_episode_v2",
    }


def experiment(
    candidate_id: str,
    outcomes: list[dict],
    *,
    selection: dict | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "base_strategy": "adaptive_ensemble",
        "manifest_sha256": f"manifest-{candidate_id}",
        "protocol": {
            "prospective_only": True,
            "round_trip_cost_bps": 30.0,
            "automatic_promotion": False,
        },
        "selection_evidence": selection or {},
        "outcomes": outcomes,
    }


def test_purged_walk_forward_requires_chronology_purge_and_embargo():
    good = purged_walk_forward_validation(
        [
            {
                "training": [observation(0, 4, 5), observation(5, 9, 10)],
                "validation": [observation(12, 14, 15), observation(15, 17, 18)],
            },
            {
                "training": [observation(0, 14, 15), observation(15, 19, 20)],
                "validation": [observation(22, 24, 25), observation(25, 27, 28)],
            },
        ],
        embargo_seconds=2.0,
    )
    assert good["purged_walk_forward_passed"] is True
    assert good["embargo_applied"] is True

    leakage = purged_walk_forward_validation(
        [{"training": [observation(0, 9, 13)], "validation": [observation(12, 14, 15)]}],
        embargo_seconds=1.0,
    )
    assert leakage["passed"] is False
    assert "fold_0_training_validation_leakage" in leakage["reasons"]

    no_embargo = purged_walk_forward_validation(
        [{"training": [observation(0, 8, 10)], "validation": [observation(11, 13, 14)]}],
        embargo_seconds=2.0,
    )
    assert no_embargo["passed"] is False
    assert "fold_0_embargo_violation" in no_embargo["reasons"]


def test_pbo_flags_selection_instability_and_fails_closed_when_unavailable():
    # Four alternatives each dominate a different contiguous quarter. The
    # strategy selected in-sample therefore ranks poorly in its complement.
    matrix = {}
    for strategy in range(4):
        values = []
        for segment in range(4):
            values.extend([0.10, 0.10] if segment == strategy else [-0.02, -0.02])
        matrix[str(strategy)] = values

    result = probability_of_backtest_overfitting(matrix, segments=4)
    assert result["valid"] is True
    assert result["pbo"] == pytest.approx(1.0)
    assert result["splits"] > 0

    missing = probability_of_backtest_overfitting({"only": [0.01] * 20})
    assert missing["valid"] is False
    assert missing["pbo"] == 1.0


def test_deflated_performance_penalizes_multiple_testing():
    returns = [0.010, 0.012, 0.009, 0.011, 0.013, 0.008] * 30
    few = deflated_performance_evidence(returns, number_of_trials=2)
    many = deflated_performance_evidence(returns, number_of_trials=500)

    assert few["valid"] is True
    assert many["valid"] is True
    assert many["deflated_performance_statistic"] < few["deflated_performance_statistic"]
    assert many["multiple_testing_adjusted_p_value"] >= few["multiple_testing_adjusted_p_value"]


def test_drift_and_edge_decay_require_stable_multi_regime_edge():
    stable = [
        {
            "recorded_at": float(index),
            "net_return": 0.010 if index % 2 else 0.008,
            "regime": "trend" if index % 3 else "range",
        }
        for index in range(120)
    ]
    stable_result = drift_and_edge_decay(stable)
    assert stable_result["valid"] is True
    assert stable_result["drift_stable"] is True
    assert stable_result["regimes"] == 2

    decayed = [
        {
            "recorded_at": float(index),
            "net_return": 0.010 if index < 60 else -0.010,
            "regime": "trend" if index % 2 else "range",
        }
        for index in range(120)
    ]
    decay_result = drift_and_edge_decay(decayed)
    assert decay_result["valid"] is True
    assert decay_result["drift_stable"] is False
    assert decay_result["recent_mean"] < 0.0


def test_holdout_freezes_existing_prefix_before_future_outcomes(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    engine = EvidenceQualificationEngine(tmp_path / "qualification.json")
    state = {
        "experiments": {
            "candidate": experiment(
                "candidate",
                [outcome(i, recorded_at=900.0 + i / 10.0) for i in range(100)],
            )
        }
    }

    first = engine.qualify(state)
    freeze = first["partitions"]["untouched_holdout"]["freeze"]
    assert freeze["frozen_outcome_count"] == 100
    assert first["partitions"]["untouched_holdout"]["samples"] == 0
    assert first["untouched_holdout_passed"] is False

    clock["now"] = 1_200.0
    state["experiments"]["candidate"]["outcomes"].extend(
        [outcome(100 + i, recorded_at=1_001.0 + i) for i in range(100)]
    )
    second = engine.qualify(state)
    assert second["partitions"]["prospective_paper"]["samples"] == 100
    assert second["partitions"]["untouched_holdout"]["samples"] == 100
    assert second["untouched_holdout_passed"] is True
    assert second["automatic_promotion"] is False
    assert second["testnet_authority"] is False
    assert second["live_authority"] is False


def test_holdout_prefix_and_manifest_are_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    base = {"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(100)])}}

    prefix_engine = EvidenceQualificationEngine(tmp_path / "prefix.json")
    prefix_engine.qualify(base)
    base["experiments"]["candidate"]["outcomes"][0]["net_return"] = 99.0
    prefix = prefix_engine.qualify(base)
    assert any("prefreeze_outcomes_mutated" in reason for reason in prefix["qualification"]["reasons"])
    assert prefix["untouched_holdout_passed"] is False

    manifest_state = {"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(100)])}}
    manifest_engine = EvidenceQualificationEngine(tmp_path / "manifest.json")
    manifest_engine.qualify(manifest_state)
    manifest_state["experiments"]["candidate"]["protocol"]["round_trip_cost_bps"] = 31.0
    mutated = manifest_engine.qualify(manifest_state)
    assert any("manifest_changed_after_freeze" in reason for reason in mutated["qualification"]["reasons"])
    assert mutated["untouched_holdout_passed"] is False


def test_partitions_persistence_hash_and_lineage_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    selection_a = {
        "purged_walk_forward_folds": [
            {
                "training": [observation(0, 5, 6), observation(6, 10, 11)],
                "validation": [observation(13, 15, 16)],
            }
        ],
        "aligned_validation_returns": [0.05, 0.05, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
    }
    selection_b = {
        **selection_a,
        "aligned_validation_returns": [-0.01, -0.01, 0.05, 0.05, -0.01, -0.01, -0.01, -0.01],
    }
    path = tmp_path / "qualification.json"
    engine = EvidenceQualificationEngine(path, embargo_seconds=2.0)
    state = {
        "experiments": {
            "a": experiment("a", [outcome(i, 0.010 + (i % 3) * 0.001) for i in range(120)], selection=selection_a),
            "b": experiment("b", [outcome(i, 0.008 + (i % 3) * 0.001) for i in range(120)], selection=selection_b),
        }
    }
    result = engine.qualify(state)

    assert set(result["partitions"]) == {"training", "validation", "prospective_paper", "untouched_holdout"}
    assert result["purged_walk_forward_passed"] is True
    assert result["embargo_applied"] is True
    assert result["evidence_reproducibility_hash"]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["history"][-1]["payload"]["evidence_reproducibility_hash"] == result["evidence_reproducibility_hash"]

    persisted["history"][0]["payload"]["independent_samples"] = 999_999
    path.write_text(json.dumps(persisted), encoding="utf-8")
    restarted = EvidenceQualificationEngine(path)
    blocked = restarted.qualify(state)
    assert restarted.health()["lineage_integrity_ok"] is False
    assert blocked["probability_backtest_overfitting"] == 1.0
    assert blocked["untouched_holdout_passed"] is False
    assert blocked["automatic_promotion"] is False
    assert blocked["testnet_authority"] is False
    assert blocked["live_authority"] is False


def test_unified_control_plane_consumes_measured_contract_and_stays_fail_closed(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    control_path = tmp_path / "vps_unified_control_plane_v141.json"
    prospective_path = tmp_path / "vps_prospective_validation.json"
    selection = {
        "purged_walk_forward_folds": [
            {
                "training": [observation(0, 5, 6), observation(6, 10, 11)],
                "validation": [observation(13, 15, 16)],
            }
        ],
        "aligned_validation_returns": [0.01, -0.005, 0.012, -0.004, 0.011, -0.003, 0.010, -0.002],
    }
    prospective_path.write_text(
        json.dumps(
            {
                "experiments": {
                    "a": experiment("a", [outcome(i, 0.010 + (i % 3) * 0.001) for i in range(120)], selection=selection),
                    "b": experiment(
                        "b",
                        [outcome(i, 0.008 + (i % 3) * 0.001) for i in range(120)],
                        selection={**selection, "aligned_validation_returns": [-0.005, 0.01, -0.004, 0.012, -0.003, 0.011, -0.002, 0.010]},
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    plane = UnifiedDecisionControlPlane(control_path)
    supplied = {
        "independent_samples": 0,
        "purged_walk_forward_passed": False,
        "embargo_applied": False,
        "untouched_holdout_passed": False,
        "multiple_testing_controlled": False,
        "prospective_net_positive": False,
        "calibration_reliable": True,
        "drift_stable": False,
        "probability_backtest_overfitting": 1.0,
        "deflated_performance_statistic": -1.0,
        "partitions": {},
    }

    measured = plane._measured_validation(supplied)
    gate = plane._promotion_gate(measured)
    assert measured["independent_samples"] == 120
    assert measured["purged_walk_forward_passed"] is True
    assert measured["embargo_applied"] is True
    assert measured["evidence_reproducibility_hash"]
    assert measured["untouched_holdout_passed"] is False
    assert gate["eligible_for_human_review"] is False
    assert gate["automatic_promotion"] is False
    assert gate["testnet_authority"] is False
    assert gate["live_authority"] is False


def test_missing_evidence_and_safety_floor_relaxation_fail_closed(tmp_path):
    engine = EvidenceQualificationEngine(tmp_path / "qualification.json")
    result = engine.qualify({"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(120)])}})
    assert result["purged_walk_forward_passed"] is False
    assert result["embargo_applied"] is False
    assert result["probability_backtest_overfitting"] == 1.0
    assert result["multiple_testing_controlled"] is False
    assert result["automatic_promotion"] is False

    with pytest.raises(ValueError, match="100-sample"):
        EvidenceQualificationEngine(tmp_path / "a.json", minimum_samples=99)
    with pytest.raises(ValueError, match="two regimes"):
        EvidenceQualificationEngine(tmp_path / "b.json", minimum_regimes=1)
    with pytest.raises(ValueError, match="30-bps"):
        EvidenceQualificationEngine(tmp_path / "c.json", modeled_round_trip_cost_bps=29.99)
