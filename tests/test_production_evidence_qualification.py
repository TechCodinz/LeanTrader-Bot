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


def observation(start: float, end: float, label_end: float) -> dict:
    return {
        "feature_start": start,
        "feature_end": end,
        "label_end": label_end,
    }


def test_purged_walk_forward_accepts_real_embargo():
    result = purged_walk_forward_validation(
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

    assert result["passed"] is True
    assert result["purged_walk_forward_passed"] is True
    assert result["embargo_applied"] is True
    assert all(row["embargo_gap_seconds"] >= 2.0 for row in result["folds"])


def test_purged_walk_forward_rejects_label_leakage():
    result = purged_walk_forward_validation(
        [
            {
                "training": [observation(0, 9, 13)],
                "validation": [observation(12, 14, 15)],
            }
        ],
        embargo_seconds=1.0,
    )

    assert result["passed"] is False
    assert "fold_0_training_validation_leakage" in result["reasons"]


def test_purged_walk_forward_rejects_missing_embargo_gap():
    result = purged_walk_forward_validation(
        [
            {
                "training": [observation(0, 8, 10)],
                "validation": [observation(11, 13, 14)],
            }
        ],
        embargo_seconds=2.0,
    )

    assert result["passed"] is False
    assert "fold_0_embargo_violation" in result["reasons"]


def test_pbo_detects_strategy_selection_instability():
    # Strategy A wins the first half and loses the second; strategy B is the
    # mirror image. CSCV repeatedly selects an in-sample winner that ranks
    # poorly out of sample.
    matrix = {
        "a": [0.08, 0.08, 0.08, 0.08, -0.08, -0.08, -0.08, -0.08],
        "b": [-0.08, -0.08, -0.08, -0.08, 0.08, 0.08, 0.08, 0.08],
    }

    result = probability_of_backtest_overfitting(matrix, segments=4)

    assert result["valid"] is True
    assert result["splits"] > 0
    assert result["pbo"] >= 0.5


def test_pbo_fails_closed_without_multiple_aligned_trials():
    assert probability_of_backtest_overfitting({"only": [0.01] * 20})["pbo"] == 1.0
    result = probability_of_backtest_overfitting(
        {"a": [0.01] * 20, "b": [0.01] * 19}
    )
    assert result["valid"] is False
    assert result["pbo"] == 1.0


def test_deflated_performance_penalizes_multiple_testing():
    returns = [0.01, 0.012, 0.009, 0.011, 0.013, 0.008] * 30

    few = deflated_performance_evidence(returns, number_of_trials=2)
    many = deflated_performance_evidence(returns, number_of_trials=500)

    assert few["valid"] is True
    assert many["valid"] is True
    assert many["deflated_performance_statistic"] < few["deflated_performance_statistic"]
    assert many["multiple_testing_adjusted_p_value"] >= few["multiple_testing_adjusted_p_value"]


def test_drift_detection_accepts_stable_multi_regime_edge():
    rows = []
    for index in range(120):
        rows.append(
            {
                "recorded_at": float(index),
                "net_return": 0.01 if index % 2 == 0 else 0.008,
                "regime": "trend" if index % 3 else "range",
            }
        )

    result = drift_and_edge_decay(rows)

    assert result["valid"] is True
    assert result["drift_stable"] is True
    assert result["regimes"] == 2


def test_drift_detection_rejects_sign_inversion():
    rows = [
        {
            "recorded_at": float(index),
            "net_return": 0.01 if index < 60 else -0.01,
            "regime": "trend" if index % 2 else "range",
        }
        for index in range(120)
    ]

    result = drift_and_edge_decay(rows)

    assert result["valid"] is True
    assert result["drift_stable"] is False
    assert result["recent_mean"] < 0.0


def experiment(candidate_id: str, outcomes: list[dict], *, selection: dict | None = None) -> dict:
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


def outcome(index: int, value: float = 0.01, regime: str | None = None) -> dict:
    return {
        "recorded_at": float(index),
        "strategy": "adaptive_ensemble",
        "symbol": "BTC/USDT",
        "regime": regime or ("trend" if index % 2 else "range"),
        "net_return": value,
        "evidence_authority": "costed_shadow_episode_v2",
    }


def test_holdout_is_frozen_before_new_outcomes(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    engine = EvidenceQualificationEngine(tmp_path / "qualification.json")
    state = {
        "experiments": {
            "candidate": experiment(
                "candidate",
                [
                    {**outcome(index), "recorded_at": 900.0 + index / 10.0}
                    for index in range(100)
                ],
            )
        }
    }

    first = engine.qualify(state)
    freeze = first["partitions"]["untouched_holdout"]["freeze"]
    assert freeze["frozen_outcome_count"] == 100
    assert first["partitions"]["untouched_holdout"]["samples"] == 0
    assert first["untouched_holdout_passed"] is False

    clock["now"] = 1_100.0
    state["experiments"]["candidate"]["outcomes"].extend(
        [
            {**outcome(100 + index), "recorded_at": 1_001.0 + index}
            for index in range(100)
        ]
    )
    second = engine.qualify(state)

    assert second["partitions"]["prospective_paper"]["samples"] == 100
    assert second["partitions"]["untouched_holdout"]["samples"] == 100
    assert second["untouched_holdout_passed"] is True


def test_holdout_prefix_mutation_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    engine = EvidenceQualificationEngine(tmp_path / "qualification.json")
    state = {"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(100)])}}
    engine.qualify(state)

    state["experiments"]["candidate"]["outcomes"][0]["net_return"] = 99.0
    result = engine.qualify(state)

    assert result["untouched_holdout_passed"] is False
    assert any("prefreeze_outcomes_mutated" in reason for reason in result["qualification"]["reasons"])
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False


def test_manifest_mutation_after_holdout_freeze_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    engine = EvidenceQualificationEngine(tmp_path / "qualification.json")
    state = {"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(100)])}}
    engine.qualify(state)

    state["experiments"]["candidate"]["protocol"]["round_trip_cost_bps"] = 31.0
    result = engine.qualify(state)

    assert any("manifest_changed_after_freeze" in reason for reason in result["qualification"]["reasons"])
    assert result["untouched_holdout_passed"] is False


def test_explicit_partitions_and_reproducibility_hash_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    selection = {
        "purged_walk_forward_folds": [
            {
                "training": [observation(0, 5, 6), observation(6, 10, 11)],
                "validation": [observation(13, 15, 16)],
            }
        ],
        "aligned_validation_returns": [0.01, -0.005, 0.012, -0.004, 0.011, -0.003, 0.010, -0.002],
    }
    state = {
        "experiments": {
            "a": experiment("a", [outcome(i) for i in range(120)], selection=selection),
            "b": experiment(
                "b",
                [outcome(i, 0.008) for i in range(120)],
                selection={
                    **selection,
                    "aligned_validation_returns": [-0.005, 0.01, -0.004, 0.012, -0.003, 0.011, -0.002, 0.010],
                },
            ),
        }
    }
    path = tmp_path / "qualification.json"
    engine = EvidenceQualificationEngine(path, embargo_seconds=2.0)

    result = engine.qualify(state)

    assert set(result["partitions"]) == {
        "training",
        "validation",
        "prospective_paper",
        "untouched_holdout",
    }
    assert result["purged_walk_forward_passed"] is True
    assert result["embargo_applied"] is True
    assert result["evidence_reproducibility_hash"]
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["history"][-1]["payload"]["evidence_reproducibility_hash"] == result["evidence_reproducibility_hash"]
    assert engine.health()["lineage_integrity_ok"] is True


def test_lineage_corruption_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    path = tmp_path / "qualification.json"
    engine = EvidenceQualificationEngine(path)
    state = {"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(100)])}}
    engine.qualify(state)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"][0]["payload"]["independent_samples"] = 999_999
    path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = EvidenceQualificationEngine(path)
    result = restarted.qualify(state)

    assert restarted.health()["lineage_integrity_ok"] is False
    assert result["qualification"]["valid"] is False
    assert result["probability_backtest_overfitting"] == 1.0
    assert result["untouched_holdout_passed"] is False
    assert result["automatic_promotion"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False


def test_missing_walk_forward_and_pbo_evidence_fail_closed(tmp_path):
    engine = EvidenceQualificationEngine(tmp_path / "qualification.json")
    state = {"experiments": {"candidate": experiment("candidate", [outcome(i) for i in range(120)])}}

    result = engine.qualify(state)

    assert result["purged_walk_forward_passed"] is False
    assert result["embargo_applied"] is False
    assert result["probability_backtest_overfitting"] == 1.0
    assert result["multiple_testing_controlled"] is False
    assert result["automatic_promotion"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False


def test_safety_floors_cannot_be_relaxed(tmp_path):
    with pytest.raises(ValueError, match="100-sample"):
        EvidenceQualificationEngine(tmp_path / "a.json", minimum_samples=99)
    with pytest.raises(ValueError, match="two regimes"):
        EvidenceQualificationEngine(tmp_path / "b.json", minimum_regimes=1)
    with pytest.raises(ValueError, match="30-bps"):
        EvidenceQualificationEngine(tmp_path / "c.json", modeled_round_trip_cost_bps=29.99)
