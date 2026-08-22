from __future__ import annotations

import json

from leantrader.production.runner import PaperRunner
from leantrader.production.unified_control_plane import UnifiedDecisionControlPlane


def _base_validation() -> dict:
    return {
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


def _measured_contract(tag: str) -> dict:
    return {
        "independent_samples": 120,
        "purged_walk_forward_passed": True,
        "embargo_applied": True,
        "untouched_holdout_passed": False,
        "multiple_testing_controlled": True,
        "prospective_net_positive": True,
        "calibration_reliable": True,
        "drift_stable": False,
        "probability_backtest_overfitting": 0.10,
        "deflated_performance_statistic": 1.5,
        "partitions": {
            "training": {"samples": 60},
            "validation": {"samples": 60},
            "prospective_paper": {"samples": 100},
            "untouched_holdout": {"samples": 0},
        },
        "qualification": {"reasons": ["untouched_holdout_not_passed"]},
        "evidence_reproducibility_hash": tag,
        "automatic_promotion": False,
        "paper_promotion_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "execution_authority": False,
    }


def test_control_plane_measures_unchanged_runtime_evidence_once(tmp_path):
    control_path = tmp_path / "vps_unified_control_plane_v141.json"
    prospective_path = tmp_path / "vps_prospective_validation.json"
    prospective_path.write_text(json.dumps({"experiments": {}}), encoding="utf-8")
    plane = UnifiedDecisionControlPlane(control_path)
    calls = {"count": 0}

    def fake_qualify_from_disk(*, base_validation):
        calls["count"] += 1
        assert base_validation["calibration_reliable"] is True
        return _measured_contract(f"hash-{calls['count']}")

    plane.evidence_qualification.qualify_from_disk = fake_qualify_from_disk
    supplied = _base_validation()
    first = plane._measured_validation(dict(supplied))
    second = plane._measured_validation(dict(supplied))

    assert calls["count"] == 1
    assert first == second
    assert first is not second
    assert plane.health()["measurement_count"] == 1
    assert plane.health()["validation_cache_reuse"] is True

    changed = dict(supplied)
    changed["prospective_net_positive"] = True
    third = plane._measured_validation(changed)
    assert calls["count"] == 2
    assert third["evidence_reproducibility_hash"] == "hash-2"


def test_runner_surfaces_measured_partitions_and_never_grants_authority():
    measured = _measured_contract("deterministic-hash")
    status = {
        "unified_decision_control_plane": {
            "validation_partitions": {"untouched_holdout": {"status": "legacy_placeholder"}},
            "automatic_promotion": False,
            "testnet_authority": False,
            "live_authority": False,
        }
    }
    health = {
        "measured_validation_applied": True,
        "measured_validation": measured,
        "measurement_count": 1,
        "validation_cache_reuse": True,
        "evidence_qualification": {"version": "1.42.0"},
        "automatic_promotion": False,
        "paper_promotion_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "execution_authority": False,
    }

    result = PaperRunner._apply_v142_measured_validation_status(status, health)
    bucket = result["unified_decision_control_plane"]
    metrics = bucket["qualification_metrics"]

    assert bucket["validation_partitions"] == measured["partitions"]
    assert bucket["measured_validation"]["evidence_reproducibility_hash"] == "deterministic-hash"
    assert metrics["independent_samples"] == 120
    assert metrics["purged_walk_forward_passed"] is True
    assert metrics["untouched_holdout_passed"] is False
    assert metrics["automatic_promotion"] is False
    assert metrics["testnet_authority"] is False
    assert metrics["live_authority"] is False
    assert result["runtime_evidence_qualification"]["execution_authority"] is False
