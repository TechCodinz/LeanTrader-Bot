from __future__ import annotations

import json
import time

import pytest

from leantrader.production.alpha_tournament import AlphaTournament, StrategyFoundry
from leantrader.production.prospective_validation import ProspectiveValidationLab
from leantrader.production.runtime_evidence_qualification import RuntimeEvidenceQualificationEngine
from leantrader.production.strategy_observatory import StrategyObservatory


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


def test_v142_observatory_enriches_closed_episode_intervals(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(
        "leantrader.production.strategy_observatory_v141.time.time",
        lambda: clock["now"],
    )
    observatory = StrategyObservatory(tmp_path / "observatory.json", round_trip_cost_bps=30.0)
    first = observatory.observe(
        "BTC/USDT",
        100.0,
        [{"engine": "trend", "score": 1.0}],
        {},
    )
    assert first["episodes_closed"] == 0

    clock["now"] = 1_010.0
    second = observatory.observe(
        "BTC/USDT",
        101.0,
        [{"engine": "trend", "score": -1.0}],
        {},
    )
    assert second["episodes_closed"] == 1
    row = second["outcomes"][0]
    assert row["strategy"] == "engine:trend"
    assert row["opened_at"] == 1_000.0
    assert row["closed_at"] == 1_010.0
    assert row["feature_start"] == 1_000.0
    assert row["feature_end"] == 1_000.0
    assert row["label_end"] == 1_010.0
    assert row["evidence_interval_complete"] is True


def test_v142_foundry_precommits_partition_protocol_and_new_identity(tmp_path):
    tournament = AlphaTournament(tmp_path / "alpha.json")
    result = tournament.evaluate(strategy_health=ranked_health())
    manifests = result["foundry_manifests"]
    assert len(manifests) == 2
    assert all(row["candidate_id"].startswith("v142-foundry-") for row in manifests)
    assert all(
        row["selection_evidence"]["selection_frozen_before_partitioned_outcomes"]
        is True
        for row in manifests
    )
    plan = manifests[0]["research_protocol"]["partition_plan"]
    assert plan["initial_training_samples"] == 60
    assert plan["walk_forward_folds"] == 3
    assert plan["validation_samples_per_fold"] == 20
    assert plan["embargo_samples_per_fold"] == 1
    assert plan["prospective_paper_samples"] == 100
    assert plan["untouched_holdout_samples"] == 100
    assert manifests[0]["research_protocol"]["freeze_before_outcome_required"] is True
    assert manifests[0]["research_protocol"]["automatic_promotion"] is False
    assert manifests[0]["testnet_authority"] is False
    assert manifests[0]["live_authority"] is False


def test_partition_assignment_is_precommitted_and_walk_forward_expanding():
    plan = StrategyFoundry.partition_plan()
    assign = ProspectiveValidationLab._partition_assignment
    assert assign(0, plan)["partition"] == "training"
    assert assign(59, plan)["partition"] == "training"
    assert assign(60, plan) == {
        "partition": "embargo",
        "walk_forward_fold": 0,
        "partition_index": 0,
    }
    assert assign(61, plan)["partition"] == "validation"
    assert assign(61, plan)["walk_forward_fold"] == 0
    assert assign(81, plan)["partition"] == "embargo"
    assert assign(82, plan)["walk_forward_fold"] == 1
    assert assign(102, plan)["partition"] == "embargo"
    assert assign(103, plan)["walk_forward_fold"] == 2
    assert assign(123, plan)["partition"] == "prospective_paper"
    assert assign(222, plan)["partition"] == "prospective_paper"
    assert assign(223, plan)["partition"] == "untouched_holdout"
    assert assign(322, plan)["partition"] == "untouched_holdout"
    assert assign(323, plan)["partition"] == "untouched_holdout"


def test_v142_prospective_lab_refuses_episode_opened_before_registration(tmp_path, monkeypatch):
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

    clock["now"] = 1_010.0
    strategy = manifest["base_strategy"]
    rejected = lab.observe_cycle(
        observatory_authority="costed_shadow_episode_v2",
        observed_round_trip_cost_bps=30.0,
        strategy_episodes=[
            {
                "strategy": strategy,
                "symbol": "BTC/USDT",
                "regime": "trend",
                "net_return": 0.01,
                "opened_at": 999.0,
                "closed_at": 1_010.0,
                "feature_start": 999.0,
                "feature_end": 999.0,
                "label_end": 1_010.0,
                "evidence_interval_complete": True,
                "evidence_authority": "costed_shadow_episode_v2",
            }
        ],
        foundry_manifests=[manifest],
        market_rows={},
    )
    assert rejected["episodes_recorded"] == 0
    assert lab.health()["partition_rejections"] == 1

    clock["now"] = 1_020.0
    accepted = lab.observe_cycle(
        observatory_authority="costed_shadow_episode_v2",
        observed_round_trip_cost_bps=30.0,
        strategy_episodes=[
            {
                "strategy": strategy,
                "symbol": "BTC/USDT",
                "regime": "range",
                "net_return": 0.01,
                "opened_at": 1_011.0,
                "closed_at": 1_020.0,
                "feature_start": 1_011.0,
                "feature_end": 1_011.0,
                "label_end": 1_020.0,
                "evidence_interval_complete": True,
                "evidence_authority": "costed_shadow_episode_v2",
            }
        ],
        foundry_manifests=[manifest],
        market_rows={},
    )
    assert accepted["episodes_recorded"] == 1
    experiment = lab.state["experiments"][manifest["candidate_id"]]
    assert experiment["outcomes"][0]["partition"] == "training"
    assert experiment["outcomes"][0]["episode_ordinal"] == 0


def v142_experiment(candidate: str, strategy: str, score: float) -> dict:
    plan = StrategyFoundry.partition_plan()
    protocol = {
        "prospective_only": True,
        "minimum_additional_shadow_samples": 100,
        "round_trip_cost_bps": 30.0,
        "walk_forward_required": True,
        "out_of_sample_required": True,
        "multiple_testing_correction_required": True,
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
    }
    return {
        "candidate_id": candidate,
        "base_strategy": strategy,
        "registered_at": 1_000.0,
        "manifest_sha256": f"manifest-{candidate}",
        "protocol": protocol,
        "selection_evidence": {
            "authority": "costed_shadow_episode_v2",
            "conservative_score": score,
            "evidence_protocol_version": "1.42",
            "selection_frozen_before_partitioned_outcomes": True,
        },
        "outcomes": [],
        "regime_returns": {},
        "status": "collecting_partitioned_evidence",
    }


def fill_outcomes(experiment: dict, *, validation_return: float, base_time: float) -> None:
    plan = experiment["protocol"]["partition_plan"]
    total = (
        int(plan["initial_training_samples"])
        + int(plan["walk_forward_folds"])
        * (
            int(plan["embargo_samples_per_fold"])
            + int(plan["validation_samples_per_fold"])
        )
        + int(plan["prospective_paper_samples"])
        + int(plan["untouched_holdout_samples"])
    )
    outcomes = []
    for ordinal in range(total):
        assignment = ProspectiveValidationLab._partition_assignment(ordinal, plan)
        opened = base_time + ordinal * 10.0
        closed = opened + 5.0
        partition = assignment["partition"]
        if partition == "validation":
            value = validation_return + (ordinal % 3) * 0.0001
        elif partition == "embargo":
            value = 0.001
        else:
            value = 0.010 + (ordinal % 3) * 0.001
        outcomes.append(
            {
                "recorded_at": closed,
                "opened_at": opened,
                "closed_at": closed,
                "feature_start": opened,
                "feature_end": opened,
                "label_end": closed,
                "evidence_interval_complete": True,
                "episode_ordinal": ordinal,
                **assignment,
                "strategy": experiment["base_strategy"],
                "symbol": "BTC/USDT",
                "regime": "trend" if ordinal % 2 else "range",
                "net_return": value,
                "evidence_authority": "costed_shadow_episode_v2",
            }
        )
    experiment["outcomes"] = outcomes


def test_partitioned_qualifier_measures_all_gates_and_opens_holdout_once(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    path = tmp_path / "qualification.json"
    engine = RuntimeEvidenceQualificationEngine(path, embargo_seconds=1.0)
    a = v142_experiment("v142-a", "engine:a", 2.0)
    b = v142_experiment("v142-b", "engine:b", 1.0)
    state = {"experiments": {"v142-a": a, "v142-b": b}}

    frozen = engine.qualify(state, base_validation={"calibration_reliable": True})
    assert frozen["untouched_holdout_passed"] is False
    assert frozen["partitions"]["untouched_holdout"]["sealed"] is True

    clock["now"] = 5_000.0
    fill_outcomes(a, validation_return=0.010, base_time=1_100.0)
    fill_outcomes(b, validation_return=0.001, base_time=1_100.0)
    qualified = engine.qualify(state, base_validation={"calibration_reliable": True})

    assert qualified["purged_walk_forward_passed"] is True
    assert qualified["embargo_applied"] is True
    assert qualified["independent_samples"] == 100
    assert qualified["prospective_net_positive"] is True
    assert qualified["drift_stable"] is True
    assert qualified["probability_backtest_overfitting"] == pytest.approx(0.0)
    assert qualified["deflated_performance_statistic"] > 0.0
    assert qualified["multiple_testing_controlled"] is True
    assert qualified["partitions"]["training"]["samples"] == 60
    assert qualified["partitions"]["validation"]["samples"] == 60
    assert qualified["partitions"]["validation"]["folds"] == 3
    assert qualified["partitions"]["prospective_paper"]["samples"] == 100
    assert qualified["partitions"]["untouched_holdout"]["opened_once"] is True
    assert qualified["partitions"]["untouched_holdout"]["samples_collected"] == 100
    assert qualified["untouched_holdout_passed"] is True
    assert qualified["evidence_reproducibility_hash"]
    assert qualified["automatic_promotion"] is False
    assert qualified["paper_promotion_authority"] is False
    assert qualified["testnet_authority"] is False
    assert qualified["live_authority"] is False

    persisted = json.loads(path.read_text(encoding="utf-8"))
    opening = persisted["holdout_openings"]["v142-a"]
    assert opening["sample_count"] == 100
    assert opening["sealed_outcomes_sha256"]

    first_holdout = next(
        row for row in a["outcomes"] if row["partition"] == "untouched_holdout"
    )
    first_holdout["net_return"] = 99.0
    corrupted = engine.qualify(state, base_validation={"calibration_reliable": True})
    assert corrupted["untouched_holdout_passed"] is False
    assert any(
        "opened_holdout_mutated" in reason
        for reason in corrupted["qualification"]["reasons"]
    )


def test_partitioned_qualifier_never_peeks_at_immature_holdout(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    engine = RuntimeEvidenceQualificationEngine(tmp_path / "qualification.json")
    experiment = v142_experiment("v142-a", "engine:a", 2.0)
    state = {"experiments": {"v142-a": experiment}}
    engine.qualify(state)

    clock["now"] = 5_000.0
    fill_outcomes(experiment, validation_return=0.01, base_time=1_100.0)
    experiment["outcomes"] = experiment["outcomes"][:-1]
    result = engine.qualify(state)
    holdout = result["partitions"]["untouched_holdout"]
    assert holdout["samples_collected"] == 99
    assert holdout["sealed"] is True
    assert holdout["opened_once"] is False
    assert holdout["mean_net_return"] is None
    assert holdout["lower_95_net_return"] is None
    assert result["untouched_holdout_passed"] is False
