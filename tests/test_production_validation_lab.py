from __future__ import annotations

import pytest

from leantrader.production.prospective_validation import ProspectiveValidationLab


def manifest(
    candidate_id: str = "foundry-a",
    strategy: str = "engine:trend",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_kind": "shadow_research_manifest",
        "base_strategy": strategy,
        "status": "awaiting_independent_validation",
        "selection_evidence": {
            "authority": "costed_shadow_episode_v2",
            "samples": 120,
            "average_net_return": 0.001,
            "ewma_net_return": 0.0008,
            "win_rate": 0.60,
            "conservative_score": 0.01,
        },
        "research_protocol": {
            "prospective_only": True,
            "minimum_additional_shadow_samples": 100,
            "round_trip_cost_bps": 30.0,
            "walk_forward_required": True,
            "out_of_sample_required": True,
            "multiple_testing_correction_required": True,
            "paper_review_required": True,
            "automatic_promotion": False,
        },
        "hypothesis_context": [],
        "executable_code": None,
        "parameter_mutation_authority": False,
        "paper_promotion_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "can_increase_risk": False,
        "execution_authority": False,
    }


def episode(
    strategy: str = "engine:trend",
    net_return: float = 0.001,
    *,
    regime: str = "trend",
) -> dict:
    return {
        "strategy": strategy,
        "symbol": "BTC/USDT",
        "regime": regime,
        "net_return": net_return,
        "evidence_authority": "costed_shadow_episode_v2",
    }


def observe(
    lab: ProspectiveValidationLab,
    *,
    episodes: list[dict] | None = None,
    manifests: list[dict] | None = None,
    market_rows: dict[str, dict] | None = None,
    authority: str = "costed_shadow_episode_v2",
    cost_bps: float = 30.0,
) -> dict:
    return lab.observe_cycle(
        observatory_authority=authority,
        observed_round_trip_cost_bps=cost_bps,
        strategy_episodes=list(episodes or []),
        foundry_manifests=list(manifests or []),
        market_rows=dict(market_rows or {}),
    )


def test_constructor_refuses_looser_research_contract(tmp_path):
    with pytest.raises(ValueError, match="100-sample"):
        ProspectiveValidationLab(tmp_path / "state.json", minimum_samples=99)
    with pytest.raises(ValueError, match="30-bps"):
        ProspectiveValidationLab(
            tmp_path / "state.json",
            round_trip_cost_bps=29.99,
        )
    with pytest.raises(ValueError, match="at least two"):
        ProspectiveValidationLab(
            tmp_path / "state.json",
            counterfactual_horizon_observations=1,
        )


def test_registration_is_strictly_prospective(tmp_path):
    lab = ProspectiveValidationLab(tmp_path / "state.json")
    first = observe(
        lab,
        manifests=[manifest()],
        episodes=[episode(net_return=0.50)],
    )
    assert first["experiments_registered"] == ["foundry-a"]
    assert first["experiments"][0]["statistics"]["samples"] == 0

    second = observe(lab, episodes=[episode(net_return=0.002)])
    assert second["episodes_recorded"] == 1
    assert second["experiments"][0]["statistics"]["samples"] == 1
    assert second["experiments"][0]["samples_remaining"] == 99


def test_invalid_or_authority_seeking_manifest_is_quarantined(tmp_path):
    lab = ProspectiveValidationLab(tmp_path / "state.json")
    unsafe = manifest()
    unsafe["live_authority"] = True
    result = observe(lab, manifests=[unsafe])
    assert result["experiments"] == []
    assert result["manifest_rejections_retained"] == 1
    assert lab.health()["experiments"] == 0
    assert lab.health()["live_authority"] is False
    assert lab.health()["execution_authority"] is False


@pytest.mark.parametrize(
    ("authority", "cost_bps"),
    [
        ("gross_directional_diagnostic", 30.0),
        ("costed_shadow_episode_v2", 29.99),
    ],
)
def test_bad_evidence_contract_blocks_registration_and_outcomes(
    tmp_path,
    authority,
    cost_bps,
):
    lab = ProspectiveValidationLab(tmp_path / "state.json")
    result = observe(
        lab,
        authority=authority,
        cost_bps=cost_bps,
        manifests=[manifest()],
        episodes=[episode()],
    )
    assert result["strategy_contract_valid"] is False
    assert result["experiments_registered"] == []
    assert result["episodes_recorded"] == 0
    assert result["experiments"] == []


def test_counterfactual_trials_are_non_overlapping_and_costed(tmp_path):
    lab = ProspectiveValidationLab(
        tmp_path / "state.json",
        counterfactual_horizon_observations=2,
    )
    first = observe(
        lab,
        market_rows={
            "BTC/USDT": {
                "price": 100.0,
                "base_enter_candidate": True,
                "final_allowed": True,
                "route_reason": "approved",
                "regime": "trend",
                "confidence": 0.8,
                "quality_score": 0.9,
            }
        },
    )
    assert first["counterfactual"]["pending_non_overlapping_trials"] == 1
    assert first["counterfactuals_opened"] == ["BTC/USDT"]

    second = observe(
        lab,
        market_rows={
            "BTC/USDT": {
                "price": 101.0,
                "base_enter_candidate": False,
                "final_allowed": False,
                "route_reason": "no_signal",
                "regime": "trend",
            }
        },
    )
    approved = second["counterfactual"]["gate_statistics"]["approved"]
    assert approved["samples"] == 1
    assert approved["average_net_return"] == pytest.approx(0.007)
    assert second["counterfactual"]["pending_non_overlapping_trials"] == 0
    assert second["counterfactual"]["observational_not_causal"] is True
    assert second["counterfactual"]["cannot_replace_strategy_holdout"] is True


def test_counterfactual_attributes_downstream_gate_and_regime(tmp_path):
    lab = ProspectiveValidationLab(
        tmp_path / "state.json",
        counterfactual_horizon_observations=2,
    )
    observe(
        lab,
        market_rows={
            "ETH/USDT": {
                "price": 100.0,
                "base_enter_candidate": True,
                "final_allowed": False,
                "route_reason": "brain:negative_expectancy",
                "regime": "range",
            }
        },
    )
    result = observe(
        lab,
        market_rows={
            "ETH/USDT": {
                "price": 99.0,
                "base_enter_candidate": False,
                "final_allowed": False,
                "route_reason": "no_signal",
                "regime": "range",
            }
        },
    )
    gate = result["counterfactual"]["gate_statistics"]["trading_brain"]
    regime = result["counterfactual"]["regime_statistics"]["range"]
    assert gate["samples"] == 1
    assert gate["average_net_return"] == pytest.approx(-0.013)
    assert regime["samples"] == 1


def test_holdout_support_requires_samples_uncertainty_and_regime_diversity(tmp_path):
    lab = ProspectiveValidationLab(tmp_path / "state.json")
    observe(
        lab,
        manifests=[
            manifest("foundry-a", "engine:trend"),
            manifest("foundry-b", "engine:mean_reversion"),
        ],
    )
    positive = [
        episode(
            "engine:trend",
            0.001,
            regime="trend" if index % 2 == 0 else "range",
        )
        for index in range(100)
    ]
    weak = [
        episode(
            "engine:mean_reversion",
            0.001 if index % 2 == 0 else -0.001,
            regime="trend" if index % 2 == 0 else "range",
        )
        for index in range(100)
    ]
    result = observe(lab, episodes=positive + weak)
    rows = {row["candidate_id"]: row for row in result["experiments"]}
    supported = rows["foundry-a"]
    rejected = rows["foundry-b"]

    assert supported["status"] == "research_supported_holdout"
    assert supported["statistics"]["samples"] == 100
    assert supported["statistics"]["lower_95_net_return"] > 0.0
    assert supported["adjusted_p_value"] < 0.05
    assert supported["observed_regimes"] == 2

    assert rejected["status"] == "not_supported_holdout"
    assert rejected["statistics"]["average_net_return"] == pytest.approx(0.0)
    assert rejected["paper_promotion_authority"] is False
    assert rejected["testnet_authority"] is False
    assert rejected["live_authority"] is False
    assert rejected["can_increase_risk"] is False
    assert rejected["execution_authority"] is False


def test_persistence_retains_research_state_without_authority(tmp_path):
    path = tmp_path / "state.json"
    lab = ProspectiveValidationLab(path)
    observe(lab, manifests=[manifest()])
    observe(lab, episodes=[episode()])
    reloaded = ProspectiveValidationLab(path)
    health = reloaded.health()
    assert health["cycles"] == 2
    assert health["experiments"] == 1
    assert health["minimum_samples"] == 100
    assert health["round_trip_cost_bps"] == 30.0
    assert health["paper_promotion_authority"] is False
    assert health["testnet_authority"] is False
    assert health["live_authority"] is False
    assert health["execution_authority"] is False
