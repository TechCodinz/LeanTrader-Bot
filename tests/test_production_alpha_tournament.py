from __future__ import annotations

import pytest

from leantrader.production.alpha_tournament import AlphaTournament


def health(
    strategies: dict,
    *,
    authority: str = "costed_shadow_episode_v2",
    cost_bps: float = 30.0,
) -> dict:
    return {
        "evidence_authority": authority,
        "round_trip_cost_bps": cost_bps,
        "strategies": strategies,
    }


def evidence(
    *,
    samples: int,
    average: float,
    ewma: float,
    win_rate: float,
) -> dict:
    wins = int(samples * win_rate)
    return {
        "samples": samples,
        "wins": wins,
        "win_rate": win_rate,
        "average_return": average,
        "ewma_net_return": ewma,
        "negative_streak": 0,
    }


def test_constructor_refuses_looser_sample_or_cost_thresholds(tmp_path):
    with pytest.raises(ValueError, match="100-sample"):
        AlphaTournament(tmp_path / "state.json", minimum_samples=99)
    with pytest.raises(ValueError, match="30-bps"):
        AlphaTournament(
            tmp_path / "state.json",
            expected_round_trip_cost_bps=29.99,
        )


def test_insufficient_evidence_waits_without_promotion(tmp_path):
    tournament = AlphaTournament(tmp_path / "state.json")
    result = tournament.evaluate(
        strategy_health=health(
            {
                "engine:trend": evidence(
                    samples=99,
                    average=0.01,
                    ewma=0.01,
                    win_rate=0.80,
                )
            }
        )
    )
    assert result["eligible_count"] == 0
    assert result["awaiting_samples"][0]["samples_remaining"] == 1
    assert result["foundry_manifests"] == []
    assert result["paper_promotion_authority"] is False
    assert result["testnet_authority"] is False
    assert result["live_authority"] is False
    assert result["execution_authority"] is False


def test_costed_positive_candidate_is_ranked_for_more_research_only(tmp_path):
    tournament = AlphaTournament(tmp_path / "state.json")
    result = tournament.evaluate(
        strategy_health=health(
            {
                "engine:trend": evidence(
                    samples=120,
                    average=0.0012,
                    ewma=0.0008,
                    win_rate=0.60,
                ),
                "timeframe:5m": evidence(
                    samples=140,
                    average=0.0007,
                    ewma=0.0006,
                    win_rate=0.55,
                ),
            }
        ),
        hypothesis_agenda=[{"question": "Does the edge survive volatility regimes?"}],
    )
    assert [row["strategy"] for row in result["ranking"]] == [
        "engine:trend",
        "timeframe:5m",
    ]
    assert all(row["execution_authority"] is False for row in result["ranking"])
    manifests = result["foundry_manifests"]
    assert len(manifests) == 2
    assert manifests[0]["research_protocol"]["minimum_additional_shadow_samples"] == 100
    assert manifests[0]["research_protocol"]["round_trip_cost_bps"] == 30.0
    assert manifests[0]["research_protocol"]["prospective_only"] is True
    assert manifests[0]["research_protocol"]["automatic_promotion"] is False
    assert manifests[0]["executable_code"] is None
    assert manifests[0]["can_increase_risk"] is False
    assert manifests[0]["execution_authority"] is False


def test_non_positive_or_low_win_rate_candidate_is_rejected(tmp_path):
    tournament = AlphaTournament(tmp_path / "state.json")
    result = tournament.evaluate(
        strategy_health=health(
            {
                "engine:negative": evidence(
                    samples=150,
                    average=-0.0001,
                    ewma=0.0002,
                    win_rate=0.70,
                ),
                "engine:fragile": evidence(
                    samples=150,
                    average=0.0004,
                    ewma=0.0002,
                    win_rate=0.49,
                ),
            }
        )
    )
    assert result["eligible_count"] == 0
    reasons = {
        row["strategy"]: row["reasons"]
        for row in result["rejected_after_costs"]
    }
    assert "non_positive_average_net_return" in reasons["engine:negative"]
    assert "win_rate_below_floor" in reasons["engine:fragile"]


@pytest.mark.parametrize(
    ("authority", "cost_bps"),
    [
        ("gross_directional_diagnostic", 30.0),
        ("costed_shadow_episode_v2", 29.99),
    ],
)
def test_evidence_contract_mismatch_blocks_all_ranking(
    tmp_path,
    authority,
    cost_bps,
):
    tournament = AlphaTournament(tmp_path / "state.json")
    result = tournament.evaluate(
        strategy_health=health(
            {
                "engine:trend": evidence(
                    samples=1_000,
                    average=0.02,
                    ewma=0.01,
                    win_rate=0.90,
                )
            },
            authority=authority,
            cost_bps=cost_bps,
        )
    )
    assert result["status"] == "blocked_evidence_contract"
    assert result["evidence_contract_valid"] is False
    assert result["ranking"] == []
    assert result["foundry_manifests"] == []


def test_manifest_identity_is_deterministic_and_state_persists(tmp_path):
    path = tmp_path / "state.json"
    strategy_health = health(
        {
            "engine:trend": evidence(
                samples=100,
                average=0.001,
                ewma=0.0007,
                win_rate=0.60,
            )
        }
    )
    first = AlphaTournament(path)
    first_id = first.evaluate(
        strategy_health=strategy_health
    )["foundry_manifests"][0]["candidate_id"]
    second_id = first.evaluate(
        strategy_health=strategy_health
    )["foundry_manifests"][0]["candidate_id"]
    reloaded = AlphaTournament(path)
    assert first_id == second_id
    assert reloaded.health()["evaluations"] == 2
    assert reloaded.health()["minimum_samples"] == 100
    assert reloaded.health()["round_trip_cost_bps"] == 30.0
