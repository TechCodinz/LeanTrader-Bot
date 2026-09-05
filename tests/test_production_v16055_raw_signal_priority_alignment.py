from pathlib import Path


SOURCE = Path(
    "src/leantrader/production/"
    "testnet_execution_first_candidates_v1619.py"
).read_text()


def test_v16055_reuses_velocity_lane_derivation():
    assert '"_velocity_state"' in SOURCE
    assert (
        'velocity_state.get(\n'
        '                        "projected_capture_bps_5s"'
    ) in SOURCE


def test_v16055_reads_real_raw_micro_path_assessments():
    assert 'micro.get("path_assessments")' in SOURCE
    assert '"raw_micro_path_priority": True' in SOURCE


def test_v16055_keeps_priority_ranking_only():
    assert '"priority_is_ranking_only": True' in SOURCE
    assert '"priority_bypasses_profit_gate": False' in SOURCE
    assert '"priority_bypasses_execution_preflight": False' in SOURCE


def test_v16055_preserves_two_second_freshness():
    assert "age_seconds <= 2.0" in SOURCE
    assert '"freshness_gate_seconds": 2.0' in SOURCE


def test_v16055_has_compatibility_capture_fallback():
    assert (
        '"recent_midpoint_trend_bps_5s"'
        in SOURCE
    )
    assert (
        '"midpoint_velocity_bps_per_second"'
        in SOURCE
    )
    assert (
        '"midpoint_acceleration_bps_per_second2"'
        in SOURCE
    )


def test_v16055_does_not_grant_execution_authority():
    assert '"raw_signal_priority_alignment": True' in SOURCE
    assert '"live_authority": False' in SOURCE
