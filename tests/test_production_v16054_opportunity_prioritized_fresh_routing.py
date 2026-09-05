
from pathlib import Path


SOURCE = Path(
    "src/leantrader/production/"
    "testnet_execution_first_candidates_v1619.py"
).read_text()


def test_v16054_has_bounded_priority_window():
    assert "priority_window = min(" in SOURCE
    assert "12," in SOURCE


def test_v16054_uses_same_two_second_freshness():
    assert "age_seconds <= 2.0" in SOURCE
    assert '"freshness_gate_seconds": 2.0' in SOURCE


def test_v16054_prioritizes_projected_capture_and_micro_edge():
    assert '"projected_capture_bps_5s"' in SOURCE
    assert '"expected_edge_bps"' in SOURCE
    assert "priority_score = max(" in SOURCE


def test_v16054_is_ranking_only():
    assert '"priority_is_ranking_only": True' in SOURCE
    assert '"priority_bypasses_profit_gate": False' in SOURCE
    assert '"priority_bypasses_execution_preflight": False' in SOURCE


def test_v16054_keeps_fresh_first_probe_contract():
    freshness = SOURCE.index(
        "if not self._signal_ready("
    )
    probe = SOURCE.index(
        "result = _probe_candidate("
    )
    assert freshness < probe
