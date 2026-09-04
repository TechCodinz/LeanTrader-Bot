
from pathlib import Path


SOURCE = Path(
    "src/leantrader/production/"
    "testnet_execution_first_candidates_v1619.py"
).read_text()


def test_v16053_checks_signal_before_network_probe():
    freshness = SOURCE.index(
        "if not self._signal_ready("
    )

    probe = SOURCE.index(
        "result = _probe_candidate("
    )

    assert freshness < probe


def test_v16053_does_not_relax_two_second_freshness():
    assert "age_seconds <= 2.0" in SOURCE
    assert '"freshness_gate_seconds": 2.0' in SOURCE


def test_v16053_stale_candidate_is_warmed_not_executed():
    assert "execution_clean_stale.append(" in SOURCE
    assert "stale_candidates_consume_network_probe" in SOURCE
    assert "normal_execution_preflight_still_required" in SOURCE


def test_v16053_warm_cohort_remains_bounded():
    assert "capacity = max(" in SOURCE
    assert "min(" in SOURCE
    assert "12," in SOURCE
