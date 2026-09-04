
from pathlib import Path


def test_v16051_separates_assessment_from_entry_cap():
    source = Path(
        "src/leantrader/production/"
        "testnet_execution_first_candidates_v1619.py"
    ).read_text()

    assert "capital_limits_entries_not_assessment" in source
    assert "assessment_target" in source
    assert "authenticated_entry_capacity_target" in source


def test_v16051_execution_pins_receive_precision_context():
    source = Path(
        "src/leantrader/agents/swarm_service.py"
    ).read_text()

    assert "_precision_pinned_context_symbols" in source
    assert "_execution_candidate_pins" in source
    assert "_execution_precision_pins" in source


def test_v16051_reuses_recent_clean_reconciliation():
    source = Path(
        "src/leantrader/production/testnet_execution.py"
    ).read_text()

    assert "recent_clean_reconciliation" in source
    assert "recent_clean_reconciliation" in source
    assert "self.reconcile()" in source


def test_v16051_refreshes_reconciliation_before_fresh_selection():
    source = Path(
        "src/leantrader/production/fast_collective_hyper.py"
    ).read_text()

    assert "recent_reconciliation_already_clear" in source
    assert "recent_clean_reconciliation" in source
