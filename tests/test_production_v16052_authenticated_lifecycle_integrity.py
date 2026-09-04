
from pathlib import Path


def test_v16052_v1608_wrapper_preserves_event_id():
    source = Path(
        "src/leantrader/production/testnet_exit_recycle.py"
    ).read_text()

    assert '"paper_event_id": str(event.get("event_id", ""))' in source


def test_v16052_terminal_recovery_requires_exact_authenticated_cycle():
    source = Path(
        "src/leantrader/production/"
        "testnet_terminal_pending_recovery_v1629.py"
    ).read_text()

    assert "_authenticated_cycle_matches_exit" in source
    assert "exact_authenticated_cycle_correlated" in source
    assert "modeled_outcome_used_for_learning" in source
    assert '"fabricated_close": False' in source


def test_v16052_terminal_learning_uses_actual_position_notional():
    source = Path(
        "src/leantrader/production/"
        "testnet_terminal_pending_recovery_v1629.py"
    ).read_text()

    assert "actual_entry_notional" in source
    assert "filled_quantity * entry_price" in source
    assert "_record_specialist_regime_outcome" in source
