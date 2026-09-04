
from pathlib import Path


def test_v16050_has_authenticated_fresh_selection_budget():
    source = Path(
        "src/leantrader/production/"
        "testnet_execution_first_candidates_v1619.py"
    ).read_text()

    assert "selection_target" in source
    assert "capital_funded_candidates" in source
    assert "fresh_selection_budgeting" in source
    assert "len(selected)" in source
    assert ">= selection_target" in source
    assert "freshness gate" in source.lower()
