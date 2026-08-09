from __future__ import annotations

import pytest

from leantrader.production.ledger import PaperLedger


def test_paper_ledger_persists_and_accounts_for_costs(tmp_path):
    path = tmp_path / "paper.json"
    ledger = PaperLedger(path, 50.0)
    buy = ledger.buy("BTC/USDT", 100.0, 5.0, 2.0, fee_bps=10.0, slippage_bps=5.0)
    assert buy["side"] == "buy"
    assert ledger.cash < 45.0

    restored = PaperLedger(path, 999.0)
    event = restored.sell("BTC/USDT", 110.0, fee_bps=10.0, slippage_bps=5.0, reason="test")
    assert event["realized_pnl"] > 0
    assert event["realized_return"] == pytest.approx(event["realized_pnl"] / 5.005)
    assert restored.cash > 50.0


def test_paper_ledger_rejects_overspend(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.json", 10.0)
    with pytest.raises(ValueError, match="insufficient"):
        ledger.buy("BTC/USDT", 100.0, 20.0, 2.0, fee_bps=10.0, slippage_bps=5.0)
