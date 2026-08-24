from leantrader.production.ledger import PaperLedger


def test_stale_capital_preservation_is_recomputed(tmp_path):
    ledger = PaperLedger(tmp_path / "ledger.json", 100.0)
    ledger.halt_reason = "capital_preservation"

    halt = ledger.risk_check(
        {},
        daily_limit=0.02,
        drawdown_limit=0.10,
    )

    assert halt is None
    assert ledger.halt_reason is None


def test_real_daily_loss_replaces_capital_preservation(tmp_path):
    ledger = PaperLedger(tmp_path / "ledger.json", 100.0)
    ledger.halt_reason = "capital_preservation"
    ledger.day_start_equity = 100.0
    ledger.peak_equity = 100.0
    ledger.cash = 97.0

    halt = ledger.risk_check(
        {},
        daily_limit=0.02,
        drawdown_limit=0.10,
    )

    assert halt is not None
    assert halt.startswith("daily_loss:")


def test_hard_ledger_halt_remains_latched(tmp_path):
    ledger = PaperLedger(tmp_path / "ledger.json", 100.0)
    ledger.halt_reason = "daily_loss:0.0300"

    halt = ledger.risk_check(
        {},
        daily_limit=0.02,
        drawdown_limit=0.10,
    )

    assert halt == "daily_loss:0.0300"
