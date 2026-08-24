from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

import leantrader.production.fast_testnet_absence_quorum as quorum
from leantrader.production.testnet_execution import BybitTestnetExecutionEngine


class QuorumBybit:
    def __init__(
        self,
        *,
        failing_history_round: int | None = None,
        positive_execution_round: int | None = None,
    ) -> None:
        self.round = 0
        self.failing_history_round = failing_history_round
        self.positive_execution_round = positive_execution_round
        self.native_calls: list[tuple[str, int, str]] = []
        self.collection_calls: list[str] = []

    def market(self, symbol: str) -> dict[str, object]:
        return {
            "id": symbol.replace("/", ""),
        }

    def fetch_open_orders(
        self,
        _symbol: str,
        _since: object,
        _limit: object,
        params: dict[str, object],
    ) -> list[dict[str, object]]:
        self.collection_calls.append(
            f"open:{params['orderLinkId']}"
        )
        return []

    def fetch_closed_orders(
        self,
        _symbol: str,
        _since: object,
        _limit: object,
        params: dict[str, object],
    ) -> list[dict[str, object]]:
        self.collection_calls.append(
            f"closed:{params['orderLinkId']}"
        )
        return []

    def fetch_canceled_orders(
        self,
        _symbol: str,
        _since: object,
        _limit: object,
        params: dict[str, object],
    ) -> list[dict[str, object]]:
        self.collection_calls.append(
            f"canceled:{params['orderLinkId']}"
        )
        return []

    def private_get_v5_order_realtime(
        self,
        params: dict[str, object],
    ) -> dict[str, object]:
        self.round += 1
        client_id = str(params["orderLinkId"])
        self.native_calls.append(
            ("realtime", self.round, client_id)
        )
        return {
            "retCode": 0,
            "result": {"list": []},
        }

    def private_get_v5_order_history(
        self,
        params: dict[str, object],
    ) -> dict[str, object]:
        client_id = str(params["orderLinkId"])
        self.native_calls.append(
            ("history", self.round, client_id)
        )
        if self.round == self.failing_history_round:
            raise RuntimeError(
                "temporary Testnet order-history failure"
            )
        return {
            "retCode": 0,
            "result": {"list": []},
        }

    def private_get_v5_execution_list(
        self,
        params: dict[str, object],
    ) -> dict[str, object]:
        client_id = str(params["orderLinkId"])
        self.native_calls.append(
            ("execution", self.round, client_id)
        )
        rows: list[dict[str, object]] = []
        if self.round == self.positive_execution_round:
            rows = [
                {
                    "orderId": "accepted-late-1",
                    "orderLinkId": client_id,
                    "execQty": "0.05",
                    "execValue": "5.0",
                    "execPrice": "100.0",
                }
            ]
        return {
            "retCode": 0,
            "result": {"list": rows},
        }

    def create_order(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "v1.60.7 reconciliation must never resubmit"
        )


def _engine(
    tmp_path: Path,
    fake: QuorumBybit,
) -> BybitTestnetExecutionEngine:
    instance = BybitTestnetExecutionEngine(
        api_key_path=tmp_path / "unused-key",
        api_secret_path=tmp_path / "unused-secret",
        state_path=tmp_path / "testnet-state.json",
        confirmation="I_UNDERSTAND_TESTNET_ONLY",
        exchange_factory=lambda _config: fake,
    )
    instance.exchange = fake
    instance.endpoint_verified = True
    instance._verify_testnet_urls = lambda: None
    return instance


def _recent_ambiguous_record(
    client_id: str,
) -> dict[str, object]:
    return {
        "client_order_id": client_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "quantity": 0.05,
        "submitted_usd": 5.0,
        "reference_price": 100.0,
        "reason": "paper_entry",
        "paper_event_timestamp": (
            dt.datetime.now(dt.UTC).isoformat()
        ),
        "submitted_at": (
            dt.datetime.now(dt.UTC).isoformat()
        ),
        "status": "submitting",
        "order_id": None,
        "filled": 0.0,
        "applied_filled": 0.0,
        "filled_cost": 0.0,
        "applied_fill_cost": 0.0,
        "average": None,
        "fee": 0.0,
        "fee_currency": None,
        "applied_fee": 0.0,
        "fill_counted": False,
    }


def test_v1607_recent_ambiguity_resolves_after_three_negative_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quorum.time,
        "sleep",
        lambda _seconds: None,
    )
    fake = QuorumBybit()
    instance = _engine(tmp_path, fake)
    client_id = "lt-fast-absence-1"
    record = _recent_ambiguous_record(client_id)

    observed = instance._recover_order_with_bounded_retry(
        record,
        client_id,
    )

    assert observed is not None
    assert observed["status"] == "rejected"
    assert fake.round == 3
    assert {
        call_client_id
        for _source, _round, call_client_id in fake.native_calls
    } == {client_id}
    assert record[
        "fast_absence_quorum_consecutive_rounds"
    ] == 3
    assert record["reconciliation_resolution"] == (
        "native_bybit_fast_authoritative_absence_quorum"
    )


def test_v1607_endpoint_failure_resets_quorum_and_stays_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quorum.time,
        "sleep",
        lambda _seconds: None,
    )
    fake = QuorumBybit(
        failing_history_round=2,
    )
    instance = _engine(tmp_path, fake)
    client_id = "lt-fast-absence-failure"
    record = _recent_ambiguous_record(client_id)

    observed = instance._recover_order_with_bounded_retry(
        record,
        client_id,
    )

    assert observed is None
    assert fake.round == 3
    assert record["status"] == "submitting"
    assert record[
        "fast_absence_quorum_consecutive_rounds"
    ] == 1
    assert "endpoint_failure" in str(
        record["fast_absence_quorum_last_reset_reason"]
    )


def test_v1607_positive_execution_can_never_be_classified_as_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quorum.time,
        "sleep",
        lambda _seconds: None,
    )
    fake = QuorumBybit(
        positive_execution_round=3,
    )
    instance = _engine(tmp_path, fake)
    client_id = "lt-fast-absence-positive"
    record = _recent_ambiguous_record(client_id)

    observed = instance._recover_order_with_bounded_retry(
        record,
        client_id,
    )

    assert observed is not None
    assert observed["id"] == "accepted-late-1"
    assert observed["status"] == "open"
    assert observed["filled"] == pytest.approx(0.05)
    assert record[
        "fast_absence_quorum_consecutive_rounds"
    ] == 0
    assert record["reconciliation_resolution"] == (
        "native_bybit_execution_link_id"
    )


def test_v1607_health_exposes_fast_fail_closed_contract(
    tmp_path: Path,
) -> None:
    fake = QuorumBybit()
    instance = _engine(tmp_path, fake)

    recovery = instance.health()[
        "automatic_reconciliation_recovery"
    ]
    fast = recovery["fast_absence_quorum"]

    assert recovery["old_ambiguity_only"] is False
    assert recovery["resubmission_allowed"] is False
    assert recovery["fail_closed"] is True
    assert fast["enabled"] is True
    assert fast["required_consecutive_rounds"] == 3
    assert fast["authoritative_sources"] == [
        "realtime_order",
        "order_history",
        "execution_history",
    ]
    assert fast["fail_closed_on_endpoint_error"] is True
    assert fast["resubmission_allowed"] is False
