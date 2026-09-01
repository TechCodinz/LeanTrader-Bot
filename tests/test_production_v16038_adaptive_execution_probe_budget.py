import threading

import leantrader.production.testnet_execution_first_candidates_v1619 as router


class FakeService:
    def collective_candidates(self, limit=8):
        return [
            "A/USDT",
            "B/USDT",
            "C/USDT",
            "D/USDT",
            "E/USDT",
        ]

    def collective_signal(self, symbol):
        return {
            "fresh": True,
            "age_seconds": 0.1,
        }


class FakeTestnet:
    max_order_usd = 10.0

    def safe_snapshot(self):
        return {
            "positions": {},
        }

    def eligible_symbols(self, quote):
        assert quote == "USDT"
        return {
            "A/USDT",
            "B/USDT",
            "C/USDT",
            "D/USDT",
            "E/USDT",
        }


class FakeLane:
    def __init__(self):
        self._lock = threading.RLock()
        self.testnet = FakeTestnet()
        self.state = {
            "account_balance": {
                "free": {
                    "USDT": 10.0,
                }
            }
        }

    def _save_locked(self):
        return None


def test_empty_selection_can_probe_beyond_normal_two_probe_budget(monkeypatch):
    lane = FakeLane()
    service = FakeService()

    monkeypatch.setattr(
        router,
        "_safe_minimum_cost",
        lambda engine, symbol: {
            "minimum_cost_usd": 0.01,
            "safe_required_usd": 0.01,
        },
    )

    monkeypatch.setattr(
        router,
        "_route_blocked_until",
        lambda lane, symbol: 0.0,
    )

    calls = []

    def probe(lane_arg, symbol):
        calls.append(symbol)

        return {
            "allowed": symbol == "C/USDT",
            "reason": (
                "round_trip_executable"
                if symbol == "C/USDT"
                else "prospective_position_not_sellable_under_stress"
            ),
        }

    monkeypatch.setattr(
        router,
        "_probe_candidate",
        probe,
    )

    proxy = router._ExecutionFirstCandidateProxy(
        service,
        lane,
        1000.0,
    )

    selected = proxy.collective_candidates(limit=1)

    assert selected == ["C/USDT"]
    assert calls == [
        "A/USDT",
        "B/USDT",
        "C/USDT",
    ]

    telemetry = lane.state[
        "v1619_last_candidate_selection"
    ]

    assert telemetry["network_probes"] == 3
    assert telemetry["selected_count"] == 1
    assert telemetry["selected"] == ["C/USDT"]


def test_empty_selection_probe_expansion_remains_bounded_to_four(monkeypatch):
    lane = FakeLane()
    service = FakeService()

    monkeypatch.setattr(
        router,
        "_safe_minimum_cost",
        lambda engine, symbol: {
            "minimum_cost_usd": 0.01,
            "safe_required_usd": 0.01,
        },
    )

    monkeypatch.setattr(
        router,
        "_route_blocked_until",
        lambda lane, symbol: 0.0,
    )

    calls = []

    def probe(lane_arg, symbol):
        calls.append(symbol)

        return {
            "allowed": False,
            "reason": "prospective_position_not_sellable_under_stress",
        }

    monkeypatch.setattr(
        router,
        "_probe_candidate",
        probe,
    )

    proxy = router._ExecutionFirstCandidateProxy(
        service,
        lane,
        1000.0,
    )

    selected = proxy.collective_candidates(limit=1)

    assert selected == []
    assert calls == [
        "A/USDT",
        "B/USDT",
        "C/USDT",
        "D/USDT",
    ]

    telemetry = lane.state[
        "v1619_last_candidate_selection"
    ]

    assert telemetry["network_probes"] == 4
    assert telemetry["selected_count"] == 0
    assert telemetry["probe_budget_deferrals"] == 1
