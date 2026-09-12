"""Paper, Testnet and Live are three destinations for one intelligence fabric.

The tracked-default hardening made every checked-in config fail-closed. This
verifies it did not do the other thing -- turn Testnet into a ceiling. An
operator who explicitly selects live, with live credentials, must still reach
the live broker path.

It also verifies the converse, which matters more: credentials must never
decide the destination. Adding an exchange API key is authentication, not a
mode selection.

No order is submitted anywhere in this module. Every test that reaches the
order path replaces the ccxt client with an in-process interceptor, so
nothing can leave the machine even if a URL were wrong.
"""

import os

import pytest

from src.leantrader.execution import preflight
from src.leantrader.execution.broker_ccxt import (
    _PROBE_CACHE,
    BrokerCCXT,
    _legacy_mode,
)
from src.leantrader.execution.router import resolve_execution_context, route_order

MODE_VARS = (
    "EXECUTION_MODE",
    "TRADING_MODE",
    "CCXT_TESTNET",
    "BYBIT_TESTNET",
    "ENABLE_LIVE",
    "ALLOW_LIVE",
    "LIVE_CONFIRM",
    "API_ENVIRONMENT",
    "EXCHANGE_ENVIRONMENT",
    "BROKER_BACKEND",
    "EXCHANGE_ID",
    "CCXT_EXCHANGE",
)

CREDENTIAL_VARS = (
    "BYBIT_API_KEY",
    "BYBIT_API_SECRET",
    "BYBIT_TESTNET_API_KEY",
    "BYBIT_TESTNET_API_SECRET",
    "BYBIT_TESTNET_API_KEY_FILE",
    "BYBIT_TESTNET_API_SECRET_FILE",
    "CCXT_API_KEY",
    "CCXT_API_SECRET",
    "API_KEY",
    "API_SECRET",
)

LIVE_GRANT = {"ENABLE_LIVE": "true", "ALLOW_LIVE": "true", "LIVE_CONFIRM": "YES"}

# Invented values. They are never sent anywhere: every test that reaches an
# order replaces the exchange client entirely.
FAKE_LIVE_KEY = "liveKEY" + "0" * 11          # secret-scan: allow
FAKE_LIVE_SECRET = "liveSECRET" + "0" * 26    # secret-scan: allow


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_TELEMETRY_PATH", str(tmp_path / "t.json"))
    for name in MODE_VARS + CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    _PROBE_CACHE.clear()
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    _PROBE_CACHE.clear()
    preflight.reset_caches()
    preflight.reset_shared_brokers()


class Interceptor:
    """Stands in for the ccxt client. Records; never sends."""

    instances = []

    def __init__(self, environment):
        self.environment = environment
        self.orders = []
        Interceptor.instances.append(self)

    def create_order(self, symbol, order_type, side, qty, price, params):
        self.orders.append((symbol, order_type, side, qty, price))
        return {
            "id": f"INTERCEPTED-{len(self.orders)}",
            "status": "closed",
            "filled": qty,
            "average": price or 64_000.0,
        }

    def fetch_order(self, order_id, symbol):
        return None


@pytest.fixture
def intercept(monkeypatch):
    """Replace the exchange client so no request can leave the process."""
    Interceptor.instances = []
    monkeypatch.setattr(
        BrokerCCXT,
        "_make_exchange",
        lambda self, environment, authenticated: Interceptor(environment),
    )
    return Interceptor


def live_credentials(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", FAKE_LIVE_KEY)
    monkeypatch.setenv("BYBIT_API_SECRET", FAKE_LIVE_SECRET)


def order_payload():
    return {
        "symbol": "BTC/USDT",
        "side": "buy",
        "qty": 0.001,
        "price": 64_000.0,
        "order_type": "market",
        "backend": "ccxt",
    }


# ------------------------------------------------- 1-3. the three destinations


def test_explicit_paper_resolves_to_paper():
    context = resolve_execution_context(order_payload(), "paper")
    assert context["execution_mode"] == "paper"
    assert context["authority"] == "paper"


def test_explicit_testnet_resolves_to_testnet(monkeypatch):
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "tnKEY" + "0" * 13)  # secret-scan: allow
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "tnSEC" + "0" * 31)  # secret-scan: allow

    broker = BrokerCCXT(execution_mode="testnet", exchange_id="bybit")
    assert broker.resolve_mode() == "testnet"
    assert broker.authority == "testnet"


def test_explicit_live_with_live_credentials_resolves_through_the_live_path(
    monkeypatch, intercept
):
    """The capability must survive the tracked-default hardening."""
    live_credentials(monkeypatch)

    broker = BrokerCCXT(execution_mode="live", exchange_id="bybit")
    assert broker.resolve_mode() == "live"
    assert broker.authority == "live"
    assert broker.has_credentials is True

    receipt = route_order(order_payload(), "live")

    assert receipt["ok"] is True
    assert receipt["authority"] == "live"
    assert receipt["execution_mode"] == "live"
    assert receipt["simulated"] is False
    assert receipt["order"]["id"].startswith("INTERCEPTED")


def test_the_live_test_submits_no_real_order(monkeypatch, intercept):
    """Requirement 4, asserted rather than assumed."""
    live_credentials(monkeypatch)
    route_order(order_payload(), "live")

    assert intercept.instances, "the order path was never reached"
    for client in intercept.instances:
        assert isinstance(client, Interceptor), "a real ccxt client was built"

    # Every recorded order landed on the interceptor, not on a network client.
    submitted = sum(len(c.orders) for c in intercept.instances)
    assert submitted == 1


# --------------------------------- 5-6. credentials are not a mode selection


def test_credentials_alone_do_not_switch_paper_to_live(monkeypatch, intercept):
    live_credentials(monkeypatch)

    broker = BrokerCCXT(execution_mode="paper", exchange_id="bybit")
    assert broker.resolve_mode() == "paper"
    assert broker.authority == "paper"

    receipt = route_order(order_payload(), "paper")
    assert receipt["authority"] == "paper"
    assert receipt["simulated"] is True


def test_credentials_alone_do_not_switch_testnet_to_live(monkeypatch, intercept):
    live_credentials(monkeypatch)

    broker = BrokerCCXT(execution_mode="testnet", exchange_id="bybit")
    assert broker.resolve_mode() == "testnet"
    assert broker.authority == "testnet"

    receipt = route_order(order_payload(), "testnet")
    assert receipt["authority"] == "testnet"
    assert [c.environment for c in intercept.instances] == ["testnet"]


def test_nothing_selected_plus_credentials_never_lands_on_live(monkeypatch):
    """Adding an API key must not move the destination to real money.

    Discovery used to probe Testnet and then live regardless, so credentials
    valid only on production resolved to live with no mode ever selected.
    """
    live_credentials(monkeypatch)

    class LiveOnly(BrokerCCXT):
        def _probe_environment(self, environment):
            return environment == "live"

    broker = LiveOnly()
    assert broker.requested_mode == "auto"
    assert broker.auto_requested is False, "nothing was selected"
    assert broker.has_credentials is True
    assert broker.resolve_mode() != "live"
    assert broker.authority == "none"


def test_explicitly_requested_discovery_may_still_reach_live(monkeypatch):
    """EXECUTION_MODE=auto is itself an operator decision.

    Writing "auto" asks to be routed wherever the credentials work. Leaving
    the variable unset asks for nothing, and those are different choices --
    only the first may be discovered into live.
    """
    live_credentials(monkeypatch)
    monkeypatch.setenv("EXECUTION_MODE", "auto")

    class LiveOnly(BrokerCCXT):
        def _probe_environment(self, environment):
            return environment == "live"

    broker = LiveOnly()
    assert broker.auto_requested is True
    assert broker.resolve_mode() == "live"
    assert broker.authority == "live"


def test_explicit_discovery_still_prefers_testnet_when_both_work(monkeypatch):
    """Discovery reaching live is a fallback, not a preference."""
    live_credentials(monkeypatch)
    monkeypatch.setenv("EXECUTION_MODE", "auto")

    class BothWork(BrokerCCXT):
        def _probe_environment(self, environment):
            return True

    assert BothWork().resolve_mode() == "testnet"


def test_auto_discovery_still_finds_testnet(monkeypatch):
    live_credentials(monkeypatch)

    class TestnetOnly(BrokerCCXT):
        def _probe_environment(self, environment):
            return environment == "testnet"

    broker = TestnetOnly()
    assert broker.resolve_mode() == "testnet"


def test_both_credential_sets_present_explicit_testnet_stays_testnet(
    monkeypatch, intercept
):
    """Requirement 6."""
    live_credentials(monkeypatch)
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "tnKEY" + "0" * 13)  # secret-scan: allow
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "tnSEC" + "0" * 31)  # secret-scan: allow

    broker = BrokerCCXT(execution_mode="testnet", exchange_id="bybit")
    assert broker.resolve_mode() == "testnet"
    # Testnet uses the testnet key, not the live one.
    assert broker.api_key.startswith("tnKEY")


def test_live_mode_does_not_use_testnet_credentials(monkeypatch):
    """Testnet keys are scoped to Testnet and must not authenticate live."""
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "tnKEY" + "0" * 13)  # secret-scan: allow
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "tnSEC" + "0" * 31)  # secret-scan: allow

    broker = BrokerCCXT(execution_mode="live", exchange_id="bybit")
    assert not broker.api_key.startswith("tnKEY")
    assert broker.has_credentials is False, (
        "live must not be authenticated by Testnet credentials"
    )
    assert broker.authority == "none"


# ------------------------------------------- 7-8. endpoints match the mode


def test_explicit_live_does_not_use_testnet_endpoints():
    broker = BrokerCCXT(execution_mode="live", exchange_id="bybit")
    exchange = broker._make_exchange("live", authenticated=False)

    assert "testnet" not in str(exchange.urls.get("api")).lower()
    assert "x-simulated-trading" not in (getattr(exchange, "headers", {}) or {})


def test_explicit_testnet_does_not_use_production_endpoints():
    broker = BrokerCCXT(execution_mode="testnet", exchange_id="bybit")
    exchange = broker._make_exchange("testnet", authenticated=False)

    assert "testnet" in str(exchange.urls.get("api")).lower()


def test_a_venue_whose_sandbox_is_a_no_op_is_refused_not_silently_live():
    """bitget accepts set_sandbox_mode(True) and keeps production URLs.

    Trusting the call would have sent real orders to the live endpoint for a
    caller who asked for Testnet.
    """
    broker = BrokerCCXT(execution_mode="testnet", exchange_id="bitget")

    with pytest.raises(RuntimeError, match="without changing where requests are sent"):
        broker._make_exchange("testnet", authenticated=False)


def test_a_header_based_sandbox_is_accepted():
    """OKX sandboxes by header, not by URL. That is still a real sandbox."""
    broker = BrokerCCXT(execution_mode="testnet", exchange_id="okx")
    exchange = broker._make_exchange("testnet", authenticated=False)

    assert (getattr(exchange, "headers", {}) or {}).get("x-simulated-trading") == "1"


# --------------------------- 9-11. one authority, no bypass, preflight always


@pytest.mark.parametrize("mode", ["paper", "testnet", "live"])
def test_route_order_is_the_sole_order_authority_in_every_mode(
    mode, monkeypatch, intercept
):
    live_credentials(monkeypatch)
    monkeypatch.setenv("BYBIT_TESTNET_API_KEY", "tnKEY" + "0" * 13)  # secret-scan: allow
    monkeypatch.setenv("BYBIT_TESTNET_API_SECRET", "tnSEC" + "0" * 31)  # secret-scan: allow

    receipt = route_order(order_payload(), mode)

    assert receipt["execution_mode"] == mode
    assert receipt["ok"] is True
    assert receipt["order"]["id"]


def test_raw_exchange_bypass_remains_blocked_in_live(monkeypatch):
    """The guard must not be relaxed for the mode that matters most."""
    import logging

    import router as router_module

    class RawExchange:
        def create_order(self, *args, **kwargs):
            raise AssertionError("a raw exchange order reached the exchange")

    client = router_module.ExchangeRouter.__new__(router_module.ExchangeRouter)
    client.id = "bybit"
    client.testnet = False
    client.live = True
    client.ex = RawExchange()

    client.apply_runtime_order_block(logging.getLogger("test"))
    receipt = client.ex.create_order("BTC/USDT", "market", "buy", 0.001)

    assert receipt["ok"] is False
    assert "bypass_blocked" in receipt["error"]


@pytest.mark.parametrize("mode", ["paper", "testnet", "live"])
def test_preflight_runs_for_every_mode(mode):
    """Requirement 11: live gets the same checks Testnet does."""

    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"

        def __init__(self, authority):
            self.authority = authority

        def resolve_mode(self):
            return "testnet" if self.authority == "paper" else self.authority

        def load_markets(self):
            return {
                "BTC/USDT": {
                    "spot": True,
                    "active": True,
                    "base": "BTC",
                    "quote": "USDT",
                    "taker": 0.001,
                    "limits": {"amount": {"min": 0.0}, "cost": {"min": 5.0}},
                }
            }

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}}

        def fetch_ticker(self, symbol):
            return {"last": 64_000.0}

        def _make_exchange(self, environment, authenticated):
            return None

    prepared, blocked = preflight.prepare_order(
        {"symbol": "BTC/USDT", "side": "buy", "price": 64_000.0, "confidence": 0.9},
        broker=Broker(mode),
    )

    assert blocked is None, f"preflight refused in {mode}: {blocked}"
    assert prepared.notional >= prepared.min_notional
    assert prepared.notional <= 500.0


def test_preflight_blocks_an_unfundable_live_order():
    """Live is not exempt from the venue minimums."""

    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "live"

        def resolve_mode(self):
            return "live"

        def load_markets(self):
            return {
                "BTC/USDT": {
                    "spot": True,
                    "active": True,
                    "base": "BTC",
                    "quote": "USDT",
                    "taker": 0.001,
                    "limits": {"amount": {"min": 0.0}, "cost": {"min": 500.0}},
                }
            }

        def fetch_balance(self):
            return {"free": {"USDT": 13.95}}

        def fetch_ticker(self, symbol):
            return {"last": 64_000.0}

        def _make_exchange(self, environment, authenticated):
            return None

    prepared, blocked = preflight.prepare_order(
        {"symbol": "BTC/USDT", "side": "buy", "price": 64_000.0}, broker=Broker()
    )

    assert prepared is None
    # Refused at the minimum-ticket stage: the venue's 500 minimum is more
    # than 13.95 can fund, and live is not exempt from that.
    assert blocked.blocker == preflight.CAPITAL_BELOW_EXECUTABLE_MINIMUM
    assert blocked.stage == "minimum_ticket"


# ------------------------- 12-13. tracked defaults vs operator configuration


def test_tracked_defaults_alone_do_not_reach_live(monkeypatch):
    """Requirement 12: what the repository ships must fail closed."""
    import pathlib

    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        pytest.skip("no .env in this checkout")

    for line in env.read_text(errors="ignore").split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().upper()
        if key in MODE_VARS:
            monkeypatch.setenv(key, value.split("#")[0].strip().strip("\"'"))

    assert _legacy_mode() != "live"


def test_an_operator_can_still_deliberately_select_live(monkeypatch):
    """Requirement 13, by both supported routes."""
    monkeypatch.setenv("EXECUTION_MODE", "live")
    assert _legacy_mode() == "live"

    monkeypatch.delenv("EXECUTION_MODE")
    for key, value in LIVE_GRANT.items():
        monkeypatch.setenv(key, value)
    assert _legacy_mode() == "live"


def test_a_venue_sandbox_flag_does_not_shadow_an_explicit_live_grant(monkeypatch):
    """The regression the tracked-default hardening introduced.

    BYBIT_TESTNET=true is now in the tracked .env and in every
    Testnet-oriented deployment file. It used to be checked before the live
    grant, so an operator who set all three live flags got Testnet silently.
    """
    for key, value in LIVE_GRANT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("BYBIT_TESTNET", "true")

    assert _legacy_mode() == "live"

    monkeypatch.delenv("BYBIT_TESTNET")
    monkeypatch.setenv("CCXT_TESTNET", "true")
    assert _legacy_mode() == "live"


def test_execution_mode_outranks_the_legacy_live_grant(monkeypatch):
    """The canonical selector is supreme in both directions."""
    for key, value in LIVE_GRANT.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setenv("EXECUTION_MODE", "testnet")
    assert _legacy_mode() == "testnet"

    monkeypatch.setenv("EXECUTION_MODE", "paper")
    assert _legacy_mode() == "paper"


def test_a_partial_live_grant_does_not_grant_live(monkeypatch):
    """All three flags are required, so live is never reached by halves."""
    monkeypatch.setenv("ENABLE_LIVE", "true")
    assert _legacy_mode() != "live"

    monkeypatch.setenv("ALLOW_LIVE", "true")
    assert _legacy_mode() != "live"

    monkeypatch.setenv("LIVE_CONFIRM", "NO")
    assert _legacy_mode() != "live"

    monkeypatch.setenv("LIVE_CONFIRM", "YES")
    assert _legacy_mode() == "live"
