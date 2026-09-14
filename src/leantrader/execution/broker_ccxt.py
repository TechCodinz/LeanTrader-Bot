from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

_log = logging.getLogger(__name__)


EXECUTION_MODES = {
    "auto",
    "paper",
    "testnet",
    "live",
}

_PROBE_CACHE: Dict[
    Tuple[str, str],
    Tuple[str, float],
] = {}

_PROBE_TTL_SECONDS = 300.0


# PASS4_SECRET_FILE_CREDENTIAL_READER
def _read_secret_file(
    path_value: Optional[str],
) -> str:
    """
    Read one mounted runtime secret.

    Secret contents are never logged or returned
    from broker status/describe methods.
    """
    path = str(
        path_value or ""
    ).strip()

    if not path:
        return ""

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as handle:
            return (
                handle
                .read()
                .strip()
            )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "configured credential file "
            f"does not exist: {path}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            "configured credential file "
            f"could not be read: {path}"
        ) from exc


def _env_bool(
    name: str,
    default: bool = False,
) -> bool:
    return os.getenv(
        name,
        "true" if default else "false",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _normalize_mode(
    value: Optional[str],
) -> str:
    value = str(
        value or ""
    ).strip().lower()

    aliases = {
        "sandbox": "testnet",
        "demo": "testnet",
        "practice": "testnet",
        "real": "live",
        "production": "live",
        "prod": "live",
        "simulation": "paper",
        "sim": "paper",
        "emu": "paper",
    }

    value = aliases.get(
        value,
        value,
    )

    if value in EXECUTION_MODES:
        return value

    return "auto"


def _legacy_mode() -> str:
    """
    Resolve execution mode from the environment, in precedence order.

    1. EXECUTION_MODE -- the canonical selector. Whatever it names wins,
       including live. Nothing below can override or shadow it.
    2. The legacy three-flag live grant: ENABLE_LIVE, ALLOW_LIVE and
       LIVE_CONFIRM=YES together. All three are required, so this is a
       deliberate operator act, never an accident.
    3. Venue sandbox hints: CCXT_TESTNET / BYBIT_TESTNET.
    4. TRADING_MODE, for old files that only carried that.
    5. "auto", meaning nothing was selected.

    The live grant is checked BEFORE the sandbox hints. It used to be
    checked after, which made a per-venue endpoint flag silently outrank an
    explicit operator decision: with BYBIT_TESTNET=true present -- as it now
    is in the tracked .env, and as it is in any Testnet-oriented deployment
    file -- an operator who set all three live flags got Testnet with no
    warning. A sandbox flag says which endpoint a venue should use; it is not
    a mode selection, and it must not shadow one.
    """
    explicit = os.getenv(
        "EXECUTION_MODE",
        "",
    ).strip()

    if explicit:
        return _normalize_mode(
            explicit
        )

    legacy_live = (
        _env_bool(
            "ENABLE_LIVE",
            False,
        )
        and _env_bool(
            "ALLOW_LIVE",
            False,
        )
        and (
            os.getenv(
                "LIVE_CONFIRM",
                "",
            ).strip().upper()
            == "YES"
        )
    )

    sandbox_hint = (
        _env_bool(
            "CCXT_TESTNET",
            False,
        )
        or _env_bool(
            "BYBIT_TESTNET",
            False,
        )
    )

    if legacy_live:
        if sandbox_hint:
            # Both were set. Say which one is being honoured rather than
            # picking one silently; an operator seeing this has a
            # contradiction in their configuration to resolve.
            _log.warning(
                "Execution mode: live requested via ENABLE_LIVE/ALLOW_LIVE/"
                "LIVE_CONFIRM while a testnet sandbox flag is also set. "
                "Honouring the explicit live grant. Set EXECUTION_MODE to "
                "state the intent unambiguously."
            )
        return "live"

    if sandbox_hint:
        return "testnet"

    trading_mode = os.getenv(
        "TRADING_MODE",
        "",
    ).strip().lower()

    if trading_mode in {
        "paper",
        "simulation",
        "sim",
    }:
        return "paper"

    return "auto"


class BrokerCCXT:
    """
    Native LeanTrader exchange execution backend.

    Strategy engines do not know whether execution is:
      - paper
      - testnet/sandbox
      - live/production

    EXECUTION_MODE controls that at runtime.

    In auto mode:
      - no credentials -> paper
      - sandbox credentials authenticate -> testnet
      - production credentials authenticate -> live
      - credentials authenticate nowhere -> invalid/no authority

    Environment detection uses read-only authenticated calls.
    """

    def __init__(
        self,
        execution_mode: Optional[str] = None,
        exchange_id: Optional[str] = None,
        credentials: Optional[
            Dict[str, Any]
        ] = None,
        market_mode: Optional[str] = None,
    ) -> None:
        """
        credentials is an optional in-memory
        account profile.

        It is never returned by describe(),
        execution_status(), or order results.
        """
        profile = dict(
            credentials or {}
        )

        def profile_value(
            *names: str,
        ) -> str:
            for name in names:
                value = profile.get(
                    name
                )

                if value is None:
                    continue

                value = str(
                    value
                ).strip()

                if value:
                    return value

            return ""

        self.exchange_id = str(
            exchange_id
            or profile_value(
                "exchange_id",
                "exchange",
                "venue",
            )
            or os.getenv(
                "CCXT_EXCHANGE"
            )
            or os.getenv(
                "EXCHANGE_ID"
            )
            or "bybit"
        ).strip().lower()

        self.market_mode = str(
            market_mode
            or profile_value(
                "market_mode",
                "exchange_mode",
            )
            or os.getenv(
                "EXCHANGE_MODE",
                "spot",
            )
            or "spot"
        ).strip().lower()

        self.requested_mode = (
            _normalize_mode(
                execution_mode
            )
            if execution_mode
            else _legacy_mode()
        )

        # "auto" arrives two ways, and they are not the same decision.
        #
        # An operator who writes EXECUTION_MODE=auto (or passes it here) has
        # asked for discovery and may be discovered into live. An operator
        # who set nothing has selected nothing -- and for them, adding an
        # exchange API key must not move the execution destination to real
        # money. Only the first may reach live.
        self.auto_requested = str(
            execution_mode
            or os.getenv(
                "EXECUTION_MODE",
                "",
            )
        ).strip().lower() in {
            "auto",
        }

        prefix = (
            self.exchange_id
            .replace("-", "_")
            .upper()
        )

        # PASS4_CANONICAL_TESTNET_CREDENTIALS
        #
        # Testnet keys remain separate from any
        # future live/production key pair.
        # PASS4_TESTNET_SECRET_FILE_PRECEDENCE
        testnet_api_key_file = (
            os.getenv(
                f"{prefix}_TESTNET_API_KEY_FILE",
                "",
            ).strip()
            if self.requested_mode == "testnet"
            else ""
        )

        testnet_api_secret_file = (
            os.getenv(
                f"{prefix}_TESTNET_API_SECRET_FILE",
                "",
            ).strip()
            if self.requested_mode == "testnet"
            else ""
        )

        testnet_api_key = (
            (
                _read_secret_file(
                    testnet_api_key_file
                )
                or os.getenv(
                    f"{prefix}_TESTNET_API_KEY",
                    "",
                ).strip()
            )
            if self.requested_mode == "testnet"
            else ""
        )

        testnet_api_secret = (
            (
                _read_secret_file(
                    testnet_api_secret_file
                )
                or os.getenv(
                    f"{prefix}_TESTNET_API_SECRET",
                    "",
                ).strip()
            )
            if self.requested_mode == "testnet"
            else ""
        )

        self.api_key = (
            profile_value(
                "apiKey",
                "api_key",
                "key",
            )
            or testnet_api_key
            or os.getenv(
                f"{prefix}_API_KEY"
            )
            or os.getenv(
                "CCXT_API_KEY"
            )
            or os.getenv(
                "API_KEY"
            )
            or ""
        )

        self.api_secret = (
            profile_value(
                "secret",
                "api_secret",
                "secret_key",
            )
            or testnet_api_secret
            or os.getenv(
                f"{prefix}_API_SECRET"
            )
            or os.getenv(
                f"{prefix}_SECRET_KEY"
            )
            or os.getenv(
                "CCXT_API_SECRET"
            )
            or os.getenv(
                "API_SECRET"
            )
            or ""
        )

        self.password = (
            profile_value(
                "password",
                "api_password",
                "passphrase",
            )
            or os.getenv(
                f"{prefix}_API_PASSWORD"
            )
            or os.getenv(
                f"{prefix}_PASSPHRASE"
            )
            or os.getenv(
                "CCXT_API_PASSWORD"
            )
            or os.getenv(
                "API_PASSWORD"
            )
            or ""
        )

        self.uid = (
            profile_value(
                "uid",
                "account_id",
            )
            or os.getenv(
                f"{prefix}_UID"
            )
            or os.getenv(
                "CCXT_UID"
            )
            or ""
        )

        self.has_credentials = bool(
            self.api_key
            and self.api_secret
        )

        self._resolved_mode: Optional[
            str
        ] = None

        self._exchange_cache: Dict[
            Tuple[str, bool],
            Any,
        ] = {}

        self._probe_errors: Dict[
            str,
            str,
        ] = {}

    def _credential_fingerprint(
        self,
    ) -> str:
        if not self.has_credentials:
            return "public"

        payload = (
            self.api_key
            + "\0"
            + self.api_secret
            + "\0"
            + self.password
        ).encode(
            "utf-8",
            errors="ignore",
        )

        return hashlib.sha256(
            payload
        ).hexdigest()[:16]

    def _ccxt_class(
        self,
    ):
        try:
            import ccxt
        except Exception as exc:
            raise RuntimeError(
                "ccxt import failed"
            ) from exc

        candidates = [
            self.exchange_id
        ]

        if self.exchange_id == "gate":
            candidates.append(
                "gateio"
            )

        if self.exchange_id == "gateio":
            candidates.append(
                "gate"
            )

        for candidate in candidates:
            klass = getattr(
                ccxt,
                candidate,
                None,
            )

            if klass is not None:
                return klass

        raise RuntimeError(
            "Unsupported CCXT exchange: "
            f"{self.exchange_id}"
        )

    def _base_options(
        self,
    ) -> Dict[str, Any]:
        options: Dict[
            str,
            Any,
        ] = {}

        if self.exchange_id == "bybit":
            # Bybit authenticated requests use a 5s receive window by
            # default. Cold/private requests on real network paths can
            # legitimately exceed that even when host and exchange clocks
            # are synchronized. Keep this venue-level and mode-neutral:
            # it applies equally to sandbox and live endpoints.
            options[
                "recvWindow"
            ] = 10000

            options[
                "defaultType"
            ] = (
                "swap"
                if self.market_mode
                in {
                    "linear",
                    "swap",
                    "futures",
                    "future",
                }
                else "spot"
            )

            if (
                self.market_mode
                == "linear"
            ):
                options[
                    "defaultSubType"
                ] = "linear"

        elif (
            self.exchange_id
            == "binance"
            and self.market_mode
            in {
                "linear",
                "swap",
                "future",
                "futures",
            }
        ):
            options[
                "defaultType"
            ] = "future"

        extra = os.getenv(
            "CCXT_OPTIONS_JSON",
            "",
        ).strip()

        if extra:
            try:
                parsed = json.loads(
                    extra
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    options.update(
                        parsed
                    )
            except Exception:
                pass

        return options

    def _make_exchange(
        self,
        environment: str,
        authenticated: bool,
    ):
        environment = (
            "testnet"
            if environment
            == "testnet"
            else "live"
        )

        key = (
            environment,
            bool(authenticated),
        )

        cached = (
            self._exchange_cache.get(
                key
            )
        )

        if cached is not None:
            return cached

        klass = self._ccxt_class()

        opts: Dict[
            str,
            Any,
        ] = {
            "enableRateLimit": True,
            "timeout": int(
                os.getenv(
                    "CCXT_TIMEOUT_MS",
                    "15000",
                )
            ),
            "options": (
                self._base_options()
            ),
        }

        if (
            authenticated
            and self.has_credentials
        ):
            opts[
                "apiKey"
            ] = self.api_key

            opts[
                "secret"
            ] = self.api_secret

            if self.password:
                opts[
                    "password"
                ] = self.password

            if self.uid:
                opts[
                    "uid"
                ] = self.uid

        exchange = klass(
            opts
        )

        if environment == "testnet":
            sandbox = getattr(
                exchange,
                "set_sandbox_mode",
                None,
            )

            if not callable(
                sandbox
            ):
                raise RuntimeError(
                    f"{self.exchange_id} "
                    "does not expose a "
                    "CCXT sandbox endpoint"
                )

            # Verify the switch actually reaches the wire, rather than
            # trusting that the call returned.
            #
            # Venues sandbox differently: most swap the API URLs, OKX keeps
            # its URL and sends an x-simulated-trading header instead. Both
            # are real. But bitget, in the installed ccxt, sets only an
            # internal options flag and changes nothing that affects where a
            # request goes -- set_sandbox_mode(True) reports success and the
            # client still points at production. A caller who asked for
            # Testnet would have sent real orders to the live endpoint
            # believing they were sandboxed.
            #
            # So the evidence required is a change to something that decides
            # where the request lands: the URLs, the hostname, or the
            # headers. An options flag on its own is not evidence.
            def _routing_fingerprint() -> str:
                return json.dumps(
                    {
                        "urls": exchange.urls,
                        "hostname": getattr(
                            exchange,
                            "hostname",
                            None,
                        ),
                        "headers": getattr(
                            exchange,
                            "headers",
                            None,
                        ),
                    },
                    sort_keys=True,
                    default=str,
                )

            before = _routing_fingerprint()

            try:
                sandbox(
                    True
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{self.exchange_id} has no usable CCXT sandbox: "
                    f"{type(exc).__name__}"
                ) from exc

            if _routing_fingerprint() == before:
                raise RuntimeError(
                    f"{self.exchange_id} accepted set_sandbox_mode(True) "
                    "without changing where requests are sent; Testnet "
                    "execution is not available on this venue"
                )

        self._exchange_cache[
            key
        ] = exchange

        return exchange

    def _probe_environment(
        self,
        environment: str,
    ) -> bool:
        """
        Read-only authenticated probe.
        No order is submitted here.
        """
        if not self.has_credentials:
            return False

        try:
            exchange = (
                self._make_exchange(
                    environment,
                    authenticated=True,
                )
            )

            result = (
                exchange.fetch_balance()
            )

            return isinstance(
                result,
                dict,
            )

        except Exception as exc:
            self._probe_errors[
                environment
            ] = (
                type(exc).__name__
            )

            return False

    def resolve_mode(
        self,
    ) -> str:
        if self._resolved_mode:
            return self._resolved_mode

        requested = (
            self.requested_mode
        )

        if requested == "paper":
            self._resolved_mode = (
                "paper"
            )

            return (
                self._resolved_mode
            )

        if requested in {
            "testnet",
            "live",
        }:
            self._resolved_mode = (
                requested
            )

            return (
                self._resolved_mode
            )

        if not self.has_credentials:
            self._resolved_mode = (
                "paper"
            )

            return (
                self._resolved_mode
            )

        hint = (
            os.getenv(
                "API_ENVIRONMENT"
            )
            or os.getenv(
                "EXCHANGE_ENVIRONMENT"
            )
            or ""
        )

        hint = _normalize_mode(
            hint
        )

        if hint in {
            "testnet",
            "live",
        }:
            if self._probe_environment(
                hint
            ):
                self._resolved_mode = (
                    hint
                )

                return (
                    self._resolved_mode
                )

            self._resolved_mode = (
                "invalid"
            )

            return (
                self._resolved_mode
            )

        cache_key = (
            self.exchange_id,
            self._credential_fingerprint(),
        )

        cached = _PROBE_CACHE.get(
            cache_key
        )

        if cached:
            cached_mode, timestamp = (
                cached
            )

            if (
                time.time()
                - timestamp
                < _PROBE_TTL_SECONDS
            ):
                self._resolved_mode = (
                    cached_mode
                )

                return (
                    self._resolved_mode
                )

        # Which environments discovery may consider.
        #
        # Testnet always. Live only when the operator explicitly asked for
        # discovery by writing "auto"; when nothing was selected at all,
        # live is not probed. That distinction is the point: an operator who
        # wrote EXECUTION_MODE=auto asked to be routed wherever their
        # credentials work, but an operator who set nothing did not, and for
        # them adding an exchange API key used to silently move the
        # execution destination to real money.
        environments = (
            ("testnet", "live")
            if self.auto_requested
            else ("testnet",)
        )

        for environment in environments:
            if self._probe_environment(
                environment
            ):
                self._resolved_mode = (
                    environment
                )

                _PROBE_CACHE[
                    cache_key
                ] = (
                    environment,
                    time.time(),
                )

                return (
                    self._resolved_mode
                )

        if not self.auto_requested:
            _log.warning(
                "Credentials for %s did not authenticate on Testnet and no "
                "execution mode was selected. Live is not probed unless "
                "discovery is requested explicitly (EXECUTION_MODE=auto) or "
                "live is selected (EXECUTION_MODE=live).",
                self.exchange_id,
            )

        self._resolved_mode = (
            "invalid"
        )

        _PROBE_CACHE[
            cache_key
        ] = (
            "invalid",
            time.time(),
        )

        return self._resolved_mode

    @property
    def authority(
        self,
    ) -> str:
        mode = self.resolve_mode()

        if mode == "paper":
            return "paper"

        if (
            mode in {
                "testnet",
                "live",
            }
            and self.has_credentials
        ):
            return mode

        return "none"

    @property
    def environment(
        self,
    ) -> str:
        return self.resolve_mode()

    def describe(
        self,
    ) -> Dict[str, Any]:
        """
        Redacted execution description.
        Never returns credentials.
        """
        return {
            "exchange": (
                self.exchange_id
            ),
            "market_mode": (
                self.market_mode
            ),
            "requested_mode": (
                self.requested_mode
            ),
            "resolved_mode": (
                self.resolve_mode()
            ),
            "authority": (
                self.authority
            ),
            "authenticated": (
                self.has_credentials
            ),
            "probe_errors": dict(
                self._probe_errors
            ),
        }

    def fetch_ticker(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        mode = self.resolve_mode()

        environment = (
            mode
            if mode
            in {
                "testnet",
                "live",
            }
            else "live"
        )

        exchange = (
            self._make_exchange(
                environment,
                authenticated=False,
            )
        )

        ticker = (
            exchange.fetch_ticker(
                symbol
            )
        )

        return ticker or {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ):
        mode = self.resolve_mode()

        environment = (
            mode
            if mode
            in {
                "testnet",
                "live",
            }
            else "live"
        )

        exchange = (
            self._make_exchange(
                environment,
                authenticated=False,
            )
        )

        return (
            exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
            )
            or []
        )

    def load_markets(
        self,
    ) -> Dict[str, Any]:
        mode = self.resolve_mode()

        environment = (
            mode
            if mode in {
                "testnet",
                "live",
            }
            else "live"
        )

        exchange = (
            self._make_exchange(
                environment,
                authenticated=False,
            )
        )

        markets = (
            exchange.load_markets()
        )

        return (
            markets
            if isinstance(
                markets,
                dict,
            )
            else {}
        )

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        mode = self.resolve_mode()

        environment = (
            mode
            if mode in {
                "testnet",
                "live",
            }
            else "live"
        )

        exchange = (
            self._make_exchange(
                environment,
                authenticated=False,
            )
        )

        result = (
            exchange.fetch_order_book(
                symbol,
                limit=limit,
            )
        )

        return (
            result
            if isinstance(
                result,
                dict,
            )
            else {}
        )

    def fetch_balance(
        self,
    ) -> Dict[str, Any]:
        mode = self.resolve_mode()

        if (
            mode not in {
                "testnet",
                "live",
            }
            or not self.has_credentials
        ):
            return {}

        exchange = (
            self._make_exchange(
                mode,
                authenticated=True,
            )
        )

        balance = (
            exchange.fetch_balance()
        )

        return balance or {}

    def order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        params: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Universal authenticated exchange order.

        The strategy decides intent.
        Execution mode and exchange environment
        remain runtime concerns.
        """
        symbol = str(
            symbol or ""
        )

        order_type = str(
            order_type or "market"
        ).strip().lower()

        side = str(
            side or ""
        ).strip().lower()

        try:
            qty = float(
                qty or 0.0
            )
        except Exception:
            qty = 0.0

        if price is not None:
            try:
                price = float(
                    price
                )
            except Exception:
                price = None

        params = dict(
            params or {}
        )

        if (
            not symbol
            or side not in {
                "buy",
                "sell",
            }
            or qty <= 0.0
        ):
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "error": (
                    "invalid_order_request"
                ),
                "symbol": symbol,
                "side": side,
                "qty": qty,
            }

        mode = self.resolve_mode()

        if mode == "paper":
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "paper",
                "execution_mode": (
                    "paper"
                ),
                "exchange": (
                    self.exchange_id
                ),
                "error": (
                    "paper_orders_are_"
                    "handled_by_router"
                ),
            }

        if mode == "invalid":
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "none",
                "execution_mode": (
                    "invalid"
                ),
                "exchange": (
                    self.exchange_id
                ),
                "error": (
                    "credentials_not_"
                    "authenticated_on_"
                    "configured_exchange"
                ),
            }

        if self.authority not in {
            "testnet",
            "live",
        }:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": "none",
                "execution_mode": mode,
                "exchange": (
                    self.exchange_id
                ),
                "error": (
                    "no_authenticated_"
                    "execution_authority"
                ),
            }

        exchange = (
            self._make_exchange(
                mode,
                authenticated=True,
            )
        )

        try:
            order_price = (
                None
                if order_type
                == "market"
                else price
            )

            order = (
                exchange.create_order(
                    symbol,
                    order_type,
                    side,
                    qty,
                    order_price,
                    params,
                )
                or {}
            )

            order_id = (
                order.get("id")
                if isinstance(
                    order,
                    dict,
                )
                else None
            )

            if (
                order_id
                and hasattr(
                    exchange,
                    "fetch_order",
                )
            ):
                for _ in range(3):
                    try:
                        time.sleep(
                            0.25
                        )

                        refreshed = (
                            exchange.fetch_order(
                                order_id,
                                symbol,
                            )
                        )

                        if refreshed:
                            order = (
                                refreshed
                            )

                        status = str(
                            (
                                order
                                if isinstance(
                                    order,
                                    dict,
                                )
                                else {}
                            ).get(
                                "status",
                                "",
                            )
                        ).lower()

                        if status in {
                            "closed",
                            "filled",
                        }:
                            break

                    except Exception:
                        break

            order_payload = (
                order
                if isinstance(order, dict)
                else {}
            )

            try:
                filled_quantity = float(
                    order_payload.get("filled") or 0.0
                )
            except (TypeError, ValueError):
                filled_quantity = 0.0

            # Testnet restoration invariant:
            # acknowledgement/order-id is not proof of execution.
            # Preserve existing Live behavior unchanged.
            executed = (
                filled_quantity > 0.0
                if mode == "testnet"
                else True
            )

            return {
                "ok": True,
                "executed": executed,
                "simulated": False,
                "authority": mode,
                "execution_mode": mode,
                "exchange": (
                    self.exchange_id
                ),
                "order_type": (
                    order_type
                ),
                "order": order_payload,
                "error": (
                    "testnet_order_acknowledged_without_fill"
                    if mode == "testnet" and not executed
                    else None
                ),
            }

        except Exception as exc:
            return {
                "ok": False,
                "executed": False,
                "simulated": False,
                "authority": (
                    self.authority
                ),
                "execution_mode": mode,
                "exchange": (
                    self.exchange_id
                ),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": (
                    order_type
                ),
                "error": str(exc),
            }

    def market(
        self,
        symbol: str,
        side: str,
        qty: float,
        ref_price: float = 0.0,
    ) -> Dict[str, Any]:
        return self.order(
            symbol=symbol,
            order_type="market",
            side=side,
            qty=qty,
            price=None,
            params={},
        )
