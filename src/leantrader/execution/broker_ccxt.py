from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Tuple


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
    Preserve compatibility with old environment files,
    without making those flags the architecture.
    """
    explicit = os.getenv(
        "EXECUTION_MODE",
        "",
    ).strip()

    if explicit:
        return _normalize_mode(
            explicit
        )

    if (
        _env_bool(
            "CCXT_TESTNET",
            False,
        )
        or _env_bool(
            "BYBIT_TESTNET",
            False,
        )
    ):
        return "testnet"

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

    if legacy_live:
        return "live"

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
    ) -> None:
        self.exchange_id = (
            exchange_id
            or os.getenv(
                "CCXT_EXCHANGE"
            )
            or os.getenv(
                "EXCHANGE_ID"
            )
            or "bybit"
        ).strip().lower()

        self.market_mode = (
            os.getenv(
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

        prefix = (
            self.exchange_id
            .replace("-", "_")
            .upper()
        )

        self.api_key = (
            os.getenv(
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
            os.getenv(
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
            os.getenv(
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
            os.getenv(
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

            sandbox(
                True
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

        for environment in (
            "testnet",
            "live",
        ):
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

            return {
                "ok": True,
                "executed": True,
                "simulated": False,
                "authority": mode,
                "execution_mode": mode,
                "exchange": (
                    self.exchange_id
                ),
                "order_type": (
                    order_type
                ),
                "order": order,
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
