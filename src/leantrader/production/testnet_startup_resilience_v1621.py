from __future__ import annotations

import datetime as dt
import math
import time
from typing import Any

from .testnet_execution import (
    TestnetSafetyError,
)


INITIAL_RETRY_SECONDS = 5.0
MAX_RETRY_SECONDS = 60.0

RECOVERABLE_PROVIDER_MARKERS = (
    '"retcode":10016',
    "'retcode': 10016",
    "internal system error",
    "exchange not available",
    "exchange unavailable",
    "service unavailable",
    "temporarily unavailable",
    "request timeout",
    "request timed out",
    "timed out",
    "rate limit",
    "too many requests",
    "connection reset",
    "connection aborted",
    "remote disconnected",
    "network error",
)


def _utc_now() -> str:
    return dt.datetime.now(
        dt.UTC
    ).isoformat()


def _ensure_state(
    engine: Any,
) -> None:
    defaults = {
        "_v1621_startup_degraded": False,
        "_v1621_startup_fatal": False,
        "_v1621_startup_failures": 0,
        "_v1621_consecutive_failures": 0,
        "_v1621_recovery_attempts": 0,
        "_v1621_recovery_successes": 0,
        "_v1621_last_error": None,
        "_v1621_last_failure_at": None,
        "_v1621_last_success_at": None,
        "_v1621_next_retry_monotonic": 0.0,
        "_v1621_retry_seconds": INITIAL_RETRY_SECONDS,
        "_v1621_recovery_in_progress": False,
    }

    for name, value in defaults.items():
        if not hasattr(
            engine,
            name,
        ):
            setattr(
                engine,
                name,
                value,
            )


def _exception_text(
    exc: BaseException,
) -> str:
    rows: list[str] = []

    seen: set[int] = set()
    current: BaseException | None = exc

    while (
        current is not None
        and id(current) not in seen
        and len(rows) < 8
    ):
        seen.add(
            id(current)
        )

        rows.append(
            f"{type(current).__name__}: {current}"
        )

        current = (
            current.__cause__
            or current.__context__
        )

    return " | ".join(
        rows
    ).lower()


def _recoverable_provider_failure(
    exc: BaseException,
) -> bool:
    text = _exception_text(
        exc
    )

    return any(
        marker in text
        for marker in (
            RECOVERABLE_PROVIDER_MARKERS
        )
    )


def _close_provider(
    engine: Any,
) -> None:
    exchange = getattr(
        engine,
        "exchange",
        None,
    )

    close = getattr(
        exchange,
        "close",
        None,
    )

    if callable(close):
        try:
            close()
        except Exception:
            pass

    engine.exchange = None
    engine.endpoint_verified = False
    engine.authenticated = False
    engine.credential_fingerprint = None
    engine._eligible_symbols = set()

    engine.api_attestation = {
        "verified": False,
    }

    engine.exchange_capabilities = {}


def _record_degraded(
    engine: Any,
    exc: BaseException,
    *,
    fatal: bool = False,
) -> None:
    _ensure_state(
        engine
    )

    engine._v1621_startup_failures += 1
    engine._v1621_consecutive_failures += 1

    engine._v1621_startup_degraded = True
    engine._v1621_startup_fatal = bool(
        fatal
    )

    try:
        detail = engine._redact(
            str(exc)
        )
    except Exception:
        detail = (
            type(exc).__name__
        )

    engine._v1621_last_error = detail
    engine._v1621_last_failure_at = (
        _utc_now()
    )

    if fatal:
        engine._v1621_retry_seconds = (
            MAX_RETRY_SECONDS
        )

        engine._v1621_next_retry_monotonic = (
            math.inf
        )

    else:
        failure_index = max(
            0,
            engine._v1621_consecutive_failures
            - 1,
        )

        delay = min(
            MAX_RETRY_SECONDS,
            INITIAL_RETRY_SECONDS
            * (
                2
                ** min(
                    failure_index,
                    4,
                )
            ),
        )

        engine._v1621_retry_seconds = (
            float(delay)
        )

        engine._v1621_next_retry_monotonic = (
            time.monotonic()
            + delay
        )

    _close_provider(
        engine
    )


def _record_ready(
    engine: Any,
    *,
    recovered: bool,
) -> None:
    _ensure_state(
        engine
    )

    engine._v1621_startup_degraded = False
    engine._v1621_startup_fatal = False
    engine._v1621_consecutive_failures = 0
    engine._v1621_retry_seconds = (
        INITIAL_RETRY_SECONDS
    )
    engine._v1621_next_retry_monotonic = (
        0.0
    )
    engine._v1621_last_error = None
    engine._v1621_last_success_at = (
        _utc_now()
    )

    if recovered:
        engine._v1621_recovery_successes += 1


def install_testnet_startup_resilience_v1621() -> None:
    from .testnet_execution import (
        BybitTestnetExecutionEngine,
    )

    if getattr(
        BybitTestnetExecutionEngine,
        "_v1621_startup_resilience_installed",
        False,
    ):
        return

    original_start = (
        BybitTestnetExecutionEngine.start
    )
    original_health = (
        BybitTestnetExecutionEngine.health
    )
    original_eligible_symbols = (
        BybitTestnetExecutionEngine.eligible_symbols
    )
    original_mirror_events = (
        BybitTestnetExecutionEngine.mirror_events
    )
    original_reconcile_required = (
        BybitTestnetExecutionEngine.reconcile_required
    )

    def run_start_attempt(
        self: Any,
        *,
        initial: bool,
    ) -> bool:
        _ensure_state(
            self
        )

        try:
            original_start(
                self
            )

        except Exception as exc:
            if _recoverable_provider_failure(
                exc
            ):
                _record_degraded(
                    self,
                    exc,
                    fatal=False,
                )

                return False

            if initial:
                # Configuration, endpoint, credential,
                # permission and other unknown startup
                # failures remain hard fail-closed.
                raise

            # A previously running process must not
            # crash-loop if a recovery attempt later
            # discovers an unsafe condition. Disable
            # execution permanently until intervention.
            _record_degraded(
                self,
                exc,
                fatal=True,
            )

            return False

        reconciliation_errors = list(
            self.state.get(
                "last_reconciliation_errors",
                [],
            )
            or []
        )

        if reconciliation_errors:
            _record_degraded(
                self,
                TestnetSafetyError(
                    "startup reconciliation "
                    "remains unresolved"
                ),
                fatal=False,
            )

            return False

        _record_ready(
            self,
            recovered=(
                not initial
            ),
        )

        return True

    def recover_if_due(
        self: Any,
    ) -> bool:
        _ensure_state(
            self
        )

        ready = bool(
            self.authenticated
            and self.endpoint_verified
            and not self._v1621_startup_degraded
            and not (
                self.state.get(
                    "last_reconciliation_errors",
                    [],
                )
                or []
            )
        )

        if ready:
            return True

        if (
            self._v1621_startup_fatal
            or self._v1621_recovery_in_progress
        ):
            return False

        if (
            time.monotonic()
            < self._v1621_next_retry_monotonic
        ):
            return False

        self._v1621_recovery_in_progress = (
            True
        )
        self._v1621_recovery_attempts += 1

        try:
            return run_start_attempt(
                self,
                initial=False,
            )

        finally:
            self._v1621_recovery_in_progress = (
                False
            )

    def start(
        self: Any,
    ) -> None:
        _ensure_state(
            self
        )

        with self._io_lock:
            run_start_attempt(
                self,
                initial=True,
            )

    def health(
        self: Any,
    ) -> dict[str, Any]:
        _ensure_state(
            self
        )

        payload = original_health(
            self
        )

        ready = bool(
            self.authenticated
            and self.endpoint_verified
            and not self._v1621_startup_degraded
            and not (
                self.state.get(
                    "last_reconciliation_errors",
                    [],
                )
                or []
            )
        )

        retry_in = 0.0

        if (
            self._v1621_startup_degraded
            and not self._v1621_startup_fatal
        ):
            retry_in = max(
                0.0,
                self._v1621_next_retry_monotonic
                - time.monotonic(),
            )

        payload[
            "healthy"
        ] = ready

        payload[
            "startup_recovery"
        ] = {
            "version": "1.60.21",
            "ready": ready,
            "degraded": bool(
                self._v1621_startup_degraded
            ),
            "fatal": bool(
                self._v1621_startup_fatal
            ),
            "automatic_retry": True,
            "bounded_exponential_backoff": True,
            "initial_retry_seconds": (
                INITIAL_RETRY_SECONDS
            ),
            "maximum_retry_seconds": (
                MAX_RETRY_SECONDS
            ),
            "current_retry_seconds": float(
                self._v1621_retry_seconds
            ),
            "retry_in_seconds": retry_in,
            "startup_failures": int(
                self._v1621_startup_failures
            ),
            "consecutive_failures": int(
                self._v1621_consecutive_failures
            ),
            "recovery_attempts": int(
                self._v1621_recovery_attempts
            ),
            "recovery_successes": int(
                self._v1621_recovery_successes
            ),
            "last_error": (
                self._v1621_last_error
            ),
            "last_failure_at": (
                self._v1621_last_failure_at
            ),
            "last_success_at": (
                self._v1621_last_success_at
            ),
            "orders_allowed_while_degraded": False,
            "fresh_sandbox_adapter_per_retry": True,
            "endpoint_reverification_required": True,
            "api_reattestation_required": True,
            "balance_refresh_required": True,
            "reconciliation_clear_required": True,
            "persistent_learning_preserved": True,
            "live_authority": False,
        }

        payload[
            "version"
        ] = "3.2"

        payload[
            "live_authority"
        ] = False

        return payload

    def safe_snapshot(
        self: Any,
    ) -> dict[str, Any]:
        with self._io_lock:
            recover_if_due(
                self
            )

            return self.health()

    def eligible_symbols(
        self: Any,
        quote: str = "USDT",
    ) -> set[str]:
        with self._io_lock:
            if not recover_if_due(
                self
            ):
                return set()

            return original_eligible_symbols(
                self,
                quote,
            )

    def mirror_events(
        self: Any,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self._io_lock:
            if not recover_if_due(
                self
            ):
                raise TestnetSafetyError(
                    "Bybit Testnet execution "
                    "is temporarily degraded; "
                    "new orders remain blocked"
                )

        return original_mirror_events(
            self,
            events,
        )

    def reconcile_required(
        self: Any,
    ) -> dict[str, Any]:
        with self._io_lock:
            if not recover_if_due(
                self
            ):
                raise TestnetSafetyError(
                    "Bybit Testnet execution "
                    "is not yet safely recovered"
                )

        return original_reconcile_required(
            self
        )

    BybitTestnetExecutionEngine.start = (
        start
    )

    BybitTestnetExecutionEngine.health = (
        health
    )

    BybitTestnetExecutionEngine.safe_snapshot = (
        safe_snapshot
    )

    BybitTestnetExecutionEngine.eligible_symbols = (
        eligible_symbols
    )

    BybitTestnetExecutionEngine.mirror_events = (
        mirror_events
    )

    BybitTestnetExecutionEngine.reconcile_required = (
        reconcile_required
    )

    BybitTestnetExecutionEngine.VERSION = (
        "3.2"
    )

    BybitTestnetExecutionEngine._v1621_recover_if_due = (
        recover_if_due
    )

    BybitTestnetExecutionEngine._v1621_startup_resilience_installed = (
        True
    )
