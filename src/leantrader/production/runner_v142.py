from __future__ import annotations

import argparse
import json
import signal
from typing import Any

from . import runner_v141 as _runner_v141
from .runner_v141 import *  # noqa: F401,F403
from .runner_v141 import (
    MarketFeed,
    PaperRunner as _V141PaperRunner,
    configure_logging,
    preflight,
)
from .settings import Settings


class PaperRunner(_V141PaperRunner):
    """v1.42 runner adapter over the exact v1.41 production runner.

    v1.41 execution, accounting, risk, routing, and paper-only behavior remain in
    ``runner_v141.py``. This adapter only makes the v1.42 measured evidence
    qualification visible in the canonical heartbeat after each cycle.
    """

    VERSION = "1.42.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Keep the v1.41 implementation byte-preserved while retaining the
        # public runner module's test/integration seam. Legacy callers patch the
        # exported Testnet engine before constructing PaperRunner; mirror that
        # injected class into the preserved module instead of bypassing secret
        # enforcement or changing the legacy tests.
        _runner_v141.BybitTestnetExecutionEngine = BybitTestnetExecutionEngine
        super().__init__(*args, **kwargs)

    @staticmethod
    def _apply_v142_measured_validation_status(
        status: dict[str, Any],
        control_health: dict[str, Any],
    ) -> dict[str, Any]:
        bucket = status.setdefault("unified_decision_control_plane", {})
        if not isinstance(bucket, dict):
            bucket = {}
            status["unified_decision_control_plane"] = bucket
        measured = control_health.get("measured_validation")
        bucket["health"] = control_health
        bucket["v1_42_measured_validation"] = True
        bucket["automatic_promotion"] = False
        bucket["paper_promotion_authority"] = False
        bucket["testnet_authority"] = False
        bucket["live_authority"] = False
        bucket["execution_authority"] = False
        if isinstance(measured, dict):
            bucket["measured_validation"] = measured
            partitions = measured.get("partitions")
            if isinstance(partitions, dict):
                bucket["validation_partitions"] = partitions
            bucket["qualification_metrics"] = {
                "independent_samples": int(measured.get("independent_samples") or 0),
                "purged_walk_forward_passed": measured.get("purged_walk_forward_passed") is True,
                "embargo_applied": measured.get("embargo_applied") is True,
                "untouched_holdout_passed": measured.get("untouched_holdout_passed") is True,
                "multiple_testing_controlled": measured.get("multiple_testing_controlled") is True,
                "probability_backtest_overfitting": measured.get("probability_backtest_overfitting"),
                "deflated_performance_statistic": measured.get("deflated_performance_statistic"),
                "calibration_reliable": measured.get("calibration_reliable") is True,
                "drift_stable": measured.get("drift_stable") is True,
                "prospective_net_positive": measured.get("prospective_net_positive") is True,
                "evidence_reproducibility_hash": measured.get("evidence_reproducibility_hash"),
                "automatic_promotion": False,
                "testnet_authority": False,
                "live_authority": False,
            }
        else:
            bucket["measured_validation"] = None
            bucket["qualification_metrics"] = {
                "available": False,
                "automatic_promotion": False,
                "testnet_authority": False,
                "live_authority": False,
            }
        status["runtime_evidence_qualification"] = {
            "version": str((control_health.get("evidence_qualification") or {}).get("version") or "1.42.0"),
            "measured_validation_applied": control_health.get("measured_validation_applied") is True,
            "measurement_count": int(control_health.get("measurement_count") or 0),
            "validation_cache_reuse": control_health.get("validation_cache_reuse") is True,
            "automatic_promotion": False,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "execution_authority": False,
        }
        return status

    def cycle(self) -> dict[str, Any]:
        status = super().cycle()
        control_health = self.unified_control_plane.health()
        self._apply_v142_measured_validation_status(status, control_health)
        # The v1.41 runner writes its heartbeat before returning. Persist the
        # v1.42 measured view atomically as the final heartbeat for this cycle.
        self._write_json_atomic(self.settings.heartbeat_path, status)
        return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeanTrader v1.42 paper runner with measured evidence qualification"
    )
    parser.add_argument("--once", action="store_true", help="run one market cycle and exit")
    parser.add_argument("--preflight", action="store_true", help="validate safe configuration without network access")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    if args.preflight:
        print(json.dumps(preflight(settings), indent=2))
        return

    runner = PaperRunner(settings, MarketFeed(settings.exchange))

    def request_stop(_signum: int, _frame: Any) -> None:
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    runner.run(once=args.once)


if __name__ == "__main__":
    main()
