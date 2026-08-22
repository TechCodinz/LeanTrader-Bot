from __future__ import annotations

import argparse
import json
import signal
from typing import Any

from . import runner_v142 as _runner_v142
from .runner_v142 import *  # noqa: F401,F403
from .runner_v142 import (
    MarketFeed,
    PaperRunner as _V142PaperRunner,
    configure_logging,
    preflight,
)
from .settings import Settings
from ..agents.fast_path import FastSwarmRuntime
from ..agents.swarm_service import ReadOnlySwarmService


class PaperRunner(_V142PaperRunner):
    """v1.43 runner: v1.42 supervision plus parallel market-swarm scouting.

    The v1.42/v1.41 execution and evidence path is preserved. A separate public
    read-only MarketFeed drives the faster swarm observer service, preventing
    market scouting from waiting behind the slower intelligence cycle. The fast
    service has no order, Testnet or live authority.
    """

    VERSION = "1.43.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Preserve the public runner test/integration seam through every frozen
        # runner layer without weakening Testnet secret enforcement.
        _runner_v142.BybitTestnetExecutionEngine = BybitTestnetExecutionEngine
        super().__init__(*args, **kwargs)
        self.fast_swarm_service: ReadOnlySwarmService | None = None

    def _build_fast_swarm_service(self) -> ReadOnlySwarmService:
        # Dedicated feed: no credentials, no order methods, independent of the
        # slow production runner's network cadence/cache.
        dedicated_feed = MarketFeed(self.settings.exchange)
        runtime = FastSwarmRuntime(
            fee_bps=2.0 * self.settings.fee_bps,
            slippage_bps=2.0 * self.settings.slippage_bps,
            adverse_selection_bps=0.0,
            max_ranked_opportunities=max(4, min(24, self.settings.market_scan_batch_size)),
            max_observer_symbols=max(2, min(8, self.settings.max_open_positions * 2)),
        )
        cadence = max(5.0, min(15.0, float(self.settings.poll_seconds) / 4.0))
        return ReadOnlySwarmService(
            feed=dedicated_feed,
            runtime=runtime,
            market_quote=self.settings.market_quote,
            min_quote_volume_usd=self.settings.market_min_quote_volume_usd,
            max_spread_bps=self.settings.market_max_spread_bps,
            scan_batch_size=max(1, self.settings.market_scan_batch_size),
            candle_limit=max(48, min(120, self.settings.candle_limit)),
            cadence_seconds=cadence,
            discovery_refresh_seconds=max(60.0, min(300.0, float(self.settings.market_refresh_seconds))),
            timeframe="1m",
            timeframe_seconds=60.0,
        )

    def start_fast_swarm(self) -> None:
        if self.fast_swarm_service is None:
            self.fast_swarm_service = self._build_fast_swarm_service()
        self.fast_swarm_service.start()

    def stop_fast_swarm(self) -> None:
        if self.fast_swarm_service is not None:
            self.fast_swarm_service.stop()

    @staticmethod
    def _inactive_swarm_status() -> dict[str, Any]:
        return {
            "version": "1.43.0",
            "configured": True,
            "running": False,
            "reason": "parallel_service_starts_with_continuous_runner",
            "cadence_role": "seconds_to_minutes",
            "movement_only_can_allocate_capital": False,
            "requires_independent_agent_qualification": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def cycle(self) -> dict[str, Any]:
        status = super().cycle()
        service = self.fast_swarm_service
        if service is None:
            swarm_health = self._inactive_swarm_status()
        else:
            swarm_health = service.health(equity=float(status.get("equity") or self.settings.starting_cash))
            swarm_health["configured"] = True
        status["market_swarm"] = swarm_health
        status["market_swarm"]["slow_control_plane_blocking_fast_scout"] = False
        status["market_swarm"]["supervisory_evidence_version"] = "1.42"
        status["market_swarm"]["automatic_promotion"] = False
        status["market_swarm"]["execution_authority"] = False
        status["market_swarm"]["testnet_authority"] = False
        status["market_swarm"]["live_authority"] = False
        self._write_json_atomic(self.settings.heartbeat_path, status)
        return status

    def run(self, once: bool = False) -> None:
        # Continuous production-paper runtime gets the parallel service. One-shot
        # deterministic/smoke cycles remain network-bounded to the canonical
        # runner unless the service is explicitly started by the caller.
        if not once:
            self.start_fast_swarm()
        try:
            super().run(once=once)
        finally:
            self.stop_fast_swarm()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeanTrader v1.43 paper runner with evidence-qualified parallel market swarm"
    )
    parser.add_argument("--once", action="store_true", help="run one canonical market cycle and exit")
    parser.add_argument("--preflight", action="store_true", help="validate safe configuration without network access")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    if args.preflight:
        payload = preflight(settings)
        payload["market_swarm"] = {
            "version": "1.43.0",
            "parallel_read_only_service": True,
            "timeframe": "1m",
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
        print(json.dumps(payload, indent=2))
        return

    runner = PaperRunner(settings, MarketFeed(settings.exchange))

    def request_stop(_signum: int, _frame: Any) -> None:
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    runner.run(once=args.once)


if __name__ == "__main__":
    main()
