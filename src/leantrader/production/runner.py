from __future__ import annotations

import argparse
import json
import signal
from typing import Any

from . import runner_v142 as _runner_v142
from .runner_v142 import *  # noqa: F401,F403
from .runner_v142 import MarketFeed, PaperRunner as _V142PaperRunner, configure_logging, preflight
from .settings import Settings
from ..agents.capital_allocator import SwarmCapitalAllocator
from ..agents.fast_path import FastSwarmRuntime
from ..agents.shared_position_graph import PositionCoordinator
from ..agents.swarm_evidence import SwarmOutcomeJournal, build_v142_swarm_manifests
from ..agents.swarm_orchestrator import MarketSwarmOrchestrator
from ..agents.swarm_service import ReadOnlySwarmService
from ..agents.swarm_shadow_portfolio import SwarmShadowPortfolio


class PaperRunner(_V142PaperRunner):
    """v1.43: v1.42 supervision plus parallel costed market-swarm shadow evidence."""

    VERSION = "1.43.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _runner_v142.BybitTestnetExecutionEngine = BybitTestnetExecutionEngine
        super().__init__(*args, **kwargs)
        self.fast_swarm_service: ReadOnlySwarmService | None = None
        base = self.settings.strategy_observatory_state_path
        self.swarm_outcome_journal = SwarmOutcomeJournal(base.with_name("vps_market_swarm_outcomes.json"))
        self.swarm_manifests_registered = False
        self.last_swarm_evidence_ingest: dict[str, Any] = {
            "submitted": 0,
            "episodes_recorded": 0,
            "qualification_refresh": "not_started",
        }

    def _swarm_round_trip_cost_bps(self) -> float:
        return max(30.0, 2.0 * (self.settings.fee_bps + self.settings.slippage_bps))

    def _register_swarm_research_manifests(self) -> dict[str, Any]:
        manifests = build_v142_swarm_manifests(
            minimum_samples=self.settings.evolution_min_shadow_samples,
            round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
        )
        result = self.prospective_validation.observe_cycle(
            observatory_authority=self.prospective_validation.EVIDENCE_AUTHORITY,
            observed_round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
            strategy_episodes=[],
            foundry_manifests=manifests,
            market_rows={},
        )
        self.swarm_manifests_registered = True
        return {
            "requested": len(manifests),
            "registered_now": len(result.get("experiments_registered") or []),
            "prospective_only": True,
            "freeze_before_outcome": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def _build_fast_swarm_service(self) -> ReadOnlySwarmService:
        dedicated_feed = MarketFeed(self.settings.exchange)
        base = self.settings.strategy_observatory_state_path
        symbol_cap = max(0.01, min(0.20, float(self.settings.max_position_pct)))
        portfolio_cap = max(
            symbol_cap,
            min(0.70, float(self.settings.max_position_pct) * max(1, int(self.settings.max_open_positions))),
        )
        coordinator = PositionCoordinator(
            max_symbol_exposure_fraction=symbol_cap,
            max_portfolio_exposure_fraction=portfolio_cap,
            state_path=base.with_name("vps_market_swarm_positions.json"),
        )
        allocator = SwarmCapitalAllocator(
            coordinator,
            profit_reinvest_fraction=self.settings.capital_profit_reinvest_fraction,
        )
        swarm = MarketSwarmOrchestrator(coordinator=coordinator, allocator=allocator)
        runtime = FastSwarmRuntime(
            swarm=swarm,
            fee_bps=2.0 * self.settings.fee_bps,
            slippage_bps=2.0 * self.settings.slippage_bps,
            adverse_selection_bps=0.0,
            max_ranked_opportunities=max(4, min(24, self.settings.market_scan_batch_size)),
            max_observer_symbols=max(2, min(8, self.settings.max_open_positions * 2)),
        )
        shadow_portfolio = SwarmShadowPortfolio(
            base.with_name("vps_market_swarm_shadow_portfolio.json"),
            starting_equity=self.settings.starting_cash,
            fee_bps=self.settings.fee_bps,
            slippage_bps=self.settings.slippage_bps,
            principal_floor_fraction=self.settings.capital_principal_floor_fraction,
            profit_reinvest_fraction=self.settings.capital_profit_reinvest_fraction,
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
            shadow_portfolio=shadow_portfolio,
            outcome_journal=self.swarm_outcome_journal,
            base_order_usd=self.settings.order_usd,
        )

    def start_fast_swarm(self) -> None:
        if not self.swarm_manifests_registered:
            self._register_swarm_research_manifests()
        if self.fast_swarm_service is None:
            self.fast_swarm_service = self._build_fast_swarm_service()
        self.fast_swarm_service.start()

    def stop_fast_swarm(self) -> None:
        if self.fast_swarm_service is not None:
            self.fast_swarm_service.stop()

    def _ingest_swarm_outcomes(self) -> dict[str, Any]:
        episodes = self.swarm_outcome_journal.pending(limit=1000)
        if not episodes:
            self.last_swarm_evidence_ingest = {
                "submitted": 0,
                "episodes_recorded": 0,
                "qualification_refresh": "no_new_closed_outcomes",
                "journal": self.swarm_outcome_journal.health(),
            }
            return dict(self.last_swarm_evidence_ingest)
        if not self.swarm_manifests_registered:
            self._register_swarm_research_manifests()
        result = self.prospective_validation.observe_cycle(
            observatory_authority=self.prospective_validation.EVIDENCE_AUTHORITY,
            observed_round_trip_cost_bps=self._swarm_round_trip_cost_bps(),
            strategy_episodes=episodes,
            foundry_manifests=[],
            market_rows={},
        )
        ids = [str(row.get("episode_id")) for row in episodes if row.get("episode_id")]
        acknowledged = self.swarm_outcome_journal.acknowledge(ids)
        self.last_swarm_evidence_ingest = {
            "submitted": len(episodes),
            "episodes_recorded": int(result.get("episodes_recorded") or 0),
            "acknowledged": acknowledged,
            "overlapping_or_ineligible_episodes_are_not_retried": True,
            "qualification_refresh": "next_supervisory_cycle",
            "journal": self.swarm_outcome_journal.health(),
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
        return dict(self.last_swarm_evidence_ingest)

    @staticmethod
    def _inactive_swarm_status() -> dict[str, Any]:
        return {
            "version": "1.43.0",
            "configured": True,
            "running": False,
            "reason": "parallel_service_starts_with_continuous_runner",
            "cadence_role": "seconds_to_minutes",
            "shadow_portfolio": "separate_from_canonical_paper_ledger",
            "movement_only_can_allocate_capital": False,
            "requires_independent_agent_qualification": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def _write_health_state(self, status: dict[str, Any]) -> None:
        swarm = status.get("market_swarm") or {}
        swarm_required = self.fast_swarm_service is not None
        swarm_ok = bool(
            isinstance(swarm, dict)
            and swarm.get("running") is True
            and swarm.get("healthy") is True
            and swarm.get("stale") is not True
        )
        payload = {
            "timestamp": status.get("timestamp"),
            "runtime": status.get("runtime"),
            "healthy": bool(
                status.get("healthy") is True
                and (not swarm_required or swarm_ok)
            ),
            "errors": status.get("errors") or [],
            "testnet_execution": status.get("testnet_execution") or {},
            "market_swarm": {
                "required": swarm_required,
                "running": swarm.get("running") is True,
                "healthy": swarm.get("healthy") is True,
                "stale": swarm.get("stale") is True,
                "cycles": int(swarm.get("cycles") or 0),
                "consecutive_failures": int(swarm.get("consecutive_failures") or 0),
                "last_success_at": swarm.get("last_success_at"),
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            },
        }
        self._write_json_atomic(
            self.settings.heartbeat_path.with_name("vps_health_state.json"),
            payload,
        )

    def cycle(self) -> dict[str, Any]:
        status = super().cycle()
        swarm_evidence = self._ingest_swarm_outcomes() if self.fast_swarm_service is not None else {
            "submitted": 0,
            "episodes_recorded": 0,
            "qualification_refresh": "parallel_service_not_running",
            "journal": self.swarm_outcome_journal.health(),
        }
        service = self.fast_swarm_service
        if service is None:
            swarm_health = self._inactive_swarm_status()
        else:
            swarm_health = service.health(equity=float(status.get("equity") or self.settings.starting_cash))
            swarm_health["configured"] = True
        status["market_swarm"] = swarm_health
        status["market_swarm"]["swarm_evidence_ingest"] = swarm_evidence
        status["market_swarm"]["slow_control_plane_blocking_fast_scout"] = False
        status["market_swarm"]["supervisory_evidence_version"] = "1.42"
        status["market_swarm"]["canonical_paper_ledger_mutation_from_fast_thread"] = False
        status["market_swarm"]["automatic_promotion"] = False
        status["market_swarm"]["execution_authority"] = False
        status["market_swarm"]["testnet_authority"] = False
        status["market_swarm"]["live_authority"] = False

        engines = status.setdefault("engines", {})
        if isinstance(engines, dict) and service is not None:
            swarm_operational = bool(
                status["market_swarm"].get("running") is True
                and status["market_swarm"].get("healthy") is True
                and status["market_swarm"].get("stale") is not True
            )
            engines["market_swarm"] = {
                "required": True,
                "healthy": swarm_operational,
                "state": "running" if swarm_operational else "degraded",
                "failures": int(status["market_swarm"].get("consecutive_failures") or 0),
            }
            portfolio = status["market_swarm"].get("shadow_portfolio")
            engines["swarm_shadow_portfolio"] = {
                "required": True,
                "healthy": isinstance(portfolio, dict),
                "state": "running" if isinstance(portfolio, dict) else "missing",
                "failures": 0 if isinstance(portfolio, dict) else 1,
            }
            journal = status["market_swarm"].get("swarm_outcome_journal")
            engines["swarm_outcome_journal"] = {
                "required": True,
                "healthy": isinstance(journal, dict),
                "state": "running" if isinstance(journal, dict) else "missing",
                "failures": 0 if isinstance(journal, dict) else 1,
            }

        self._write_json_atomic(self.settings.heartbeat_path, status)
        self._write_health_state(status)
        return status

    def run(self, once: bool = False) -> None:
        if not once:
            self.start_fast_swarm()
        try:
            super().run(once=once)
        finally:
            self.stop_fast_swarm()


def main() -> None:
    parser = argparse.ArgumentParser(description="LeanTrader v1.43 paper runner with evidence-qualified parallel market swarm")
    parser.add_argument("--once", action="store_true", help="run one canonical market cycle and exit")
    parser.add_argument("--preflight", action="store_true", help="validate safe configuration without network access")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    if args.preflight:
        payload = preflight(settings)
        payload["market_swarm"] = {
            "version": "1.43.0",
            "parallel_read_only_market_feed": True,
            "separate_persistent_shadow_portfolio": True,
            "closed_outcome_evidence_journal": True,
            "canonical_paper_ledger_mutation_from_fast_thread": False,
            "timeframe": "1m",
            "modeled_round_trip_cost_floor_bps": max(30.0, 2.0 * (settings.fee_bps + settings.slippage_bps)),
            "prospective_minimum_samples": max(100, settings.evolution_min_shadow_samples),
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
