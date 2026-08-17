from __future__ import annotations

import json
import time

from leantrader.production.active_research import ActiveResearchPlanner
from leantrader.production.evolution_fabric import EvolutionFabric


def make_pack(now: float, *, pack_id: str = "macro-sidecar", score: float = 0.4) -> dict:
    return {
        "schema_version": 1,
        "pack_id": pack_id,
        "version": "1.0.0",
        "producer": "bounded-test-sidecar",
        "generated_at": now,
        "expires_at": now + 1800,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "can_add_credentials": False,
        "capabilities": ["rates_fx_cross_asset"],
        "observations": [
            {
                "symbol": "BTC/USDT",
                "kind": "signal",
                "score": score,
                "confidence": 0.8,
                "source": "test-public-source",
                "provenance": "deterministic fixture",
                "observed_at": now,
                "horizon_seconds": 60,
            }
        ],
    }


def test_hot_pack_is_ingested_without_execution_authority(tmp_path):
    now = time.time()
    inbox = tmp_path / "evolution" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "macro.json").write_text(json.dumps(make_pack(now)), encoding="utf-8")
    fabric = EvolutionFabric(
        tmp_path / "state.json",
        inbox,
        max_pack_age_seconds=3600,
        minimum_shadow_samples=10,
    )
    fabric.start()
    snap = fabric.refresh(prices={"BTC/USDT": 100.0})
    assert snap["active_packs"] == 1
    assert snap["hot_reload_supported"] is True
    assert snap["core_restart_required_for_new_pack"] is False
    assert snap["execution_authority"] is False
    assert snap["can_increase_upstream_risk"] is False
    assert snap["arbitrary_code_execution"] is False
    assert snap["capabilities"]["rates_fx_cross_asset"]["status"] == "available_external_shadow"


def test_pack_requesting_authority_is_quarantined(tmp_path):
    now = time.time()
    inbox = tmp_path / "evolution" / "inbox"
    inbox.mkdir(parents=True)
    pack = make_pack(now)
    pack["execution_authority"] = True
    (inbox / "unsafe.json").write_text(json.dumps(pack), encoding="utf-8")
    fabric = EvolutionFabric(tmp_path / "state.json", inbox)
    fabric.start()
    snap = fabric.refresh(prices={"BTC/USDT": 100.0})
    assert snap["active_packs"] == 0
    assert snap["quarantined_packs"] == 1


def test_stale_pack_is_quarantined(tmp_path):
    now = time.time()
    inbox = tmp_path / "evolution" / "inbox"
    inbox.mkdir(parents=True)
    pack = make_pack(now - 7200)
    pack["expires_at"] = now + 100
    (inbox / "stale.json").write_text(json.dumps(pack), encoding="utf-8")
    fabric = EvolutionFabric(tmp_path / "state.json", inbox, max_pack_age_seconds=3600)
    fabric.start()
    snap = fabric.refresh(prices={"BTC/USDT": 100.0})
    assert snap["active_packs"] == 0
    assert snap["quarantined_packs"] == 1


def test_shadow_signal_is_costed_and_never_promoted_to_execution(tmp_path, monkeypatch):
    base = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: base)
    inbox = tmp_path / "evolution" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "challenger.json").write_text(json.dumps(make_pack(base, pack_id="challenger")), encoding="utf-8")
    fabric = EvolutionFabric(
        tmp_path / "state.json",
        inbox,
        max_pack_age_seconds=3600,
        minimum_shadow_samples=10,
        round_trip_cost_bps=30.0,
    )
    fabric.start()
    snap = fabric.refresh(prices={"BTC/USDT": 100.0})
    assert snap["pending_shadow_episodes"] == 1
    monkeypatch.setattr(time, "time", lambda: base + 61)
    snap = fabric.refresh(prices={"BTC/USDT": 101.0})
    metrics = snap["shadow_metrics"]["challenger"]
    assert metrics["samples"] == 1
    assert 0 < metrics["average_net_return"] < 0.01
    assert metrics["execution_authority"] is False
    assert metrics["can_enable_live"] is False
    assert metrics["can_increase_risk"] is False


def test_research_demand_exports_machine_readable_sidecar_requests(tmp_path):
    fabric = EvolutionFabric(tmp_path / "state.json", tmp_path / "evolution" / "inbox")
    fabric.start()
    payload = fabric.sync_research_demand(
        adapter_backlog=[
            {"source": "macro_calendar", "requests": 12, "max_priority": 0.9, "description": "macro"},
        ],
        research_agenda=[{"task_id": "a", "question": "What changes?", "priority": 0.9}],
        world_market={"rare_scope_symbols": 1},
    )
    assert payload["desired_capabilities"][0]["capability"] == "macro_calendar"
    assert payload["sidecar_contract"]["hot_reload"] is True
    assert payload["sidecar_contract"]["core_restart_required"] is False
    assert fabric.requests_path.exists()


def test_active_research_accepts_external_shadow_capability_as_satisfied(tmp_path):
    planner = ActiveResearchPlanner(tmp_path / "research.json")
    world = {
        "latent_patterns": ["narrative_price_divergence"],
        "unknowns": [],
        "timeframe_coverage": 1.0,
        "senses": {},
    }
    result = planner.plan_symbol(
        symbol="BTC/USDT",
        world=world,
        self_model={"unknowns": [], "uncertainty": 0.1},
        council={"disagreement": 0.1},
        critic={},
        hypotheses={},
        engine_health={
            "market_data": {"healthy": True},
            "memory_retention": {"healthy": True},
            "strategy_observatory": {"healthy": True},
        },
        public_context_health={"market_data_fresh": True, "news_fresh": True},
        arbitrage={"available": True},
        sensor_snapshot={"source_status": {}},
        external_capabilities={
            "rates_fx_cross_asset": {"status": "available_external_shadow", "pack_id": "macro"},
            "macro_calendar": {"status": "available_external_shadow", "pack_id": "macro"},
        },
    )
    assert result["source_status"]["rates_fx_cross_asset"] == "available_external_shadow"
    assert "rates_fx_cross_asset" not in result["missing_adapters"]
    assert "rates_fx_cross_asset" not in result["degraded_sources"]
