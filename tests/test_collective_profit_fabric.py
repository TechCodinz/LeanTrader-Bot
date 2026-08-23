from __future__ import annotations

import threading
import time

from leantrader.agents.swarm_service import ReadOnlySwarmService
from leantrader.production.decision_router import (
    BoundedDecisionRouter,
    MarketEvidenceGate,
)
from leantrader.production.evolution_fabric import EvolutionFabric
from leantrader.production.unified_control_plane_v141 import (
    UnifiedDecisionControlPlane,
)
from leantrader.production.ledger import Position
from leantrader.production.runner_v141 import (
    _collective_route_reversal,
    _legacy_trend_exit_applies,
    _position_entry_origin,
)


def router(tmp_path):
    return BoundedDecisionRouter(
        MarketEvidenceGate(
            tmp_path / "router.json",
            minimum_samples=3,
            rolling_window=10,
        ),
        minimum_advanced_confidence=0.20,
        minimum_combined_score=0.20,
        negative_consensus_veto=-0.25,
    )


def strong_advanced():
    return {
        "swarm": {
            "score": 0.40,
            "confidence": 0.70,
        },
        "signals": [
            {
                "engine": "a",
                "score": 0.4,
                "confidence": 0.6,
            },
            {
                "engine": "b",
                "score": 0.4,
                "confidence": 0.6,
            },
            {
                "engine": "c",
                "score": 0.4,
                "confidence": 0.6,
            },
        ],
        "news_blackout": False,
    }


def test_legacy_no_collective_path_remains_compatible(tmp_path):
    result = router(tmp_path).route(
        symbol="BTC/USDT",
        base_enter=False,
        base_score=0.40,
        base_confidence=0.80,
        advanced=strong_advanced(),
    )

    assert result["allowed"] is False
    assert result["reason"] == "adaptive_signal_not_ready"
    assert result["entry_origin"] == "none"
    assert result["live_authority"] is False


def test_collective_costed_engines_can_originate_testnet_route(tmp_path):
    advanced = {
        "swarm": {
            "score": 0.10,
            "confidence": 0.40,
        },
        "signals": [
            {
                "engine": "technical_structure",
                "score": 0.10,
                "confidence": 0.40,
            }
        ],
        "news_blackout": False,
    }

    collective = {
        "fast_swarm": {
            "fresh": True,
            "ranked_opportunity": {
                "qualified": True,
                "quality_multiplier": 0.80,
            },
            "timeframe_assessments": {
                "5m": {
                    "direction": "long",
                    "independently_qualified": True,
                    "confidence": 0.80,
                    "expected_edge_bps": 50.0,
                    "modeled_round_trip_cost_bps": 30.0,
                }
            },
            "micro_proposals": [],
        },
        "evolution_evidence": [
            {
                "kind": "signal",
                "pack_id": "mtf_5m_long_60m",
                "score": 0.70,
                "confidence": 0.80,
                "research_validated": True,
                "shadow_samples": 120,
                "average_net_return": 0.004,
                "ewma_net_return": 0.003,
            }
        ],
        "sensor_context": {},
        "arbitrage_quotes": [],
        "alpha_tournament": {},
    }

    result = router(tmp_path).route(
        symbol="BTC/USDT",
        base_enter=False,
        base_score=0.05,
        base_confidence=0.60,
        advanced=advanced,
        collective=collective,
    )

    assert result["allowed"] is True
    assert result["collective_origin_ready"] is True
    assert result["entry_origin"] == "collective_profit_fabric"
    assert result["paper_authority"] is True
    assert result["testnet_authority"] is True
    assert result["live_authority"] is False


def test_sensor_context_cannot_originate_without_costed_edge(tmp_path):
    collective = {
        "fast_swarm": {
            "fresh": False,
        },
        "evolution_evidence": [],
        "sensor_context": {
            "flow_intelligence": {
                "status": "available",
                "confidence": 0.90,
                "values": {
                    "flow_score": 0.90,
                },
            },
            "liquidations": {
                "status": "available",
                "confidence": 0.90,
                "values": {
                    "events": 20,
                    "liquidation_imbalance": 0.90,
                },
            },
        },
        "arbitrage_quotes": [],
        "alpha_tournament": {},
    }

    result = router(tmp_path).route(
        symbol="BTC/USDT",
        base_enter=False,
        base_score=0.0,
        base_confidence=0.5,
        advanced={
            "swarm": {
                "score": 0.0,
                "confidence": 0.5,
            },
            "signals": [],
            "news_blackout": False,
        },
        collective=collective,
    )

    assert result["allowed"] is False
    assert result["collective_origin_ready"] is False


def test_fast_swarm_collective_snapshot_is_thread_safe():
    service = ReadOnlySwarmService.__new__(
        ReadOnlySwarmService
    )
    service._lock = threading.RLock()
    service.cadence_seconds = 15.0
    service.last_success_at = time.time()
    service.cycles = 3
    service.last_step = {
        "ranked": [
            {
                "symbol": "BTC/USDT",
                "qualified": True,
                "quality_multiplier": 0.8,
            }
        ],
        "timeframe_assessments": {
            "BTC/USDT": {
                "5m": {
                    "independently_qualified": True,
                    "direction": "long",
                }
            }
        },
        "micro_agent_foundry_proposals": [
            {
                "symbol": "BTC/USDT",
                "evidence_qualified": True,
                "independently_qualified": True,
            }
        ],
        "microstructure": {
            "BTC/USDT": {
                "features": {
                    "midpoint": 100.0,
                }
            }
        },
    }

    result = service.collective_signal(
        "BTC/USDT"
    )

    assert result["fresh"] is True
    assert result["qualified_timeframe_paths"] == 1
    assert result["qualified_micro_proposals"] == 1
    assert result["canonical_router_input"] is True
    assert result["live_authority"] is False


def test_evolution_evidence_exposes_validated_shadow_economics(tmp_path):
    fabric = EvolutionFabric(
        tmp_path / "state.json",
        tmp_path / "inbox",
        minimum_shadow_samples=100,
        round_trip_cost_bps=30.0,
    )

    now = time.time()

    fabric.state = {
        "packs": {
            "winner": {
                "status": "active",
                "expires_at": now + 600,
                "version": "1",
                "producer": "test",
                "observations": [
                    {
                        "symbol": "BTC/USDT",
                        "kind": "signal",
                        "score": 0.7,
                        "confidence": 0.8,
                        "observed_at": now,
                    }
                ],
            }
        },
        "shadow_metrics": {
            "winner": {
                "samples": 120,
                "win_rate": 0.60,
                "average_net_return": 0.004,
                "ewma_net_return": 0.003,
                "research_validated": True,
            }
        },
    }

    row = fabric.evidence_for(
        "BTC/USDT"
    )[0]

    assert row["pack_id"] == "winner"
    assert row["research_validated"] is True
    assert row["shadow_samples"] == 120
    assert row["average_net_return"] == 0.004


def test_unified_control_plane_accepts_testnet_as_sandbox_state(tmp_path):
    plane = UnifiedDecisionControlPlane(
        tmp_path / "ucp.json"
    )

    reasons = plane._safety_reasons(
        {
            "trading_mode": "paper",
            "enable_live": False,
            "allow_live": False,
            "live_confirm": "NO",
            "testnet_enabled": True,
            "runtime_integrity_ok": True,
            "heartbeat_fresh": True,
        },
        {
            "equity": 50.0,
            "peak_equity": 50.0,
            "daily_pnl": 0.0,
            "loss_streak": 0,
            "volatility_ratio": 1.0,
        },
    )

    assert "testnet_execution_not_disabled" not in reasons
    assert "testnet_execution_state_invalid" not in reasons


def test_collective_origin_has_origin_aware_exit_contract():
    collective = Position(
        quantity=1.0,
        entry_price=100.0,
        peak_price=100.0,
        atr=1.0,
        metadata={
            "decision_route": {
                "entry_origin": "collective_profit_fabric",
            }
        },
    )

    legacy = Position(
        quantity=1.0,
        entry_price=100.0,
        peak_price=100.0,
        atr=1.0,
        metadata={},
    )

    assert (
        _position_entry_origin(collective)
        == "collective_profit_fabric"
    )
    assert _legacy_trend_exit_applies(collective) is False

    assert _position_entry_origin(legacy) == "adaptive"
    assert _legacy_trend_exit_applies(legacy) is True

    assert (
        _collective_route_reversal(
            {
                "collective_profit_fabric": {
                    "ensemble_score": 0.35,
                    "ensemble_confidence": 0.75,
                }
            }
        )
        is False
    )

    assert (
        _collective_route_reversal(
            {
                "collective_profit_fabric": {
                    "ensemble_score": -0.35,
                    "ensemble_confidence": 0.75,
                }
            }
        )
        is True
    )
