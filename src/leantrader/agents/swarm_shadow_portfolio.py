from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from .swarm_evidence import strategy_name


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass
class ShadowTranche:
    tranche_id: str
    agent_id: str
    role: str
    timeframe: str
    symbol: str
    side: str
    entry_reference_price: float
    entry_fill_price: float
    quantity: float
    notional: float
    entry_fee: float
    confidence: float
    expected_edge_bps: float
    modeled_round_trip_cost_bps: float
    target_net_bps: float
    stop_net_bps: float
    max_holding_seconds: float
    opened_at: float
    regime: str = "unknown"
    last_mark_price: float | None = None

    @property
    def strategy(self) -> str:
        return strategy_name(self.role, self.timeframe)


class SwarmShadowPortfolio:
    """Persistent paper-shadow account for the fast swarm.

    This account is deliberately separate from the canonical PaperLedger. It is
    protected by one process-local lock, persists atomically, models entry/exit
    fee and slippage, and emits closed net-of-cost outcomes without any order or
    Testnet/live authority.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    HORIZON_SECONDS = {
        "1m": 12 * 60.0,
        "5m": 8 * 5 * 60.0,
        "15m": 6 * 15 * 60.0,
        "1h": 4 * 60 * 60.0,
        "4h": 3 * 4 * 60 * 60.0,
    }

    def __init__(
        self,
        state_path: Path,
        *,
        starting_equity: float,
        fee_bps: float,
        slippage_bps: float,
        principal_floor_fraction: float = 0.70,
        profit_reinvest_fraction: float = 0.50,
    ) -> None:
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if fee_bps < 0 or slippage_bps < 0:
            raise ValueError("shadow costs cannot be negative")
        if not 0.0 <= principal_floor_fraction <= 1.0:
            raise ValueError("principal_floor_fraction must be in [0, 1]")
        if not 0.0 <= profit_reinvest_fraction <= 1.0:
            raise ValueError("profit_reinvest_fraction must be in [0, 1]")
        self.state_path = state_path
        self.starting_equity = float(starting_equity)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.principal_floor_fraction = float(principal_floor_fraction)
        self.profit_reinvest_fraction = float(profit_reinvest_fraction)
        self._lock = threading.RLock()
        self.state = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "starting_equity": self.starting_equity,
            "realized_pnl": 0.0,
            "locked_profit": 0.0,
            "peak_equity": self.starting_equity,
            "max_drawdown": 0.0,
            "closed_tranches": 0,
            "open": {},
            "last_close": None,
        }

    def _load(self) -> dict[str, Any]:
        empty = self._empty()
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) != self.SCHEMA_VERSION:
                return empty
            if abs(_finite(payload.get("starting_equity")) - self.starting_equity) > 1e-9:
                return empty
            payload.setdefault("realized_pnl", 0.0)
            payload.setdefault("locked_profit", 0.0)
            payload.setdefault("peak_equity", self.starting_equity)
            payload.setdefault("max_drawdown", 0.0)
            payload.setdefault("closed_tranches", 0)
            payload.setdefault("open", {})
            payload.setdefault("last_close", None)
            return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)

    @staticmethod
    def _side(side: str) -> str:
        side = str(side).lower()
        if side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        return side

    def _entry_fill(self, reference_price: float, side: str) -> float:
        slip = self.slippage_bps / 10_000.0
        return reference_price * (1.0 + slip if side == "long" else 1.0 - slip)

    def _exit_fill(self, reference_price: float, side: str) -> float:
        slip = self.slippage_bps / 10_000.0
        return reference_price * (1.0 - slip if side == "long" else 1.0 + slip)

    def _open_records(self) -> list[ShadowTranche]:
        rows: list[ShadowTranche] = []
        for payload in (self.state.get("open") or {}).values():
            if isinstance(payload, dict):
                rows.append(ShadowTranche(**payload))
        return rows

    def open_symbols(self) -> set[str]:
        with self._lock:
            return {row.symbol for row in self._open_records()}

    def has_open_agent(self, agent_id: str) -> bool:
        with self._lock:
            return any(row.agent_id == str(agent_id) for row in self._open_records())

    def open_notional(self) -> float:
        with self._lock:
            return sum(row.notional for row in self._open_records())

    def _unrealized_pnl(self, marks: Mapping[str, float] | None = None) -> float:
        marks = marks or {}
        total = 0.0
        for row in self._open_records():
            mark = _finite(marks.get(row.symbol), _finite(row.last_mark_price, row.entry_reference_price))
            if mark <= 0:
                mark = row.entry_reference_price
            fill = self._exit_fill(mark, row.side)
            direction = 1.0 if row.side == "long" else -1.0
            gross = direction * (fill - row.entry_fill_price) * row.quantity
            exit_fee = row.quantity * fill * self.fee_bps / 10_000.0
            total += gross - row.entry_fee - exit_fee
        return total

    def equity(self, marks: Mapping[str, float] | None = None) -> float:
        with self._lock:
            value = self.starting_equity + _finite(self.state.get("realized_pnl")) + self._unrealized_pnl(marks)
            peak = max(_finite(self.state.get("peak_equity"), self.starting_equity), value)
            self.state["peak_equity"] = peak
            drawdown = 0.0 if peak <= 0 else max(0.0, (peak - value) / peak)
            self.state["max_drawdown"] = max(_finite(self.state.get("max_drawdown")), drawdown)
            return value

    def remaining_deployable_notional(self, marks: Mapping[str, float] | None = None) -> float:
        with self._lock:
            equity = max(0.0, self.equity(marks))
            base_floor = self.starting_equity * self.principal_floor_fraction
            protected = min(equity, base_floor + max(0.0, _finite(self.state.get("locked_profit"))))
            deployable = max(0.0, equity - protected)
            return max(0.0, deployable - self.open_notional())

    def open_tranche(
        self,
        *,
        tranche_id: str,
        agent_id: str,
        role: str,
        timeframe: str,
        symbol: str,
        side: str,
        reference_price: float,
        notional: float,
        confidence: float,
        expected_edge_bps: float,
        modeled_round_trip_cost_bps: float,
        marks: Mapping[str, float] | None = None,
        regime: str = "unknown",
    ) -> dict[str, Any]:
        side = self._side(side)
        reference_price = float(reference_price)
        notional = float(notional)
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError("reference_price must be positive and finite")
        if not math.isfinite(notional) or notional <= 0:
            raise ValueError("notional must be positive and finite")
        with self._lock:
            if tranche_id in (self.state.get("open") or {}):
                raise ValueError("shadow tranche is already open")
            available = self.remaining_deployable_notional(marks)
            if notional > available + 1e-9:
                raise ValueError("shadow tranche exceeds protected deployable capital")
            entry_fill = self._entry_fill(reference_price, side)
            quantity = notional / entry_fill
            entry_fee = notional * self.fee_bps / 10_000.0
            modeled_cost = max(30.0, float(modeled_round_trip_cost_bps))
            expected_edge = max(modeled_cost, float(expected_edge_bps))
            target = max(modeled_cost * 1.25, expected_edge * 0.75)
            stop = max(modeled_cost * 0.75, min(target * 0.75, expected_edge * 0.50))
            horizon = float(self.HORIZON_SECONDS.get(str(timeframe), 60.0 * 60.0))
            record = ShadowTranche(
                tranche_id=str(tranche_id),
                agent_id=str(agent_id),
                role=str(role),
                timeframe=str(timeframe),
                symbol=str(symbol).upper(),
                side=side,
                entry_reference_price=reference_price,
                entry_fill_price=entry_fill,
                quantity=quantity,
                notional=notional,
                entry_fee=entry_fee,
                confidence=max(0.0, min(1.0, float(confidence))),
                expected_edge_bps=expected_edge,
                modeled_round_trip_cost_bps=modeled_cost,
                target_net_bps=target,
                stop_net_bps=stop,
                max_holding_seconds=horizon,
                opened_at=time.time(),
                regime=str(regime)[:80],
                last_mark_price=reference_price,
            )
            self.state.setdefault("open", {})[record.tranche_id] = asdict(record)
            self._save()
            return asdict(record)

    def preview_net_return(self, tranche_id: str, *, reference_price: float) -> dict[str, float]:
        with self._lock:
            payload = (self.state.get("open") or {}).get(str(tranche_id))
            if not isinstance(payload, dict):
                raise KeyError(f"unknown shadow tranche: {tranche_id}")
            row = ShadowTranche(**payload)
            exit_fill = self._exit_fill(float(reference_price), row.side)
            direction = 1.0 if row.side == "long" else -1.0
            gross = direction * (exit_fill - row.entry_fill_price) * row.quantity
            exit_fee = row.quantity * exit_fill * self.fee_bps / 10_000.0
            net_pnl = gross - row.entry_fee - exit_fee
            entry_cost = row.notional + row.entry_fee
            return {
                "exit_fill": exit_fill,
                "exit_fee": exit_fee,
                "net_pnl": net_pnl,
                "net_return": net_pnl / max(entry_cost, 1e-12),
                "net_bps": net_pnl / max(entry_cost, 1e-12) * 10_000.0,
            }

    def exit_candidates(
        self,
        *,
        marks: Mapping[str, float],
        timeframe_assessments: Mapping[str, Mapping[str, Any]] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        now = time.time() if now is None else float(now)
        assessments = timeframe_assessments or {}
        decisions: list[dict[str, Any]] = []
        with self._lock:
            for row in self._open_records():
                mark = _finite(marks.get(row.symbol))
                if mark <= 0:
                    continue
                payload = self.preview_net_return(row.tranche_id, reference_price=mark)
                reason: str | None = None
                assessment = (assessments.get(row.symbol) or {}).get(row.timeframe)
                if payload["net_bps"] >= row.target_net_bps:
                    reason = "shadow_take_profit"
                elif payload["net_bps"] <= -row.stop_net_bps:
                    reason = "shadow_stop"
                elif isinstance(assessment, dict):
                    if assessment.get("independently_qualified") is True and str(assessment.get("direction") or "") not in {"", row.side}:
                        reason = "timeframe_reversal"
                if reason is None and now - row.opened_at >= row.max_holding_seconds:
                    reason = "precommitted_time_horizon"
                self.state["open"][row.tranche_id]["last_mark_price"] = mark
                if reason:
                    decisions.append({"tranche_id": row.tranche_id, "agent_id": row.agent_id, "symbol": row.symbol, "reference_price": mark, "reason": reason})
            if self.state.get("open"):
                self._save()
        return decisions

    def close_tranche(self, tranche_id: str, *, reference_price: float, reason: str) -> dict[str, Any]:
        with self._lock:
            payload = (self.state.get("open") or {}).get(str(tranche_id))
            if not isinstance(payload, dict):
                raise KeyError(f"unknown shadow tranche: {tranche_id}")
            row = ShadowTranche(**payload)
            result = self.preview_net_return(row.tranche_id, reference_price=float(reference_price))
            closed_at = time.time()
            self.state["realized_pnl"] = _finite(self.state.get("realized_pnl")) + result["net_pnl"]
            if result["net_pnl"] > 0:
                self.state["locked_profit"] = _finite(self.state.get("locked_profit")) + result["net_pnl"] * (1.0 - self.profit_reinvest_fraction)
            self.state["closed_tranches"] = int(self.state.get("closed_tranches") or 0) + 1
            self.state.get("open", {}).pop(row.tranche_id, None)
            episode = {
                "strategy": row.strategy,
                "agent_id": row.agent_id,
                "symbol": row.symbol,
                "regime": row.regime,
                "role": row.role,
                "timeframe": row.timeframe,
                "side": row.side,
                "opened_at": row.opened_at,
                "closed_at": closed_at,
                "entry_reference_price": row.entry_reference_price,
                "exit_reference_price": float(reference_price),
                "entry_fill_price": row.entry_fill_price,
                "exit_fill_price": result["exit_fill"],
                "notional": row.notional,
                "net_pnl": result["net_pnl"],
                "net_return": result["net_return"],
                "round_trip_cost_bps_floor": row.modeled_round_trip_cost_bps,
                "fee_bps_per_side": self.fee_bps,
                "slippage_bps_per_side": self.slippage_bps,
                "confidence": row.confidence,
                "expected_edge_bps": row.expected_edge_bps,
                "exit_reason": str(reason),
                "evidence_interval_complete": True,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
            self.state["last_close"] = dict(episode)
            self._save()
            return episode

    def health(self, marks: Mapping[str, float] | None = None) -> dict[str, Any]:
        with self._lock:
            current_equity = self.equity(marks)
            return {
                "version": self.VERSION,
                "starting_equity": self.starting_equity,
                "equity": current_equity,
                "realized_pnl": _finite(self.state.get("realized_pnl")),
                "locked_profit": _finite(self.state.get("locked_profit")),
                "open_tranches": len(self.state.get("open") or {}),
                "open_notional": self.open_notional(),
                "remaining_deployable_notional": self.remaining_deployable_notional(marks),
                "closed_tranches": int(self.state.get("closed_tranches") or 0),
                "peak_equity": _finite(self.state.get("peak_equity"), self.starting_equity),
                "max_drawdown": _finite(self.state.get("max_drawdown")),
                "principal_floor_fraction": self.principal_floor_fraction,
                "profit_reinvest_fraction": self.profit_reinvest_fraction,
                "fee_bps_per_side": self.fee_bps,
                "slippage_bps_per_side": self.slippage_bps,
                "persistent": True,
                "thread_safe_single_process": True,
                "canonical_paper_ledger_mutation": False,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
