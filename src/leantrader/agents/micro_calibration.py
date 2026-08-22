from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class MicroCalibrationJournal:
    """Prospective, non-executable labels for sub-minute assessments."""

    VERSION = "1.47.0"
    SCHEMA_VERSION = 1
    MAX_PENDING = 10_000
    MAX_RESOLVED = 50_000

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "pending": [],
            "resolved": [],
        }
        if not self.path.exists():
            return empty
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if int(data.get("schema_version") or 0) == self.SCHEMA_VERSION:
                data.setdefault("pending", [])
                data.setdefault("resolved", [])
                return data
        except Exception:
            pass
        return empty

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        tmp.write_text(
            json.dumps(self.state, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def register(
        self,
        *,
        symbol: str,
        midpoint: float,
        assessments: list[dict[str, Any]],
        observed_at: float | None = None,
    ) -> int:
        now = float(observed_at or time.time())
        midpoint = _finite(midpoint)
        if midpoint <= 0:
            return 0

        added = 0
        with self._lock:
            pending = list(self.state.get("pending") or [])
            existing = {
                str(row.get("assessment_id"))
                for row in pending
                if isinstance(row, dict)
            }

            for raw in assessments:
                if not isinstance(raw, dict):
                    continue

                horizon = int(raw.get("horizon_seconds") or 0)
                direction = str(raw.get("direction") or "")
                if horizon not in {5, 15, 30, 60}:
                    continue
                if direction not in {"long", "short", "flat"}:
                    continue

                canonical = (
                    f"{symbol}|{now:.6f}|{horizon}|{direction}|"
                    f"{raw.get('pressure_score')}|{midpoint:.12f}"
                )
                assessment_id = (
                    "micro-cal-"
                    + hashlib.sha256(canonical.encode()).hexdigest()[:24]
                )
                if assessment_id in existing:
                    continue

                pending.append(
                    {
                        "assessment_id": assessment_id,
                        "registered_at": now,
                        "due_at": now + horizon,
                        "symbol": str(symbol).upper(),
                        "horizon_seconds": horizon,
                        "direction": direction,
                        "entry_midpoint": midpoint,
                        "confidence": _finite(raw.get("confidence")),
                        "pressure_score": _finite(raw.get("pressure_score")),
                        "expected_edge_bps": _finite(
                            raw.get("expected_edge_bps")
                        ),
                        "modeled_round_trip_cost_bps": max(
                            30.0,
                            _finite(
                                raw.get("modeled_round_trip_cost_bps"),
                                30.0,
                            ),
                        ),
                        "originally_qualified": (
                            raw.get("independently_qualified") is True
                        ),
                        "original_reason": str(
                            raw.get("reason") or "unknown"
                        ),
                        "specialist": str(
                            raw.get("specialist") or "unknown"
                        ),
                        "regime": str(raw.get("regime") or "unknown"),
                        "automatic_promotion": False,
                        "execution_authority": False,
                        "testnet_authority": False,
                        "live_authority": False,
                    }
                )
                existing.add(assessment_id)
                added += 1

            self.state["pending"] = pending[-self.MAX_PENDING :]
            if added:
                self._save()

        return added

    def resolve(
        self,
        *,
        marks: dict[str, float],
        observed_at: float | None = None,
    ) -> int:
        now = float(observed_at or time.time())
        resolved_count = 0

        with self._lock:
            remaining: list[dict[str, Any]] = []
            resolved = list(self.state.get("resolved") or [])

            for row in self.state.get("pending") or []:
                if not isinstance(row, dict):
                    continue

                if now < _finite(row.get("due_at")):
                    remaining.append(row)
                    continue

                symbol = str(row.get("symbol") or "").upper()
                exit_mid = _finite(marks.get(symbol))
                entry_mid = _finite(row.get("entry_midpoint"))

                # Do not fabricate an outcome if the future mark was not seen.
                if exit_mid <= 0 or entry_mid <= 0:
                    remaining.append(row)
                    continue

                raw_return_bps = (
                    (exit_mid - entry_mid) / entry_mid * 10_000.0
                )
                direction = str(row.get("direction") or "")
                if direction == "long":
                    directional_bps = raw_return_bps
                elif direction == "short":
                    directional_bps = -raw_return_bps
                else:
                    directional_bps = 0.0

                cost = max(
                    30.0,
                    _finite(
                        row.get("modeled_round_trip_cost_bps"),
                        30.0,
                    ),
                )

                outcome = {
                    **row,
                    "resolved_at": now,
                    "exit_midpoint": exit_mid,
                    "raw_return_bps": raw_return_bps,
                    "directional_return_bps": directional_bps,
                    "net_after_modeled_cost_bps": directional_bps - cost,
                    "direction_correct": directional_bps > 0,
                    "cost_clearing": directional_bps > cost,
                    "prospective_label": True,
                    "is_trade": False,
                    "automatic_promotion": False,
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                }
                resolved.append(outcome)
                resolved_count += 1

            self.state["pending"] = remaining[-self.MAX_PENDING :]
            self.state["resolved"] = resolved[-self.MAX_RESOLVED :]

            if resolved_count:
                self._save()

        return resolved_count

    @staticmethod
    def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "samples": 0,
                "directional_accuracy": 0.0,
                "cost_clear_rate": 0.0,
                "average_directional_bps": 0.0,
                "average_net_after_cost_bps": 0.0,
            }

        return {
            "samples": len(rows),
            "directional_accuracy": (
                sum(row.get("direction_correct") is True for row in rows)
                / len(rows)
            ),
            "cost_clear_rate": (
                sum(row.get("cost_clearing") is True for row in rows)
                / len(rows)
            ),
            "average_directional_bps": (
                sum(_finite(row.get("directional_return_bps")) for row in rows)
                / len(rows)
            ),
            "average_net_after_cost_bps": (
                sum(
                    _finite(row.get("net_after_modeled_cost_bps"))
                    for row in rows
                )
                / len(rows)
            ),
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            pending = [
                dict(row)
                for row in self.state.get("pending") or []
                if isinstance(row, dict)
            ]
            resolved = [
                dict(row)
                for row in self.state.get("resolved") or []
                if isinstance(row, dict)
            ]

        by_horizon: dict[str, dict[str, Any]] = {}
        for horizon in (5, 15, 30, 60):
            rows = [
                row
                for row in resolved
                if int(row.get("horizon_seconds") or 0) == horizon
            ]
            by_horizon[str(horizon)] = self._metrics(rows)

        return {
            "version": self.VERSION,
            "pending_labels": len(pending),
            "resolved_labels": len(resolved),
            "metrics": self._metrics(resolved),
            "by_horizon": by_horizon,
            "minimum_evidence_before_calibration": 100,
            "labels_are_not_trades": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
