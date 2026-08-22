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

    VERSION = "1.49.0"
    SCHEMA_VERSION = 1
    MAX_PENDING = 10_000
    MAX_RESOLVED = 50_000
    MAX_RESOLUTION_DELAY_SECONDS = 3.0

    def __init__(
        self,
        path: Path,
        *,
        accepted_horizons: tuple[int, ...] = (5, 15, 30, 60),
    ) -> None:
        horizons = tuple(
            sorted({
                int(value)
                for value in accepted_horizons
                if int(value) > 0
            })
        )
        if not horizons:
            raise ValueError("accepted_horizons cannot be empty")
        self.path = path
        self.accepted_horizons = horizons
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
                if horizon not in self.accepted_horizons:
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

    def due_symbols(
        self,
        *,
        observed_at: float | None = None,
        lookahead_seconds: float = 0.5,
        limit: int = 4,
    ) -> list[str]:
        now = float(observed_at or time.time())
        cutoff = now + max(0.0, float(lookahead_seconds))

        with self._lock:
            rows = [
                row
                for row in self.state.get("pending") or []
                if isinstance(row, dict)
                and _finite(row.get("due_at")) <= cutoff
                and now - _finite(row.get("due_at"))
                    <= self.MAX_RESOLUTION_DELAY_SECONDS
            ]

        rows.sort(key=lambda row: _finite(row.get("due_at")))

        symbols: list[str] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
                if len(symbols) >= max(1, int(limit)):
                    break
        return symbols

    def censor_expired(
        self,
        *,
        observed_at: float | None = None,
    ) -> int:
        now = float(observed_at or time.time())
        censored_count = 0

        with self._lock:
            remaining: list[dict[str, Any]] = []
            resolved = list(self.state.get("resolved") or [])

            for row in self.state.get("pending") or []:
                if not isinstance(row, dict):
                    continue

                due_at = _finite(row.get("due_at"))
                delay = now - due_at

                if delay <= self.MAX_RESOLUTION_DELAY_SECONDS:
                    remaining.append(row)
                    continue

                resolved.append(
                    {
                        **row,
                        "resolved_at": now,
                        "target_due_at": due_at,
                        "actual_observed_at": None,
                        "resolution_delay_seconds": max(0.0, delay),
                        "timing_valid": False,
                        "timing_censored": True,
                        "censor_reason": "missed_resolution_window",
                        "exit_midpoint": None,
                        "raw_return_bps": None,
                        "directional_return_bps": None,
                        "net_after_modeled_cost_bps": None,
                        "direction_correct": None,
                        "cost_clearing": None,
                        "prospective_label": True,
                        "is_trade": False,
                        "automatic_promotion": False,
                        "execution_authority": False,
                        "testnet_authority": False,
                        "live_authority": False,
                    }
                )
                censored_count += 1

            self.state["pending"] = remaining[-self.MAX_PENDING :]
            self.state["resolved"] = resolved[-self.MAX_RESOLVED :]

            if censored_count:
                self._save()

        return censored_count

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
                    "target_due_at": _finite(row.get("due_at")),
                    "actual_observed_at": now,
                    "resolution_delay_seconds": max(
                        0.0, now - _finite(row.get("due_at"))
                    ),
                    "timing_valid": (
                        max(0.0, now - _finite(row.get("due_at")))
                        <= self.MAX_RESOLUTION_DELAY_SECONDS
                    ),
                    "timing_censored": (
                        max(0.0, now - _finite(row.get("due_at")))
                        > self.MAX_RESOLUTION_DELAY_SECONDS
                    ),
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

    def evidence_rankings(
        self,
        *,
        minimum_samples: int = 30,
    ) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = [
                dict(row)
                for row in self.state.get("resolved") or []
                if isinstance(row, dict)
                and row.get("timing_valid") is True
                and str(row.get("original_reason") or "") not in {
                    "insufficient_depth",
                    "spread_too_wide",
                    "flat_micro_pressure",
                }
            ]

        groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

        for row in rows:
            key = (
                str(row.get("specialist") or "unknown"),
                int(row.get("horizon_seconds") or 0),
                str(row.get("regime") or "unknown"),
            )
            groups.setdefault(key, []).append(row)

        output: dict[str, dict[str, Any]] = {}

        for (specialist, horizon, regime), group in groups.items():
            values = [
                _finite(row.get("net_after_modeled_cost_bps"))
                for row in group
            ]
            n = len(values)
            if not values:
                continue

            mean = sum(values) / n

            if n > 1:
                variance = sum(
                    (value - mean) ** 2
                    for value in values
                ) / (n - 1)
                standard_error = math.sqrt(
                    max(0.0, variance) / n
                )
            else:
                standard_error = float("inf")

            conservative_net = (
                mean - 1.96 * standard_error
                if math.isfinite(standard_error)
                else float("-inf")
            )

            correct = sum(
                row.get("direction_correct") is True
                for row in group
            )
            cost_clear = sum(
                row.get("cost_clearing") is True
                for row in group
            )

            key = f"{specialist}|{horizon}|{regime}"

            evidence_qualified = (
                n >= max(1, int(minimum_samples))
                and mean > 0.0
                and conservative_net > 0.0
            )

            output[key] = {
                "specialist": specialist,
                "horizon_seconds": horizon,
                "regime": regime,
                "samples": n,
                "directional_accuracy": correct / n,
                "cost_clear_rate": cost_clear / n,
                "average_net_after_cost_bps": mean,
                "standard_error_net_bps": (
                    standard_error
                    if math.isfinite(standard_error)
                    else None
                ),
                "conservative_net_after_cost_bps": (
                    conservative_net
                    if math.isfinite(conservative_net)
                    else None
                ),
                "minimum_samples": max(
                    1,
                    int(minimum_samples),
                ),
                "evidence_qualified": evidence_qualified,
                "action": (
                    "shadow_candidate"
                    if evidence_qualified
                    else "no_trade"
                ),
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }

        return output

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
            timing_valid = [
                row for row in resolved
                if row.get("timing_valid") is True
            ]
            timing_censored = [
                row for row in resolved
                if row.get("timing_censored") is True
            ]
            legacy_unclassified = [
                row for row in resolved
                if "timing_valid" not in row
                and "timing_censored" not in row
            ]

        by_horizon: dict[str, dict[str, Any]] = {}
        for horizon in self.accepted_horizons:
            rows = [
                row
                for row in timing_valid
                if int(row.get("horizon_seconds") or 0) == horizon
            ]
            by_horizon[str(horizon)] = self._metrics(rows)

        rankings = self.evidence_rankings()
        eligible = [
            row for row in rankings.values()
            if row.get("evidence_qualified") is True
        ]

        return {
            "version": self.VERSION,
            "accepted_horizons_seconds": list(
                self.accepted_horizons
            ),
            "evidence_ranked_groups": len(rankings),
            "evidence_qualified_groups": len(eligible),
            "evidence_no_trade_groups": (
                len(rankings) - len(eligible)
            ),
            "pending_labels": len(pending),
            "resolved_labels": len(resolved),
            "metrics": self._metrics(timing_valid),
            "timing_valid_labels": len(timing_valid),
            "timing_censored_labels": len(timing_censored),
            "legacy_unclassified_labels": len(legacy_unclassified),
            "maximum_resolution_delay_seconds": (
                self.MAX_RESOLUTION_DELAY_SECONDS
            ),
            "by_horizon": by_horizon,
            "minimum_evidence_before_calibration": 100,
            "labels_are_not_trades": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }
