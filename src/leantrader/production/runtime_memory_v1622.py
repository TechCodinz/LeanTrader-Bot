from __future__ import annotations

from pathlib import Path
from typing import Any


CGROUP_ROOT = Path(
    "/sys/fs/cgroup"
)


def _read_text(
    root: Path,
    name: str,
) -> str:
    try:
        return (
            (root / name)
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )
    except OSError:
        return ""


def _read_number(
    root: Path,
    name: str,
) -> int | None:
    value = _read_text(
        root,
        name,
    )

    if (
        not value
        or value == "max"
    ):
        return None

    try:
        return max(
            0,
            int(value),
        )
    except ValueError:
        return None


def _events(
    root: Path,
) -> dict[str, int]:
    rows: dict[str, int] = {}

    for line in _read_text(
        root,
        "memory.events",
    ).splitlines():
        parts = line.split()

        if len(parts) != 2:
            continue

        try:
            rows[
                parts[0]
            ] = int(
                parts[1]
            )
        except ValueError:
            continue

    return rows


def cgroup_memory_snapshot(
    root: Path = CGROUP_ROOT,
) -> dict[str, Any]:
    current = (
        _read_number(
            root,
            "memory.current",
        )
        or 0
    )

    limit = _read_number(
        root,
        "memory.max",
    )

    peak = _read_number(
        root,
        "memory.peak",
    )

    swap_current = (
        _read_number(
            root,
            "memory.swap.current",
        )
        or 0
    )

    swap_limit = _read_number(
        root,
        "memory.swap.max",
    )

    events = _events(
        root
    )

    usage_pct = (
        (
            current
            / limit
            * 100.0
        )
        if (
            limit is not None
            and limit > 0
        )
        else None
    )

    available = (
        max(
            0,
            limit - current,
        )
        if limit is not None
        else None
    )

    if usage_pct is None:
        pressure = "unbounded"

    elif usage_pct >= 95.0:
        pressure = "critical"

    elif usage_pct >= 85.0:
        pressure = "high"

    elif usage_pct >= 70.0:
        pressure = "elevated"

    else:
        pressure = "normal"

    return {
        "version": "1.60.22",
        "source": "cgroup_v2",
        "current_bytes": current,
        "limit_bytes": limit,
        "available_bytes": available,
        "peak_bytes": peak,
        "usage_pct": usage_pct,
        "pressure": pressure,
        "swap_current_bytes": (
            swap_current
        ),
        "swap_limit_bytes": (
            swap_limit
        ),
        "events": events,
        "max_events": int(
            events.get(
                "max",
                0,
            )
        ),
        "oom_events": int(
            events.get(
                "oom",
                0,
            )
        ),
        "oom_kill_events": int(
            events.get(
                "oom_kill",
                0,
            )
        ),
        "telemetry_only": True,
        "trading_authority_changed": False,
        "live_authority": False,
    }
