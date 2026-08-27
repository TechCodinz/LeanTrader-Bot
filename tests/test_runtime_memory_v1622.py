from pathlib import Path

from leantrader.production.runtime_memory_v1622 import (
    cgroup_memory_snapshot,
)


def write(
    root: Path,
    name: str,
    value: str,
) -> None:
    (
        root / name
    ).write_text(
        value,
        encoding="utf-8",
    )


def test_cgroup_memory_snapshot_reports_pressure(
    tmp_path,
):
    write(
        tmp_path,
        "memory.current",
        "3221225472\n",
    )

    write(
        tmp_path,
        "memory.max",
        "4294967296\n",
    )

    write(
        tmp_path,
        "memory.peak",
        "3758096384\n",
    )

    write(
        tmp_path,
        "memory.swap.current",
        "1048576\n",
    )

    write(
        tmp_path,
        "memory.swap.max",
        "2147483648\n",
    )

    write(
        tmp_path,
        "memory.events",
        (
            "low 0\n"
            "high 2\n"
            "max 7\n"
            "oom 1\n"
            "oom_kill 1\n"
        ),
    )

    row = (
        cgroup_memory_snapshot(
            tmp_path
        )
    )

    assert row[
        "limit_bytes"
    ] == 4294967296

    assert row[
        "usage_pct"
    ] == 75.0

    assert row[
        "pressure"
    ] == "elevated"

    assert row[
        "max_events"
    ] == 7

    assert row[
        "oom_kill_events"
    ] == 1

    assert row[
        "trading_authority_changed"
    ] is False

    assert row[
        "live_authority"
    ] is False


def test_unlimited_cgroup_is_supported(
    tmp_path,
):
    write(
        tmp_path,
        "memory.current",
        "12345\n",
    )

    write(
        tmp_path,
        "memory.max",
        "max\n",
    )

    write(
        tmp_path,
        "memory.events",
        "max 0\noom 0\noom_kill 0\n",
    )

    row = (
        cgroup_memory_snapshot(
            tmp_path
        )
    )

    assert row[
        "limit_bytes"
    ] is None

    assert row[
        "usage_pct"
    ] is None

    assert row[
        "pressure"
    ] == "unbounded"


def test_compose_persists_four_gib_envelope():
    text = Path(
        "docker-compose.yml"
    ).read_text(
        encoding="utf-8"
    )

    assert "memory: 4G" in text
    assert "memory: 1G" not in text
