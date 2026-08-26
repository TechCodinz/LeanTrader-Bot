from __future__ import annotations

import json
import threading

from leantrader.production import (
    runner_v141,
)


def test_concurrent_atomic_json_writers_use_unique_temporary_files(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "vps_health_state.json"
    )

    barrier = threading.Barrier(
        2
    )

    real_replace = (
        runner_v141.os.replace
    )

    def synchronized_replace(
        source,
        destination,
    ):
        # Force both writers to reach the
        # atomic rename simultaneously.
        barrier.wait(
            timeout=5.0
        )

        return real_replace(
            source,
            destination,
        )

    monkeypatch.setattr(
        runner_v141.os,
        "replace",
        synchronized_replace,
    )

    errors: list[
        BaseException
    ] = []

    def writer(
        writer_id: int,
    ) -> None:
        try:
            runner_v141.PaperRunner._write_json_atomic(
                path,
                {
                    "writer": writer_id,
                    "healthy": True,
                },
            )
        except BaseException as exc:
            errors.append(
                exc
            )

    threads = [
        threading.Thread(
            target=writer,
            args=(1,),
        ),
        threading.Thread(
            target=writer,
            args=(2,),
        ),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(
            timeout=10.0
        )

    assert all(
        not thread.is_alive()
        for thread in threads
    )

    assert errors == []

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "healthy"
    ] is True

    assert payload[
        "writer"
    ] in {
        1,
        2,
    }

    leftovers = [
        child
        for child in (
            tmp_path.iterdir()
        )
        if child.name.endswith(
            ".tmp"
        )
    ]

    assert leftovers == []
