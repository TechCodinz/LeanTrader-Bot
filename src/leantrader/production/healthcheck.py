from __future__ import annotations

import json
import os
import time
from pathlib import Path


def main() -> None:
    path = Path(os.getenv("HEARTBEAT_PATH", "runtime/vps_heartbeat.json"))
    max_age = float(os.getenv("HEARTBEAT_MAX_AGE", "180"))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - float(payload["timestamp"])
    except Exception as exc:
        print(f"unhealthy: heartbeat unavailable: {exc}")
        raise SystemExit(1) from exc
    if age > max_age:
        print(f"unhealthy: heartbeat is {age:.1f}s old")
        raise SystemExit(1)
    if payload.get("healthy") is not True:
        print(f"unhealthy: market cycle errors: {payload.get('errors', {})}")
        raise SystemExit(1)
    print(f"healthy: paper heartbeat is {age:.1f}s old")


if __name__ == "__main__":
    main()
