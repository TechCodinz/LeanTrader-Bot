from __future__ import annotations

import json
import os
import time
from pathlib import Path


DEFAULT_RUNTIME_ID = "verified-multi-engine-v12.8.2-deep-flow-self-model-hygiene"


def main() -> None:
    path = Path(os.getenv("HEARTBEAT_PATH", "runtime/vps_heartbeat.json"))
    max_age = float(os.getenv("HEARTBEAT_MAX_AGE", "180"))
    expected_runtime = os.getenv("EXPECTED_RUNTIME_ID", DEFAULT_RUNTIME_ID).strip()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - float(payload["timestamp"])
    except Exception as exc:
        print(f"unhealthy: heartbeat unavailable: {exc}")
        raise SystemExit(1) from exc
    if age > max_age:
        print(f"unhealthy: heartbeat is {age:.1f}s old")
        raise SystemExit(1)
    if expected_runtime and payload.get("runtime") != expected_runtime:
        print(
            "unhealthy: heartbeat runtime mismatch: "
            f"expected={expected_runtime} actual={payload.get('runtime')}"
        )
        raise SystemExit(1)
    if payload.get("healthy") is not True:
        print(f"unhealthy: market cycle errors: {payload.get('errors', {})}")
        raise SystemExit(1)
    testnet = bool((payload.get("testnet_execution") or {}).get("enabled"))
    mode = "paper + Bybit testnet mirror" if testnet else "paper"
    print(f"healthy: {mode} heartbeat is {age:.1f}s old")


if __name__ == "__main__":
    main()
