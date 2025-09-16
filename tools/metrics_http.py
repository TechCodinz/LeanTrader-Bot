from __future__ import annotations

import argparse
import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from prometheus_client import start_http_server, Gauge  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
LOG_METRICS = RUNTIME / "metrics.json"

# Gauges
G_TICKS = Gauge("ultra_ticks", "Continuous demo tick count parsed from logs")
G_DRYRUN = Gauge("ultra_dryrun_skips", "Dry-run skip count parsed from logs")
G_CASH = Gauge("ultra_cash", "Paper account cash parsed from state")
G_TRAINED = Gauge("ultra_training_models_today", "Number of models trained today")


def _today_training_file() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return RUNTIME / "training_daily" / f"{day}.json"


def _sample_once() -> None:
    # metrics.json
    try:
        if LOG_METRICS.exists():
            data = json.loads(LOG_METRICS.read_text(encoding="utf-8"))
            G_TICKS.set(float(data.get("ticks", 0)))
            G_DRYRUN.set(float(data.get("dryrun_skips", 0)))
            G_CASH.set(float(data.get("cash", 0.0)))
    except Exception:
        pass
    # training daily
    try:
        tf = _today_training_file()
        if tf.exists():
            js = json.loads(tf.read_text(encoding="utf-8"))
            results = js.get("results", []) or []
            ok = 0
            for r in results:
                if isinstance(r, dict) and "ensemble" in r:
                    ok += 1
            G_TRAINED.set(float(ok))
    except Exception:
        pass


def _sampler(stop_event: threading.Event, interval: int) -> None:
    while not stop_event.is_set():
        try:
            _sample_once()
        except Exception:
            pass
        stop_event.wait(interval)


def main() -> int:
    p = argparse.ArgumentParser(description="Start Prometheus HTTP metrics exporter")
    p.add_argument("--port", type=int, default=int(os.getenv("METRICS_PORT", "9000")))
    p.add_argument("--addr", default=os.getenv("METRICS_ADDR", "0.0.0.0"))
    p.add_argument("--interval", type=int, default=int(os.getenv("METRICS_INTERVAL", "10")))
    args = p.parse_args()

    start_http_server(args.port, addr=args.addr)
    print(f"metrics exporter listening on {args.addr}:{args.port}")
    stop = threading.Event()
    t = threading.Thread(target=_sampler, args=(stop, args.interval), daemon=True)
    t.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

