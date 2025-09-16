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
PAPER_STATE = RUNTIME / "paper_state.json"

# Gauges
G_TICKS = Gauge("ultra_ticks", "Continuous demo tick count parsed from logs")
G_DRYRUN = Gauge("ultra_dryrun_skips", "Dry-run skip count parsed from logs")
G_CASH = Gauge("ultra_cash", "Paper account cash parsed from state")
G_TRAINED = Gauge("ultra_training_models_today", "Number of models trained today")
G_KPI_WIN = Gauge("ultra_kpi_win_rate", "Win rate (0..1) rolling")
G_KPI_RR = Gauge("ultra_kpi_avg_rr", "Average risk-reward rolling")
G_KPI_SHARPE = Gauge("ultra_kpi_sharpe", "Sharpe ratio rolling")
G_KPI_DD = Gauge("ultra_kpi_max_drawdown", "Max drawdown rolling (0..1)")


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
    # KPIs from paper_state.json (placeholder calc if no full trade log)
    try:
        if PAPER_STATE.exists():
            js = json.loads(PAPER_STATE.read_text(encoding="utf-8"))
            # If history available, compute rough win rate and dd
            hist = js.get("history") or []
            wins = 0
            total = 0
            pnl_list = []
            for h in hist:
                try:
                    pnl = float(h.get("pnl_pct", 0.0))
                    pnl_list.append(pnl)
                    total += 1
                    if pnl > 0:
                        wins += 1
                except Exception:
                    continue
            win_rate = (wins / total) if total else 0.0
            G_KPI_WIN.set(win_rate)
            # crude Sharpe: mean/std of daily-like pnl series
            try:
                import statistics as _stats
                mean = _stats.mean(pnl_list) if pnl_list else 0.0
                std = _stats.pstdev(pnl_list) if len(pnl_list) > 1 else 1.0
                sharpe = (mean / std) if std else 0.0
            except Exception:
                sharpe = 0.0
            G_KPI_SHARPE.set(sharpe)
            # max drawdown placeholder from cash vs starting cash
            try:
                start_cash = float(js.get("start_cash", js.get("cash", 0.0)))
                cash = float(js.get("cash", 0.0))
                dd = max(0.0, (start_cash - cash) / max(1.0, start_cash))
            except Exception:
                dd = 0.0
            G_KPI_DD.set(dd)
            # avg RR not available without per-trade SL/TP; set 0
            G_KPI_RR.set(0.0)
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

