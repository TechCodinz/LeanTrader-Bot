import json
import time
import pathlib
import csv
from typing import List, Dict, Any

INBOX = pathlib.Path("/opt/leantrader/inbox_signals")
OUT = pathlib.Path("/opt/leantrader/out/copy_alpha")
INBOX.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def load_all() -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []
    for p in INBOX.glob("*.json"):
        try:
            sigs += json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            pass
    for p in INBOX.glob("*.csv"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    sigs.append(r)
        except Exception:
            pass
    return sigs


def normalize(sig: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts": float(sig.get("timestamp", time.time())),
        "ex": (sig.get("exchange") or "").lower(),
        "sym": sig.get("symbol"),
        "side": sig.get("side"),
        "conf": float(sig.get("confidence", 0.5)),
        "entry": float(sig.get("entry", 0)),
        "sl": float(sig.get("sl", 0)),
        "tp": float(sig.get("tp", 0)),
        "src": sig.get("source", "external"),
    }


def merge_to_store(sigs: List[Dict[str, Any]]) -> None:
    buckets = {}
    for s in sigs:
        n = normalize(s)
        ex = n["ex"] or "unknown"
        buckets.setdefault(ex, []).append(n)
    for ex, rows in buckets.items():
        path = OUT / f"{ex}_signals.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sigs = load_all()
    merge_to_store(sigs)
    print(f"merged {len(sigs)} signals")


