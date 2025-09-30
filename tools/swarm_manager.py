import os
import pathlib
import subprocess
import sys
from typing import Dict

import yaml  # type: ignore

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "exchanges.yml"


def load_cfg() -> Dict:
    return yaml.safe_load(open(CFG, "r", encoding="utf-8"))


def env_from_file(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path:
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            env[k] = v
    return env


def start() -> None:
    cfg = load_cfg()
    for name, ex in (cfg.get("exchanges", {}) or {}).items():
        if not ex.get("enabled"):
            continue
        env_file = ex.get("env_file")
        mode = ex.get("mode", "paper")
        if mode == "live":
            print(f"skip {name} live (router handles it)")
            continue
        env = os.environ.copy()
        env.update(env_from_file(env_file))
        # Optional: per-exchange artifact isolation
        env["LEDGER_PATH"] = ex.get("data_root", "/opt/leantrader/data") + "/ledger.csv"
        env["REPORTS_DIR"] = ex.get("out_root", "/opt/leantrader/out") + "/reports"
        env["MODEL_DIR"] = ex.get("out_root", "/opt/leantrader/out") + "/models"
        env["CHECKPOINT_DIR"] = ex.get("out_root", "/opt/leantrader/out") + "/checkpoints"
        args = [sys.executable, str(ROOT / "ultra_launcher.py"), "--mode", "paper"]
        meta = pathlib.Path("/opt/leantrader/out/meta")
        meta.mkdir(parents=True, exist_ok=True)
        log = open(meta / f"swarm_{name}.log", "ab")
        log.write(f"\n=== START {name} ===\n".encode())
        subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"started {name}")


if __name__ == "__main__":
    start()


