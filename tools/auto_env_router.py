#!/usr/bin/env python3
import os
import sys
import subprocess
import pathlib
from typing import Dict, Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "exchanges.yml"


def _ensure_deps() -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=False)
    try:
        import ccxt  # type: ignore
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ccxt"], check=False)


def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_env_file(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def _probe_balance_usd(env: Dict[str, str]) -> float:
    try:
        import ccxt  # type: ignore
    except Exception:
        return -1.0

    ex_id = (env.get("EXCHANGE_ID") or "").lower().replace("gate.io", "gateio")
    if not hasattr(ccxt, ex_id):
        return -1.0

    # Map keys
    key_var = f"{ex_id.upper()}_API_KEY"
    sec_var = f"{ex_id.upper()}_SECRET"
    if ex_id == "gateio":
        key_var, sec_var = "GATEIO_API_KEY", "GATEIO_SECRET"

    api_key = env.get(key_var)
    secret = env.get(sec_var)
    if not api_key or not secret:
        return -1.0

    ex = getattr(ccxt, ex_id)({"apiKey": api_key, "secret": secret})
    if ex_id == "bybit" and env.get("BYBIT_TESTNET", "false").lower() == "true":
        ex.set_sandbox_mode(True)
    if ex_id == "binance" and env.get("BINANCE_TESTNET", "false").lower() == "true":
        ex.set_sandbox_mode(True)

    try:
        bal = ex.fetch_balance()
        for k in ("USDT", "USD", "BUSD", "USDC"):
            if isinstance(bal.get("free"), dict) and k in bal["free"]:
                return float(bal["free"][k])
            v = bal.get(k)
            if isinstance(v, dict) and "total" in v:
                return float(v["total"])
    except Exception:
        return -1.0
    return -1.0


def _exec_with_env(env_path: str, launcher: str = "ultra_launcher.py", extra_args=None) -> None:
    child_env = os.environ.copy()
    child_env.update(_load_env_file(env_path))
    args = [sys.executable, str(ROOT / launcher)]
    if extra_args:
        args += list(extra_args)
    print(f"\u23ef\ufe0f  Exec: {args} with env={env_path}")
    os.execve(sys.executable, args, child_env)


def main() -> None:
    _ensure_deps()
    cfg = _load_yaml(CONFIG_PATH)
    router = cfg.get("router", {})
    min_live = float(router.get("min_live_balance_usdt", 40))
    live_env = router.get("live_env_file")
    test_env = router.get("testnet_env_file")
    fallback = bool(router.get("fallback_to_testnet", True))

    selected = None

    if live_env:
        live_vars = _load_env_file(live_env)
        bal = _probe_balance_usd(live_vars)
        print(f"\ud83d\udd0e Live balance probe: {live_vars.get('EXCHANGE_ID','?')} balance={bal}")
        if bal >= min_live or bal < 0:
            selected = ("live", live_env)

    if not selected and fallback and test_env:
        test_vars = _load_env_file(test_env)
        bal = _probe_balance_usd(test_vars)
        print(f"\ud83d\udd0e Testnet probe: {test_vars.get('EXCHANGE_ID','?')} balance={bal}")
        selected = ("testnet", test_env)

    if not selected:
        print("\u274c No viable environment profile found. Check configs/exchanges.yml and env files.")
        sys.exit(1)

    mode, env_file = selected
    extra = ("--mode", "paper") if mode == "testnet" else None
    _exec_with_env(env_file, launcher="ultra_launcher.py", extra_args=extra)


if __name__ == "__main__":
    main()


