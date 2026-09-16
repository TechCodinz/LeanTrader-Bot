#!/usr/bin/env python3
"""Find code that invents market data, and say which of it the bot can reach.

Two numbers matter and they are different. The first is how much fabrication
exists anywhere in the repository. The second is how much of it the running
bot can actually import -- because only the second can affect a trade.

Some randomness is correct and is not reported: neural-network weight
initialisation, particle-swarm starting positions, epsilon-greedy exploration
in a bandit, and the paper broker and emulator, whose whole job is to
simulate. The rule this enforces is narrower: a number presented to a caller
as if it were measured must have been measured.

    python3 tools/fabrication_audit.py            # reachable code only
    python3 tools/fabrication_audit.py --all      # whole repository
"""

from __future__ import annotations

import ast
import collections
import os
import sys

FABRICATORS = {"uniform", "choice", "random", "randint", "randn", "normal", "rand", "gauss"}

# Randomness that is correct where it is.
ALLOWED = {
    "paper_broker.py",
    "src/leantrader/execution/broker_emulator.py",
    "ultra_god_mode.py",          # neural weights, swarm positions
    "alpha_engines.py",           # epsilon-greedy exploration
    "nobel_ai_models.py",         # __main__ demo fixture only
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             "learned_data_backup", "_incoming", "tests"}


def fabrication_sites(path: str) -> int:
    try:
        tree = ast.parse(open(path, errors="replace").read())
    except Exception:
        return 0
    count = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in FABRICATORS:
            continue
        value = node.func.value
        base = value.id if isinstance(value, ast.Name) else getattr(value, "attr", "")
        if base in {"random", "np"}:
            count += 1
    return count


def reachable_from(entry: str) -> list:
    def path_for(mod):
        for candidate in (mod.replace(".", "/") + ".py", mod.replace(".", "/") + "/__init__.py"):
            if os.path.exists(candidate):
                return candidate
        return None

    seen, queue, found = set(), collections.deque([entry]), []
    while queue:
        module = queue.popleft()
        if module in seen:
            continue
        seen.add(module)
        path = path_for(module)
        if not path:
            continue
        found.append(path)
        try:
            tree = ast.parse(open(path, errors="replace").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                queue.append(node.module)
    return found


def main() -> int:
    everywhere = "--all" in sys.argv
    reachable = set(reachable_from("REAL_PROFIT_BOT"))

    if everywhere:
        paths = []
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            paths += [
                os.path.join(root, f).lstrip("./") for f in files if f.endswith(".py")
            ]
    else:
        paths = sorted(reachable)

    offenders = {}
    for path in paths:
        if path in ALLOWED:
            continue
        count = fabrication_sites(path)
        if count:
            offenders[path] = count

    scope = "REPOSITORY" if everywhere else "REACHABLE FROM REAL_PROFIT_BOT"
    print(f"=== FABRICATION AUDIT — {scope} ===")
    print(f"modules scanned: {len(paths)}")

    in_path = {p: n for p, n in offenders.items() if p in reachable}
    print(f"reachable modules that fabricate: {len(in_path)}")

    if in_path:
        print("\n!! THESE CAN AFFECT A TRADE:")
        for path, count in sorted(in_path.items(), key=lambda kv: -kv[1]):
            print(f"   {count:4}  {path}")

    dormant = {p: n for p, n in offenders.items() if p not in reachable}
    if dormant and everywhere:
        print(f"\n   dormant (not imported by the bot): {len(dormant)} module(s)")
        for path, count in sorted(dormant.items(), key=lambda kv: -kv[1])[:15]:
            print(f"   {count:4}  {path}")

    if in_path:
        print("\nFAIL: the trading path can reach fabricated data.")
        return 1
    print("\nPASS: nothing the bot imports invents market data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
