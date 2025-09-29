"""Curated repo checks: syntax compile for runtime modules.

Usage:
  py -3.13 tools/run_checks.py
"""
from __future__ import annotations

from pathlib import Path

CURATED_DIRS = [
    Path("src"),
    Path("runtime"),
    Path("reporting"),
    Path("risk"),
    Path("allocators"),
    Path("signals"),
    Path("features"),
    Path("tools"),
]

EXCLUDE_PATTERNS = (
    ".venv",
    "env",
    "venv",
    "site-packages",
    "_incoming",
    str(Path("runtime") / "backups"),
    "traders_core",
    "lt_plugins",
    "cli",
    "scripts",
    "tests",
)


def _allowed(path: Path) -> bool:
    s = str(path)
    return not any(p in s for p in EXCLUDE_PATTERNS)


def compile_curated() -> bool:
    ok = True
    for d in CURATED_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*.py"):
            if not _allowed(f):
                continue
            try:
                compile(f.read_text(encoding="utf-8"), str(f), "exec")
            except Exception as e:
                ok = False
                print("SyntaxError:", f)
                print(e)
    return ok


def main() -> int:
    ok = compile_curated()
    print("curated compile:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
