#!/usr/bin/env python3
"""Audit every module: does it parse, import, and tell the truth.

Four questions per file, in the order they matter:

1. Does it parse at all?
2. Does it use names it never imports or defines? That is a NameError the
   moment the module is touched. Three files in core/ had their import blocks
   stripped exactly this way, which is why the real indicator and risk engines
   were unreachable for months without anyone seeing an error.
3. Does it invent market data?
4. Does it carry a credential literal?

Run against either repository:

    python3 tools/full_audit.py [path]           # summary
    python3 tools/full_audit.py [path] --full    # every finding
"""

from __future__ import annotations

import ast
import builtins
import os
import re
import sys

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "learned_data_backup", "_incoming", "site-packages", ".mypy_cache",
    ".pytest_cache", "build", "dist",
}

FABRICATORS = {"uniform", "choice", "random", "randint", "randn", "normal", "gauss"}

# Randomness that is correct where it is: weight initialisation, bandit
# exploration, Monte Carlo, and the simulators whose job is to simulate.
FABRICATION_ALLOWED = {
    "paper_broker.py",
    "ultra_god_mode.py",
    "alpha_engines.py",
    "nobel_ai_models.py",
    "nobel_risk_management.py",
    "src/leantrader/execution/broker_emulator.py",
}

# Names that are commonly used without import because they come from a
# star-import or are injected; flagging them is noise, not signal.
IGNORE_NAMES = {"__file__", "__name__", "__doc__", "__package__", "_"}

CREDENTIAL_PATTERNS = (
    re.compile(r"""['"][0-9a-fA-F]{32,}['"]"""),
    re.compile(r"""['"]\d{8,10}:AA[\w-]{30,}['"]"""),   # telegram bot token
    re.compile(r"""['"]sk-[A-Za-z0-9_-]{20,}['"]"""),
)

CREDENTIAL_KEYS = re.compile(
    r"""(api_?key|secret|passphrase|password|token)\s*[:=]\s*['"]([^'"]{16,})['"]""",
    re.IGNORECASE,
)


def python_files(root: str):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(base, name)
                yield path, os.path.relpath(path, root)


def undefined_names(tree: ast.AST) -> set:
    """Names read at module or class level that are never bound anywhere.

    Deliberately conservative: only module-level and class-level reads, never
    inside a function, because a function can legitimately receive anything
    through a closure or a global set elsewhere. A name missing at class level
    fires at import time, which is the failure this is hunting.
    """
    bound = set(dir(builtins)) | IGNORE_NAMES

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)

    used = set()
    script_only = set()

    def is_main_guard(node) -> bool:
        """`if __name__ == "__main__":` -- runs on execution, not on import."""
        if not isinstance(node, ast.If):
            return False
        return "__name__" in {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }

    def scan(node, inside_function=False):
        for child in ast.iter_child_nodes(node):
            if is_main_guard(child):
                # Collected separately: a name missing here breaks `python
                # file.py`, but never breaks importing the module, so it
                # cannot be what silently disconnected an engine.
                collect(child, script_only)
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Decorators and defaults evaluate at definition time.
                for d in child.decorator_list:
                    collect(d, used)
                for d in (child.args.defaults or []):
                    collect(d, used)
                collect(child.returns, used) if child.returns else None
                for a in list(child.args.args) + list(child.args.kwonlyargs):
                    if a.annotation:
                        collect(a.annotation, used)
                scan(child, True)
            elif isinstance(child, ast.ClassDef):
                for d in child.decorator_list:
                    collect(d, used)
                for b in child.bases:
                    collect(b, used)
                scan(child, inside_function)
            elif not inside_function:
                collect(child, used)
                scan(child, inside_function)
            else:
                scan(child, True)

    def collect(node, out):
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                out.add(sub.id)

    scan(tree)
    importable = {n for n in used - bound if not n.startswith("__")}
    entrypoint = {n for n in script_only - bound - used if not n.startswith("__")}
    return importable, entrypoint


def fabrication_count(tree: ast.AST) -> int:
    total = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in FABRICATORS:
            continue
        value = node.func.value
        base = value.id if isinstance(value, ast.Name) else getattr(value, "attr", "")
        if base in {"random", "np"}:
            total += 1
    return total


def credential_hits(source: str) -> list:
    hits = []
    for match in CREDENTIAL_KEYS.finditer(source):
        value = match.group(2)
        if value.startswith(("${", "os.", "<", "your", "YOUR", "xxx", "XXX")):
            continue
        if value.lower() in {"none", "null", "changeme", "placeholder", "redacted"}:
            continue
        hits.append(match.group(1).lower())
    for pattern in CREDENTIAL_PATTERNS:
        if pattern.search(source):
            hits.append("literal")
    return sorted(set(hits))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else "."
    full = "--full" in sys.argv

    broken, undefined, fabricating, credentials, total = [], [], [], [], 0
    script_only_broken = []

    for path, rel in python_files(root):
        total += 1
        try:
            source = open(path, errors="replace").read()
        except OSError:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            broken.append((rel, f"line {exc.lineno}: {exc.msg}"))
            continue

        missing, entry_missing = undefined_names(tree)
        if missing:
            undefined.append((rel, sorted(missing)[:6]))
        if entry_missing:
            script_only_broken.append((rel, sorted(entry_missing)[:6]))

        if rel not in FABRICATION_ALLOWED:
            count = fabrication_count(tree)
            if count:
                fabricating.append((rel, count))

        creds = credential_hits(source)
        if creds:
            credentials.append((rel, creds))

    print(f"=== FULL AUDIT — {os.path.abspath(root)} ===")
    print(f"python modules: {total}\n")

    def section(title, rows, render):
        print(f"{title}: {len(rows)}")
        shown = rows if full else rows[:10]
        for row in shown:
            print(f"    {render(row)}")
        if not full and len(rows) > len(shown):
            print(f"    ... {len(rows) - len(shown)} more (use --full)")
        print()

    section("SYNTAX ERRORS", broken, lambda r: f"{r[0]}  {r[1]}")
    section("NAMEERROR ON IMPORT (used but never bound)", undefined,
            lambda r: f"{r[0]}  missing: {', '.join(r[1])}")
    section("BREAKS ONLY WHEN RUN DIRECTLY (import is fine)", script_only_broken,
            lambda r: f"{r[0]}  missing: {', '.join(r[1])}")
    section("FABRICATES MARKET DATA", fabricating, lambda r: f"{r[1]:4}  {r[0]}")
    section("CREDENTIAL LITERALS", credentials, lambda r: f"{r[0]}  [{', '.join(r[1])}]")

    critical = len(broken) + len(undefined)
    print(f"critical (cannot import): {critical}")
    print(f"fabricating: {len(fabricating)}   credentials: {len(credentials)}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
