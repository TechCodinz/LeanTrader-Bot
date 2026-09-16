#!/usr/bin/env python3
"""Restore import lines that were stripped from modules.

Across this repository 175 modules use names they never import -- Dict, Path,
List, Any, pd, np, dataclass -- and raise NameError the moment anything
imports them. It is the same damage found in core/strategy_engine.py,
core/risk_manager.py and core/order_manager.py, which is why the real
indicator and risk engines sat unreachable without ever logging an error.

This only ever ADDS an import for a name the module already uses and never
binds. It adds nothing else, removes nothing, and refuses any name not in the
table below -- an unknown name is reported for a human rather than guessed at.
Every file is re-parsed afterwards, and reverted if the edit did not resolve
the name or broke the parse.

    python3 tools/repair_imports.py            # report only
    python3 tools/repair_imports.py --apply    # write the fixes
"""

from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_audit import SKIP_DIRS, undefined_names  # noqa: E402

# name -> the import line that binds it. Standard library and the two data
# libraries this project uses throughout; nothing project-specific, because
# guessing a local import is how you create a circular one.
KNOWN = {
    "Any": "from typing import Any",
    "Dict": "from typing import Dict",
    "List": "from typing import List",
    "Optional": "from typing import Optional",
    "Tuple": "from typing import Tuple",
    "Set": "from typing import Set",
    "Callable": "from typing import Callable",
    "Mapping": "from typing import Mapping",
    "Sequence": "from typing import Sequence",
    "Iterable": "from typing import Iterable",
    "Iterator": "from typing import Iterator",
    "Union": "from typing import Union",
    "Literal": "from typing import Literal",
    "pd": "import pandas as pd",
    "np": "import numpy as np",
    "Path": "from pathlib import Path",
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import field",
    "asdict": "from dataclasses import asdict",
    "Enum": "from enum import Enum",
    "ABC": "from abc import ABC",
    "abstractmethod": "from abc import abstractmethod",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "timezone": "from datetime import timezone",
    "defaultdict": "from collections import defaultdict",
    "deque": "from collections import deque",
    "contextmanager": "from contextlib import contextmanager",
    "SimpleNamespace": "from types import SimpleNamespace",
    "os": "import os",
    "sys": "import sys",
    "time": "import time",
    "json": "import json",
    "math": "import math",
    "re": "import re",
    "asyncio": "import asyncio",
    "logging": "import logging",
    "threading": "import threading",
    "random": "import random",
    "traceback": "import traceback",
    "uuid": "import uuid",
    "hashlib": "import hashlib",
    "subprocess": "import subprocess",
    "sqlite3": "import sqlite3",
    "warnings": "import warnings",
    "itertools": "import itertools",
    "functools": "import functools",
    "copy": "import copy",
    "pickle": "import pickle",
    "csv": "import csv",
    "requests": "import requests",
    "ccxt": "import ccxt",
    # Third-party names this project already depends on elsewhere.
    "BaseModel": "from pydantic import BaseModel",
    "Field": "from pydantic import Field",
    "FastAPI": "from fastapi import FastAPI",
    "Request": "from fastapi import Request",
    "Gauge": "from prometheus_client import Gauge",
    "Counter": "from prometheus_client import Counter",
    "Histogram": "from prometheus_client import Histogram",
    "Decimal": "from decimal import Decimal",
    "dt": "import datetime as dt",
}

# A module-level logger needs a statement, not an import.
LOGGER_LINES = ("import logging", "logger = logging.getLogger(__name__)")


def insertion_point(source: str, tree: ast.AST) -> int:
    """The line index just after the docstring and any __future__ import."""
    lines = source.splitlines(keepends=True)
    index = 0

    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        index = tree.body[0].end_lineno

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            index = max(index, node.end_lineno)

    # Keep any shebang and encoding line at the top.
    while index < len(lines) and lines[index].startswith("#!"):
        index += 1
    return index


def repair(path: str, apply: bool):
    try:
        source = open(path, errors="replace").read()
        tree = ast.parse(source)
    except Exception:
        return None

    missing, _ = undefined_names(tree)
    if not missing:
        return None

    fixable = sorted(n for n in missing if n in KNOWN or n == "logger")
    unknown = sorted(n for n in missing if n not in KNOWN and n != "logger")
    if not fixable:
        return ("unknown", path, unknown)

    additions = []
    for name in fixable:
        if name == "logger":
            additions.extend(
                line for line in LOGGER_LINES if line not in source
            )
        else:
            line = KNOWN[name]
            if line not in source:
                additions.append(line)

    if not additions:
        return None

    if not apply:
        return ("would-fix", path, fixable + [f"?{n}" for n in unknown])

    lines = source.splitlines(keepends=True)
    at = insertion_point(source, tree)
    block = (
        "\n# Restored: these names were used below but never imported, so this\n"
        "# module raised NameError on import.\n"
        + "\n".join(dict.fromkeys(additions))
        + "\n"
    )
    lines.insert(at, block)
    updated = "".join(lines)

    try:
        new_tree = ast.parse(updated)
    except SyntaxError:
        return ("failed-parse", path, fixable)

    still_missing, _ = undefined_names(new_tree)
    if still_missing & set(fixable):
        return ("failed-bind", path, sorted(still_missing & set(fixable)))

    open(path, "w").write(updated)
    return ("fixed", path, fixable + [f"?{n}" for n in unknown])


def main() -> int:
    apply = "--apply" in sys.argv
    root = next((a for a in sys.argv[1:] if not a.startswith("--")), ".")

    results = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if name.endswith(".py"):
                outcome = repair(os.path.join(base, name), apply)
                if outcome:
                    results.append(outcome)

    by_kind = {}
    for kind, path, names in results:
        by_kind.setdefault(kind, []).append((path, names))

    for kind in ("fixed", "would-fix", "failed-parse", "failed-bind", "unknown"):
        rows = by_kind.get(kind, [])
        if not rows:
            continue
        print(f"\n{kind.upper()}: {len(rows)}")
        for path, names in rows[:20]:
            print(f"    {os.path.relpath(path, root)}  {', '.join(names[:6])}")
        if len(rows) > 20:
            print(f"    ... {len(rows) - 20} more")

    print(f"\n{'applied' if apply else 'dry run'} — {len(results)} module(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
