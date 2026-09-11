#!/usr/bin/env python3
"""Fail if any tracked file can grant real-money authority by being loaded.

A deployment safety gate stopped a rollout because tracked configuration
still carried live-enabling values. This is the same check, in the
repository, so the next one fails in CI instead.

What counts as loadable: anything whose assignments execute when something
reads it -- .env files and their variants, *.env templates, .cfg/.conf/.ini
/.properties, shell scripts that export these variables, and Python that
emits such a file. Markdown is documentation: it cannot be sourced, and a
setup guide showing an operator the live recipe is the point.

There are no per-file exemptions. A template that documents live deployment
keeps its explanatory comments and ships fail-closed assignments; that is the
whole design, and an exemption would be a hole in it.

    python -m tools.config_posture          # exit 1 on any offender
    python -m tools.config_posture --list   # show what is being checked
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Set, Tuple

# Assignments that grant, or widen the path to, real-money authority.
#
# The first group is the live-authority grant itself: all three are required
# together, so any one of them being enabled in a tracked file is a step
# toward it that nobody chose deliberately.
FORBIDDEN: Dict[str, Set[str]] = {
    "ENABLE_LIVE": {"true", "1", "yes", "on"},
    "ALLOW_LIVE": {"true", "1", "yes", "on"},
    "LIVE_CONFIRM": {"yes", "true", "1"},
    # Mode selectors that name the production environment outright.
    "TRADING_MODE": {"live", "real", "production", "prod"},
    "EXECUTION_MODE": {"live", "real", "production", "prod"},
    # Sandbox switches. Turning one off points a credentialed client at a
    # production endpoint, which is the same hazard by a different route.
    "BYBIT_TESTNET": {"false", "0", "no", "off"},
    "CCXT_TESTNET": {"false", "0", "no", "off"},
    "BINANCE_TESTNET": {"false", "0", "no", "off"},
    "GATEIO_SANDBOX": {"false", "0", "no", "off"},
    "OKX_SANDBOX": {"false", "0", "no", "off"},
}

# Required posture for a tracked file that mentions the live-authority flags
# at all: if it assigns one, it must assign it fail-closed.
SAFE_VALUES: Dict[str, str] = {
    "ENABLE_LIVE": "false",
    "ALLOW_LIVE": "false",
    "LIVE_CONFIRM": "NO",
    "TRADING_MODE": "testnet or paper",
    "EXECUTION_MODE": "testnet or paper",
    "BYBIT_TESTNET": "true",
    "CCXT_TESTNET": "true",
    "BINANCE_TESTNET": "true",
    "GATEIO_SANDBOX": "true",
    "OKX_SANDBOX": "true",
}

CONFIG_SUFFIXES = {".env", ".cfg", ".conf", ".ini", ".properties"}
SHELL_SUFFIXES = {".sh", ".bash", ".zsh"}

ASSIGNMENT = re.compile(
    r"""^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"""
)


def is_config_file(path: pathlib.PurePath) -> bool:
    """Files whose KEY=VALUE lines are configuration when loaded."""
    name = path.name
    if name.startswith(".env") or name.startswith("env."):
        return True
    if name in {"env", ".env", "environment"}:
        return True
    if path.suffix in CONFIG_SUFFIXES:
        return True
    return False


def is_shell_file(path: pathlib.PurePath) -> bool:
    return path.suffix in SHELL_SUFFIXES


def is_generator(path: pathlib.PurePath) -> bool:
    """Python that writes configuration is configuration one step removed."""
    return path.suffix == ".py"


def tracked_files(root: pathlib.Path) -> List[pathlib.Path]:
    names = subprocess.check_output(
        ["git", "ls-files"], cwd=str(root), text=True
    ).split("\n")
    return [root / name for name in names if name]


def _clean_value(raw: str) -> str:
    """The value as a loader would see it: unquoted, inline comment removed."""
    value = raw.strip()
    if value[:1] in {'"', "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end].strip().lower()
    value = value.split("#", 1)[0]
    return value.strip().strip("\"'").lower()


def scan_text(text: str, *, generator: bool = False) -> Iterable[Tuple[int, str, str]]:
    """Yield (line number, key, value) for each forbidden assignment.

    In a generator, the assignment usually sits inside a string literal that
    will be written to a file, so indentation and surrounding quotes are not
    a reason to skip it.
    """
    for number, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        candidate = stripped
        if generator:
            # Strip a leading quote from a line inside a triple-quoted block.
            candidate = candidate.lstrip("\"'")

        match = ASSIGNMENT.match(candidate)
        if not match:
            continue

        key = match.group("key").upper()
        forbidden = FORBIDDEN.get(key)
        if not forbidden:
            continue

        value = _clean_value(match.group("value"))
        if not value:
            continue
        # A shell default such as ${ENABLE_LIVE:-false} is not a literal.
        if value.startswith("$"):
            continue
        if value in forbidden:
            yield number, key, value


def scan(paths: Iterable[pathlib.Path], root: pathlib.Path) -> List[str]:
    findings: List[str] = []

    for path in paths:
        if any(part in {".git", "__pycache__", "node_modules"} for part in path.parts):
            continue

        config = is_config_file(path)
        shell = is_shell_file(path)
        generator = is_generator(path)
        if not (config or shell or generator):
            continue

        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        # A Python file only matters here if it is emitting configuration.
        if generator and not config and "TRADING_MODE" not in text and (
            "ENABLE_LIVE" not in text and "ALLOW_LIVE" not in text
        ):
            continue

        try:
            name = path.relative_to(root)
        except ValueError:
            name = path

        for number, key, value in scan_text(text, generator=generator and not config):
            findings.append(
                f"{name}:{number}  {key}={value}  "
                f"(must be {SAFE_VALUES.get(key, 'fail-closed')})"
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail on tracked configuration that can enable live trading."
    )
    parser.add_argument(
        "--list", action="store_true", help="list the files being checked"
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    paths = tracked_files(root)

    if args.list:
        for path in sorted(paths):
            if is_config_file(path) or is_shell_file(path):
                print(path.relative_to(root))
        return 0

    findings = scan(paths, root)

    if not findings:
        print(f"Tracked configuration is fail-closed ({len(paths)} files checked).")
        return 0

    print(f"{len(findings)} live-enabling assignment(s) in tracked files:\n")
    for finding in sorted(set(findings)):
        print(f"  {finding}")
    print(
        "\nReal-money live mode requires a separate operator-controlled "
        "configuration.\nNever enable it from a tracked repository template."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
