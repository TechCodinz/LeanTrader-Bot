#!/usr/bin/env python3
"""Fail if a tracked file contains a concrete credential literal.

A deployment safety gate found live exchange keys committed across scripts,
deployment notes and .env. They are being rotated externally, and this exists
so the next one is caught in CI rather than by a gate at deploy time.

It looks for credential-named variables assigned a concrete-looking value,
and for the shapes of secrets that do not need a variable name to be
dangerous (Telegram bot tokens, AWS access keys, hex private keys).

Placeholders are allowed by design -- a repository needs to show the shape of
its configuration. A value is a placeholder if it is obviously not a secret:
YOUR_..., <...>, ${...}, changeme, REDACTED_..., and similar.

    python -m tools.secret_scan          # scan tracked files, exit 1 on a hit
    python -m tools.secret_scan --all    # scan the working tree too
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Iterable, List, Tuple

KEY_NAME = (
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"(?:API_?KEY|API_?SECRET|SECRET_?KEY|SECRET|TOKEN|PASSPHRASE|PRIVATE_?KEY|PASSWORD)"
    r"[A-Za-z0-9_]*"
)

ASSIGNMENT = re.compile(
    rf"""(?P<key>{KEY_NAME})\s*[:=]\s*(?P<q>['"]?)(?P<val>[A-Za-z0-9_\-:./+]{{12,}})(?P=q)""",
    re.IGNORECASE,
)

# Shapes that are secrets regardless of the name they are bound to.
SHAPES = (
    ("telegram_bot_token", re.compile(r"\b\d{8,11}:[A-Za-z0-9_-]{30,40}\b")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("hex_private_key", re.compile(r"\b0x[a-fA-F0-9]{64}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)

# A 32-byte hex value is far more often a public constant than a key: event
# topics, transaction and block hashes, merkle roots, commit ids. A shape hit
# on a line saying so is not reported.
# Underscore is a word character, so TRANSFER_TOPIC needs an explicit
# boundary set rather than \b.
NOT_A_SECRET_CONTEXT = re.compile(
    r"(?:^|[^A-Za-z])(topic|hash|digest|signature|sighash|selector|checksum|"
    r"commit|merkle|root|address|contract|txid|blockhash|event)"
    r"(?:[^A-Za-z]|$)",
    re.IGNORECASE,
)

# Telegram's own documentation token, used in every setup guide.
BOTFATHER_EXAMPLE = "1234567890:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"

# A value that is plainly not a secret.
PLACEHOLDER = re.compile(
    r"""^(
        your | xxx+ | placeholder | example | changeme | change_me | insert | paste
      | redacted | dummy | fake | sample | none | null | true | false | test_
      | <  | \$ | \{ | 0x[Yy]our | \.\.\. | abc123 | secret_?key$ | api_?key$
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Environment-variable references, not values: ${X}, $X, os.getenv names.
ENV_REFERENCE = re.compile(r"^[A-Z][A-Z0-9_]*$")

SKIP_SUFFIXES = (".lock", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz")
SKIP_PARTS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def tracked_files(root: pathlib.Path) -> List[pathlib.Path]:
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=str(root), text=True
    ).split("\n")
    return [root / name for name in out if name]


def all_files(root: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# A value that is code rather than data: settings.bybit_api_key,
# cipher.decrypt(...), self._api_key, np.random.uniform(...).
CODE_EXPRESSION = re.compile(
    r"""^(
        [A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_]      # attribute access
      | [A-Za-z_][A-Za-z0-9_]*\(               # a call
      | (os\.)?(getenv|environ)                # an env lookup
    )""",
    re.VERBOSE,
)


def _is_placeholder(value: str, key: str = "") -> bool:
    if value == BOTFATHER_EXAMPLE:
        return True
    if key and value.lower() == key.lower():
        # foo(api_secret=api_secret): a keyword argument passing a variable.
        return True
    if PLACEHOLDER.match(value):
        return True
    if "REDACTED" in value.upper():
        return True
    if CODE_EXPRESSION.match(value):
        # The literal is an expression, so the secret is not in this file.
        return True
    if ENV_REFERENCE.match(value) and value.count("_") >= 1:
        # BYBIT_API_KEY assigned to the name of another variable.
        return True
    return False


def scan_text(text: str) -> Iterable[Tuple[int, str, str]]:
    """Yield (line number, finding kind, masked value)."""
    for number, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith(("#", "//", "*")) and "REDACTED" in line.upper():
            continue

        for kind, pattern in SHAPES:
            for match in pattern.finditer(line):
                value = match.group(0)
                if _is_placeholder(value) or value == BOTFATHER_EXAMPLE:
                    continue
                if NOT_A_SECRET_CONTEXT.search(line):
                    continue
                yield number, kind, _mask(value)

        for match in ASSIGNMENT.finditer(line):
            value = match.group("val")
            key = match.group("key")
            if _is_placeholder(value, key):
                continue
            yield number, f"assignment:{key}", _mask(value)


def _mask(value: str) -> str:
    """Never print a secret, even when reporting one."""
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:3]}…{value[-2:]} (len {len(value)})"


def scan(paths: Iterable[pathlib.Path], root: pathlib.Path) -> List[str]:
    findings: List[str] = []
    for path in paths:
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            name = path.relative_to(root)
        except ValueError:
            name = path
        for number, kind, masked in scan_text(text):
            findings.append(f"{name}:{number}  {kind}  {masked}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for committed secrets.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan the whole working tree, not only tracked files",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    paths = all_files(root) if args.all else tracked_files(root)
    findings = scan(paths, root)

    if not findings:
        print(f"No credential literals found in {len(paths)} files.")
        return 0

    print(f"{len(findings)} possible credential literal(s):\n")
    for finding in sorted(set(findings)):
        print(f"  {finding}")
    print(
        "\nValues are masked. Move real credentials into the mounted secret "
        "files described in SECRETS.md and leave a placeholder here."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
