"""Resolve a curl_cffi impersonation target against the installed build.

curl_cffi's set of browser fingerprints changes with every release, and the
bare aliases ("chrome", "safari", "firefox") are not present on all of them --
older builds require a concrete, versioned target such as "chrome110". A call
site that hardcodes either form raises on a build that does not list it, which
takes down the FX and commodity chart fetches entirely.

This asks the installed curl_cffi what it actually supports and picks from
that, so the same source works across versions. It never invents a version
string, and when nothing can be resolved it raises rather than letting a
caller quietly fall back to data it did not fetch.

Order of preference:
  1. CURL_IMPERSONATE_TARGET, when the operator sets one and it is supported.
  2. The caller's requested target, if supported.
  3. The bare family alias ("chrome"), if supported.
  4. The highest-numbered concrete build in that family that is supported.
"""

from __future__ import annotations

import os
import re
import typing
from functools import lru_cache
from typing import List, Optional, Tuple

DEFAULT_FAMILY = "chrome"


@lru_cache(maxsize=1)
def supported_targets() -> Tuple[str, ...]:
    """Every impersonation target the installed curl_cffi lists.

    Reads the package's own declaration rather than a table kept here, which
    would go stale the moment curl_cffi is upgraded. Returns () when the
    installed build exposes no discoverable list.
    """
    candidates: List[str] = []

    try:
        from curl_cffi.requests.impersonate import (  # type: ignore
            BrowserTypeLiteral,
        )

        candidates.extend(str(v) for v in typing.get_args(BrowserTypeLiteral))
    except Exception:
        pass

    if not candidates:
        try:
            from curl_cffi.requests.impersonate import (  # type: ignore
                BrowserType,
            )

            candidates.extend(str(m.value) for m in BrowserType)
        except Exception:
            pass

    if not candidates:
        try:
            from curl_cffi.requests import BrowserType  # type: ignore

            candidates.extend(str(m.value) for m in BrowserType)
        except Exception:
            pass

    seen = set()
    ordered = []
    for name in candidates:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _family_of(target: str) -> str:
    match = re.match(r"^([a-z_]+?)\d", target)
    return match.group(1) if match else target


def _version_key(target: str) -> Tuple[int, ...]:
    """Sort concrete builds by their embedded version numbers."""
    return tuple(int(part) for part in re.findall(r"\d+", target)) or (0,)


def newest_in_family(family: str) -> Optional[str]:
    """The highest-numbered concrete build of ``family``, or None.

    Mobile and beta variants are excluded: a desktop chart endpoint served a
    mobile fingerprint gets a different response shape.
    """
    family = family.strip().lower()
    matches = [
        target
        for target in supported_targets()
        if _family_of(target) == family
        and re.search(r"\d", target)
        and not target.endswith(("_android", "_ios"))
        and "beta" not in target
    ]
    if not matches:
        return None
    return sorted(matches, key=_version_key)[-1]


def resolve_impersonation(requested: Optional[str] = None) -> str:
    """Return a target this curl_cffi supports.

    Raises RuntimeError naming what was tried when none resolves, so the
    failure says the fingerprint is unavailable rather than surfacing as an
    opaque error from inside the request.
    """
    supported = supported_targets()
    tried: List[str] = []

    override = os.getenv("CURL_IMPERSONATE_TARGET", "").strip()
    for candidate in (override, (requested or "").strip(), DEFAULT_FAMILY):
        if not candidate:
            continue
        tried.append(candidate)
        if not supported or candidate in supported:
            # An empty list means this build exposes no declaration to check
            # against; the caller's target is passed through unchanged rather
            # than refused on the strength of a failed introspection.
            return candidate

    for candidate in (requested, DEFAULT_FAMILY):
        if not candidate:
            continue
        newest = newest_in_family(_family_of(candidate.strip().lower()))
        if newest:
            return newest

    raise RuntimeError(
        "curl_cffi exposes no usable impersonation target "
        f"(tried: {', '.join(tried) or 'none'}; "
        f"supported: {len(supported)} listed)"
    )
