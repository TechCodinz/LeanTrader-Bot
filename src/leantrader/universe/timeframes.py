"""One authoritative translation from internal interval names to venue ones.

Strategy code is free to speak in whatever interval vocabulary suits it --
the continuous trader uses MetaTrader-style M1/M5/M15/M30/H1/H4/D1, other
engines use ccxt-style 1m/5m/1h. What must not happen is either of those
reaching an exchange unchanged, which is what produced a steady stream of
"Invalid period!" from Bybit for BTC/USDT, ETH/USDT, SOL/USDT and every other
perfectly valid symbol the continuous trader was studying.

The boundary is here:

    internal label -> canonical timeframe -> venue-supported timeframe
      -> native request

A venue that does not support the requested interval is answered before the
call, either by aggregating from a supported shorter interval where that is
mathematically exact, or by classifying TIMEFRAME_NOT_SUPPORTED. A materially
different interval is never substituted silently: an engine asking for 4h and
being handed 1h would compute the wrong thing and never know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

TIMEFRAME_NOT_SUPPORTED = "TIMEFRAME_NOT_SUPPORTED"

# Canonical form is ccxt's, because that is what the adapters speak.
# Minutes are the ordering key and the basis for legitimate aggregation.
CANONICAL_MINUTES: Dict[str, int] = {
    "1s": 0,          # sub-minute; never aggregated from, never aggregated to
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}

# Internal vocabularies seen across this codebase, mapped to canonical.
# MetaTrader-style labels come from the continuous trader and the DSL; the
# rest are spellings that appear in individual engines.
_ALIASES: Dict[str, str] = {
    # MetaTrader style
    "m1": "1m", "m3": "3m", "m5": "5m", "m15": "15m", "m30": "30m",
    "h1": "1h", "h2": "2h", "h4": "4h", "h6": "6h", "h8": "8h", "h12": "12h",
    "d1": "1d", "d3": "3d", "w1": "1w", "mn1": "1M", "mn": "1M",
    # Verbose spellings
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
    "60min": "1h", "1hour": "1h", "4hour": "4h", "1day": "1d", "daily": "1d",
    "1week": "1w", "weekly": "1w", "monthly": "1M",
    "hourly": "1h", "minute": "1m", "day": "1d", "week": "1w",
    # Numeric-minute spellings
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "480": "8h",
    "720": "12h", "1440": "1d", "10080": "1w",
}

_PATTERN = re.compile(r"^(?P<count>\d+)\s*(?P<unit>[a-zA-Z]+)$")

_UNIT_MINUTES = {
    "s": 0,
    "m": 1,
    "min": 1,
    "h": 60,
    "hr": 60,
    "hour": 60,
    "d": 1440,
    "day": 1440,
    "w": 10080,
    "week": 10080,
}


@dataclass(frozen=True)
class TimeframePlan:
    """How to satisfy a request for one interval on one venue."""

    requested: str
    canonical: str
    venue_timeframe: str = ""
    supported: bool = False
    aggregate_from: str = ""
    aggregate_factor: int = 1
    classification: str = ""
    detail: str = ""

    @property
    def direct(self) -> bool:
        return self.supported and not self.aggregate_from

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "canonical": self.canonical,
            "venue_timeframe": self.venue_timeframe,
            "supported": self.supported,
            "aggregate_from": self.aggregate_from,
            "aggregate_factor": self.aggregate_factor,
            "classification": self.classification,
            "detail": self.detail,
        }


def canonical_timeframe(value: Any) -> str:
    """Return the canonical ccxt interval for any internal label, or "".

    Case-insensitive, whitespace-tolerant, and deliberately strict about what
    it will not guess: an unrecognised label returns "" so the caller
    classifies rather than sending something plausible-looking to a venue.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""

    # "1M" means one month in ccxt and must not be folded into "1m".
    if raw in CANONICAL_MINUTES:
        return raw

    lowered = raw.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    if lowered in CANONICAL_MINUTES:
        return lowered

    match = _PATTERN.match(lowered)
    if match:
        try:
            count = int(match.group("count"))
        except ValueError:
            return ""
        unit = match.group("unit")
        minutes = _UNIT_MINUTES.get(unit)
        if minutes:
            total = count * minutes
            for name, canonical_minutes in CANONICAL_MINUTES.items():
                if canonical_minutes == total and canonical_minutes > 0:
                    return name
    return ""


def timeframe_minutes(value: Any) -> int:
    canonical = canonical_timeframe(value)
    return CANONICAL_MINUTES.get(canonical, 0)


def venue_timeframes(exchange: Any) -> Tuple[str, ...]:
    """Intervals a venue actually supports, from its own metadata.

    ccxt publishes this as ``timeframes``. When a venue does not, the caller
    gets () and should treat every interval as unverified rather than assume
    support -- which is the honest answer, not a reason to guess.
    """
    declared = getattr(exchange, "timeframes", None)
    if not declared:
        return ()
    if isinstance(declared, dict):
        names = list(declared.keys())
    elif isinstance(declared, (list, tuple, set)):
        names = list(declared)
    else:
        return ()
    return tuple(str(n) for n in names if n)


def plan_timeframe(
    requested: Any,
    supported: Optional[Sequence[str]] = None,
    allow_aggregation: bool = True,
) -> TimeframePlan:
    """Decide how to serve ``requested`` given what a venue supports.

    ``supported`` empty or None means the venue publishes no interval list.
    The canonical interval is then passed through unverified rather than
    refused, because refusing everything would be worse than letting the
    venue answer for itself.
    """
    raw = str(requested or "").strip()
    canonical = canonical_timeframe(raw)

    if not canonical:
        return TimeframePlan(
            requested=raw,
            canonical="",
            supported=False,
            classification=TIMEFRAME_NOT_SUPPORTED,
            detail=f"unrecognised interval {raw!r}",
        )

    if not supported:
        return TimeframePlan(
            requested=raw,
            canonical=canonical,
            venue_timeframe=canonical,
            supported=True,
            detail="venue publishes no interval list; passed through",
        )

    available = {str(s) for s in supported}

    if canonical in available:
        return TimeframePlan(
            requested=raw,
            canonical=canonical,
            venue_timeframe=canonical,
            supported=True,
            detail="natively supported",
        )

    # A venue may spell the same interval differently. Match by duration
    # rather than by text, so 60 and 1h are recognised as one interval.
    wanted_minutes = CANONICAL_MINUTES.get(canonical, 0)
    for candidate in available:
        if wanted_minutes and timeframe_minutes(candidate) == wanted_minutes:
            return TimeframePlan(
                requested=raw,
                canonical=canonical,
                venue_timeframe=candidate,
                supported=True,
                detail=f"venue spells this interval {candidate!r}",
            )

    if allow_aggregation and wanted_minutes:
        # Aggregation is only legitimate when the shorter interval divides
        # the longer one exactly: 4h from 1h is four candles, 4h from 90m is
        # not 4h at all. Prefer the longest exact divisor, so the fewest
        # candles are fetched.
        divisors = [
            (timeframe_minutes(candidate), candidate)
            for candidate in available
            if timeframe_minutes(candidate) > 0
            and wanted_minutes % timeframe_minutes(candidate) == 0
            and timeframe_minutes(candidate) < wanted_minutes
        ]
        if divisors:
            minutes, candidate = max(divisors)
            return TimeframePlan(
                requested=raw,
                canonical=canonical,
                venue_timeframe=candidate,
                supported=True,
                aggregate_from=candidate,
                aggregate_factor=wanted_minutes // minutes,
                detail=(
                    f"aggregated from {candidate} "
                    f"({wanted_minutes // minutes} candles per bar)"
                ),
            )

    return TimeframePlan(
        requested=raw,
        canonical=canonical,
        supported=False,
        classification=TIMEFRAME_NOT_SUPPORTED,
        detail=(
            f"{canonical} is not supported and cannot be aggregated "
            f"from {sorted(available)[:8]}"
        ),
    )


def aggregate_ohlcv(rows: Sequence[Sequence[float]], factor: int) -> List[List[float]]:
    """Combine ``factor`` candles into one, preserving OHLCV semantics.

    Open from the first, close from the last, high and low across the group,
    volume summed. A trailing partial group is dropped rather than emitted as
    a complete bar, because a half-formed candle is not the interval asked
    for.
    """
    if factor <= 1:
        return [list(row) for row in rows]

    aggregated: List[List[float]] = []
    for start in range(0, len(rows) - factor + 1, factor):
        group = rows[start : start + factor]
        if len(group) < factor:
            break
        try:
            aggregated.append(
                [
                    group[0][0],
                    float(group[0][1]),
                    max(float(row[2]) for row in group),
                    min(float(row[3]) for row in group),
                    float(group[-1][4]),
                    sum(float(row[5]) for row in group),
                ]
            )
        except (IndexError, TypeError, ValueError):
            continue
    return aggregated
