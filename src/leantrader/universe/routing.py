"""Choose the execution venue before calling one, not after failing on one.

The rule this enforces: resolve canonical symbol → venue notation → market
type → venue availability → environment availability → capability state,
*then* decide whether to make the call. A market the venue does not list is
an expected routing decision with a structured classification, never an
exchange error and never a retry.

When the selected venue cannot execute an opportunity, the intelligence is
not thrown away. It is classified -- another venue may list it, paper may
still trade it, the swarm may still learn from it -- and only a destination
with real authority, real metadata and real balance is chosen.

Nothing here fabricates authority. A venue LeanTrader can see but cannot
authenticate against is reported as exactly that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .venues import (
    AUTH_REQUIRED,
    UNKNOWN_NO_RUNTIME_SNAPSHOT,
    DELISTED,
    LISTED,
    NOT_LISTED,
    NOT_SUPPORTED_IN_TESTNET,
    SPOT,
    TEMPORARILY_SUSPENDED,
    capabilities,
)

# Routing classifications. Each is a state, not a failure.
MARKET_NOT_LISTED_ON_VENUE = "MARKET_NOT_LISTED_ON_VENUE"
VENUE_NOT_EXECUTION_ELIGIBLE = "VENUE_NOT_EXECUTION_ELIGIBLE"
PUBLIC_INTELLIGENCE_ONLY = "PUBLIC_INTELLIGENCE_ONLY"
PAPER_EXECUTABLE = "PAPER_EXECUTABLE"
TESTNET_NOT_AVAILABLE = "TESTNET_NOT_AVAILABLE"
VENUE_AVAILABLE_NOT_AUTHENTICATED = "VENUE_AVAILABLE_NOT_AUTHENTICATED"
EXECUTABLE = "EXECUTABLE"
NO_ELIGIBLE_VENUE = "NO_ELIGIBLE_VENUE"
VENUE_STATE_UNKNOWN = UNKNOWN_NO_RUNTIME_SNAPSHOT

# Attention lanes. Research stays broad; execution follows what can actually
# be traded from here. Both run at once -- narrowing research to the
# execution venue would throw away the reason for watching other venues.
RESEARCH = "research"
EXECUTION = "execution"


@dataclass
class VenueDecision:
    """Which venue should take this order, and why the others cannot."""

    canonical_symbol: str
    market_type: str
    environment: str
    chosen_venue: str = ""
    classification: str = NO_ELIGIBLE_VENUE
    reason: str = ""
    considered: Dict[str, str] = field(default_factory=dict)
    research_venues: List[str] = field(default_factory=list)

    @property
    def executable(self) -> bool:
        return bool(self.chosen_venue) and self.classification in {
            EXECUTABLE,
            PAPER_EXECUTABLE,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "canonical_symbol": self.canonical_symbol,
            "market_type": self.market_type,
            "environment": self.environment,
            "chosen_venue": self.chosen_venue,
            "classification": self.classification,
            "reason": self.reason,
            "considered": dict(self.considered),
            "research_venues": list(self.research_venues),
            "executable": self.executable,
        }


def authenticated_venues() -> Dict[str, str]:
    """Venues with credentials configured, and how they were supplied.

    Presence only -- no value is read, returned or logged. Holding
    credentials is authentication, not permission: execution mode still
    decides the destination.
    """
    found: Dict[str, str] = {}
    for venue in _configured_venues():
        prefix = venue.replace("-", "_").upper()
        if os.getenv(f"{prefix}_TESTNET_API_KEY_FILE") or os.getenv(
            f"{prefix}_API_KEY_FILE"
        ):
            found[venue] = "secret_file"
        elif os.getenv(f"{prefix}_TESTNET_API_KEY") or os.getenv(
            f"{prefix}_API_KEY"
        ):
            found[venue] = "environment"
    return found


def _configured_venues() -> Tuple[str, ...]:
    raw = os.getenv("PUBLIC_DISCOVERY_VENUES", "").strip()
    if raw:
        return tuple(v.strip().lower() for v in raw.split(",") if v.strip())
    return ("bybit", "binance", "okx", "kucoin", "gateio", "mexc", "bitget")


def execution_venue() -> str:
    """The venue the universal router is currently authenticated against."""
    from ..execution import preflight

    try:
        return preflight.shared_broker().exchange_id
    except Exception:
        return (
            os.getenv("CCXT_EXCHANGE")
            or os.getenv("EXCHANGE_ID")
            or "bybit"
        ).strip().lower()


def may_call_venue(
    venue: str,
    canonical_symbol: str,
    market_type: str = SPOT,
    environment: str = "live",
) -> Tuple[bool, str, str]:
    """The guard that belongs in front of every venue call.

    Returns (allowed, classification, detail). A False here means resolve
    locally and move on -- do not call, do not raise, do not retry.
    """
    resolution = capabilities.resolve(
        venue, canonical_symbol, market_type, environment
    )
    if resolution.callable:
        return True, resolution.classification, resolution.detail

    classification = resolution.classification
    if classification in {NOT_LISTED, DELISTED}:
        classification = MARKET_NOT_LISTED_ON_VENUE
    elif classification == NOT_SUPPORTED_IN_TESTNET:
        classification = TESTNET_NOT_AVAILABLE

    return False, classification, resolution.detail


def venue_state_known(venue: str, environment: str) -> bool:
    """Whether anything at all has been learned about this venue."""
    return capabilities.venue_metadata_known(venue, environment)


def select_venue(
    canonical_symbol: str,
    environment: str,
    market_type: str = SPOT,
    preferred: Optional[str] = None,
    require_authentication: bool = True,
) -> VenueDecision:
    """Pick a venue that can genuinely execute this market, or explain why not.

    Order of preference: the venue the router is authenticated against, then
    any other authenticated venue that lists the market in this environment.
    A venue that lists the market but has no credentials is reported as
    available-not-authenticated rather than being used or discarded.
    """
    preferred = (preferred or execution_venue()).strip().lower()
    decision = VenueDecision(
        canonical_symbol=canonical_symbol,
        market_type=market_type,
        environment=environment,
    )

    # Paper does not touch a venue at all. Anything in the intelligence
    # universe is paper-executable, which is the point of paper.
    if environment == "paper":
        decision.chosen_venue = preferred
        decision.classification = PAPER_EXECUTABLE
        decision.reason = "paper execution does not require venue listing"
        decision.research_venues = capabilities.venues_listing(
            canonical_symbol, market_type
        )
        return decision

    authenticated = authenticated_venues()
    listing = capabilities.venues_listing(
        canonical_symbol, market_type, environment
    )
    decision.research_venues = capabilities.venues_listing(
        canonical_symbol, market_type
    )

    candidates = [preferred] + [v for v in listing if v != preferred]

    for venue in candidates:
        record = capabilities.record_for(
            venue, canonical_symbol, market_type, environment
        )

        if record is None:
            # Absence of evidence is not evidence of absence. If this venue's
            # metadata has never been read -- no discovery in this process and
            # no snapshot loaded -- the honest answer is that the state is
            # unknown. Reporting NOT_LISTED here is how genuinely listed
            # markets came to be explained as missing.
            decision.considered[venue] = (
                MARKET_NOT_LISTED_ON_VENUE
                if capabilities.venue_metadata_known(venue, environment)
                else VENUE_STATE_UNKNOWN
            )
            continue
        if record.state == TEMPORARILY_SUSPENDED:
            decision.considered[venue] = TEMPORARILY_SUSPENDED
            continue
        if record.state != LISTED:
            decision.considered[venue] = (
                MARKET_NOT_LISTED_ON_VENUE
                if record.state in {NOT_LISTED, DELISTED}
                else record.state
            )
            continue

        if require_authentication and venue not in authenticated:
            decision.considered[venue] = VENUE_AVAILABLE_NOT_AUTHENTICATED
            continue

        decision.chosen_venue = venue
        decision.classification = EXECUTABLE
        decision.reason = (
            "lists the market in this environment"
            + (" and is authenticated" if require_authentication else "")
        )
        decision.considered[venue] = EXECUTABLE
        return decision

    # Nothing executable. Say which kind of nothing, because the answers
    # differ: a market nobody lists is not the same as one we simply cannot
    # authenticate against, and the second is a configuration decision.
    if all(
        state == VENUE_STATE_UNKNOWN for state in decision.considered.values()
    ) and decision.considered:
        decision.classification = VENUE_STATE_UNKNOWN
        decision.reason = (
            "no venue metadata has been read and no runtime snapshot was "
            "found; this is unknown, not absent"
        )
    elif any(
        state == VENUE_AVAILABLE_NOT_AUTHENTICATED
        for state in decision.considered.values()
    ):
        decision.classification = AUTH_REQUIRED
        decision.reason = (
            "listed on a connected venue with no execution credentials"
        )
    elif decision.research_venues:
        decision.classification = PUBLIC_INTELLIGENCE_ONLY
        decision.reason = (
            "observed on other venues but not tradable in this environment"
        )
    else:
        decision.classification = NO_ELIGIBLE_VENUE
        decision.reason = "not listed on any connected venue"

    return decision


def attention_lane(
    canonical_symbol: str,
    environment: str,
    market_type: str = SPOT,
) -> str:
    """Which lane should spend compute on this market.

    Execution-focused agents should work markets they can actually trade from
    here. Everything else stays in research, where it continues to inform
    models, comparisons and paper trading without consuming execution
    attention.
    """
    if environment == "paper":
        return EXECUTION

    venue = execution_venue()
    record = capabilities.record_for(
        venue, canonical_symbol, market_type, environment
    )
    if record is not None and record.state == LISTED:
        return EXECUTION
    return RESEARCH
