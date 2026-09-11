"""Canonical market universe: one normalized view of every market seen.

Discovery itself stays where it already lives -- DynamicMarketScanner pulls
bulk tickers from the connected venues. This package is what was missing
between that and the engines: a single normalized record per market, the
ranking that decides what is worth studying and what is worth trading with a
small balance, and the scheduler that spreads the universe across the swarm.
"""

from .registry import (  # noqa: F401
    Market,
    MarketUniverse,
    WorkItem,
    universe,
)
