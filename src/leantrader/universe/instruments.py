"""What kind of instrument this is, and who can legitimately price it.

A crypto exchange is not a foreign-exchange data provider. The runtime was
asking Bybit for EUR/USD, GBP/USD, USD/JPY and others -- not because any
engine believed Bybit lists them, but because the router happens to execute
through Bybit and the market-data path inherited that choice. Execution venue
and data venue are different questions and had been collapsed into one.

    canonical instrument -> asset class -> market type
      -> compatible data venues -> venue symbol -> timeframe capability

A crypto venue serves crypto. FX, metals and commodities come from a provider
that actually quotes them. When nothing legitimate can price an instrument the
answer is DATA_PROVIDER_UNAVAILABLE -- never a number from somewhere else, and
never a fabricated one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Asset classes. Deliberately coarse: the distinction that matters here is
# which kind of provider can quote the instrument.
CRYPTO = "crypto"
FX = "fx"
METAL = "metal"
COMMODITY = "commodity"
EQUITY = "equity"
INDEX = "index"
UNKNOWN = "unknown"

DATA_PROVIDER_UNAVAILABLE = "DATA_PROVIDER_UNAVAILABLE"
DATA_VENUE_INCOMPATIBLE = "DATA_VENUE_INCOMPATIBLE"
CONFIG_REQUIRED = "CONFIG_REQUIRED"

# Fiat currencies. A pair of two of these is foreign exchange, whoever is
# being asked about it.
FIAT = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "ZAR", "MXN", "TRY", "RUB", "INR", "BRL", "KRW", "THB",
        "HUF", "CZK", "ILS", "AED", "SAR",
    }
)

# Stablecoins are crypto quote assets, not fiat, even when they are named
# after one. USDT/USD is a crypto market; EUR/USD is not.
STABLECOINS = frozenset(
    {
        "USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDD",
        "PYUSD", "EURT", "EURS", "USDE", "USDS",
    }
)

METALS = frozenset({"XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER", "PLATINUM", "PALLADIUM"})

COMMODITIES = frozenset(
    {"USOIL", "UKOIL", "WTI", "BRENT", "NGAS", "NATGAS", "COPPER", "HG", "CL", "NG"}
)

# Which venue kinds can serve which asset class. Crypto exchanges quote
# crypto; everything else needs a provider that actually covers it.
_CCXT_VENUES = frozenset(
    {"bybit", "binance", "okx", "kucoin", "gateio", "gate", "mexc", "bitget",
     "coinbase", "coinbaseexchange", "kraken", "htx", "huobi", "bitfinex"}
)

# The public chart adapter in this repo covers FX, metals and commodities.
_YAHOO_ASSET_CLASSES = frozenset({FX, METAL, COMMODITY, EQUITY, INDEX})

YAHOO = "yahoo"


@dataclass(frozen=True)
class DataRoute:
    """Where an instrument's market data may legitimately come from."""

    canonical_symbol: str
    asset_class: str
    venue: str = ""
    provider: str = ""
    available: bool = False
    classification: str = ""
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "canonical_symbol": self.canonical_symbol,
            "asset_class": self.asset_class,
            "venue": self.venue,
            "provider": self.provider,
            "available": self.available,
            "classification": self.classification,
            "detail": self.detail,
        }


def _parts(symbol: str) -> Tuple[str, str]:
    text = str(symbol or "").strip().upper()
    if "/" not in text:
        return text, ""
    base, _, quote = text.partition("/")
    return base.strip(), quote.split(":")[0].strip()


def classify_asset(symbol: str, market: Optional[Dict[str, Any]] = None) -> str:
    """Asset class for a canonical symbol.

    Venue metadata wins when it is available -- an exchange saying a market
    is spot crypto is better evidence than any naming heuristic. Otherwise
    the base and quote decide, with stablecoins treated as crypto rather than
    as the fiat they are named after.
    """
    if isinstance(market, dict):
        if market.get("spot") or market.get("swap") or market.get("future"):
            base, quote = _parts(market.get("symbol") or symbol)
            if base in METALS or quote in METALS:
                return METAL
            if base in FIAT and quote in FIAT:
                return FX
            return CRYPTO

    base, quote = _parts(symbol)
    if not base:
        return UNKNOWN

    if base in METALS or (quote in METALS and base not in STABLECOINS):
        return METAL
    if base in COMMODITIES or quote in COMMODITIES:
        return COMMODITY
    if base in STABLECOINS or quote in STABLECOINS:
        return CRYPTO
    if base in FIAT and quote in FIAT:
        return FX
    if quote in FIAT and base not in FIAT:
        # BTC/USD and similar: a crypto asset quoted in fiat.
        return CRYPTO
    if not quote:
        return COMMODITY if base in COMMODITIES else UNKNOWN
    return CRYPTO


def is_crypto_venue(venue: str) -> bool:
    return str(venue or "").strip().lower() in _CCXT_VENUES


def venue_serves_asset(venue: str, asset_class: str) -> bool:
    """Can this venue legitimately quote this kind of instrument?"""
    venue = str(venue or "").strip().lower()
    if venue == YAHOO:
        return asset_class in _YAHOO_ASSET_CLASSES
    if is_crypto_venue(venue):
        return asset_class == CRYPTO
    return False


def resolve_data_route(
    canonical_symbol: str,
    execution_venue: str = "",
    market: Optional[Dict[str, Any]] = None,
) -> DataRoute:
    """Where this instrument's data may come from, given the execution venue.

    The execution venue is only a candidate, never an assumption. It is used
    when it genuinely serves the asset class, and otherwise a compatible
    provider is named -- or the absence of one is.
    """
    asset_class = classify_asset(canonical_symbol, market)

    if asset_class == UNKNOWN:
        return DataRoute(
            canonical_symbol=canonical_symbol,
            asset_class=asset_class,
            available=False,
            classification=DATA_PROVIDER_UNAVAILABLE,
            detail="asset class could not be determined from the symbol",
        )

    if execution_venue and venue_serves_asset(execution_venue, asset_class):
        return DataRoute(
            canonical_symbol=canonical_symbol,
            asset_class=asset_class,
            venue=str(execution_venue).strip().lower(),
            provider="ccxt",
            available=True,
            detail="execution venue also serves this asset class",
        )

    if asset_class in _YAHOO_ASSET_CLASSES:
        if _yahoo_covers(canonical_symbol):
            return DataRoute(
                canonical_symbol=canonical_symbol,
                asset_class=asset_class,
                venue=YAHOO,
                provider="yahoo_public_chart",
                available=True,
                detail="public chart adapter covers this instrument",
            )
        return DataRoute(
            canonical_symbol=canonical_symbol,
            asset_class=asset_class,
            available=False,
            classification=CONFIG_REQUIRED,
            detail=(
                "no configured provider maps this instrument; a data "
                "adapter is required"
            ),
        )

    if asset_class == CRYPTO:
        return DataRoute(
            canonical_symbol=canonical_symbol,
            asset_class=asset_class,
            available=False,
            classification=DATA_PROVIDER_UNAVAILABLE,
            detail="no connected crypto venue lists this market",
        )

    return DataRoute(
        canonical_symbol=canonical_symbol,
        asset_class=asset_class,
        available=False,
        classification=DATA_PROVIDER_UNAVAILABLE,
        detail=f"no provider configured for {asset_class}",
    )


def _yahoo_covers(canonical_symbol: str) -> bool:
    """Whether the public chart adapter can map this instrument.

    Asked of the adapter rather than duplicated here, so the two cannot
    drift apart.
    """
    try:
        from real_yahoo_adapter import normalize_symbol as yahoo_normalize
    except Exception:
        return False

    try:
        return bool(yahoo_normalize(canonical_symbol))
    except Exception:
        return False


def may_use_venue_for_data(
    venue: str,
    canonical_symbol: str,
    market: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """The guard for a market-data call, before the venue is contacted.

    Returns (allowed, classification, detail). This is the asset-class half
    of the check; the listing half lives in routing.may_call_venue, and a
    caller needs both.
    """
    asset_class = classify_asset(canonical_symbol, market)

    if asset_class == UNKNOWN:
        return (
            False,
            DATA_PROVIDER_UNAVAILABLE,
            f"unclassifiable instrument {canonical_symbol}",
        )

    if venue_serves_asset(venue, asset_class):
        return True, asset_class, ""

    return (
        False,
        DATA_VENUE_INCOMPATIBLE,
        f"{venue} does not quote {asset_class} instruments",
    )


def asset_classes_present(symbols: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for symbol in symbols:
        asset_class = classify_asset(symbol)
        counts[asset_class] = counts.get(asset_class, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
