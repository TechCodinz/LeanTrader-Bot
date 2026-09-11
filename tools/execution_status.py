#!/usr/bin/env python3
"""Report where execution is stopping, from inside the running container.

The question this answers is the one that is hard to answer from logs: the
engines produce signals, and no order reaches the exchange -- at which stage
does it stop, and how often?

Three sections:

  ENVIRONMENT  the resolved venue, execution mode and authority. Whether
               credentials are present, never what they are.
  UNIVERSE     how many markets were discovered, how many the authenticated
               venue can actually execute, and how much of it is being
               studied -- the answer to "of the thousands we found, how many
               are we using?"
  COUNTERS     the persisted lifecycle funnel and blocker breakdown written
               by src/leantrader/execution/preflight.py, plus stage timings.
  PREFLIGHT    an optional live dry run for one symbol that walks the same
               checks a real order would and names the first that refuses.

The dry run reads markets, a ticker and the account balance. It never submits
an order, and --symbol is required before it will touch the network at all.

    python -m tools.execution_status
    python -m tools.execution_status --symbol BTC/USDT
    python -m tools.execution_status --symbol BTC/USDT --side buy --confidence 0.85
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict


def _repo_root_on_path() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)


_repo_root_on_path()

from src.leantrader.execution import preflight  # noqa: E402
from src.leantrader.execution.broker_ccxt import BrokerCCXT  # noqa: E402
from src.leantrader.universe.registry import universe  # noqa: E402


def _rule(title: str) -> None:
    print()
    print(title)
    print("-" * max(len(title), 60))


def _age(timestamp: float) -> str:
    if not timestamp:
        return "never"
    delta = max(0.0, time.time() - float(timestamp))
    if delta < 90:
        return f"{delta:.0f}s ago"
    if delta < 5400:
        return f"{delta / 60:.0f}m ago"
    return f"{delta / 3600:.1f}h ago"


def report_environment() -> BrokerCCXT:
    _rule("ENVIRONMENT")

    broker = preflight.shared_broker()
    described = broker.describe()

    for key in (
        "exchange",
        "market_mode",
        "requested_mode",
        "resolved_mode",
        "authority",
        "authenticated",
    ):
        print(f"  {key:16} {described.get(key)}")

    probe_errors = described.get("probe_errors") or {}
    if probe_errors:
        print(f"  {'probe_errors':16} {probe_errors}")

    # Which configuration inputs are set. Values are never printed.
    prefix = broker.exchange_id.replace("-", "_").upper()
    relevant = [
        "EXECUTION_MODE",
        "EXCHANGE_ID",
        "CCXT_EXCHANGE",
        "EXCHANGE_MODE",
        "BROKER_BACKEND",
        "EXECUTION_EXCHANGE_OVERRIDE",
        f"{prefix}_TESTNET_API_KEY_FILE",
        f"{prefix}_TESTNET_API_SECRET_FILE",
        f"{prefix}_TESTNET_API_KEY",
        f"{prefix}_TESTNET_API_SECRET",
    ]
    print()
    print("  configuration inputs (presence only):")
    for name in relevant:
        value = os.getenv(name)
        if value is None:
            state = "unset"
        elif name.endswith("_FILE"):
            state = f"set -> {value}" if os.path.exists(value) else f"set -> MISSING FILE {value}"
        elif "KEY" in name or "SECRET" in name:
            state = "set"
        else:
            state = f"set = {value}"
        print(f"    {name:38} {state}")

    if described.get("authority") not in {"testnet", "live"}:
        print()
        print("  >> No authenticated execution authority. Orders will be")
        print("     refused before they reach the exchange.")

    return broker


def report_universe(capital: float = 0.0) -> Dict[str, Any]:
    _rule("UNIVERSE")

    telemetry = universe.telemetry(capital_quote=capital)
    markets = telemetry["markets_discovered_by_venue"]

    if not markets:
        print("  No markets discovered yet.")
        print()
        print("  >> Discovery has not run, or its output never reached the")
        print("     registry. Every engine that trades from a pair list will")
        print("     be running on whatever it was seeded with.")
        return telemetry

    print("  discovered by venue:")
    for venue, count in sorted(markets.items(), key=lambda kv: -kv[1]):
        print(f"    {venue:<14} {count:>6}")

    print()
    for label, key in (
        ("normalized unique symbols", "normalized_unique_symbols"),
        ("active spot USDT markets", "active_spot_usdt_markets"),
        ("assigned to swarm shards", "markets_assigned_to_swarm"),
        ("studied in last 1m", "markets_analyzed_last_1m"),
        ("studied in last 5m", "markets_analyzed_last_5m"),
        ("studied ever", "markets_analyzed_ever"),
        ("micro candidates", "micro_candidates"),
        ("major candidates", "major_candidates"),
        ("signals generated", "signals_generated"),
    ):
        print(f"  {label:<28} {telemetry[key]}")

    for label, key in (
        ("signals by strategy", "signals_by_strategy"),
        ("signals by timeframe", "signals_by_timeframe"),
        ("signals by symbol (top)", "signals_by_symbol"),
    ):
        breakdown = telemetry.get(key) or {}
        if breakdown:
            print(f"  {label}:")
            for name, count in list(breakdown.items())[:10]:
                print(f"    {name:<24} {count}")

    print()
    print(f"  execution venue              {telemetry['execution_venue'] or '(none)'}")
    print(f"  execution eligible           {telemetry['execution_eligible']}")
    print(f"  execution ineligible         {telemetry['execution_ineligible']}")
    print(f"  eligibility unknown          {telemetry['execution_eligibility_unknown']}")

    reasons = telemetry["ineligibility_reasons"]
    if reasons:
        print()
        print("  not executable here, by reason:")
        width = max(len(r) for r in reasons)
        for reason, count in reasons.items():
            print(f"    {reason:<{width}}  {count:>6}")

    total = telemetry["normalized_unique_symbols"]
    studied = telemetry["markets_analyzed_ever"]
    if total:
        print()
        print(
            f"  >> Studying {studied}/{total} discovered markets "
            f"({studied / total * 100:.1f}%)."
        )

    return telemetry


def report_counters() -> Dict[str, Any]:
    _rule("COUNTERS")

    snapshot = preflight.telemetry_snapshot()
    path = os.getenv(
        "EXECUTION_TELEMETRY_PATH", "runtime/execution_telemetry.json"
    )
    print(f"  source            {path}")
    print(f"  last updated      {_age(snapshot.get('updated_at', 0))}")

    attempts = int(snapshot.get("attempts", 0))
    prepared = int(snapshot.get("prepared", 0))
    submitted = int(snapshot.get("submitted", 0))
    acknowledged = int(snapshot.get("acknowledged", 0))

    print()
    print("  lifecycle funnel:")
    print(f"    attempts        {attempts}")
    print(f"    prepared        {prepared}")
    print(f"    submitted       {submitted}")
    print(f"    acknowledged    {acknowledged}")
    print(f"    fills           {int(snapshot.get('fills', 0))}")
    print(f"    closes          {int(snapshot.get('closes', 0))}")

    if attempts == 0:
        print()
        print("  >> No execution attempts recorded at all. Nothing is reaching")
        print("     the execution stage; the break is upstream of preflight.")

    blockers = snapshot.get("blockers") or {}
    if blockers:
        print()
        print("  blockers (most frequent first):")
        rows = sorted(
            blockers.items(),
            key=lambda kv: int(kv[1].get("count", 0)),
            reverse=True,
        )
        width = max(len(name) for name, _ in rows)
        for name, entry in rows:
            count = int(entry.get("count", 0))
            seen = _age(entry.get("last_seen", 0))
            detail = str(entry.get("last_detail", ""))[:70]
            print(f"    {name:<{width}}  {count:>6}  {seen:>10}  {detail}")
    else:
        print()
        print("  blockers: none recorded")

    stages = snapshot.get("stages") or {}
    if stages:
        print()
        print("  stage timings (count, mean, max seconds):")
        width = max(len(name) for name in stages)
        for name, entry in sorted(stages.items()):
            count = int(entry.get("count", 0)) or 1
            total = float(entry.get("total_seconds", 0.0))
            worst = float(entry.get("max_seconds", 0.0))
            print(
                f"    {name:<{width}}  {count:>6}  "
                f"{total / count:>8.3f}  {worst:>8.3f}"
            )

    return snapshot


def report_preflight(args) -> int:
    _rule(f"PREFLIGHT DRY RUN  {args.symbol} {args.side}")
    print("  No order is submitted. This walks the same checks a real order")
    print("  would and reports the first stage that refuses.")
    print()

    intent = {
        "symbol": args.symbol,
        "side": args.side,
        "confidence": args.confidence,
        "order_type": "market",
    }
    if args.price:
        intent["price"] = args.price

    started = time.time()
    prepared, blocked = preflight.prepare_order(intent)
    elapsed = time.time() - started

    if prepared is None:
        print(f"  BLOCKED at stage '{blocked.stage}'")
        print(f"    class   {blocked.blocker}")
        print(f"    detail  {blocked.detail}")
        print(f"    took    {elapsed:.2f}s")
        return 1

    print("  PREPARED (not submitted)")
    for label, value in (
        ("symbol", prepared.symbol),
        ("side", prepared.side),
        ("amount", prepared.amount),
        ("price", prepared.price),
        ("notional", f"{prepared.notional:.8f} {prepared.quote_currency}"),
        ("free balance", f"{prepared.free_quote:.8f} {prepared.quote_currency}"),
        ("venue min notional", prepared.min_notional),
        ("venue min amount", prepared.min_amount),
        ("taker fee", prepared.fee_rate),
        ("venue", prepared.exchange_id),
        ("execution mode", prepared.execution_mode),
        ("sizing", prepared.sizing_reason),
        ("took", f"{elapsed:.2f}s"),
    ):
        print(f"    {label:20} {value}")

    print()
    print("  This order would be submitted through route_order. Nothing was")
    print("  sent by this command.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report where execution is stopping."
    )
    parser.add_argument(
        "--symbol",
        help="run a live preflight dry run for this symbol (reads markets, "
        "ticker and balance; submits nothing)",
    )
    parser.add_argument("--side", default="buy", choices=["buy", "sell"])
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--price", type=float, default=None)
    parser.add_argument(
        "--explain",
        metavar="SYMBOL",
        help="say why this particular market has or has not reached execution",
    )
    args = parser.parse_args()

    report_environment()

    capital = 0.0
    try:
        broker = preflight.shared_broker()
        if broker.authority in {"testnet", "live"}:
            balance = preflight.fetch_balance_cached(broker) or {}
            free = balance.get("free")
            if isinstance(free, dict):
                capital = float(free.get("USDT") or 0.0)
    except Exception:
        capital = 0.0

    report_universe(capital)
    report_counters()

    if args.explain:
        _rule(f"WHY {args.explain}")
        for key, value in universe.explain(args.explain).items():
            print(f"  {key:<22} {value}")

    if not args.symbol:
        print()
        print("Pass --symbol BTC/USDT to run a live preflight dry run,")
        print("or --explain DOGE/USDT to trace one market's eligibility.")
        return 0

    return report_preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
