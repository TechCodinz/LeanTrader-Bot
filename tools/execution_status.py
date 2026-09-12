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
from src.leantrader.universe.registry import universe  # noqa: E402
from src.leantrader.universe import routing  # noqa: E402
from src.leantrader.universe import venues as venue_capabilities  # noqa: E402
from src.leantrader.execution import intent as execution_intent  # noqa: E402
from src.leantrader.execution import inventory as inventory_reconciler  # noqa: E402


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


def report_environment():
    """Print the resolved execution environment, or why it cannot resolve.

    Building the broker can fail outright -- most often a *_FILE credential
    path that is not mounted, which raises by design so a broken mount does
    not run unauthenticated. That is precisely the situation this command
    exists to diagnose, so the failure is reported rather than propagated.
    """
    _rule("ENVIRONMENT")

    broker = None
    described = {}
    try:
        broker = preflight.shared_broker()
        described = broker.describe()
    except Exception as e:
        print(f"  !! cannot resolve execution environment: {type(e).__name__}")
        print(f"     {e}")
        print()
        print("  >> Nothing can execute until this is fixed. A missing")
        print("     credential file is the usual cause; see SECRETS.md.")

    if described:
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
    exchange_id = getattr(broker, "exchange_id", None) or (
        os.getenv("CCXT_EXCHANGE") or os.getenv("EXCHANGE_ID") or "bybit"
    )
    prefix = str(exchange_id).replace("-", "_").upper()
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

    if described and described.get("authority") not in {"testnet", "live"}:
        print()
        print("  >> No authenticated execution authority. Orders will be")
        print("     refused before they reach the exchange.")

    return broker


def load_runtime_snapshot() -> Dict[str, Any]:
    """Read what the running process discovered.

    Both registries are in-process singletons, so this command saw an empty
    universe no matter how much the runtime had found. The runtime writes a
    snapshot; this reads it. Nothing is fabricated when the file is absent --
    the report says the runtime has not written one.
    """
    payload = venue_capabilities.read_snapshot()
    if not payload:
        return {}
    try:
        venue_capabilities.capabilities.load_snapshot(payload)
    except Exception:
        pass
    return payload


def report_universe(capital: float = 0.0) -> Dict[str, Any]:
    _rule("UNIVERSE")

    snapshot = load_runtime_snapshot()
    if snapshot:
        age = venue_capabilities.snapshot_age_seconds(snapshot)
        print(f"  snapshot source    {snapshot.get('snapshot_path', '?')}")
        print(f"  snapshot written   {_age(snapshot.get('written_at', 0))}")
        print(f"  markets persisted  {len(snapshot.get('markets') or [])}")
        if age is not None and age > 300:
            print()
            print("  !! STALE. This is the last state the runtime wrote, not")
            print("     current truth. Everything below is as of that moment.")
        print()
    else:
        print("  No runtime snapshot found; showing this process only.")
        print()

    telemetry = snapshot.get("universe") or universe.telemetry(
        capital_quote=capital
    )
    markets = telemetry.get("markets_discovered_by_venue") or {}

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
    print(f"    {'GLOBAL':<14} {telemetry.get('normalized_unique_symbols', 0):>6}")

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


def report_venue_capabilities(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    _rule("VENUE CAPABILITIES")

    telemetry = (
        snapshot.get("capabilities")
        or venue_capabilities.capabilities.telemetry()
    )

    listed = telemetry.get("listed_by_venue_environment") or {}
    if listed:
        print("  markets listed, by venue and environment:")
        width = max(len(k) for k in listed)
        for label, count in listed.items():
            print(f"    {label:<{width}}  {count:>6}")
    else:
        print("  No venue capability metadata recorded yet.")

    not_listed = telemetry.get("known_not_listed_by_venue") or {}
    if not_listed:
        print()
        print("  known NOT listed (resolved locally, never called):")
        width = max(len(k) for k in not_listed)
        for venue, count in not_listed.items():
            print(f"    {venue:<{width}}  {count:>6}")

    suppressed = telemetry.get("suppressed_calls_by_reason") or {}
    total = telemetry.get("suppressed_calls_total", 0)
    print()
    print(f"  venue calls avoided     {total}")
    if suppressed:
        width = max(len(k) for k in suppressed)
        for reason, count in suppressed.items():
            print(f"    {reason:<{width}}  {count:>6}")

    by_state = telemetry.get("by_state") or {}
    if by_state:
        print()
        print("  capability states:")
        width = max(len(k) for k in by_state)
        for state, count in by_state.items():
            print(f"    {state:<{width}}  {count:>6}")

    if total:
        print()
        print(
            f"  >> {total} exchange calls were answered from capability memory"
        )
        print("     instead of becoming errors. That is the routing working.")

    return telemetry


def report_inventory() -> Dict[str, Any]:
    _rule("INVENTORY")

    try:
        broker = preflight.shared_broker()
    except Exception as exc:
        print(f"  cannot read the account: {type(exc).__name__}")
        return {}

    report = inventory_reconciler.reconcile_from_broker(broker)

    if not report.get("available"):
        print(f"  {report.get('reason', 'unavailable')}")
        return report

    summary = report["summary"]
    capital = report["capital"]

    print(f"  venue / environment  {report['venue']} / {report['environment']}")
    print(f"  non-zero assets      {summary['nonzero_assets']}")
    for label, key in (
        ("managed positions", "managed_positions"),
        ("orphaned positions", "orphaned_positions"),
        ("dust assets", "dust_assets"),
        ("non-executable", "non_executable"),
        ("exit eligible", "exit_eligible"),
        ("exit pending", "exit_pending"),
    ):
        print(f"  {label:<20} {summary[key]}")

    print()
    print(f"  free quote           {capital['free_quote']}")
    print(f"  spendable            {capital['spendable_quote']}")
    print(f"  inventory value      {capital['inventory_value']}")
    print(f"  reclaimable capital  {capital['reclaimable_capital']}")
    print(f"  portfolio value      {capital['portfolio_value']}")
    print(f"  cash fraction        {capital['cash_fraction']}")

    items = report.get("items") or []
    if items:
        print()
        print("  holdings:")
        width = max(len(i["asset"]) for i in items)
        for item in items:
            print(
                f"    {item['asset']:<{width}}  {item['classification']:<24} "
                f"value={item['liquidation_value']:.6f} "
                f"exit={item['exit_state'] or '-'} "
                f"owner={item['owner'] or 'unknown'}"
            )
            if item["exit_blocked_reason"]:
                print(f"    {'':<{width}}  └─ {item['exit_blocked_reason']}")

    if summary["orphaned_positions"]:
        print()
        print("  >> Orphaned inventory has no live owner and no exit logic.")
        print("     Capital stays committed until something adopts it.")

    return report


def report_intents() -> Dict[str, Any]:
    _rule("EXECUTION INTENTS")

    summary = execution_intent.outcome_summary()
    if not summary["traced_intents"]:
        print("  No traced intents in this process.")
        return summary

    print(f"  traced intents       {summary['traced_intents']}")

    for label, key in (
        ("stopped at stage", "stopped_by_stage"),
        ("stopped with outcome", "stopped_by_outcome"),
        ("by source engine", "by_source_engine"),
    ):
        breakdown = summary[key]
        if not breakdown:
            continue
        print()
        print(f"  {label}:")
        width = max(len(k) for k in breakdown)
        for name, count in breakdown.items():
            print(f"    {name:<{width}}  {count:>6}")

    return summary


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

    recovered = int(snapshot.get("recovered_orders", 0))
    external = int(snapshot.get("reconciled_external_orders", 0))
    if recovered or external:
        print()
        print("  recovered from previous runs (not this run's submissions):")
        print(f"    recovered_orders            {recovered}")
        print(f"    reconciled_external_orders  {external}")

    current = preflight.current_run_counters()
    if current:
        print()
        print(f"  this run ({preflight.RUN_ID}):")
        for name in preflight.LIFECYCLE_ORDER:
            print(f"    {name:<15} {int(current.get(name, 0))}")

    violations = preflight.lifecycle_violations(snapshot)
    if violations:
        print()
        print("  !! LIFECYCLE INVARIANT VIOLATED:")
        for violation in violations:
            print(f"     {violation}")
        print("     Cumulative totals span restarts; compare the per-run block")
        print("     above, or look for a counter incremented at the wrong place.")

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

    snapshot = load_runtime_snapshot()
    report_universe(capital)
    report_venue_capabilities(snapshot)
    report_inventory()
    report_counters()
    report_intents()

    if args.explain:
        _rule(f"WHY {args.explain}")

        from src.leantrader.universe.registry import normalize_symbol

        canonical = normalize_symbol(args.explain) or args.explain
        print(f"  canonical symbol       {canonical}")

        for key, value in universe.explain(args.explain).items():
            if key != "canonical_symbol":
                print(f"  {key:<22} {value}")

        capability = venue_capabilities.capabilities.explain(canonical)
        observed = capability.get("venues") or {}

        print()
        print(f"  observed on venues     {sorted(observed) or '(none recorded)'}")
        for venue, environments in observed.items():
            for environment, detail in environments.items():
                print(
                    f"    {venue}:{environment:<8} {detail['state']:<24} "
                    f"{detail['venue_symbol'] or '-'}"
                )
                print(
                    f"      evidence={detail['evidence']} "
                    f"verified={detail['verification_count']}x "
                    f"refresh_in={detail['next_refresh_in_seconds']}s"
                )

        try:
            broker = preflight.shared_broker()
            environment = broker.resolve_mode()
        except Exception:
            environment = "unknown"

        print()
        print(f"  execution environment  {environment}")

        if environment in {"paper", "testnet", "live"}:
            decision = routing.select_venue(canonical, environment)
            print(f"  chosen venue           {decision.chosen_venue or '(none)'}")
            print(f"  classification         {decision.classification}")
            print(f"  reason                 {decision.reason}")
            if decision.considered:
                print("  venues considered:")
                width = max(len(v) for v in decision.considered)
                for venue, verdict in decision.considered.items():
                    print(f"    {venue:<{width}}  {verdict}")
            print(f"  research venues        {decision.research_venues or '(none)'}")
            print(
                f"  attention lane         "
                f"{routing.attention_lane(canonical, environment)}"
            )

    if not args.symbol:
        print()
        print("Pass --symbol BTC/USDT to run a live preflight dry run,")
        print("or --explain DOGE/USDT to trace one market's eligibility.")
        return 0

    return report_preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
