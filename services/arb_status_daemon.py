
import os
import time
from typing import List

from scanners.arbitrage import cross_exchange_spreads

def _env_list(name: str, default: str) -> List[str]:
    v = os.getenv(name, default)
    return [s.strip() for s in v.split(",") if s.strip()]

def main() -> int:
    # Configure
    symbols = _env_list("ARB_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
    venues = _env_list("ARB_VENUES", os.getenv("EXCHANGE_SET", "binance,bybit,okx,kraken,gateio"))
    min_bps = float(os.getenv("ARB_MIN_BPS", "12"))
    period = int(os.getenv("ARB_PERIOD_SEC", "120"))

    # Telegram (optional)
    try:
        from integrations.telegram import send_telegram_message
        tg_send = send_telegram_message
    except Exception:
        tg_send = None

    # Announce startup
    try:
        if tg_send:
            tg_send(
                f"📡 Arb daemon online: symbols={len(symbols)} venues={len(venues)} min_bps={min_bps}"
            )
    except Exception:
        pass

    while True:
        try:
            opps = cross_exchange_spreads(symbols, venues, min_bps=min_bps)
            if opps and tg_send:
                # Post top 3 opportunities
                opps = sorted(opps, key=lambda o: float(o.get("spread_bps", 0.0)), reverse=True)[:3]
                lines = [
                    f"{o['symbol']}: {o['buy_venue']}→{o['sell_venue']} {float(o['spread_bps']):.1f}bps size≈{float(o['size_cap']):.4f}"
                    for o in opps
                ]
                msg = "🤖 Arbitrage opportunities:\n" + "\n".join(lines)
                try:
                    tg_send(msg)
                except Exception:
                    pass
        except Exception:
            # Best-effort; do not crash the loop
            pass
        time.sleep(max(10, period))

if __name__ == "__main__":
    raise SystemExit(main())

