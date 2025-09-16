from __future__ import annotations

import os
import time
from pathlib import Path

from integrations.telegram_bot import send_signal


LOG_PATH = Path(os.getenv("ULTRA_LOG_PATH", "runtime/ultra_paper.log"))
PERIOD_SEC = int(os.getenv("STATUS_PERIOD_SEC", "300"))  # default 5 min
TAIL_LINES = int(os.getenv("STATUS_TAIL_LINES", "60"))


def tail_text(path: Path, n: int) -> str:
    if not path.exists():
        return "(log not found)"
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(error reading log: {e})"


def main() -> int:
    # We will send via integrations.telegram_bot; if creds missing, it will no-op

    print(f"[status_daemon] Started. period={PERIOD_SEC}s log={LOG_PATH}")
    while True:
        try:
            txt = tail_text(LOG_PATH, TAIL_LINES)
            # Build message with optional training summary
            msg = "📡 Status update (paper mode):\n``" + "`\n" + txt + "\n```"
            try:
                day = time.strftime("%Y%m%d")
                p = Path("runtime/training_daily/") / f"{day}.json"
                if p.exists():
                    import json as _json
                    js = _json.loads(p.read_text())
                    results = js.get("results", [])
                    ok = sum(1 for r in results if isinstance(r, dict) and "ensemble" in r)
                    msg += f"\n🧠 Training: {ok}/{len(results)} models today"
            except Exception:
                pass
            try:
                send_signal(msg, vip=False)
            except Exception:
                pass
        except Exception as e:
            print(f"[status_daemon] send error: {e}")
        time.sleep(max(10, PERIOD_SEC))


if __name__ == "__main__":
    raise SystemExit(main())

