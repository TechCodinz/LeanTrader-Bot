from __future__ import annotations

import os
import time
from pathlib import Path

from notifier import TelegramNotifier


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
    notif = TelegramNotifier()
    if not (notif.enabled and notif.token and notif.chat_id):
        print("[status_daemon] Telegram not enabled or missing credentials.")
        return 1

    print(f"[status_daemon] Started. period={PERIOD_SEC}s log={LOG_PATH}")
    while True:
        try:
            txt = tail_text(LOG_PATH, TAIL_LINES)
            notif.note("📡 Status update (paper mode):\n``" + "`\n" + txt + "\n```")
        except Exception as e:
            print(f"[status_daemon] send error: {e}")
        time.sleep(max(10, PERIOD_SEC))


if __name__ == "__main__":
    raise SystemExit(main())

