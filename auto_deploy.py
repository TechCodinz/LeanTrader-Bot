"""Deprecated legacy deployment entry point.

The historical version mixed Python, shell heredocs, credentials, and a root
systemd unit in one syntactically invalid file. Production deployment is now
defined by Docker Compose and VPS_RUNBOOK.md so it can be reviewed and tested.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "auto_deploy.py is retired. Follow VPS_RUNBOOK.md and use "
        "`docker compose up -d --build` for the supported paper-only runtime.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
