from __future__ import annotations

import hashlib
import re
from pathlib import Path


def test_vps_installer_bootstrap_hash_matches_pinned_script():
    bootstrap = Path("scripts/bootstrap_verified_vps.sh").read_bytes()
    installer = Path("scripts/install_vps_ops_bridge.sh").read_text(encoding="utf-8")

    match = re.search(r'readonly BOOTSTRAP_SHA="([0-9a-f]{64})"', installer)
    assert match is not None
    assert match.group(1) == hashlib.sha256(bootstrap).hexdigest()
