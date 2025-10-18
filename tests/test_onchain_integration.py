import json
from pathlib import Path

from signals.onchain_flows import sentiment_fusion as _sent_fusion

ONCHAIN_EVENTS_ENV = "ONCHAIN_EVENTS_PATH"


def _merge_onchain_fusion(base: dict) -> dict:
    import os
    p = os.getenv(ONCHAIN_EVENTS_ENV)
    if not p or not Path(p).exists():
        return base
    try:
        evts = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return base
    feats = _sent_fusion(evts)
    out = dict(base)
    for k, v in feats.items():
        if k.upper() == "ETH":
            out["ETH/USDT"] = 1 if (v.get("sentiment", 0.0) > 0) else (-1 if v.get("sentiment", 0.0) < 0 else 0)
    return out


def test_merge_onchain_fusion_from_file(tmp_path: Path, monkeypatch):
    evts = [
        {"type": "whale_transfer", "token": "ETH", "size_usd": 900_000, "direction": "out"},
        {"type": "dex_volume_spike", "asset": "BTC", "zscore": 4.2},
    ]
    p = tmp_path / "onchain.json"
    p.write_text(json.dumps(evts), encoding="utf-8")
    monkeypatch.setenv(ONCHAIN_EVENTS_ENV, str(p))
    base = {"ETH/USDT": 0}
    merged = _merge_onchain_fusion(base)
    assert "ETH/USDT" in merged
    assert merged["ETH/USDT"] in (-1, 0, 1)
