import json
import pathlib
import yaml  # type: ignore

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "exchange_profiles.yml"
CACHE_DIR = ROOT / "runtime" / "exchange_intel"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_yaml(p: pathlib.Path):
    return yaml.safe_load(open(p, "r", encoding="utf-8")) if p.exists() else {}


def _save_json(p: pathlib.Path, data):
    json.dump(data, open(p, "w", encoding="utf-8"))


def fetch_describe(ex):
    d = ex.describe()
    return {k: d.get(k) for k in ["id", "name", "rateLimit", "fees", "precision", "limits", "timeframes"]}


def normalize(desc):
    limits = desc.get("limits") or {}
    fees = desc.get("fees") or {}
    prec = desc.get("precision") or {}
    return {
        "rate_limit_ms": desc.get("rateLimit", 50),
        "maker_fee_pct": (fees.get("trading", {}).get("maker", 0) or 0) * 100,
        "taker_fee_pct": (fees.get("trading", {}).get("taker", 0) or 0) * 100,
        "price_precision": prec.get("price", "auto"),
        "amount_precision": prec.get("amount", "auto"),
        "min_amount": (limits.get("amount", {}) or {}).get("min", 0),
        "max_amount": (limits.get("amount", {}) or {}).get("max", None),
        "min_cost": (limits.get("cost", {}) or {}).get("min", 0),
        "max_cost": (limits.get("cost", {}) or {}).get("max", None),
    }


def build_profile(exchange_id, ex_obj, overrides):
    desc = fetch_describe(ex_obj)
    base = normalize(desc)
    prof = {"id": exchange_id, **base, **(overrides or {})}
    return prof


def refresh(exchange_id, ex_obj, overrides):
    prof = build_profile(exchange_id, ex_obj, overrides)
    _save_json(CACHE_DIR / f"{exchange_id}.json", prof)
    return prof


def load_cached(exchange_id):
    p = CACHE_DIR / f"{exchange_id}.json"
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}


def get_profile(exchange_id, ex_factory, force=False):
    cfg = _load_yaml(PROFILES).get(exchange_id, {})
    if not force:
        cached = load_cached(exchange_id)
        if cached:
            return {**cfg, **cached}
    ex = ex_factory()
    return {**cfg, **refresh(exchange_id, ex, cfg)}


