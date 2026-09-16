"""Simple model registry helpers: list models and read metadata."""

# Restored: these names were used below but never imported, so this
# module raised NameError on import.
from typing import Dict
from typing import List
from pathlib import Path

def list_models() -> List[Path]:
    p = Path("runtime") / "models"
    if not p.exists():
        return []
    return sorted(
        [x for x in p.iterdir() if x.suffix == ".pkl"],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

def read_meta(model_path: Path) -> Dict[str, object]:
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        import json

        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

if __name__ == "__main__":
    for m in list_models():
        print(m)
        print(read_meta(m))
