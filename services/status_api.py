from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse, JSONResponse

LOG_PATH = Path(os.getenv("ULTRA_LOG_PATH", "runtime/ultra_paper.log"))

app = FastAPI(title="Ultra Status API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/tail", response_class=PlainTextResponse)
def tail(n: int = Query(100, ge=1, le=1000)) -> str:
    if not LOG_PATH.exists():
        return "(log not found)"
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(error reading log: {e})"


@app.get("/status")
def status() -> JSONResponse:
    # Simple status derived from last status block in the log
    if not LOG_PATH.exists():
        return JSONResponse({"ok": False, "error": "log not found"})
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
        # find last occurrence of header line
        marker = "ULTRA TRADING SYSTEM STATUS"
        idx = text.rfind(marker)
        if idx == -1:
            return JSONResponse({"ok": True, "note": "no status block yet"})
        block = text[idx:].splitlines()[:30]
        return JSONResponse({"ok": True, "block": block})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

