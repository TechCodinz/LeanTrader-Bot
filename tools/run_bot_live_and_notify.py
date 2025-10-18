import os, sys, json, subprocess, time, urllib.parse, urllib.request

def tg_send(text: str) -> None:
    if os.getenv("TELEGRAM_ENABLED","false").lower() != "true":
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN","")
    chat  = os.getenv("TG_ADMIN_CHAT_ID","")
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat, "text": text}
    try:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode("utf-8"))
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass

def pick_last(d):
    if isinstance(d, list) and d:
        return d[-1]
    return d

def extract_exec(summary: dict) -> dict:
    # Try common fields first
    sym  = summary.get("symbol") or summary.get("market") or summary.get("pair")
    side = summary.get("side") or summary.get("action") or summary.get("signal")
    qty  = summary.get("amount") or summary.get("qty") or summary.get("size")
    px   = summary.get("avg_fill_price") or summary.get("fill_price") or summary.get("price")
    # Fall back to orders/trades arrays
    if not (sym and side and (qty or px)):
        orders = summary.get("orders") or summary.get("order") or []
        o = pick_last(orders)
        if isinstance(o, dict):
            sym  = sym  or o.get("symbol") or o.get("market") or o.get("pair")
            side = side or o.get("side")
            qty  = qty  or o.get("filled") or o.get("amount") or o.get("qty") or o.get("size")
            px   = px   or o.get("avg_fill_price") or o.get("price") or o.get("fill_price")
        trades = summary.get("trades") or summary.get("trade") or []
        t = pick_last(trades)
        if isinstance(t, dict):
            sym  = sym  or t.get("symbol") or t.get("market") or t.get("pair")
            side = side or t.get("side")
            qty  = qty  or t.get("amount") or t.get("qty") or t.get("size")
            px   = px   or t.get("price")
    # Additional metrics
    tp   = summary.get("take_profit") or summary.get("tp")
    sl   = summary.get("stop_loss") or summary.get("sl")
    conf = summary.get("confidence") or summary.get("prob") or summary.get("score")
    delta= summary.get("delta") or summary.get("pnl_delta") or summary.get("pnl")
    rpnl = summary.get("realized_pnl") or summary.get("realizedPnL") or summary.get("pnl_realized")
    upnl = summary.get("unrealized_pnl") or summary.get("unrealizedPnL") or summary.get("pnl_unrealized")
    bal  = summary.get("balance_after") or summary.get("balance") or summary.get("equity")
    sess = summary.get("session_weight") or summary.get("session") or summary.get("sessionTag")
    return {
        "sym": sym, "side": side, "qty": qty, "px": px,
        "tp": tp, "sl": sl, "conf": conf, "delta": delta,
        "rpnl": rpnl, "upnl": upnl, "bal": bal, "sess": sess
    }

def main():
    cmd = [sys.executable, os.path.join("tools","run_bot_live.py")]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out, err = (proc.stdout or "").strip(), (proc.stderr or "").strip()
        if out:
            print(out, flush=True)  # preserve stdout for daemon parsing

        # Parse last JSON-looking line, else try whole stdout
        summary = {}
        try:
            candidate = out.strip().splitlines()[-1] if out else ""
            summary = json.loads(candidate) if candidate.startswith("{") else json.loads(out)
        except Exception:
            if err:
                tg_send(f"[live] stderr:\n{err[:1800]}")
            return

        info = extract_exec(summary)
        parts = []
        parts.append(f"[live exec] {info.get('sym') or '-'} {info.get('side') or '-'} qty={info.get('qty') or '-'} px={info.get('px') or '-'}")
        if info.get("tp") or info.get("sl"): parts.append(f"tp={info['tp']} sl={info['sl']}")
        if isinstance(info.get("conf"), (int,float)): parts.append(f"conf={round(info['conf'],4)}")
        if info.get("sess"): parts.append(f"sess={info['sess']}")
        if info.get("delta") is not None: parts.append(f"ΔPnL={info['delta']}")
        if info.get("rpnl") is not None: parts.append(f"realized={info['rpnl']}")
        if info.get("upnl") is not None: parts.append(f"unrealized={info['upnl']}")
        if info.get("bal") is not None: parts.append(f"balance={info['bal']}")
        msg = " | ".join(parts)
        tg_send(msg[:4000])
    except Exception as e:
        tg_send(f"[live] wrapper error: {e}")

if __name__ == "__main__":
    main()
