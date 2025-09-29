try:
    from flask import Flask, jsonify, request  # type: ignore
except Exception:
    Flask = None  # type: ignore
    def jsonify(x):  # type: ignore
        return x
    class _Req:  # type: ignore
        json = {}
    request = _Req()  # type: ignore

try:
    from router import ExchangeRouter  # type: ignore
    from ultra_core import UltraCore  # type: ignore
except Exception:
    ExchangeRouter = None  # type: ignore
    UltraCore = None  # type: ignore

app = Flask(__name__) if Flask else None
core = UltraCore(ExchangeRouter(), None) if (UltraCore and ExchangeRouter) else None

@app.route("/signals", methods=["GET"]) if app else (lambda f: f)
def get_signals():
    if not core:
        return jsonify({"ok": False, "error": "core unavailable"}), 503
    signals = core.scout_opportunities(core.scan_markets())
    return jsonify(signals)

@app.route("/trade", methods=["POST"]) if app else (lambda f: f)
def execute_trade():
    if not core:
        return jsonify({"ok": False, "error": "core unavailable"}), 503
    data = request.json or {}
    plans = core.plan_trades([data])
    results = core.enter_trades(plans)
    return jsonify(results)

if __name__ == "__main__":
    if app:
        app.run(host="0.0.0.0", port=5000)
    else:
        print("Flask not available; mobile_trading_api is disabled.")
