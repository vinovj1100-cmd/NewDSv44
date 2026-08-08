from flask import Flask, jsonify, request, render_template, Response
import json
from db import connect, get_orders, get_inventory, complete_pick_task, complete_cycle_count
from floor_ops import PackStationEngine, AndonBoard
import realtime

app = Flask(__name__, template_folder='templates', static_folder='static')
_pack = PackStationEngine()
_andon = AndonBoard()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/events")
def events():
    return Response(realtime.sse_stream(), mimetype="text/event-stream")

@app.route("/api/kpis")
def api_kpis():
    orders = get_orders()
    with connect() as conn:
        my = conn.execute("SELECT COUNT(*) c FROM pick_tasks WHERE status='Pending'").fetchone()["c"]
        pack = conn.execute("SELECT COUNT(*) c FROM pack_tasks WHERE status IN ('Open','Packing')").fetchone()["c"]
        andon = conn.execute("SELECT COUNT(*) c FROM andon_alerts WHERE status IN ('open','ack')").fetchone()["c"]
    return jsonify({"kpis": {
        "oee_pct": 78,
        "pick_completion_pct": 85,
        "open_picks": my,
        "pending_orders": int((orders["status"] == "Pending").sum()) if not orders.empty else 0,
        "stockout_risk_skus": 3,
        "cycle_accuracy_pct": 96
    }})

@app.route("/api/pick-tasks")
def api_pick_tasks():
    with connect() as conn:
        rows = conn.execute("SELECT task_id, wave_id, sku, order_id, location, qty FROM pick_tasks "
                            "WHERE status='Pending' ORDER BY created_at LIMIT 50").fetchall()
    return jsonify({"ok": True, "tasks": [dict(r) for r in rows]})

@app.route("/api/pick-tasks/<task_id>/complete", methods=["POST"])
def api_pick_complete(task_id):
    d = request.json
    complete_pick_task(task_id, d.get("picked_qty", 1), d.get("operator", "floor"))
    return jsonify({"ok": True})

@app.route("/api/cycle-counts")
def api_cycle_counts():
    with connect() as conn:
        rows = conn.execute("SELECT count_id, sku, location, expected_qty, abc_class FROM cycle_counts "
                            "WHERE status='Scheduled' ORDER BY created_at").fetchall()
    return jsonify({"ok": True, "counts": [dict(r) for r in rows]})

@app.route("/api/cycle-counts/<count_id>/complete", methods=["POST"])
def api_cycle_complete(count_id):
    d = request.json
    complete_cycle_count(count_id, d.get("counted_qty", 0), d.get("operator", "floor"))
    return jsonify({"ok": True})

@app.route("/api/inventory")
def api_inventory():
    q = request.args.get("q", "").strip()
    df = get_inventory()
    if q and not df.empty:
        df = df[df["sku"].str.contains(q, case=False, na=False) | df["product"].str.contains(q, case=False, na=False)]
    items = df.head(20).to_dict("records")
    return jsonify({"ok": True, "items": items})

@app.route("/api/pack/active")
def api_pack_active():
    return jsonify({"packs": _pack.active_packs()})

@app.route("/api/pack/<pack_id>")
def api_pack_get(pack_id):
    with connect() as conn:
        t = conn.execute("SELECT * FROM pack_tasks WHERE pack_id=?", (pack_id,)).fetchone()
    if not t:
        return jsonify({})
    import json as _json
    d = dict(t)
    d["expected"] = _json.loads(d.get("expected_items") or "{}")
    d["scanned"] = _json.loads(d.get("scanned_items") or "{}")
    d["carton_type"] = d.get("carton_type") or _pack.suggest_carton(pack_id)
    return jsonify(d)

@app.route("/api/pack/weight", methods=["POST"])
def api_pack_weight():
    d = request.json
    return jsonify(_pack.set_weight(d["pack_id"], d.get("grams", 0)))

@app.route("/api/andon/open")
def api_andon_open():
    return jsonify({"alerts": _andon.open_alerts().to_dict("records")})

@app.route("/api/andon/raise", methods=["POST"])
def api_andon_raise():
    d = request.json
    _andon.raise_alert(d.get("zone"), d.get("kind"), d.get("message"),
                       d.get("operator", "floor"), d.get("severity", "high"))
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
