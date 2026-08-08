"""Floor Operations Engine — v4.2.
Scan-to-seal pack verification, scored putaway, andon escalation with MTTR,
cutoff-aware SLA aging. Schema lives in db.init_db(); domain CRUD lives here.
Every state change publishes a realtime event.
"""
import json
import re
import uuid
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from db import connect, get_orders, enqueue_action, add_action_log
import realtime


def _now():
    return datetime.now().isoformat()


# ══════════════════════════ PACK STATION ══════════════════════════════

@dataclass
class ScanResult:
    ok: bool
    message: str
    progress: float = 0.0
    duplicate: bool = False
    unexpected: bool = False


class PackStationEngine:
    """open_for_order -> scan_item (per unit) -> set_weight -> complete.
    Wrong/duplicate SKUs hard-reject; weight outside tolerance blocks sealing."""
    UNIT_WEIGHT_G = 250
    CARTON_TYPES = [("Mailer S", 20, 15, 5, 6), ("Box M", 35, 25, 15, 32),
                    ("Box L", 50, 40, 30, 90), ("Oversize", 80, 60, 50, 240)]

    def open_for_order(self, order_id, station="PACK-1", operator=None):
        with connect() as conn:
            row = conn.execute("SELECT items FROM orders WHERE order_id=?", (order_id,)).fetchone()
        if not row:
            return {"ok": False, "message": f"Order {order_id} not found"}
        skus = [s.strip().upper() for s in (row["items"] or "").split(",") if s.strip()]
        if not skus:
            return {"ok": False, "message": f"Order {order_id} has no items"}
        expected = dict(Counter(skus))
        pack_id = "PK-" + hashlib.md5(order_id.encode()).hexdigest()[:8].upper()
        with connect() as conn:
            # FIX: parameterized station, no string replacement
            conn.execute("""INSERT INTO pack_tasks
                (pack_id, order_id, station, status, expected_items, expected_weight_g, packed_by)
                VALUES (?,?,?,?,'Open',?,?,?)
                ON CONFLICT(pack_id) DO UPDATE SET status='Open', flag_reason=NULL""",
                (pack_id, order_id, station, json.dumps(expected),
                 sum(q * self.UNIT_WEIGHT_G for q in expected.values()), operator))
        realtime.publish("PACK_OPENED", operator or "tower",
                         f"Carton {pack_id} opened for {order_id} @ {station}", ref_id=pack_id)
        return {"ok": True, "pack_id": pack_id,
                "message": f"Carton {pack_id} opened — {sum(expected.values())} units to pack"}

    def scan_item(self, pack_id, sku):
        with connect() as conn:
            t = conn.execute("SELECT * FROM pack_tasks WHERE pack_id=?", (pack_id,)).fetchone()
            if not t:
                return ScanResult(False, "Unknown carton — open it first")
            if t["status"] == "Done":
                return ScanResult(False, "Carton already sealed")
            expected = json.loads(t["expected_items"] or "{}")
            scanned = json.loads(t["scanned_items"] or "{}")
            sku_u = sku.strip().upper()
            if sku_u not in expected:
                realtime.publish("PACK_FLAG", t["packed_by"] or "floor",
                                 f"Wrong item: {sku_u} scanned into {t['order_id']}",
                                 ref_id=pack_id, severity="warning")
                return ScanResult(False, f"❌ {sku_u} is NOT in order {t['order_id']}", unexpected=True)
            if scanned.get(sku_u, 0) >= expected[sku_u]:
                return ScanResult(False, f"⚠️ {sku_u} already packed ×{expected[sku_u]} — duplicate", duplicate=True)
            scanned[sku_u] = scanned.get(sku_u, 0) + 1
            conn.execute("UPDATE pack_tasks SET scanned_items=?, status='Packing' WHERE pack_id=?",
                         (json.dumps(scanned), pack_id))
        done, total = sum(scanned.values()), sum(expected.values())
        msg = f"✅ {sku_u} packed ({done}/{total})" + (" — all items in, weigh & seal" if done == total else "")
        return ScanResult(True, msg, progress=done / total)

    def suggest_carton(self, pack_id):
        with connect() as conn:
            t = conn.execute("SELECT expected_items FROM pack_tasks WHERE pack_id=?", (pack_id,)).fetchone()
        units = sum(json.loads(t["expected_items"] or "{}").values()) if t else 1
        vol = units * 1500
        for name, l, w, h, _ in self.CARTON_TYPES:
            if l * w * h * 0.7 >= vol:
                return name
        return "Oversize"

    def set_weight(self, pack_id, grams):
        with connect() as conn:
            t = conn.execute("SELECT expected_weight_g, weight_tolerance_pct FROM pack_tasks WHERE pack_id=?",
                             (pack_id,)).fetchone()
            exp, tol = t["expected_weight_g"], t["weight_tolerance_pct"]
            dev = abs(grams - exp) / exp * 100 if exp else 0
            conn.execute("UPDATE pack_tasks SET actual_weight_g=? WHERE pack_id=?", (int(grams), pack_id))
        ok = dev <= tol
        return {"ok": ok, "deviation_pct": round(dev, 1),
                "message": (f"⚖️ Weight within tolerance (±{dev:.1f}%)" if ok
                            else f"⚠️ Weight off by {dev:.1f}% (tol ±{tol:.0f}%) — verify contents")}

    def complete(self, pack_id, operator):
        with connect() as conn:
            t = conn.execute("SELECT * FROM pack_tasks WHERE pack_id=?", (pack_id,)).fetchone()
            if not t:
                return {"ok": False, "message": "Unknown carton"}
            expected = json.loads(t["expected_items"] or "{}")
            scanned = json.loads(t["scanned_items"] or "{}")
            missing = {s: q - scanned.get(s, 0) for s, q in expected.items() if scanned.get(s, 0) < q}
            if missing:
                conn.execute("UPDATE pack_tasks SET status='Flagged', flag_reason=? WHERE pack_id=?",
                             (f"missing {missing}", pack_id))
                realtime.publish("PACK_FLAG", operator, f"{pack_id} flagged — missing {missing}",
                                 ref_id=pack_id, severity="critical")
                return {"ok": False, "message": f"🚫 Cannot seal — missing items: {missing}"}
            if (t["actual_weight_g"] and t["expected_weight_g"]
                    and abs(t["actual_weight_g"] - t["expected_weight_g"]) / t["expected_weight_g"] * 100 > t["weight_tolerance_pct"]):
                conn.execute("UPDATE pack_tasks SET status='Flagged', flag_reason='weight' WHERE pack_id=?", (pack_id,))
                return {"ok": False, "message": "🚫 Weight outside tolerance — re-check carton"}
            carton = t["carton_type"] or self.suggest_carton(pack_id)
            conn.execute("UPDATE pack_tasks SET status='Done', carton_type=?, completed_at=? WHERE pack_id=?",
                         (carton, _now(), pack_id))
            conn.execute("UPDATE orders SET status='Shipped', updated_at=? WHERE order_id=?", (_now(), t["order_id"]))
        realtime.publish("PACK_DONE", operator, f"Order {t['order_id']} sealed in {carton}",
                         ref_id=pack_id, payload={"order_id": t["order_id"], "carton": carton})
        enqueue_action("pack_done", {"pack_id": pack_id, "order_id": t["order_id"], "carton": carton})
        add_action_log("pack_done", pack_id, t["order_id"], operator)
        return {"ok": True, "carton": carton, "message": f"📦 Sealed & shipped — {carton}"}

    def active_packs(self):
        with connect() as conn:
            rows = conn.execute("""SELECT pack_id, order_id, station, status, expected_items, scanned_items,
                expected_weight_g, actual_weight_g, carton_type FROM pack_tasks
                WHERE status IN ('Open','Packing','Flagged') ORDER BY created_at DESC LIMIT 20""").fetchall()
        return [dict(r) for r in rows]

    def flagged_count(self):
        with connect() as conn:
            return conn.execute("SELECT COUNT(*) c FROM pack_tasks WHERE status='Flagged'").fetchone()["c"]

    @staticmethod
    def progress(task):
        exp = json.loads(task["expected_items"] or "{}")
        sc = json.loads(task["scanned_items"] or "{}")
        return sum(min(sc.get(k, 0), v) for k, v in exp.items()) / (sum(exp.values()) or 1)


# ══════════════════════════ PUTAWAY ═══════════════════════════════════

class PutawayEngine:
    """Bin scoring: +40 consolidation · +30 ABC affinity · +20 headroom · +10 dock proximity."""
    DOCK_DIST = {"A": 1, "B": 3, "C": 6, "D": 9}

    def _abc_for(self, sku):
        with connect() as conn:
            row = conn.execute("SELECT abc_class FROM inventory_policy WHERE sku=?", (sku,)).fetchone()
        return row["abc_class"] if row else "C"

    def suggest(self, sku, qty):
        abc = self._abc_for(sku)
        with connect() as conn:
            bins = conn.execute("SELECT * FROM bins WHERE active=1").fetchall()
            home = conn.execute("SELECT location FROM inventory WHERE sku=?", (sku,)).fetchone()
        scored = []
        for b in bins:
            headroom = b["capacity_units"] - b["current_units"]
            if headroom < qty:
                continue
            score, why = 0.0, []
            if home and b["location"] == home["location"]:
                score += 40; why.append("consolidate")
            if b["abc_affinity"] == abc:
                score += 30; why.append(f"ABC-{abc} zone")
            score += 20 * headroom / b["capacity_units"]
            score += max(0.0, 10 - self.DOCK_DIST.get(b["zone"], 9))
            why.append(f"{headroom} free")
            scored.append((round(score, 1), b["location"], " · ".join(why[:3])))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:3]

    def generate_from_asn(self, asn_id, operator=None):
        with connect() as conn:
            asn = conn.execute("SELECT * FROM asns WHERE asn_id=?", (asn_id,)).fetchone()
        if not asn:
            return []
        created = []
        for chunk in (asn["expected_items"] or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r"(.+?)\s*[x×]\s*(\d+)$", chunk, re.I)
            sku, qty = (m.group(1).strip().upper(), int(m.group(2))) if m else (chunk.upper(), 1)
            task_id = "PA-" + hashlib.md5((asn_id + sku).encode()).hexdigest()[:8].upper()
            top = self.suggest(sku, qty)
            with connect() as conn:
                conn.execute("""INSERT INTO putaway_tasks
                    (task_id, asn_id, sku, qty, from_location, suggested_location, assigned_to)
                    VALUES (?,?,?,?,?,?,?) ON CONFLICT(task_id) DO NOTHING""",
                    (task_id, asn_id, sku, qty, asn["dock_door"] or "DOCK",
                     top[0][1] if top else None, operator))
            created.append(task_id)
        return created

    def pending_tasks(self):
        with connect() as conn:
            return pd.read_sql_query("SELECT * FROM putaway_tasks WHERE status='Pending' ORDER BY created_at", conn)

    def pending_count(self):
        with connect() as conn:
            return conn.execute("SELECT COUNT(*) c FROM putaway_tasks WHERE status='Pending'").fetchone()["c"]

    def confirm(self, task_id, location, operator):
        with connect() as conn:
            t = conn.execute("SELECT * FROM putaway_tasks WHERE task_id=?", (task_id,)).fetchone()
            if not t:
                return {"ok": False, "message": "Unknown putaway task"}
            conn.execute("UPDATE putaway_tasks SET status='Done', actual_location=?, completed_at=? WHERE task_id=?",
                         (location, _now(), task_id))
            conn.execute("UPDATE bins SET current_units=MIN(capacity_units, current_units+?) WHERE location=?",
                         (t["qty"], location))
            conn.execute("UPDATE inventory SET location=?, updated_at=? WHERE sku=?", (location, _now(), t["sku"]))
        realtime.publish("PUTAWAY_DONE", operator, f"{t['qty']}× {t['sku']} → {location}",
                         ref_id=task_id, payload={"sku": t["sku"], "location": location})
        enqueue_action("putaway_done", {"task_id": task_id, "sku": t["sku"], "location": location})
        return {"ok": True, "message": f"🏗️ {t['sku']} stored at {location}"}


# ══════════════════════════ ANDON ═════════════════════════════════════

class AndonBoard:
    """Floor raises -> supervisor acks -> resolves. MTTR tracked automatically."""
    KINDS = {"blocked_bin": "🧱", "damage": "💥", "shortage": "📉", "equipment": "🔧", "other": "❓"}

    def raise_alert(self, zone, kind, message, raised_by, severity="high"):
        alert_id = f"AN-{uuid.uuid4().hex[:6].upper()}"
        with connect() as conn:
            conn.execute("INSERT INTO andon_alerts (alert_id, zone, kind, message, raised_by, severity) VALUES (?,?,?,?,?,?)",
                         (alert_id, zone, kind, message, raised_by, severity))
        realtime.publish("ANDON_RAISED", raised_by,
                         f"{self.KINDS.get(kind, '❓')} {kind.replace('_', ' ')} @ {zone} — {message}",
                         ref_id=alert_id, severity="critical" if severity == "high" else "warning")
        enqueue_action("andon_raised", {"alert_id": alert_id, "zone": zone, "kind": kind})
        return alert_id

    def ack(self, alert_id, supervisor):
        with connect() as conn:
            conn.execute("UPDATE andon_alerts SET status='ack', acked_at=?, resolved_by=? WHERE alert_id=?",
                         (_now(), supervisor, alert_id))

    def resolve(self, alert_id, supervisor):
        with connect() as conn:
            row = conn.execute("SELECT created_at FROM andon_alerts WHERE alert_id=?", (alert_id,)).fetchone()
            conn.execute("UPDATE andon_alerts SET status='resolved', resolved_at=?, resolved_by=? WHERE alert_id=?",
                         (_now(), supervisor, alert_id))
        mttr = round((datetime.now() - datetime.fromisoformat(row["created_at"])).total_seconds() / 60, 1) if row else None
        realtime.publish("ANDON_RESOLVED", supervisor, f"Alert {alert_id} resolved in {mttr} min", ref_id=alert_id)
        return mttr

    def open_alerts(self):
        with connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM andon_alerts WHERE status IN ('open','ack') ORDER BY created_at DESC", conn)

    def open_count(self):
        with connect() as conn:
            return conn.execute("SELECT COUNT(*) c FROM andon_alerts WHERE status IN ('open','ack')").fetchone()["c"]

    def stats(self):
        with connect() as conn:
            mttr = conn.execute("""SELECT AVG((julianday(resolved_at)-julianday(created_at))*1440) m
                FROM andon_alerts WHERE status='resolved'""").fetchone()["m"]
            zones = conn.execute("SELECT zone, COUNT(*) n FROM andon_alerts GROUP BY zone ORDER BY n DESC LIMIT 3").fetchall()
        return {"mttr_min": round(mttr or 0, 1), "hot_zones": [(r["zone"], r["n"]) for r in zones]}


# ══════════════════════════ SLA AGING ═════════════════════════════════

class SLAAging:
    def __init__(self, cutoff_hours=24):
        self.cutoff_hours = cutoff_hours

    def band(self):
        out = {"fresh": 0, "warm": 0, "hot": 0, "breached": 0, "breached_ids": [], "oldest_min": 0}
        df = get_orders()
        if df.empty:
            return out
        now = datetime.now()
        for _, r in df[df["status"] == "Pending"].iterrows():
            try:
                age = (now - datetime.fromisoformat(str(r["created_at"]))).total_seconds() / 60
            except (ValueError, TypeError):
                continue
            out["oldest_min"] = max(out["oldest_min"], int(age))
            if age > self.cutoff_hours * 60:
                out["breached"] += 1; out["breached_ids"].append(r["order_id"])
            elif age > 240:
                out["hot"] += 1
            elif age > 60:
                out["warm"] += 1
            else:
                out["fresh"] += 1
        return out
