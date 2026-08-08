"""Ops Copilot — rule-based NL analytics over live state. Zero API keys."""
import re
from datetime import datetime
import pandas as pd

from db import (connect, get_orders, get_inventory_full, get_pick_tasks,
                get_labor_summary, get_cycle_counts, get_dock_appointments,
                get_waves, get_label_jobs, queue_status)
from floor_ops import AndonBoard, SLAAging, PutawayEngine

INTENTS = [
    (r"backlog|pending orders|how many orders",                     "backlog"),
    (r"late|breach|cutoff|aging|oldest",                            "aging"),
    (r"stock.?out|low stock|reorder|short",                         "stockout"),
    (r"top (?:picker|operator|packer)|productivity|labor|who did",  "labor"),
    (r"andon|alert|issue|blocked|escalat",                          "andon"),
    (r"dock|appointment|inbound",                                   "dock"),
    (r"accuracy|variance|cycle count",                              "accuracy"),
    (r"wave",                                                       "waves"),
    (r"pack|station|seal",                                          "pack"),
    (r"putaway|bin|store",                                          "putaway"),
    (r"label|sequenc|\bwb\b|ozon",                                  "labels"),
    (r"health|guardian|system status|degraded",                     "health"),
    (r"help|what can you",                                          "help"),
]


class OpsCopilot:
    def __init__(self):
        self._andon = AndonBoard()
        self._sla = SLAAging()
        self._putaway = PutawayEngine()

    def ask(self, question):
        for pattern, intent in INTENTS:
            if re.search(pattern, question, re.I):
                return getattr(self, f"_q_{intent}")()
        return {"answer": ("I report on: backlog · cutoff aging · stockouts · labor · andon · "
                           "dock · accuracy · waves · packing · putaway · labels · system health."),
                "followup": None, "data": None}

    def _q_backlog(self):
        df = get_orders()
        pending = df[df["status"] == "Pending"] if not df.empty else pd.DataFrame()
        n = len(pending)
        level = "critical 🚨" if n > 25 else "elevated ⚡" if n > 15 else "healthy ✅"
        return {"answer": f"{n} pending orders — backlog is {level}.",
                "followup": "Generate a pick wave to burn it down." if n > 10 else None,
                "data": pending.head(15) if n else None}

    def _q_aging(self):
        b = self._sla.band()
        return {"answer": (f"Order aging: {b['fresh']} fresh · {b['warm']} warm · {b['hot']} hot · "
                           f"{b['breached']} past cutoff. Oldest pending: {b['oldest_min']} min."),
                "followup": f"Breached: {', '.join(b['breached_ids'][:8])}" if b["breached"] else None,
                "data": None}

    def _q_stockout(self):
        df = get_inventory_full()
        if df.empty:
            return {"answer": "No inventory registered yet.", "followup": None, "data": None}
        risk = df[df["stock"] <= df["reorder_point"]]
        return {"answer": f"{len(risk)} SKUs at or below reorder point.",
                "followup": "Run the Replenishment engine." if len(risk) else None,
                "data": risk[["sku", "stock", "reorder_point", "abc_class"]].head(15) if len(risk) else None}

    def _q_labor(self):
        df = get_labor_summary()
        if df.empty:
            return {"answer": "No labor events logged yet.", "followup": None, "data": None}
        top = df.iloc[0]
        return {"answer": f"Top operator: {top['operator']} — {int(top['units'])} units across {int(top['events'])} events.",
                "followup": None, "data": df.head(10)}

    def _q_andon(self):
        alerts = self._andon.open_alerts()
        s = self._andon.stats()
        ans = f"{len(alerts)} open andon alert(s). MTTR: {s['mttr_min']} min."
        if s["hot_zones"]:
            ans += f" Hot zones: {', '.join(z for z, _ in s['hot_zones'])}."
        return {"answer": ans, "followup": "Acknowledge from the Andon Board." if len(alerts) else None,
                "data": alerts if len(alerts) else None}

    def _q_dock(self):
        df = get_dock_appointments()
        today = datetime.now().strftime("%Y-%m-%d")
        todays = df[df["appointment_time"].astype(str).str.startswith(today)] if not df.empty else pd.DataFrame()
        return {"answer": f"{len(todays)} appointment(s) today, {len(df)} total scheduled.",
                "followup": None, "data": todays if len(todays) else (df.head(10) if not df.empty else None)}

    def _q_accuracy(self):
        df = get_cycle_counts(status="Completed")
        if df.empty:
            return {"answer": "No completed cycle counts yet.", "followup": "Schedule counts from the Control Tower.", "data": None}
        exp, var = df["expected_qty"].sum(), df["variance"].abs().sum()
        acc = round(100 * (1 - var / exp), 2) if exp else 100.0
        return {"answer": f"Inventory accuracy: {acc}% across {len(df)} counts ({int(var)} units variance).",
                "followup": None,
                "data": df[["count_id", "sku", "expected_qty", "counted_qty", "variance"]].head(10)}

    def _q_waves(self):
        df = get_waves()
        open_w = df[df["status"] == "Open"] if not df.empty else pd.DataFrame()
        tasks = get_pick_tasks(status="Pending")
        return {"answer": f"{len(open_w)} open wave(s), {len(tasks)} pending pick tasks.",
                "followup": None, "data": open_w.head(10) if len(open_w) else None}

    def _q_pack(self):
        with connect() as conn:
            row = conn.execute("""SELECT
                SUM(CASE WHEN status IN ('Open','Packing') THEN 1 ELSE 0 END) active,
                SUM(CASE WHEN status='Flagged' THEN 1 ELSE 0 END) flagged,
                SUM(CASE WHEN status='Done' THEN 1 ELSE 0 END) done FROM pack_tasks""").fetchone()
        return {"answer": (f"Pack stations: {row['active'] or 0} in progress, "
                           f"{row['flagged'] or 0} flagged, {row['done'] or 0} sealed."),
                "followup": "Investigate flagged cartons." if row["flagged"] else None, "data": None}

    def _q_putaway(self):
        n = self._putaway.pending_count()
        return {"answer": f"{n} pending putaway task(s) from receiving.",
                "followup": "Generate putaway from a received ASN in the Control Tower." if n == 0 else None,
                "data": self._putaway.pending_tasks().head(10) if n else None}

    def _q_labels(self):
        df = get_label_jobs(10)
        if df.empty:
            return {"answer": "No label sequencing jobs recorded yet.", "followup": None, "data": None}
        rate = round(df["matched_count"].sum() / max(1, df["targets_count"].sum()) * 100, 1)
        return {"answer": f"{len(df)} recent sequencing job(s) — {rate}% overall match rate.",
                "followup": None,
                "data": df[["job_id", "marketplace", "mode", "targets_count", "matched_count",
                            "missing_count", "avg_confidence", "created_at"]]}

    def _q_health(self):
        qs = queue_status()
        orders = get_orders()
        pending = int((orders["status"] == "Pending").sum()) if not orders.empty else 0
        inv = get_inventory_full()
        low = len(inv[inv["stock"] <= inv["reorder_point"]]) if not inv.empty else 0
        andon = self._andon.open_count()
        score = 1.0 - min(0.8, pending * 0.01 + qs["queued"] * 0.004 + low * 0.01 + andon * 0.07)
        status = "HEALTHY" if score > 0.8 else "DEGRADED" if score > 0.5 else "CRITICAL"
        return {"answer": (f"System {status} (~{int(score * 100)}%): {pending} pending orders, "
                           f"{qs['queued']} queued, {low} low-stock SKUs, {andon} andon open."),
                "followup": "Open the Guardian tab for the full Ops Center." if status != "HEALTHY" else None,
                "data": None}

    def _q_help(self):
        return {"answer": ("Ask about: backlog · cutoff aging · stockouts · labor · andon · dock · "
                           "accuracy · waves · packing · putaway · labels · system health."),
                "followup": None, "data": None}
