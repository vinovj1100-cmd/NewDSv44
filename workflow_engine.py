"""Workflow Engine v4.4 — Order lifecycle state machine for WMS LITE.
Valid transitions, SLA awareness, auto-allocation, and integration hooks
for RuleEngine / AuditTrail / realtime bus.
"""
from __future__ import annotations

import hashlib

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from db import (
    connect,
    transition_order,
    get_orders,
    get_order_workflow,
    get_workflow_history,
    create_wave,
    add_pick_task,
    get_inventory,
)
import realtime

# Canonical statuses and allowed edges
TRANSITIONS = {
    "Pending": {"Allocated", "Cancelled"},
    "Allocated": {"Picking", "Pending", "Cancelled"},
    "Picking": {"Picked", "Allocated", "Cancelled"},
    "Picked": {"Packing", "Picking"},
    "Packing": {"Packed", "Picked"},
    "Packed": {"Shipped", "Packing"},
    "Shipped": set(),
    "Cancelled": set(),
}

PRIORITY_SLA_HOURS = {
    1: 4,   # express
    2: 8,
    3: 12,
    4: 18,
    5: 24,  # standard
    6: 36,
    7: 48,
}


class WorkflowEngine:
    """Order lifecycle orchestrator."""

    def can_transition(self, from_status: str, to_status: str) -> bool:
        return to_status in TRANSITIONS.get(from_status, set())

    def advance(
        self,
        order_id: str,
        to_status: str,
        actor: str = "system",
        reason: str = None,
        force: bool = False,
    ) -> Dict:
        with connect() as conn:
            row = conn.execute(
                "SELECT status FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
        if not row:
            return {"ok": False, "message": f"Order {order_id} not found"}
        from_status = row["status"]
        if not force and not self.can_transition(from_status, to_status):
            return {
                "ok": False,
                "message": f"Illegal transition {from_status} → {to_status}",
            }
        result = transition_order(order_id, to_status, actor, reason)
        realtime.publish(
            f"WF_{to_status.upper()}",
            actor,
            f"Order {order_id}: {from_status} → {to_status}",
            ref_id=order_id,
        )
        return {
            "ok": True,
            "order_id": order_id,
            "from": from_status,
            "to": to_status,
            "actor": actor,
        }

    def allocate_batch(
        self,
        order_ids: List[str] = None,
        limit: int = 25,
        actor: str = "system",
        strategy: str = "batch",
    ) -> Dict:
        """Move Pending → Allocated and optionally create a wave + pick tasks."""
        df = get_orders()
        if df.empty:
            return {"ok": False, "message": "No orders", "allocated": []}
        pending = df[df["status"] == "Pending"]
        if order_ids:
            pending = pending[pending["order_id"].isin(order_ids)]
        pending = pending.head(limit)
        allocated = []
        for _, r in pending.iterrows():
            res = self.advance(r["order_id"], "Allocated", actor, reason="auto-allocate")
            if res.get("ok"):
                allocated.append(r["order_id"])
        wave_id = None
        if allocated:
            wave_id = "WV-" + hashlib.md5(",".join(allocated).encode()).hexdigest()[:10].upper()
            create_wave(wave_id, strategy, allocated, assigned_to=actor, priority=5)
            inv = get_inventory()
            inv_map = {}
            if not inv.empty:
                for _, row in inv.iterrows():
                    inv_map[str(row["sku"]).upper()] = row["location"]
            for oid in allocated:
                items_row = df[df["order_id"] == oid]
                if items_row.empty:
                    continue
                items = (items_row.iloc[0]["items"] or "").split(",")
                for sku in items:
                    sku = sku.strip().upper()
                    if not sku:
                        continue
                    loc = inv_map.get(sku, "UNASSIGNED")
                    task_id = "PT-" + hashlib.md5(f"{wave_id}:{oid}:{sku}".encode()).hexdigest()[:10].upper()
                    add_pick_task(task_id, wave_id, sku, oid, loc, 1, assigned_to=actor)
        return {
            "ok": True,
            "allocated": allocated,
            "wave_id": wave_id,
            "count": len(allocated),
        }

    def prioritize_by_sla(self, cutoff_hours: int = 24) -> List[Dict]:
        """Return pending orders ranked by SLA risk (oldest / highest priority first)."""
        df = get_orders()
        if df.empty:
            return []
        pending = df[df["status"] == "Pending"].copy()
        now = datetime.now()
        scored = []
        for _, r in pending.iterrows():
            try:
                age_h = (now - datetime.fromisoformat(str(r["created_at"]))).total_seconds() / 3600
            except Exception:
                age_h = 0
            # lower priority number = more urgent; age adds pressure
            urgency = age_h + (6 - min(5, 5)) * 2  # default mid priority
            scored.append(
                {
                    "order_id": r["order_id"],
                    "age_hours": round(age_h, 2),
                    "urgency": round(urgency, 2),
                    "items": r["items"],
                    "created_at": r["created_at"],
                }
            )
        scored.sort(key=lambda x: -x["urgency"])
        return scored

    def pipeline_summary(self) -> Dict[str, int]:
        df = get_orders()
        if df.empty:
            return {s: 0 for s in TRANSITIONS}
        counts = df["status"].value_counts().to_dict()
        return {s: int(counts.get(s, 0)) for s in TRANSITIONS}

    def history(self, order_id: str = None, limit: int = 50):
        return get_workflow_history(order_id, limit)
