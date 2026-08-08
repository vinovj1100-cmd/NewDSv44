"""Guardian v4.2 — system health monitoring with floor-ops awareness.
New in v4.2: andon/SLA/pack-flag checks, silent-bus detection, throttled
snapshot persistence (guardian_snapshots), and CRITICAL alerts rebroadcast
onto the realtime event bus with per-message cooldown dedupe.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import realtime
from db import save_guardian_snapshot


@dataclass
class GuardianConfig:
    health_threshold: float = 0.7
    alert_cooldown_minutes: int = 5
    snapshot_min_interval_sec: int = 60
    backlog_warn: int = 15
    backlog_critical: int = 25
    queue_warn: int = 20
    queue_critical: int = 50
    bus_silent_warn_sec: int = 1800


class Guardian:
    def __init__(self, config: Optional[GuardianConfig] = None):
        self.config = config or GuardianConfig()
        self.health = 1.0
        self.alerts, self.suggestions, self.recovery = [], [], []
        self.tuner, self.ozone = {}, {}
        self._started = False
        self._last_published: Dict[str, datetime] = {}
        self._last_snapshot: Optional[datetime] = None

    def start(self):
        self._started = True
        self.health = 1.0

    def analyze(self, context: Dict[str, Any]) -> "Guardian":
        cfg = self.config
        self.alerts, self.suggestions, self.recovery = [], [], []
        h = 1.0

        inv = context.get("inventory", {})
        total_skus = inv.get("total_skus", 0)
        total_stock = inv.get("total_stock", 0)
        pending = context.get("pending_orders", 0)
        queued = context.get("sync_queue", {}).get("queued", 0)
        low_stock = context.get("low_stock_skus", [])
        andon_open = context.get("andon_open", 0)
        sla_breached = context.get("sla_breached", 0)
        pack_flagged = context.get("pack_flagged", 0)
        bus_silent = context.get("event_bus_silent_sec")

        def alert(level, message):
            self.alerts.append({"level": level, "message": message})

        # inventory
        if total_skus == 0:
            h -= 0.3; alert("WARNING", "Inventory is empty — no SKUs registered")
        elif total_stock < 10:
            h -= 0.2; alert("WARNING", f"Low total stock: {total_stock} units")
        # backlog
        if pending > cfg.backlog_critical:
            h -= 0.25; alert("CRITICAL", f"Order backlog critical: {pending} pending orders")
        elif pending > cfg.backlog_warn:
            h -= 0.15; alert("WARNING", f"Order backlog elevated: {pending} pending orders")
        # sync queue
        if queued > cfg.queue_critical:
            h -= 0.2; alert("CRITICAL", f"Sync queue backlog: {queued} items pending")
        elif queued > cfg.queue_warn:
            h -= 0.1; alert("WARNING", f"Sync queue growing: {queued} items pending")
        # low stock
        if len(low_stock) > 5:
            h -= 0.1; alert("INFO", f"{len(low_stock)} SKUs below reorder point")
        # ── v4.2 floor-ops awareness ──
        if andon_open:
            h -= min(0.2, 0.07 * andon_open)
            alert("CRITICAL" if andon_open >= 2 else "WARNING",
                  f"{andon_open} open andon alert(s) on the floor")
        if sla_breached:
            h -= min(0.25, 0.08 * sla_breached)
            alert("CRITICAL", f"{sla_breached} order(s) past SLA cutoff — expedite picking")
        if pack_flagged:
            h -= 0.1
            alert("WARNING", f"{pack_flagged} carton(s) flagged at pack stations")
        if bus_silent is not None and bus_silent > cfg.bus_silent_warn_sec and pending > 0:
            h -= 0.1
            alert("WARNING", f"Floor event bus silent for {int(bus_silent // 60)} min while {pending} orders wait")

        # suggestions
        if pending > 10:
            self.suggestions.append("Generate a pick wave to burn down the backlog")
        if queued > 20:
            self.suggestions.append("Run manual sync or enable auto-sync to clear the queue")
        if low_stock:
            self.suggestions.append(f"Run the Replenishment engine for {len(low_stock)} low-stock SKUs")
        if andon_open:
            self.suggestions.append("Acknowledge open andon alerts from the Control Tower")
        if sla_breached:
            self.suggestions.append("Prioritize breached orders in the next wave (priority 1)")
        if pack_flagged:
            self.suggestions.append("Inspect flagged cartons before they miss carrier cutoff")

        # recovery
        if h < 0.5:
            self.recovery.append("Immediate: process pending orders and drain the sync queue")
            self.recovery.append("Review staffing levels for order fulfillment")
        if h < 0.7:
            self.recovery.append("Review inventory levels and place emergency reorders")

        self.health = max(0.0, min(1.0, h))
        self.tuner = {
            "health_score": round(self.health, 2), "total_skus": total_skus,
            "total_stock": total_stock, "pending_orders": pending, "queued_items": queued,
            "low_stock_count": len(low_stock), "andon_open": andon_open,
            "sla_breached": sla_breached, "pack_flagged": pack_flagged,
            "failed_logins": context.get("failed_logins_last_5m", 0),
        }
        self.ozone = {
            "status": "HEALTHY" if self.health > 0.8 else "DEGRADED" if self.health > 0.5 else "CRITICAL",
            "uptime": "99.8%", "last_check": datetime.now().strftime("%H:%M:%S"), "next_check": "Auto (60 s)",
        }
        self._persist()
        self._broadcast()
        return self

    def _persist(self):
        """Throttled snapshot for the dashboard sparkline."""
        now = datetime.now()
        if self._last_snapshot and (now - self._last_snapshot).total_seconds() < self.config.snapshot_min_interval_sec:
            return
        self._last_snapshot = now
        save_guardian_snapshot({
            "health_score": self.health, "status": self.ozone["status"],
            "alerts_count": len(self.alerts),
            "critical_count": len([a for a in self.alerts if a["level"] == "CRITICAL"]),
            "pending_orders": self.tuner.get("pending_orders", 0),
            "queued_items": self.tuner.get("queued_items", 0),
            "low_stock_count": self.tuner.get("low_stock_count", 0),
            "andon_open": self.tuner.get("andon_open", 0),
            "sla_breached": self.tuner.get("sla_breached", 0),
            "pack_flagged": self.tuner.get("pack_flagged", 0),
        })

    def _broadcast(self):
        """Rebroadcast CRITICAL alerts onto the event bus, deduped by cooldown."""
        cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
        for a in self.alerts:
            if a["level"] != "CRITICAL":
                continue
            last = self._last_published.get(a["message"])
            if last and datetime.now() - last < cooldown:
                continue
            self._last_published[a["message"]] = datetime.now()
            realtime.publish("GUARDIAN_ALERT", "guardian", a["message"], severity="critical")
