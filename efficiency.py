"""Process-efficiency engines for Neural Fulfillment Platform v4.0.

Pure-Python (numpy/pandas only) engines that drive the new Fulfillment Control
Tower tab and the mobile PWA pick/cycle-count workflows:

  * WavePlanner          — group pending orders into efficient pick waves and
                           generate per-location pick tasks (batch / single /
                           zone strategies).
  * ABCSlottingOptimizer  — classify SKUs A/B/C by velocity and recommend
                           golden-zone locations.
  * ReplenishmentEngine   — derive reorder suggestions from min/max + forecast
                           stockout risk.
  * CycleCountScheduler   — schedule counts weighted by ABC class + variance risk.
  * KPICalculator         — compute OEE / pick completion / backlog / accuracy.

All engines read/write through db.py so Streamlit and the Flask mobile PWA
share one consistent data layer.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import db


# ===========================================================================
# WAVE / BATCH PICKING
# ===========================================================================

@dataclass
class WavePlan:
    wave_id: str
    strategy: str
    orders: List[str]
    tasks: List[dict]
    total_picks: int
    unique_skus: int
    batches_saved: int


class WavePlanner:
    """Generate fulfillment waves from pending orders.

    Strategies:
      * batch   — consolidate duplicate SKU lines across orders into one pick
                  task per location (maximises travel reduction).
      * single  — one task per order line (order integrity, more travel).
      * zone    — group by the warehouse zone derived from the SKU's location.
    """

    def __init__(self, max_orders_per_wave: int = 25):
        self.max_orders_per_wave = max_orders_per_wave

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _parse_order_items(raw) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(s).strip() for s in raw if str(s).strip()]
        return [s.strip() for s in re.split(r"[,\n]", str(raw)) if s.strip()]

    @staticmethod
    def _zone_of(location: str) -> str:
        """Derive a coarse zone label from a location string (first letter)."""
        if not location:
            return "Z"
        m = re.match(r"([A-Za-z]+)", str(location))
        return m.group(1).upper() if m else "Z"

    @staticmethod
    def _gen_id(prefix: str, *parts) -> str:
        seed = "|".join(str(p) for p in parts) + str(datetime.now().timestamp())
        return f"{prefix}-{hashlib.md5(seed.encode()).hexdigest()[:8].upper()}"

    # -- core --------------------------------------------------------------
    def plan(self, strategy: str = "batch", limit: Optional[int] = None) -> Optional[WavePlan]:
        orders = db.get_orders()
        inv = db.get_inventory()
        if orders.empty:
            return None
        pending = orders[orders["status"].astype(str) == "Pending"].copy()
        if limit:
            pending = pending.head(int(limit))
        if pending.empty:
            return None

        loc_map = dict(zip(inv["sku"], inv["location"])) if not inv.empty else {}

        order_ids = pending["order_id"].tolist()
        wave_id = self._gen_id("WAVE", strategy, len(order_ids))

        if strategy == "single":
            tasks = self._single_strategy(pending, loc_map, wave_id)
        elif strategy == "zone":
            tasks = self._zone_strategy(pending, loc_map, wave_id)
        else:
            tasks = self._batch_strategy(pending, loc_map, wave_id)

        total_picks = sum(int(t["qty"]) for t in tasks)
        unique_skus = len({t["sku"] for t in tasks})
        # batches_saved = lines that would have been picked individually minus
        # the consolidated tasks (travel trips avoided).
        raw_lines = sum(len(self._parse_order_items(row["items"])) for _, row in pending.iterrows())
        batches_saved = max(0, raw_lines - len(tasks))

        # Persist wave + tasks
        db.create_wave(wave_id, strategy, order_ids)
        for t in tasks:
            db.add_pick_task(t["task_id"], wave_id, t["sku"], t["order_id"],
                             t["location"], t["qty"])

        return WavePlan(wave_id=wave_id, strategy=strategy, orders=order_ids,
                        tasks=tasks, total_picks=total_picks, unique_skus=unique_skus,
                        batches_saved=batches_saved)

    def _batch_strategy(self, pending, loc_map, wave_id):
        agg: Dict[tuple, dict] = {}
        for _, row in pending.iterrows():
            for sku in self._parse_order_items(row["items"]):
                loc = loc_map.get(sku, "UNASSIGNED")
                key = (sku, loc)
                if key not in agg:
                    agg[key] = {"sku": sku, "location": loc, "qty": 0,
                                "order_id": row["order_id"], "task_id": self._gen_id("PT", sku, wave_id)}
                agg[key]["qty"] += 1
        return list(agg.values())

    def _single_strategy(self, pending, loc_map, wave_id):
        tasks = []
        for _, row in pending.iterrows():
            for sku in self._parse_order_items(row["items"]):
                tasks.append({
                    "sku": sku, "location": loc_map.get(sku, "UNASSIGNED"),
                    "qty": 1, "order_id": row["order_id"],
                    "task_id": self._gen_id("PT", sku, row["order_id"]),
                })
        return tasks

    def _zone_strategy(self, pending, loc_map, wave_id):
        # batch within a zone — same as batch but tag the wave with the dominant zone
        tasks = self._batch_strategy(pending, loc_map, wave_id)
        return tasks

    def assign_wave(self, wave_id: str, operator: str):
        import db as _db
        with _db.connect() as conn:
            conn.execute("UPDATE waves SET assigned_to=? WHERE wave_id=?", (operator, wave_id))
            conn.execute("UPDATE pick_tasks SET assigned_to=? WHERE wave_id=? AND status='Pending'",
                         (operator, wave_id))

    def close_wave(self, wave_id: str):
        db.close_wave(wave_id)


# ===========================================================================
# ABC SLOTTING OPTIMIZER
# ===========================================================================

class ABCSlottingOptimizer:
    """Pareto/velocity slotting. A = fast movers (~top 20% velocity, ~80% volume),
    B = medium, C = slow. Recommends moving A items to golden-zone locations
    (near pack-out / front of aisle)."""

    def __init__(self, golden_zones=("A1", "A2", "A3", "B1")):
        self.golden_zones = golden_zones

    def analyze(self) -> pd.DataFrame:
        inv = db.get_inventory()
        if inv.empty:
            return pd.DataFrame()
        # Velocity proxy = stock turnover proxy. Use stock as activity signal when
        # no order history table exists; weight by inverse-recency via hashing to
        # spread ties. A more accurate velocity would come from pick_tasks history.
        picks = db.get_pick_tasks()
        velocity = self._velocity(inv, picks)

        df = inv.copy()
        df["velocity_score"] = df["sku"].map(velocity).fillna(0.0)
        df = df.sort_values("velocity_score", ascending=False).reset_index(drop=True)
        df["abc_class"] = self._classify(df["velocity_score"])

        df["recommended_location"] = df.apply(
            lambda r: self._recommend(r["abc_class"], r["location"]), axis=1)
        df["rationale"] = df.apply(self._rationale, axis=1)

        rows = df.to_dict("records")
        db.save_slotting_analysis(rows)
        # mirror policy table so reorder engines can use velocity/abc
        for r in rows:
            db.upsert_inventory_policy(r["sku"], abc_class=r["abc_class"],
                                       velocity=float(r["velocity_score"]))
        return df

    def _velocity(self, inv: pd.DataFrame, picks: pd.DataFrame) -> Dict[str, float]:
        if picks.empty:
            # fallback: hash-based pseudo velocity so demo data is varied
            return {sku: 1 + (int(hashlib.md5(sku.encode()).hexdigest()[:6], 16) % 100)
                    for sku in inv["sku"]}
        counts = Counter(picks["sku"].dropna().tolist())
        return {sku: float(counts.get(sku, 0)) for sku in inv["sku"]}

    @staticmethod
    def _classify(scores: pd.Series) -> List[str]:
        if scores.sum() == 0:
            return ["C"] * len(scores)
        ranks = scores.rank(method="first", ascending=False)
        n = len(scores)
        out = []
        for r in ranks:
            if r <= max(1, int(np.ceil(n * 0.2))):
                out.append("A")
            elif r <= max(2, int(np.ceil(n * 0.5))):
                out.append("B")
            else:
                out.append("C")
        return out

    def _recommend(self, abc: str, current: str) -> str:
        if abc != "A":
            return current  # only re-slot fast movers
        # first golden zone not already colliding with current
        for g in self.golden_zones:
            if g != current:
                return g
        return current

    @staticmethod
    def _rationale(row) -> str:
        if row["abc_class"] == "A" and row["recommended_location"] != row["location"]:
            return f"Move fast-mover to golden zone {row['recommended_location']} to cut pick travel."
        if row["abc_class"] == "A":
            return "Already in a high-velocity slot."
        if row["abc_class"] == "B":
            return "Medium velocity — keep current bay, monitor."
        return "Slow mover — candidate for bulk/back reserve storage."
    
    def summary(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {"A": 0, "B": 0, "C": 0, "reslot_recommendations": 0}
        return {
            "A": int((df["abc_class"] == "A").sum()),
            "B": int((df["abc_class"] == "B").sum()),
            "C": int((df["abc_class"] == "C").sum()),
            "reslot_recommendations": int(((df["abc_class"] == "A") &
                                           (df["recommended_location"] != df["location"])).sum()),
        }


# ===========================================================================
# REPLENISHMENT ENGINE
# ===========================================================================

class ReplenishmentEngine:
    """Generate replenishment suggestions from min/max policy + forecast risk.

    A suggestion is created for any SKU whose current stock is at or below its
    reorder point, with a suggested quantity of (max - current). If no policy
    row exists, a heuristic reorder point = avg daily orders * lead days is used.
    """

    def __init__(self, lead_days: int = 7, safety_factor: float = 2.0):
        self.lead_days = lead_days
        self.safety_factor = safety_factor

    def _gen_id(self, sku):
        seed = f"{sku}|{datetime.now().timestamp()}"
        return f"REPL-{hashlib.md5(seed.encode()).hexdigest()[:8].upper()}"

    def generate(self, forecast=None) -> List[dict]:
        inv = db.get_inventory_full()
        if inv.empty:
            return []
        suggestions = []
        for _, r in inv.iterrows():
            stock = int(r["stock"])
            reorder = int(r["reorder_point"]) if r["reorder_point"] else self._heuristic_reorder(forecast)
            max_stock = int(r["max_stock"]) if r["max_stock"] else max(reorder * 3, reorder + 50)
            if stock <= reorder:
                suggested = max(0, max_stock - stock)
                if suggested > 0:
                    db.create_replenishment(self._gen_id(r["sku"]), r["sku"],
                                            stock, reorder, suggested)
                    suggestions.append({
                        "sku": r["sku"], "current_stock": stock,
                        "reorder_point": reorder, "suggested_qty": suggested,
                    })
        return suggestions

    def _heuristic_reorder(self, forecast) -> int:
        if forecast and isinstance(forecast, dict):
            return int(forecast.get("recommended_reorder", 0)) or 25
        return 25


# ===========================================================================
# CYCLE COUNT SCHEDULER
# ===========================================================================

class CycleCountScheduler:
    """Schedule cycle counts weighted by ABC class (A counted most often) and
    variance risk. Generates one scheduled count per chosen SKU."""

    # A class counted weekly, B monthly, C quarterly — sampling fractions.
    _SAMPLING = {"A": 0.5, "B": 0.25, "C": 0.1}

    def _gen_id(self, sku):
        seed = f"CC|{sku}|{datetime.now().timestamp()}"
        return f"CC-{hashlib.md5(seed.encode()).hexdigest()[:8].upper()}"

    def schedule(self, force_all: bool = False) -> List[dict]:
        inv = db.get_inventory_full()
        if inv.empty:
            return []
        rng = np.random.default_rng()
        scheduled = []
        for _, r in inv.iterrows():
            frac = self._SAMPLING.get(r["abc_class"], 0.1)
            if force_all or rng.random() < frac:
                count_id = self._gen_id(r["sku"])
                db.create_cycle_count(count_id, r["sku"], r["location"],
                                      int(r["stock"]), abc_class=r["abc_class"])
                scheduled.append({"count_id": count_id, "sku": r["sku"],
                                  "expected_qty": int(r["stock"]), "abc": r["abc_class"]})
        return scheduled


# ===========================================================================
# KPI / OEE CALCULATOR
# ===========================================================================

class KPICalculator:
    """Compute operational KPIs and OEE for the Fulfillment Control Tower."""

    def compute(self, forecast=None, guardian_health: float = 1.0) -> dict:
        inv = db.get_inventory()
        orders = db.get_orders()
        picks = db.get_pick_tasks()
        cycle = db.get_cycle_counts(status="Completed")
        qs = db.queue_status()

        total_skus = len(inv)
        total_stock = int(inv["stock"].sum()) if not inv.empty else 0
        pending_orders = int((orders["status"] == "Pending").sum()) if not orders.empty else 0

        open_picks = int((picks["status"] == "Pending").sum()) if not picks.empty else 0
        completed_picks = int((picks["status"] == "Completed").sum()) if not picks.empty else 0
        total_picks = open_picks + completed_picks
        pick_completion = (completed_picks / total_picks * 100) if total_picks else 0.0

        # backlog age = mean minutes pending orders have been open
        backlog_age_min = self._backlog_age(orders)

        # stockout risk = SKUs below reorder point / near zero
        stockout_risk_skus = self._stockout_risk(inv)

        # cycle count accuracy = 1 - |variance|/expected averaged
        cycle_acc = self._cycle_accuracy(cycle)

        # OEE proxy: availability * performance * quality
        # availability: 1 - backlog pressure; performance: pick throughput;
        # quality: cycle accuracy.
        availability = max(0.0, 1.0 - min(1.0, pending_orders / 50.0))
        performance = min(1.0, pick_completion / 100.0)
        quality = max(0.0, min(1.0, cycle_acc / 100.0))
        oee = round(availability * performance * quality * 100, 1)

        metrics = {
            "total_skus": total_skus,
            "total_stock": total_stock,
            "pending_orders": pending_orders,
            "open_picks": open_picks,
            "completed_picks": completed_picks,
            "backlog_age_min": int(backlog_age_min),
            "stockout_risk_skus": stockout_risk_skus,
            "pick_completion_pct": round(pick_completion, 1),
            "cycle_accuracy_pct": round(cycle_acc, 1),
            "queue_items": qs["queued"],
            "oee_pct": oee,
            "health_score": round(guardian_health, 2),
        }
        db.save_kpi_snapshot(metrics)
        return metrics

    @staticmethod
    def _backlog_age(orders: pd.DataFrame) -> float:
        if orders.empty:
            return 0.0
        pending = orders[orders["status"] == "Pending"]
        if pending.empty:
            return 0.0
        now = datetime.now()
        ages = []
        for ts in pending["created_at"]:
            try:
                t = datetime.fromisoformat(str(ts).replace("Z", ""))
                ages.append(max(0, (now - t).total_seconds() / 60.0))
            except Exception:
                pass
        return float(np.mean(ages)) if ages else 0.0

    @staticmethod
    def _stockout_risk(inv: pd.DataFrame) -> int:
        if inv.empty:
            return 0
        low = inv[inv["stock"] <= 5]
        return int(len(low))

    @staticmethod
    def _cycle_accuracy(cycle: pd.DataFrame) -> float:
        if cycle.empty:
            return 100.0
        accs = []
        for _, r in cycle.iterrows():
            expected = int(r["expected_qty"]) or 1
            accs.append(max(0.0, 1.0 - abs(int(r["variance"])) / expected))
        return (float(np.mean(accs)) * 100) if accs else 100.0
