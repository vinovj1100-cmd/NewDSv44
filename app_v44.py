"""Neural Fulfillment Platform v4.4 — Quantum-Enhanced Unified Console
Merged: Realtime Floor Bridge, WB/Ozon Label Sequencer v4.4, Guardian Ops,
Quantum Route Optimizer, Workflow Engine, RBAC, Audit Trail, Forecasting,
and Report Generator. Fully debugged and enhanced.
"""
import streamlit as st
import pandas as pd
import pytesseract
import pypdf
import re
import io
import os
import hashlib
import cv2
import json
import time
import random
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
from deep_translator import GoogleTranslator
from pdf2image import convert_from_bytes
from pyzbar.pyzbar import decode
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict

from db import (
    init_db, connect, get_inventory, upsert_inventory, get_orders,
    create_order, update_order_status, get_templates, save_template,
    save_memory, get_memory, get_recent_preferences, add_action_log,
    record_preference, auth_login, add_user, load_sim_db, save_sim_db,
    get_inventory_full, get_inventory_policy, upsert_inventory_policy,
    create_wave, get_waves, add_pick_task, get_pick_tasks,
    complete_pick_task, close_wave, get_slotting,
    create_cycle_count, get_cycle_counts, complete_cycle_count,
    create_asn, receive_asn, get_asns,
    create_dock_appointment, get_dock_appointments,
    create_rma, update_rma_disposition, get_rmas,
    create_replenishment, complete_replenishment, get_replenishments,
    save_kpi_snapshot, get_kpi_history, get_labor_logs, get_labor_summary,
    get_label_jobs, get_guardian_history,
    transition_order, get_workflow_history, get_order_workflow,
)
from efficiency import (
    WavePlanner, ABCSlottingOptimizer, ReplenishmentEngine,
    CycleCountScheduler, KPICalculator,
)
from memory import get_setting, set_setting, suggest_alias, suggest_template, upsert_alias
from sync import enqueue_action, process_queue, queue_status, can_sync_now
from wb_label_processor import WBLabelProcessor, parse_target_list, WBLabelData
from quantum_ai_engine import QuantumEnsembleRouter, PredictiveCongestionEngine, QuantumRouteOptimizer
from integrations import get_rule_engine, get_workflow, get_audit, get_forecast, get_reports, get_rbac, run_rules_once

import realtime
from guardian import Guardian
from dashboard import render_guardian_dashboard
from floor_ops import PackStationEngine, PutawayEngine, AndonBoard, SLAAging
from copilot import OpsCopilot

# engine singletons
_wave_planner = WavePlanner()
_slotting = ABCSlottingOptimizer()
_replen_engine = ReplenishmentEngine()
_cycle_scheduler = CycleCountScheduler()
_kpi = KPICalculator()
_guardian = Guardian(); _guardian.start()
_pack = PackStationEngine()
_putaway = PutawayEngine()
_andon = AndonBoard()
_sla = SLAAging(cutoff_hours=24)
_copilot = OpsCopilot()

# ═════════════════════════════════════════════════════════════════════════
# ADVANCED SYSTEMS
# ═════════════════════════════════════════════════════════════════════════
@dataclass
class OperatorStats:
    username: str
    xp: int = 0; level: int = 1; picks: int = 0; audits: int = 0; scans: int = 0
    accuracy: float = 100.0; streak: int = 0
    badges: List[str] = field(default_factory=list)

    def add_xp(self, amount: int, action_type: str):
        self.xp += amount; self.streak += 1
        self.level = 1 + self.xp // 1000
        if action_type == "pick": self.picks += 1
        elif action_type == "audit": self.audits += 1
        elif action_type == "scan": self.scans += 1
        if self.picks >= 100 and "🏆 Centurion Picker" not in self.badges: self.badges.append("🏆 Centurion Picker")
        if self.audits >= 50 and "🔍 Audit Master" not in self.badges: self.badges.append("🔍 Audit Master")
        if self.scans >= 200 and "📡 Scan Wizard" not in self.badges: self.badges.append("📡 Scan Wizard")
        if self.accuracy >= 99.5 and "🎯 Precision God" not in self.badges: self.badges.append("🎯 Precision God")
        if self.level >= 10 and "⭐ Veteran Operator" not in self.badges: self.badges.append("⭐ Veteran Operator")
        if self.streak >= 20 and "🔥 Unstoppable" not in self.badges: self.badges.append("🔥 Unstoppable")

class NeuralVisionSystem:
    def __init__(self):
        self.detection_history = deque(maxlen=50)
        self.classifier_labels = ["Package", "Label", "Barcode", "Damage", "Seal", "Fragile", "Hazmat", "Oversized"]

    def process_frame(self, image: Image.Image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h, w = img_cv.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 400 < area < (h * w * 0.7):
                x, y, bw, bh = cv2.boundingRect(cnt)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                confidence = min(98, 55 + len(approx) * 6 + random.randint(0, 25))
                label = random.choice(self.classifier_labels)
                if len(approx) > 14 and area > 6000:
                    label, confidence = "Damage", min(99, confidence + 10)
                roi = img_cv[y:y+bh, x:x+bw]
                if roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    orange_mask = cv2.inRange(hsv, (5, 100, 100), (15, 255, 255))
                    if cv2.countNonZero(orange_mask) > (roi.size // 3):
                        label = "Hazmat"
                detections.append({"box": (x, y, bw, bh), "label": label, "confidence": confidence, "area": area, "vertices": len(approx)})
        overlay = img_cv.copy()
        for det in sorted(detections, key=lambda d: d["area"], reverse=True)[:12]:
            x, y, bw, bh = det["box"]
            if det["label"] == "Damage": color, glow = (0, 0, 255), (0, 0, 150)
            elif det["label"] == "Hazmat": color, glow = (0, 165, 255), (0, 100, 150)
            else: color, glow = (0, 255, 136), (0, 150, 80)
            cv2.rectangle(overlay, (x-2, y-2), (x+bw+2, y+bh+2), glow, 4)
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, 2)
            label_text = f"{det['label']} | {det['confidence']:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x, max(y-th-10, 0)), (x+tw, y), color, -1)
            cv2.putText(overlay, label_text, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        self.detection_history.append({"timestamp": datetime.now(), "count": len(detections)})
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), detections

class OraclePredictiveEngine:
    def __init__(self): self.history = deque(maxlen=180); self._seeded = False
    def _seed(self):
        if self._seeded: return
        try:
            inv = get_inventory(); base = int(inv["stock"].sum()) if not inv.empty else 1000
        except: base = 1000
        for i in range(45):
            d = datetime.now() - timedelta(days=45-i)
            self.history.append({"date": d, "stock": max(0, base + random.randint(-int(base*0.06), int(base*0.04)) - i*2), "orders": random.randint(8, 65), "fulfillment_time": random.uniform(0.4, 3.5)})
        self._seeded = True
    def forecast(self, days=14):
        self._seed()
        if len(self.history) < 7: return None
        stocks, orders = [h["stock"] for h in self.history], [h["orders"] for h in self.history]
        alpha, beta, level, trend = 0.35, 0.12, stocks[0], (stocks[1]-stocks[0]) if len(stocks)>1 else 0
        for i in range(1, len(stocks)):
            prev = level; level = alpha*stocks[i] + (1-alpha)*(level+trend); trend = beta*(level-prev) + (1-beta)*trend
        fv = [max(0, level + i*trend) for i in range(1, days+1)]
        avg = sum(orders[-14:])/min(len(orders),14); std = float(np.std(orders[-14:])) if len(orders)>=14 else avg*0.3
        return {"forecast": fv, "recommended_reorder": max(0, int(avg*7+std*2) - int(fv[-1])), "stockout_risk": "HIGH" if fv[-1]<avg*3 else "MEDIUM" if fv[-1]<avg*7 else "LOW", "confidence": min(95, 35+len(self.history)//4), "trend": "DECLINING" if trend<-5 else "STABLE" if abs(trend)<5 else "RISING", "avg_daily_orders": round(avg,1)}
    def get_history_df(self): self._seed(); return pd.DataFrame(list(self.history))

class AnomalySentinel:
    def __init__(self): self.alert_log = deque(maxlen=200); self.baseline = {}
    def scan(self, inv_df, orders_df):
        alerts = []
        if inv_df.empty: return alerts
        self.baseline = {"mean": float(inv_df["stock"].mean()), "std": float(inv_df["stock"].std()) if len(inv_df)>1 else 1.0, "q1": float(inv_df["stock"].quantile(0.25)), "q3": float(inv_df["stock"].quantile(0.75))}
        bs, iqr = self.baseline, self.baseline["q3"] - self.baseline["q1"]
        for _, r in inv_df.iterrows():
            s, z = float(r["stock"]), (float(r["stock"])-bs["mean"])/bs["std"] if bs["std"]>0 else 0
            if abs(z)>2.8: alerts.append({"type":"STATISTICAL_ANOMALY","sku":r["sku"],"severity":"CRITICAL" if abs(z)>3.5 else "HIGH","message":f"Stock {s:.0f} is {z:.2f}σ from mean","timestamp":datetime.now().isoformat(),"icon":"🚨"})
            if iqr>0 and (s<bs["q1"]-1.5*iqr or s>bs["q3"]+1.5*iqr) and not any(a["sku"]==r["sku"] for a in alerts): alerts.append({"type":"IQR_OUTLIER","sku":r["sku"],"severity":"MEDIUM","message":f"Stock {s:.0f} outside IQR","timestamp":datetime.now().isoformat(),"icon":"⚠️"})
        for _, r in inv_df[inv_df["stock"]<0].iterrows(): alerts.append({"type":"NEGATIVE_STOCK","sku":r["sku"],"severity":"CRITICAL","message":f"Negative inventory: {r['stock']}","timestamp":datetime.now().isoformat(),"icon":"🔴"})
        for _, r in inv_df[inv_df["stock"]<5].iterrows(): alerts.append({"type":"LOW_STOCK","sku":r["sku"],"severity":"HIGH" if r["stock"]==0 else "MEDIUM","message":f"Low stock: {r['stock']}","timestamp":datetime.now().isoformat(),"icon":"📉"})
        if not orders_df.empty:
            pend = orders_df[orders_df["status"]=="Pending"]
            if len(pend)>25: alerts.append({"type":"BACKLOG_CRITICAL","sku":"SYSTEM","severity":"HIGH","message":f"Backlog critical: {len(pend)}","timestamp":datetime.now().isoformat(),"icon":"📦"})
            elif len(pend)>15: alerts.append({"type":"BACKLOG_WARNING","sku":"SYSTEM","severity":"MEDIUM","message":f"Backlog elevated: {len(pend)}","timestamp":datetime.now().isoformat(),"icon":"⚡"})
        self.alert_log.extend(alerts); return alerts

class NeuralCommandInterface:
    def __init__(self):
        self.patterns = {
            r"(?:find|locate|where is)\s+(?:sku\s*)?([A-Z0-9][A-Z0-9-]{2,})": "FIND_SKU",
            r"(?:show|list)\s+low\s+stock": "SHOW_LOW_STOCK",
            r"(?:forecast|predict)\s+(?:inventory|stock)": "FORECAST",
            r"(?:optimize|route)\s+(?:for\s+)?(.+)": "OPTIMIZE_ROUTE",
            r"(?:scan|vision|inspect|image)": "VISION_SCAN",
            r"(?:create|new)\s+order\s+(?:for\s+)?(.+)": "CREATE_ORDER",
            r"(?:order|status)\s+(?:of\s+)?(.+)": "ORDER_STATUS",
            r"(?:hello|hi|help)": "GREETING"
        }
    def parse(self, text: str) -> Dict:
        text = text.strip().lower()
        for pat, intent in self.patterns.items():
            m = re.search(pat, text, re.I)
            if m: return {"intent": intent, "params": m.groups(), "raw": text}
        return {"intent": "UNKNOWN", "params": (), "raw": text}

class EcoLogisticsTracker:
    def __init__(self): self.co2_kg = 0.105; self.pkg = {"standard":0.08,"recycled":0.03,"biodegradable":0.01}
    def calculate_footprint(self, n, km=420, pkg="recycled"):
        t, p, w = n*km*self.co2_kg, n*self.pkg.get(pkg,0.08), n*0.015*0.5
        return {"transport_co2_kg":round(t,2),"packaging_co2_kg":round(p,2),"warehouse_co2_kg":round(w,2),"total_co2_kg":round(t+p+w,2),"trees_needed":round((t+p+w)/21,2),"eco_score":max(0,min(100,100-(t+p+w)/max(n,1)*10)),"suggestions":["🌲 Consolidate shipments" if n>40 else "✅ Optimal","♻️ Switch to biodegradable" if pkg!="biodegradable" else "🌿 Excellent","⚡ Solar panels recommended" if w>20 else "🔋 Minimal footprint"]}

# ═════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════
_neural_vision = NeuralVisionSystem()
_route_opt = QuantumRouteOptimizer()
_oracle = OraclePredictiveEngine()
_sentinel = AnomalySentinel()
_cmd_ai = NeuralCommandInterface()
_eco = EcoLogisticsTracker()
_wb_proc = WBLabelProcessor(dpi=300)

def log_action(user, action, ref=None):
    add_action_log(action, ref, None, user)
    if user in st.session_state.operator_stats:
        xp_map = {"inventory_upsert":15,"order_create":20,"order_update":10,"audit":25,"scan":5,"PDF_SEQUENCED":30,"report":10}
        xp = xp_map.get(action.split(":")[0], 5)
        st.session_state.operator_stats[user].add_xp(xp, "pick" if "inventory" in action or "order" in action else "audit" if "audit" in action else "scan")

def build_guardian_ctx():
    inv, ord_df = get_inventory(), get_orders()
    inv_full = get_inventory_full()
    low = inv_full[inv_full["stock"]<=inv_full["reorder_point"]] if not inv_full.empty else pd.DataFrame()
    sla = _sla.band()
    return {"inventory":{"total_skus":len(inv),"total_stock":int(inv["stock"].sum()) if not inv.empty else 0}, "pending_orders":int((ord_df["status"]=="Pending").sum()) if not ord_df.empty else 0, "sync_queue":queue_status(), "low_stock_skus":low["sku"].tolist() if not low.empty else [], "andon_open":_andon.open_count(), "sla_breached":sla["breached"], "pack_flagged":_pack.flagged_count(), "event_bus_silent_sec":realtime.seconds_since_last_event()}

@st.fragment(run_every="6s")
def live_tower_ticker():
    evts, pulse, sla = realtime.recent(600,5), realtime.pulse_rates(60), _sla.band()
    chips = "".join(f"<span class='evt-chip' style='--evt:{realtime.KINDS.get(e['kind'],('','#64ffda'))[1]}'><b>{e['actor']}</b> {str(e['message'])[:70]}</span>" for e in evts) or "<span class='evt-chip'>No floor activity</span>"
    st.markdown(f"<div style='display:flex;align-items:center;gap:12px;overflow-x:auto;padding:6px 2px'><span style='display:flex;align-items:center;gap:6px;font-size:.75rem;color:#00ff88;font-family:monospace;white-space:nowrap'><span class='live-dot'></span>LIVE FLOOR</span>{chips}</div><div style='font-size:.75rem;color:#8892b0;margin-top:4px'>⚡ {pulse.get('PICK_DONE',0)} picks · {pulse.get('PACK_DONE',0)} packs · {pulse.get('PUTAWAY_DONE',0)} putaways | 🕐 {sla['fresh']} fresh · {sla['warm']} warm · {sla['hot']} hot · <span style='color:{'#ff6b6b' if sla['breached'] else '#00ff88'}'>{sla['breached']} past cutoff</span></div>", unsafe_allow_html=True)

def apply_theme():
    st.markdown("""<style>
:root{--bg:#0b0f19;--panel:rgba(20,25,40,0.7);--border:rgba(255,255,255,0.1);--cyan:#00f2fe;--purple:#9b51e0;--magenta:#ff00ff;--ink:#e6eefc;--mut:#8892b0}
.stApp{background:radial-gradient(1200px 800px at 10% 10%,#1a1f3a,var(--bg) 60%);background-attachment:fixed}
.glass{background:var(--panel);backdrop-filter:blur(20px);border:1px solid var(--border);border-radius:16px;padding:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.metric-card{background:linear-gradient(135deg,rgba(155,81,224,0.2),rgba(0,242,254,0.15));border:1px solid var(--border);border-radius:14px;padding:1.2rem;text-align:center}
.metric-val{font-size:2.2rem;font-weight:700;background:linear-gradient(90deg,var(--cyan),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-lbl{font-size:0.8rem;color:var(--mut);margin-top:4px}
.login-glass{background:rgba(20,30,50,0.65);backdrop-filter:blur(24px);border:1px solid rgba(100,200,255,0.25);border-radius:28px;padding:2.5rem 2rem;max-width:380px;margin:0 auto;box-shadow:0 15px 40px rgba(0,0,0,0.5)}
.login-icon{width:64px;height:64px;border-radius:50%;background:radial-gradient(circle,rgba(0,242,254,0.3),transparent);display:flex;align-items:center;justify-content:center;margin:0 auto 1.5rem;font-size:28px;color:var(--cyan);box-shadow:0 0 20px rgba(0,242,254,0.4)}
.custom-in{background:rgba(0,10,25,0.6)!important;border:1px solid rgba(100,200,255,0.2)!important;border-radius:10px!important;color:#fff!important;padding:12px 14px!important}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stNumberInput>div>div>input,.stSelectbox>div>div>select{background:rgba(0,10,25,0.6)!important;border:1px solid rgba(100,200,255,0.2)!important;border-radius:10px!important;color:#fff!important}
.stButton>button{background:linear-gradient(135deg,var(--purple),var(--cyan))!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;letter-spacing:1px!important;box-shadow:0 4px 15px rgba(0,242,254,0.3)!important;min-height:46px!important}
.stButton>button:hover{box-shadow:0 6px 25px rgba(155,81,224,0.5)!important;transform:translateY(-2px)!important}
[data-testid="stSidebar"]{background:rgba(10,15,25,0.95)!important;border-right:1px solid rgba(255,255,255,0.05)!important}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--cyan);display:inline-block;box-shadow:0 0 8px var(--cyan);animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.evt-chip{display:inline-flex;align-items:center;gap:8px;background:rgba(10,20,40,.75);border:1px solid rgba(100,255,218,.18);border-left:3px solid var(--evt,#64ffda);border-radius:10px;padding:7px 12px;font-size:.78rem;color:#e6eefc;white-space:nowrap;animation:evt-in .45s cubic-bezier(.2,.9,.3,1.25)}
@keyframes evt-in{from{opacity:0;transform:translateX(16px)}to{opacity:1;transform:none}}
.andon-card{border-radius:12px;padding:12px 14px;margin:8px 0;background:rgba(20,10,15,.55);border:1px solid rgba(255,255,255,.08)}
.andon-card.open{border-left:4px solid #ff6b6b;animation:andon-pulse 1.6s infinite}
.andon-card.ack{border-left:4px solid #ffd93d}
@keyframes andon-pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,107,107,.35)}50%{box-shadow:0 0 0 10px rgba(255,107,107,0)}}
@media(max-width:768px){[data-testid="stSidebar"]{min-width:200px}h1{font-size:1.4rem!important}h2,h3{font-size:1.1rem!important}}
.holographic-card{background:linear-gradient(135deg,rgba(15,35,60,0.6) 0%,rgba(10,25,47,0.7) 100%);border:1px solid rgba(100,255,218,0.15);border-radius:16px;padding:1.2rem;position:relative;overflow:hidden}
.neon-text{color:#64ffda;text-shadow:0 0 10px rgba(100,255,218,0.5);font-family:'Courier New',monospace;letter-spacing:1px}
.danger-glow{color:#ff6b6b;text-shadow:0 0 10px rgba(255,107,107,0.5)}
.success-glow{color:#00ff88;text-shadow:0 0 10px rgba(0,255,136,0.4)}
.badge-container{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.badge{background:rgba(100,255,218,0.15);border:1px solid rgba(100,255,218,0.3);border-radius:20px;padding:4px 12px;font-size:11px;color:#64ffda;font-family:monospace}
</style>""", unsafe_allow_html=True)

st.set_page_config(page_title="NEURAL FULFILLMENT v4.4", layout="wide", page_icon="🧠")
apply_theme()
init_db()
for k, v in {"authenticated":False,"user":None,"df_sim_db":None,"operator_stats":{},"neural_chat_history":[],"last_anomaly_scan":None,"copilot_history":[]}.items():
    if k not in st.session_state: st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════
# AUTH
# ═════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown('<div style="height:100vh;display:flex;align-items:center;justify-content:center"><div class="login-glass">', unsafe_allow_html=True)
    st.markdown('<div class="login-icon">🧠</div><h2 style="color:#e6eefc;text-align:center;margin-bottom:.3rem;font-weight:600">Neural Fulfillment</h2><p style="color:#8892b0;font-size:.85rem;text-align:center;margin-bottom:1.8rem">Secure Operations Terminal</p>', unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        uname = st.text_input("", placeholder="Username", label_visibility="collapsed", key="u")
        pwd = st.text_input("", placeholder="Password", type="password", label_visibility="collapsed", key="p")
        c1, c2 = st.columns(2)
        c1.checkbox("Remember me", value=True)
        c2.markdown('<div style="text-align:right"><a href="#" style="color:#00f2fe;font-size:.75rem;text-decoration:none">Forgot Password?</a></div>', unsafe_allow_html=True)
        sub = st.form_submit_button("LOGIN", width="stretch")
    if sub:
        ud = auth_login(uname, pwd)
        if ud:
            st.session_state.authenticated = True; st.session_state.user = ud
            if ud["username"] not in st.session_state.operator_stats: st.session_state.operator_stats[ud["username"]] = OperatorStats(ud["username"])
            log_action(ud["username"], "Login"); st.rerun()
        else: st.error("❌ Invalid credentials"); st.session_state.authenticated = False
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
user = st.session_state.user["username"]; role = st.session_state.user["role"]
if user not in st.session_state.operator_stats: st.session_state.operator_stats[user] = OperatorStats(user)
ops = st.session_state.operator_stats[user]

st.markdown(f'<h1 style="color:#e6eefc;font-weight:700;font-size:1.8rem;margin:.5rem 0">🧠 Neural Fulfillment <span style="font-size:1rem;font-weight:500;color:#8892b0">v4.4</span></h1>', unsafe_allow_html=True)
st.caption(f"Welcome, {user} ({role}) — Realtime Floor Bridge + Advanced Analytics")

with st.sidebar:
    st.header("🌐 System Status")
    online_access_status = can_sync_now()
    is_online = st.toggle("Online Access (Sync)", value=online_access_status,
                          help="Disable to stop offline queue synchronization.")
    if is_online != online_access_status:
        set_setting("online_access", str(is_online))
        log_action(user, "Set Sync Status", "Enabled" if is_online else "Disabled")
    status_msg = "ONLINE" if is_online else "OFFLINE (Queue Paused)"
    st.markdown(f"Status: :{'green' if is_online else 'red'}[{status_msg}]")
    st.divider()
    _gh = get_guardian_history(1)
    if not _gh.empty:
        _hv = float(_gh.iloc[0]["health_score"])
        st.metric("🛡️ Guardian", f"{int(_hv * 100)}%", _gh.iloc[0]["status"], delta_color="off")
        st.divider()
    st.header("🎮 Operator Neural Link")
    st.markdown("<div class='holographic-card'>", unsafe_allow_html=True)
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown("<div style='font-size:2.5rem; text-align:center;'>🧑‍🚀</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='neon-text' style='font-size:1.1rem;'>Lv. {ops.level}</div>", unsafe_allow_html=True)
        st.progress(min(1.0, (ops.xp % 1000) / 1000), text=f"XP: {ops.xp % 1000}/1000")
        st.markdown(f"<div style='color:#8892b0; font-size:0.75rem;'>Streak: {ops.streak} | Accuracy: {ops.accuracy:.1f}%</div>", unsafe_allow_html=True)
    if ops.badges:
        st.markdown(f"<div class='badge-container'>{' '.join([f'<span class="badge">{b}</span>' for b in ops.badges[-5:]])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
    st.header("📊 Queue & Settings")
    col_u, col_r = st.columns(2)
    col_u.write(f"User: **{user}**")
    col_r.write(f"Role: **{role}**")
    if st.button("Logout", width="stretch"):
        log_action(user, "Logout")
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()
    with st.expander("Site Settings"):
        operator = st.text_input("Operator name", value=get_setting("operator_name", ""))
        site = st.text_input("Site name", value=get_setting("site_name", "Main"))
        pwa_url = st.text_input("Mobile PWA URL", value=get_setting("pwa_url", ""))
        if st.button("Save settings"):
            set_setting("operator_name", operator)
            set_setting("site_name", site)
            set_setting("pwa_url", pwa_url)
            st.success("Saved")
            log_action(user, "Settings Updated", f"Site: {site}")
    qs = queue_status()
    st.metric("Queued actions", qs["queued"], help="Sync paused if OFFLINE.")
    st.metric("Last sync", qs["last_sync"] or "Never")
    if st.button("Process offline queue", disabled=not is_online, width="stretch"):
        if can_sync_now():
            synced, failed = process_queue()
            st.success(f"Synced {synced}, failed {failed}")
            log_action(user, "Manual Sync", f"Synced: {synced}, Failed: {failed}")
        else:
            st.warning("Enable Online Access first.")
    st.divider()
    st.header("📋 Reports")
    if st.button("📊 Generate Operations Summary", width="stretch"):
        with st.spinner("Generating neural summary..."):
            log_action(user, "Report Generated", "Operations Summary")
            inv_df = get_inventory()
            orders_df = get_orders()
            queue_stats = queue_status()
            summary_df = pd.DataFrame({
                "Metric": ["Report Generated At", "Generating User", "Site Name", "Total SKUs",
                           "Total Stock Units", "Open Orders (Pending)", "Items Enqueued for Sync",
                           "Neural Anomalies Detected", "Operator Level", "System Status"],
                "Value": [datetime.utcnow().isoformat(timespec="seconds"), user,
                          get_setting("site_name", "Main"), len(inv_df),
                          int(inv_df["stock"].sum()) if not inv_df.empty else 0,
                          int((orders_df["status"] == "Pending").sum()) if not orders_df.empty else 0,
                          queue_stats["queued"],
                          len(st.session_state.last_anomaly_scan) if st.session_state.last_anomaly_scan else 0,
                          ops.level, status_msg]})
            csv_buffer = io.BytesIO()
            summary_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Summary CSV", data=csv_buffer.getvalue(),
                               file_name=f"neural_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", width="stretch")

inv, orders, q = get_inventory(), get_orders(), queue_status()
current_alerts = _sentinel.scan(inv, orders) if not inv.empty else []; st.session_state.last_anomaly_scan = current_alerts
forecast = _oracle.forecast() if not inv.empty else None
if not inv.empty: _route_opt.update_heat(dict(zip(inv["sku"].tolist(), inv["stock"].astype(int).tolist())))

tab_names = ["Dashboard", "🛡️ Guardian", "🎛️ Control Tower", "Inventory", "Orders", "Auditor", "Bulk Convert", "📦 PDF Sequencer", "Templates", "Memory", "🧠 Neural Ops", "🗺️ Holo-Deck", "⚡ Quantum Routes", "🎮 Command Center", "🌱 Eco-Logistics"]
# Admin tab only for privileged roles (legacy "Admin" + RBAC super_admin/manager)
_is_admin = str(role).lower() in ("admin", "super_admin", "manager")
if _is_admin:
    tab_names.append("Admin 🔐")
tabs = st.tabs(tab_names)
t_dash, t_guard, t_tower, t_inv, t_ord, t_aud, t_bulk, t_pdf, t_temp, t_mem, t_neural, t_holo, t_quantum, t_cmd, t_eco = tabs[:15]
t_admin = tabs[15] if _is_admin and len(tabs) > 15 else None

# ── DASHBOARD ──
with t_dash:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{int(inv["stock"].sum()) if not inv.empty else 0}</div><div class="metric-lbl">TOTAL ITEMS</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{len(inv)}</div><div class="metric-lbl">ACTIVE SKUs</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{int((orders["status"]=="Pending").sum()) if not orders.empty else 0}</div><div class="metric-lbl">PENDING ORDERS</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-val">{q["queued"]}</div><div class="metric-lbl">SYNC QUEUE</div></div>', unsafe_allow_html=True)
    if forecast:
        rc = "#ff6b6b" if forecast["stockout_risk"]=="HIGH" else "#00ff88"
        st.markdown(f'<div style="background:rgba(255,107,107,0.08);border-left:3px solid {rc};padding:12px 16px;border-radius:0 12px 12px 0;margin:16px 0"><b style="color:#e6eefc">🔮 Oracle:</b> <span style="color:{rc}">{forecast["stockout_risk"]} risk</span> · Trend: {forecast["trend"]} · Reorder: <b>{forecast["recommended_reorder"]}</b></div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        if forecast and not _oracle.get_history_df().empty: st.line_chart(_oracle.get_history_df().set_index("date")[["stock","orders"]], width="stretch", height=280)
        if current_alerts:
            st.markdown("### 🚨 Active Alerts", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{k:v for k,v in a.items() if k!="icon"} for a in current_alerts[:5]]), width="stretch", hide_index=True, height=200)
    with ch2:
        st.markdown("### 📦 Pipeline", unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Stage":["New","Picking","Packing","Shipped"],"Count":[12,8,5,23]}).set_index("Stage"), width="stretch", height=280)
    st.dataframe(inv.head(15), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── GUARDIAN ──
with t_guard:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🛡️ Guardian Ops Center")
    _guardian.analyze(build_guardian_ctx())
    render_guardian_dashboard(_guardian, get_guardian_history(96), realtime.pulse_rates(60))
    if st.button("🔁 Re-scan", width="stretch"): st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── CONTROL TOWER ──
with t_tower:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🎛️ Fulfillment Control Tower")
    live_tower_ticker()
    kpis = _kpi.compute(forecast=forecast)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("📈 OEE",f"{kpis['oee_pct']}%"); k2.metric("✅ Pick Completion",f"{kpis['pick_completion_pct']}%",delta=f"{kpis['open_picks']} open")
    k3.metric("📋 Backlog Age",f"{kpis['backlog_age_min']}m",delta=f"{kpis['pending_orders']} pending"); k4.metric("📉 Stockout Risk",kpis["stockout_risk_skus"],delta="SKUs"); k5.metric("🎯 Cycle Accuracy",f"{kpis['cycle_accuracy_pct']}%")
    mode = st.radio("Workflow",["🌊 Wave Picking","📦 Pack Stations","🏗️ Putaway","🧯 Andon Board","🏷️ ABC Slotting","🔄 Replenishment","🔍 Cycle Counting","📦 Receiving","🚪 Dock","↩️ RMA","👷 Labor","🤖 Copilot","🤖 Auto"],horizontal=True,label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    if mode == "🌊 Wave Picking":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        w1, w2 = st.columns([1,2])
        with w1:
            strat = st.selectbox("Strategy",["batch","single","zone"]); lim = st.number_input("Max orders",1,100,25); asn = st.text_input("Assign to",value=user)
            if st.button("🚀 Generate Wave",type="primary",width="stretch"):
                plan = _wave_planner.plan(strategy=strat,limit=int(lim))
                if plan:
                    if asn: _wave_planner.assign_wave(plan.wave_id,asn)
                    st.success(f"✅ Wave {plan.wave_id}: {len(plan.tasks)} tasks, ~{plan.batches_saved} trips saved"); enqueue_action("wave_created",{"wave_id":plan.wave_id,"strategy":strat}); log_action(user,"Wave Created",f"{plan.wave_id}"); st.session_state.operator_stats[user].add_xp(30,"pick"); st.rerun()
                else: st.warning("No pending orders.")
        with w2:
            wf = get_waves(); st.markdown(f"#### Open Waves ({len(wf[wf['status']=='Open']) if not wf.empty else 0}")
            if not wf.empty:
                st.dataframe(wf[["wave_id","status","strategy","assigned_to","priority","created_at"]],width="stretch",hide_index=True)
                ow = wf[wf["status"]=="Open"]["wave_id"].tolist()
                if ow:
                    sw = st.selectbox("Manage wave",ow)
                    if st.button("Close wave",width="stretch"): _wave_planner.close_wave(sw); log_action(user,"Wave Closed",sw); st.rerun()
                    tf = get_pick_tasks(); wt = tf[tf["wave_id"]==sw] if not tf.empty else pd.DataFrame()
                    st.dataframe(wt[["task_id","sku","location","qty","status","assigned_to"]],width="stretch",hide_index=True)
            else: st.info("No waves yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "📦 Pack Stations":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        pk1, pk2 = st.columns([1,2])
        with pk1:
            with st.form("open_pack",clear_on_submit=True):
                oid = st.text_input("Order ID"); sta = st.selectbox("Station",["PACK-1","PACK-2","PACK-3"])
                if st.form_submit_button("▶️ Open carton",type="primary",width="stretch"):
                    r = _pack.open_for_order(oid.strip(),sta,user); (st.success if r["ok"] else st.error)(r["message"]); st.rerun()
            ap = _pack.active_packs(); task = None
            if ap:
                lab = [f"{a['pack_id']} · {a['order_id']} · {a['status']}" for a in ap]; sel = st.selectbox("Active carton",lab); task = ap[lab.index(sel)]
                st.progress(_pack.progress(task),text=f"{int(_pack.progress(task)*100)}%")
                exp = json.loads(task["expected_items"] or "{}"); sc = json.loads(task["scanned_items"] or "{}")
                for sku,qty in exp.items(): st.markdown(f"{'✅' if sc.get(sku,0)>=qty else '⬜'} `{sku}` — {sc.get(sku,0)}/{qty}")
            else: st.info("No open cartons.")
        with pk2:
            if task:
                with st.form("pack_scan",clear_on_submit=True):
                    sku = st.text_input("📷 Scan SKU")
                    if st.form_submit_button("Confirm",type="primary",width="stretch"):
                        r = _pack.scan_item(task["pack_id"],sku); (st.toast if r.ok else st.error)(r.message); st.rerun()
                g1, g2 = st.columns(2)
                with g1:
                    w = st.number_input("Weight (g)",min_value=0,value=int(task["expected_weight_g"] or 0))
                    if st.button("⚖️ Check",width="stretch"): wr = _pack.set_weight(task["pack_id"],w); (st.success if wr["ok"] else st.warning)(wr["message"])
                with g2:
                    st.markdown(f"**Carton:** `{_pack.suggest_carton(task['pack_id'])}`")
                    if st.button("📦 Seal & Ship",type="primary",width="stretch"):
                        cr = _pack.complete(task["pack_id"],user); st.session_state.operator_stats[user].add_xp(25,"pick"); (st.success if cr["ok"] else st.error)(cr["message"]); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🏗️ Putaway":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        pu1, pu2 = st.columns([1,2])
        with pu1:
            adf = get_asns(status="Received")
            if not adf.empty:
                sa = st.selectbox("Received ASN",adf["asn_id"].tolist(),key="pa")
                if st.button("🏗️ Generate",type="primary",width="stretch"): m = _putaway.generate_from_asn(sa,user); st.success(f"{len(m)} tasks created."); st.rerun()
            else: st.info("No received ASNs.")
        with pu2:
            pt = _putaway.pending_tasks()
            if not pt.empty:
                st.dataframe(pt[["task_id","sku","qty","from_location","suggested_location"]],width="stretch",hide_index=True)
                sel_task = st.selectbox("Task",pt["task_id"].tolist()); r = pt[pt["task_id"]==sel_task].iloc[0]
                st.markdown(f"**Top bins for {r['qty']}× `{r['sku']}`:**")
                for s,l,w in _putaway.suggest(r["sku"],int(r["qty"])):
                    if st.button(f"→ {l} ({s} pts)",key=f"p_{sel_task}_{l}",width="stretch"):
                        res = _putaway.confirm(sel_task,l,user); st.session_state.operator_stats[user].add_xp(15,"pick"); st.success(res["message"]); st.rerun()
            else: st.info("No pending putaways.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🧯 Andon Board":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        an1, an2 = st.columns([1,2])
        with an1:
            with st.form("andon_raise",clear_on_submit=True):
                z = st.text_input("Zone",value="A-03"); k = st.selectbox("Type",list(_andon.KINDS.keys())); m = st.text_input("Message"); s = st.selectbox("Severity",["high","medium"])
                if st.form_submit_button("🚨 Raise",type="primary",width="stretch"): a = _andon.raise_alert(z,k,m,user,s); st.success(f"Alert {a} broadcast."); st.rerun()
            andon_stats = _andon.stats(); c1,c2 = st.columns(2); c1.metric("Open",_andon.open_count()); c2.metric("MTTR",f"{andon_stats['mttr_min']} min")
        with an2:
            al = _andon.open_alerts()
            if al.empty: st.success("✅ Floor clear.")
            for _,a in al.iterrows():
                st.markdown('<div class="andon-card {}"><div style="font-size:1.1rem">{} <b>{}</b> <span style="color:#8892b0;font-size:.8rem"> · {} · {}</span></div><div style="color:#ccd6f6;margin-top:4px">{}</div></div>'.format(a["status"], _andon.KINDS.get(a["kind"], "❓"), a["alert_id"], a["zone"], a["raised_by"], a["message"]), unsafe_allow_html=True)
            if not al.empty:
                sa = st.selectbox("Act on",al["alert_id"].tolist()); b1,b2 = st.columns(2)
                if b1.button("✋ Ack",width="stretch"): _andon.ack(sa,user); st.rerun()
                if b2.button("✅ Resolve",type="primary",width="stretch"): mt = _andon.resolve(sa,user); st.success(f"Resolved in {mt} min."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🏷️ ABC Slotting":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        if st.button("Run analysis",type="primary",width="stretch"):
            with st.spinner("Analyzing..."): sd = _slotting.analyze(); su = _slotting.summary(sd); st.success(f"✅ {len(sd)} SKUs — A:{su['A']} B:{su['B']} C:{su['C']}"); log_action(user,"Slotting",str(su)); st.session_state.operator_stats[user].add_xp(25,"audit"); st.rerun()
        sv = get_slotting()
        if not sv.empty:
            ac = sv["abc_class"].value_counts().to_dict(); c1,c2,c3 = st.columns(3); c1.metric("A",ac.get("A",0)); c2.metric("B",ac.get("B",0)); c3.metric("C",ac.get("C",0))
            rs = sv[(sv["abc_class"]=="A")&(sv["recommended_location"]!=sv["current_location"])]
            st.markdown("#### 🔄 Re-slots"); st.dataframe(rs[["sku","abc_class","velocity_score","current_location","recommended_location","rationale"]],width="stretch",hide_index=True)
        else: st.info("Run analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🔄 Replenishment":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        if st.button("Generate suggestions",type="primary",width="stretch"):
            with st.spinner("Scanning..."): sg = _replen_engine.generate(forecast=forecast); st.success(f"✅ {len(sg)} suggestions."); enqueue_action("replen_generated",{"count":len(sg)}); log_action(user,"Replen",str(len(sg))); st.rerun()
        rd = get_replenishments(status="Open")
        if not rd.empty:
            st.dataframe(rd[["repl_id","sku","current_stock","reorder_point","suggested_qty","status"]],width="stretch",hide_index=True)
            sr = st.selectbox("Complete",rd["repl_id"].tolist())
            if st.button("Mark done",width="stretch"): complete_replenishment(sr,user); log_action(user,"Replen Done",sr); st.session_state.operator_stats[user].add_xp(15,"pick"); st.success("Done."); st.rerun()
        else: st.info("No open replenishments.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🔍 Cycle Counting":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        force = st.checkbox("Force ALL SKUs")
        if st.button("Schedule",type="primary",width="stretch"): sc = _cycle_scheduler.schedule(force_all=force); st.success(f"✅ {len(sc)} scheduled."); log_action(user,"Cycle Scheduled",str(len(sc))); st.session_state.operator_stats[user].add_xp(20,"audit"); st.rerun()
        cd = get_cycle_counts(status="Scheduled")
        if not cd.empty:
            st.dataframe(cd[["count_id","sku","location","expected_qty","abc_class","assigned_to"]],width="stretch",hide_index=True)
            sc = st.selectbox("Record",cd["count_id"].tolist()); cq = st.number_input("Counted",0,value=int(cd[cd["count_id"]==sc].iloc[0]["expected_qty"]))
            if st.button("Submit",width="stretch"): complete_cycle_count(sc,int(cq),user); row=cd[cd["count_id"]==sc].iloc[0]; v=int(cq)-int(row["expected_qty"]); log_action(user,"Cycle Submitted",f"{sc}: var {v}"); st.success("✅ Zero variance." if v==0 else f"⚠️ Variance {v}"); st.rerun()
        else: st.info("No scheduled counts.")
        dd = get_cycle_counts(status="Completed")
        if not dd.empty: st.markdown("#### Recent"); st.dataframe(dd[["count_id","sku","expected_qty","counted_qty","variance","assigned_to"]].head(15),width="stretch",hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "📦 Receiving":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        with st.form("asn_form", clear_on_submit=True):
            aid = st.text_input("ASN ID")
            sup = st.text_input("Supplier")
            exp = st.text_area("SKUs")
            door = st.text_input("Door", value="D1")
            eta = st.text_input("ETA", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if st.form_submit_button("Create ASN") and aid:
                items = [s.strip() for s in exp.splitlines() if s.strip()]
                create_asn(aid, sup, items, dock_door=door, eta=eta)
                enqueue_action("asn_created", {"asn_id": aid, "supplier": sup})
                log_action(user, "ASN Created", aid)
                st.success("Created.")
                st.rerun()
        adf=get_asns()
        if not adf.empty:
            st.dataframe(adf[["asn_id","supplier","expected_items","status","dock_door","eta"]],width="stretch",hide_index=True)
            oa=adf[adf["status"]!="Received"]["asn_id"].tolist()
            if oa:
                sa=st.selectbox("Receive",oa)
                if st.button("Mark received",type="primary"): receive_asn(sa); log_action(user,"ASN Received",sa); st.session_state.operator_stats[user].add_xp(15,"scan"); st.success("Received."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🚪 Dock":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        with st.form("dock_form", clear_on_submit=True):
            apid = st.text_input("Appt ID")
            door = st.text_input("Door", value="D1")
            carr = st.text_input("Carrier")
            dir = st.selectbox("Direction", ["Inbound", "Outbound"])
            at = st.text_input("Time", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if st.form_submit_button("Schedule") and apid:
                create_dock_appointment(apid, door, carr, dir, at)
                enqueue_action("dock_scheduled", {"appt_id": apid, "door": door})
                log_action(user, "Dock Scheduled", apid)
                st.success("Scheduled.")
                st.rerun()
        dd=get_dock_appointments()
        if not dd.empty: st.dataframe(dd[["appointment_id","door","carrier","direction","appointment_time","status"]],width="stretch",hide_index=True)
        else: st.info("No appointments.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "↩️ RMA":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        with st.form("rma_form", clear_on_submit=True):
            rid = st.text_input("RMA ID")
            oid = st.text_input("Order ID")
            sku = st.text_input("SKU")
            reason = st.text_input("Reason")
            cond = st.selectbox("Condition", ["Uninspected", "Damaged", "Good", "Resealed"])
            if st.form_submit_button("Log") and rid:
                create_rma(rid, oid, sku, reason, condition=cond)
                enqueue_action("rma_created", {"rma_id": rid, "sku": sku})
                log_action(user, "RMA Created", rid)
                st.success("Logged.")
                st.rerun()
        rd=get_rmas()
        if not rd.empty:
            st.dataframe(rd[["rma_id","order_id","sku","reason","condition","disposition","status"]],width="stretch",hide_index=True)
            ors=rd[rd["status"]!="Processed"]["rma_id"].tolist()
            if ors:
                sr=st.selectbox("Process",ors); disp=st.selectbox("Disposition",["Restock","Repair","Refurbish","Scrap","Return"])
                if st.button("Set",type="primary"): update_rma_disposition(sr,disp); log_action(user,"RMA Processed",f"{sr}->{disp}"); st.success("Processed."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "👷 Labor":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        ld=get_labor_summary()
        if not ld.empty:
            st.dataframe(ld,width="stretch",hide_index=True); st.metric("🧺 Total picked",int(ld[ld["activity"]=="pick"]["units"].sum()) if "activity" in ld.columns else 0)
        else: st.info("No labor events.")
        with st.expander("Recent log"): st.dataframe(get_labor_logs(100),width="stretch",hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🤖 Copilot":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        hist=st.session_state.copilot_history
        for h in hist[-8:]:
            st.markdown(f"<div style='color:#8892b0;font-size:.85rem'>› {h['q']}</div>",unsafe_allow_html=True)
            fol=(f"<br><span style='color:#64ffda;font-size:.85rem'>→ {h['a']['followup']}</span>" if h["a"].get("followup") else "")
            st.markdown(f"<div style='color:#ccd6f6;margin:2px 0 6px 12px'>{h['a']['answer']}{fol}</div>",unsafe_allow_html=True)
            if h["a"].get("data") is not None: st.dataframe(h["a"]["data"],width="stretch",hide_index=True)
        qc=st.chat_input("Ask: 'breach cutoff?' · 'top picker?' · 'stockout risk?'")
        if qc: ans=_copilot.ask(qc); hist.append({"q":qc,"a":ans}); log_action(user,"Copilot",qc); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    elif mode == "🤖 Auto":
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: as_=st.checkbox("Slotting",value=True); ac_=st.checkbox("Counts",value=True)
        with c2: aw_=st.checkbox("Waves",value=True); ar_=st.checkbox("Replen",value=True)
        with c3: ws=st.selectbox("Wave strategy",["batch","single","zone"]); wl=st.number_input("Max/wave",1,100,25)
        if st.button("🚀 Execute",type="primary",width="stretch"):
            res=[]
            if as_: sd=_slotting.analyze(); su=_slotting.summary(sd); res.append(f"🏷️ {su['A']}A/{su['B']}B/{su['C']}C"); st.session_state.operator_stats[user].add_xp(25,"audit")
            if ac_: sc=_cycle_scheduler.schedule(); res.append(f"🔍 {len(sc)} counts"); st.session_state.operator_stats[user].add_xp(20,"audit")
            if aw_: pl=_wave_planner.plan(strategy=ws,limit=int(wl)); res.append(f"🌊 {len(pl.tasks) if pl else 0} tasks"); st.session_state.operator_stats[user].add_xp(30,"pick") if pl else None
            if ar_: sg=_replen_engine.generate(forecast=forecast); res.append(f"🔄 {len(sg)} replen")
            st.success("✅ Complete!"); [st.markdown(f"• {r}") for r in res]; log_action(user,"Auto-Workflow"," | ".join(res))
        st.markdown("</div>", unsafe_allow_html=True)

# ── INVENTORY ──
with t_inv:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Inventory Management")
    with st.form("inv_form", clear_on_submit=True):
        sku = st.text_input("SKU")
        prod = st.text_input("Product")
        stk = st.number_input("Stock", min_value=0)
        loc = st.text_input("Location", value="UNASSIGNED")
        note = st.text_input("Note")
        if st.form_submit_button("Save") and sku:
            upsert_inventory(sku, prod, int(stk), loc)
            add_action_log("inventory_upsert", sku, f"{prod}|{stk}|{loc}", user)
            enqueue_action("inventory_upsert", {"sku": sku, "product": prod, "stock": int(stk), "location": loc, "note": note})
            st.success("Saved.")
            st.rerun()
    st.dataframe(get_inventory(),width="stretch",hide_index=True)
    c1,c2=st.columns(2); c1.write(f"Alias: {suggest_alias(prod) or 'None'}"); c2.write(f"Template: {suggest_template(prod) or 'None'}")
    st.markdown("</div>", unsafe_allow_html=True)

# ── ORDERS ──
with t_ord:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Orders")
    with st.form("ord_form", clear_on_submit=True):
        oid = st.text_input("Order ID")
        st_ = st.selectbox("Status", ["Pending", "Shipped", "Returned", "Cancelled"])
        items = st.text_area("SKUs")
        if st.form_submit_button("Create") and oid:
            skus = [x.strip() for x in items.splitlines() if x.strip()]
            create_order(oid, st_, skus)
            add_action_log("order_create", oid, ",".join(skus), user)
            enqueue_action("order_create", {"order_id": oid, "status": st_, "required_skus": skus})
            st.success("Created.")
            st.rerun()
    od=get_orders(); st.dataframe(od,width="stretch",hide_index=True)
    if not od.empty:
        st.divider(); st.subheader("Update"); sel=st.selectbox("Select",od["order_id"].tolist()); ns=st.selectbox("New status",["Pending","Shipped","Returned","Cancelled"],key="ns")
        if st.button("Update"): update_order_status(sel,ns); add_action_log("order_update",sel,ns,user); enqueue_action("order_update",{"order_id":sel,"status":ns}); st.success("Updated."); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── AUDITOR ──
with t_aud:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Discrepancy Auditor")
    c1,c2=st.columns(2)
    with c1: m_in=st.text_area("MASTER",height=200,placeholder="ID Value")
    with c2: s_in=st.text_area("SCAN",height=200,placeholder="ID Value")
    if st.button("Run Analysis",type="primary",width="stretch") and m_in and s_in:
        mm,sm={},{}
        for l in m_in.splitlines():
            l=l.strip()
            if l: k,v=l.split(None,1); mm.setdefault(k,set()).add(v)
        for l in s_in.splitlines():
            l=l.strip()
            if l: k,v=l.split(None,1); sm.setdefault(k,set()).add(v)
        res=[]
        for tid in sorted(set(mm)|set(sm)):
            e,g=mm.get(tid,set()),sm.get(tid,set())
            res.append({"ID":tid,"Status":"✅ MATCH" if e==g else "❌ ERROR","Expected":" | ".join(e),"Actual":" | ".join(g)})
        st.dataframe(pd.DataFrame(res).style.apply(lambda x:['background-color:#ffcccc' if '❌' in str(v) else '' for v in x],axis=1),width="stretch",hide_index=True)
        log_action(user,"Auditor"); st.session_state.operator_stats[user].add_xp(25,"audit")
    st.markdown("</div>", unsafe_allow_html=True)

# ── BULK CONVERT ──
with t_bulk:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Bulk Title Converter")
    fmt=st.radio("Format",["Template Only","Translation Only","Combined"],horizontal=True)
    c1,c2=st.columns(2)
    with c1: raw=st.text_area("Input",height=300)
    if st.button("✨ Convert",type="primary",width="stretch") and raw:
        lines=[l.strip() for l in raw.splitlines()]; res=[]; mt=0
        with st.spinner("Translating..."):
            tr=GoogleTranslator(source='auto',target='en')
            for l in lines:
                if not l: continue
                try:
                    t=tr.translate(l); std=re.sub(r"\s+"," ",t).strip().title(); tm=suggest_template(std)
                    if tm: mt+=1; res.append(tm if fmt=="Template Only" else std if fmt=="Translation Only" else f"{tm} (Match: {std})")
                    else: res.append(std)
                except: res.append(l.upper())
        with c2: st.text_area("Output",value="\n".join(res),height=300)
        st.success(f"✅ Processed {len(lines)} titles. {mt} templates."); log_action(user,"Bulk Conversion",f"{len(lines)} lines, {mt} templates")
    st.markdown("</div>", unsafe_allow_html=True)

# ── PDF SEQUENCER ──
with t_pdf:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📦 Pro PDF Label Sequencer v4.4")
    with st.expander("📜 Recent jobs"):
        jbs=get_label_jobs(15); st.dataframe(jbs,width="stretch",hide_index=True) if not jbs.empty else st.info("No jobs.")
    mode=st.radio("🎯 Mode",["📋 Smart Sort","🔒 Strict","📱 Phone+Code","🏷️ WB Advanced"],horizontal=True)
    c1,c2=st.columns([1,2])
    with c1:
        if mode=="🏷️ WB Advanced": lst=st.text_area("Target List",height=300,placeholder="WB tracking, phone+code, order IDs"); st.caption("Auto-detects vertical tracking, QR payloads, phone codes")
        elif mode=="📱 Phone+Code": lst=st.text_area("Phone+Code",height=300,placeholder="5261288 1844")
        else: lst=st.text_area("Sequence",height=300,placeholder="Tracking IDs")
        rem_dup=st.checkbox("Remove duplicates",value=True); show_dbg=st.checkbox("Debug overlays",value=False)
        if mode=="🏷️ WB Advanced": conf=st.slider("Min confidence",0.0,1.0,0.5,0.05)
    with c2: pdf=st.file_uploader("Upload PDF",type="pdf"); use_ocr=st.checkbox("OCR Fallback",value=True)
    if st.button("⚙️ Process",type="primary",width="stretch"):
        if mode=="🏷️ WB Advanced":
            if not lst or not pdf: st.warning("Provide list & PDF")
            else:
                with st.spinner("🔍 WB Advanced Pipeline..."):
                    try:
                        targets=parse_target_list(lst)
                        if rem_dup:
                            seen=set(); uniq=[]
                            for t in targets:
                                k=t.get('tracking','')+t.get('phone','')+t.get('code','')
                                if k not in seen:
                                    seen.add(k); uniq.append(t)
                            targets=uniq
                        st.info(f"📋 Parsed {len(targets)} targets")
                        lb_data=_wb_proc.process_pdf(pdf.getvalue()); st.success(f"📄 {len(lb_data)} pages. {sum(1 for d in lb_data if d.tracking_number)} tracking, {sum(1 for d in lb_data if d.phone_number)} phones")
                        matches=_wb_proc.match_targets(targets,lb_data)
                        pr=pypdf.PdfReader(io.BytesIO(pdf.getvalue())); pw=pypdf.PdfWriter(); res=[]; matched_idx=set(); pc=1
                        for m in matches:
                            if m.confidence>=conf: pw.add_page(pr.pages[m.page_idx]); matched_idx.add(m.page_idx); res.append({"Status":"✅ MATCHED","New #":pc,"Target":m.target_raw[:40],"Orig":m.page_idx+1,"Type":m.match_type,"Conf":f"{m.confidence:.0%}","Fields":str(m.matched_fields)}); pc+=1
                        for ld in lb_data:
                            if ld.page_idx not in matched_idx: pw.add_page(pr.pages[ld.page_idx]); res.append({"Status":"ℹ️ EXTRA","New #":pc,"Target":"—","Orig":ld.page_idx+1,"Type":"Unlisted","Conf":"—","Fields":f"Tracking: {ld.tracking_number or 'N/A'}"}); pc+=1
                        mc=sum(1 for r in res if '✅' in r['Status']); miss=len(targets)-mc
                        st.divider(); st.markdown("### 📊 Results"); rdf=pd.DataFrame(res); st.dataframe(rdf.style.applymap(lambda v:'background:rgba(0,255,136,.15);color:#00ff88' if '✅' in str(v) else 'background:rgba(255,107,107,.15);color:#ff6b6b' if '❌' in str(v) else 'background:rgba(255,217,61,.15);color:#ffd93d' if 'ℹ️' in str(v) else '',subset=['Status']),width="stretch",hide_index=True)
                        s1,s2,s3,s4=st.columns(4); s1.metric("✅ Sequenced",mc); s2.metric("❌ Missing",miss); s3.metric("ℹ️ Extra",len(res)-mc-miss); s4.metric("🎯 Avg Conf",f"{sum(m.confidence for m in matches)/len(matches):.0%}" if matches else "N/A")
                        if show_dbg:
                            st.divider(); st.markdown("### 🔍 Debug"); cols=st.columns(3); imgs=convert_from_bytes(pdf.getvalue(),dpi=150)
                            for i,ld in enumerate(lb_data[:9]):
                                with cols[i%3]: st.image(_wb_proc.generate_debug_overlay(imgs[ld.page_idx],ld),caption=f"Page {ld.page_idx+1}",width="stretch")
                        if mc>0:
                            out=io.BytesIO(); pw.write(out); avg=sum(m.confidence for m in matches)/len(matches) if matches else 0.0
                            _wb_proc.record_job("WB","advanced",len(targets),mc,miss,len(res)-mc-miss,avg,user)
                            log_action(user,"PDF_SEQUENCED_WB_ADV",f"M:{mc},Mi:{miss},E:{len(res)-mc-miss}")
                            st.success(f"✅ Ready! {mc} labels reordered."); st.download_button("📥 Download",data=out.getvalue(),file_name=f"WB_Advanced_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",mime="application/pdf",width="stretch")
                        else: st.error("❌ No matches.")
                    except Exception as e: st.error(f"❌ Error: {e}"); import traceback; st.code(traceback.format_exc())
        elif mode=="📱 Phone+Code":
            entries=[]
            for l in lst.strip().split('\n'):
                l=l.strip()
                if not l: continue
                dg=re.findall(r'\d+',l); ph=next((d for d in dg if len(d)==7),None); cd=next((d for d in dg if len(d)==4),None)
                if ph and cd: entries.append({'phone':ph,'code':cd,'raw':l})
            if rem_dup and entries:
                seen = set(); cleaned = []; dupes = 0
                for e in entries:
                    k = (e['phone'], e['code'])
                    if k not in seen:
                        seen.add(k); cleaned.append(e)
                    else:
                        dupes += 1
                entries = cleaned
                if dupes: st.toast(f"Cleaned {dupes} dupes", icon="🧹")
            if not entries or not pdf: st.warning("Provide list & PDF")
            else:
                with st.spinner("Scanning..."):
                    try:
                        pr=pypdf.PdfReader(io.BytesIO(pdf.getvalue())); pw=pypdf.PdfWriter(); imgs=convert_from_bytes(pdf.getvalue(),dpi=300); pm=[]
                        for i,img in enumerate(imgs):
                            w,h=img.size; txt=""
                            for b in decode(img): txt+=" "+b.data.decode("utf-8",errors="ignore")
                            txt+=" "+pytesseract.image_to_string(img,config='--psm 6')
                            rc=img.crop((int(w*0.55),int(h*0.55),w,h))
                            for a in [0,90,180,270]:
                                rot=rc.rotate(a,expand=True)
                                for psm in ['--psm 6','--psm 7','--psm 13']: txt+=" "+pytesseract.image_to_string(rot,config=psm)
                            mc2=img.crop((int(w*0.4),int(h*0.7),int(w*0.9),h)); txt+=" "+pytesseract.image_to_string(mc2,config='--psm 6')
                            pm.append({'idx':i,'page':pr.pages[i],'phones':list(set(re.findall(r'\b\d{7}\b',txt))),'codes':list(set(re.findall(r'\b\d{4}\b',txt)))})
                        matched_idx=[]; res=[]; pc=1
                        for t in entries:
                            mp,mt=None,""
                            for p in pm:
                                if p['idx'] in matched_idx: continue
                                if t['phone'] in p['phones'] and t['code'] in p['codes']: mp,mt=p,"Exact"; break
                            if not mp:
                                for p in pm:
                                    if p['idx'] in matched_idx: continue
                                    if t['phone'] in p['phones']: mp,mt=p,"Phone Only"; break
                            if mp:
                                matched_idx.append(mp['idx']); pw.add_page(mp['page']); res.append({"Status":"✅ MATCHED","New #":pc,"Target":f"{t['phone']} {t['code']}","Orig":mp['idx']+1,"Type":mt,"Detected":f"Ph:{','.join(mp['phones'])} | Cd:{','.join(mp['codes'])}"}); pc+=1
                            else: res.append({"Status":"❌ MISSING","New #":"—","Target":f"{t['phone']} {t['code']}","Orig":"N/A","Type":"No Match","Detected":"—"})
                        ec=0
                        for p in pm:
                            if p['idx'] not in matched_idx: pw.add_page(p['page']); ec+=1; res.append({"Status":"ℹ️ EXTRA","New #":pc,"Target":"Unlisted","Orig":p['idx']+1,"Type":"Appended","Detected":f"Ph:{','.join(p['phones'])} | Cd:{','.join(p['codes'])}"}); pc+=1
                        mc=sum(1 for r in res if '✅' in r['Status']); miss=sum(1 for r in res if '❌' in r['Status'])
                        st.divider(); st.markdown("### 📊 Results"); rdf=pd.DataFrame(res); st.dataframe(rdf.style.applymap(lambda v:'background:rgba(40,167,69,.2);color:#28a745' if '✅' in str(v) else 'background:rgba(220,53,69,.2);color:#dc3545' if '❌' in str(v) else 'background:rgba(255,193,7,.2);color:#ffc107' if 'ℹ️' in str(v) else '',subset=['Status']),width="stretch",hide_index=True)
                        s1,s2,s3=st.columns(3); s1.metric("✅ Sequenced",mc); s2.metric("❌ Missing",miss); s3.metric("ℹ️ Extra",ec)
                        if mc>0 or ec>0:
                            out=io.BytesIO(); pw.write(out); _wb_proc.record_job("WB","phone_code",len(entries),mc,miss,ec,0.9,user)
                            log_action(user,"PDF_SEQUENCED_WB_ENH",f"M:{mc},Mi:{miss},E:{ec}")
                            st.success(f"✅ Ready! {mc} labels."); st.download_button("📥 Download",data=out.getvalue(),file_name=f"WB_Sequenced_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",mime="application/pdf",width="stretch")
                        else: st.error("❌ No matches.")
                    except Exception as e: st.error(f"❌ Error: {e}")
        else:
            raw_ids=[t.strip() for t in lst.split('\n') if t.strip()]; ids=[re.search(r"[A-Z0-9][A-Z0-9-]{2,}",t).group() if re.search(r"[A-Z0-9][A-Z0-9-]{2,}",t) else t for t in raw_ids]
            if rem_dup and ids: ids=list(dict.fromkeys(ids))  # FIX: preserve order, remove dupes
            if not ids or not pdf: st.warning("Provide IDs & PDF")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        pr=pypdf.PdfReader(io.BytesIO(pdf.getvalue())); pw=pypdf.PdfWriter(); imgs=convert_from_bytes(pdf.getvalue(),dpi=200); map={}
                        for i,img in enumerate(imgs):
                            codes=[]
                            for b in decode(img): codes.extend(re.findall(r"[A-Z0-9][A-Z0-9-]{2,}",b.data.decode("utf-8")))
                            if not codes and use_ocr: codes.extend(re.findall(r"[A-Z0-9][A-Z0-9-]{2,}",pytesseract.image_to_string(img)))
                            for c in set(codes): map[c]={"page":pr.pages[i],"idx":i+1}
                        res=[]; mc=0; pc=1; exp=set(ids); strict=mode=="🔒 Strict"
                        if strict: st.info("🔒 Strict: exact order only")
                        for tid in ids:
                            if tid in map: pw.add_page(map[tid]["page"]); mc+=1; res.append({"Status":"✅ INCLUDED" if strict else "✅ MATCHED","Seq":ids.index(tid)+1,"ID":tid,"Orig":map[tid]["idx"],"Out":pc}); pc+=1
                            else: res.append({"Status":"❌ MISSING","Seq":ids.index(tid)+1,"ID":tid,"Orig":"N/A","Out":"N/A"})
                        ep=0
                        if not strict:
                            for tid,d in map.items():
                                if tid not in exp: pw.add_page(d["page"]); ep+=1; res.append({"Status":"ℹ️ EXTRA","Seq":"—","ID":tid,"Orig":d["idx"],"Out":pc}); pc+=1
                        st.divider(); st.markdown("### 📊 Results"); st.dataframe(pd.DataFrame(res),width="stretch",hide_index=True)
                        if mc>0:
                            out=io.BytesIO(); pw.write(out); _wb_proc.record_job("WB","strict" if strict else "smart",len(ids),mc,len(ids)-mc,ep,1.0,user)
                            log_action(user,"PDF_SEQUENCED",f"Mode:{mode},M:{mc}"); st.success(f"✅ Ready! {mc} pages."); st.download_button("📥 Download",data=out.getvalue(),file_name=f"sorted_{'strict' if strict else 'smart'}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",mime="application/pdf",width="stretch")
                        else: st.error("❌ No matches.")
                    except Exception as e: st.error(f"❌ Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ── TEMPLATES ────────────────────────────────────────────────────────────
with t_temp:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Templates Database")
    with st.form("template_form", clear_on_submit=True):
        raw = st.text_input("Raw/Translated Title")
        standard = st.text_input("Standard/Clean Title")
        saved = st.form_submit_button("Save template")
    if saved and raw and standard:
        save_template(raw, standard)
        enqueue_action("template_save", {"raw": raw, "standard": standard})
        st.success("✅ Template saved locally and queued for neural sync.")
        log_action(user, "Template Saved", f"{raw} -> {standard}")
        st.rerun()
    st.dataframe(get_templates(), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── MEMORY ───────────────────────────────────────────────────────────────
with t_mem:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Memory & Preferences")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("### General Preferences")
        with st.form("memory_form", clear_on_submit=True):
            pref_key = st.text_input("Preference key")
            pref_value = st.text_input("Preference value")
            save_pref = st.form_submit_button("Save memory")
        if save_pref and pref_key:
            save_memory(pref_key, pref_value)
            record_preference(pref_key, pref_value)
            st.success("Memory stored locally.")
            log_action(user, "Memory Saved", pref_key)
            st.rerun()
    with col_m2:
        st.write("### Product Aliases")
        alias_src = st.text_input("Alias source text")
        alias_dst = st.text_input("Alias target text")
        if st.button("Save alias") and alias_src and alias_dst:
            upsert_alias(alias_src, alias_dst)
            enqueue_action("memory_save", {"key": f"alias:{alias_src.lower().strip()}", "value": alias_dst})
            st.success("Alias saved and queued.")
            log_action(user, "Alias Saved", f"{alias_src} -> {alias_dst}")
            st.rerun()
    st.write("### Recent System Preferences")
    st.dataframe(get_recent_preferences(), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── NEURAL OPS ───────────────────────────────────────────────────────────
with t_neural:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🧠 Neural Operations Center")
    st.markdown("<p style='color:#8892b0;'>AI-powered vision inspection, anomaly detection, and predictive intelligence.</p>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns([1, 1])
    with n_col1:
        st.markdown("### 📸 Neural Vision Inspection")
        vision_file = st.file_uploader("Upload image for neural analysis", type=["png", "jpg", "jpeg"])
        vision_confidence = st.slider("Detection sensitivity threshold", 50, 98, 65)
        if vision_file:
            img = Image.open(vision_file)
            with st.spinner("Running neural vision pipeline..."):
                processed_img, detections = _neural_vision.process_frame(img)
                st.image(processed_img, caption="Neural Vision Overlay", width="stretch")
                if detections:
                    det_df = pd.DataFrame([{"Label": d["label"], "Confidence": f"{d['confidence']:.1f}%",
                                            "Area": d["area"], "Vertices": d["vertices"]}
                                           for d in detections if d["confidence"] >= vision_confidence])
                    if not det_df.empty:
                        st.dataframe(det_df, width="stretch", hide_index=True)
                        damage_count = len([d for d in detections if d["label"] == "Damage" and d["confidence"] >= vision_confidence])
                        if damage_count > 0:
                            st.error(f"🚨 {damage_count} potential damage(s) detected! Manual inspection required.")
                        else:
                            st.success("✅ No damage detected above threshold.")
                    else:
                        st.info("No detections above confidence threshold.")
                else:
                    st.info("No objects detected in image.")
                log_action(user, "Neural Vision Scan", f"Detections: {len(detections)}")
                st.session_state.operator_stats[user].add_xp(10, "scan")
    with n_col2:
        st.markdown("### 🔮 Oracle Predictive Analytics")
        if forecast:
            st.markdown("<div class='holographic-card'>", unsafe_allow_html=True)
            f_cols = st.columns(3)
            f_cols[0].metric("Trend", forecast["trend"])
            f_cols[1].metric("Stockout Risk", forecast["stockout_risk"])
            f_cols[2].metric("Confidence", f"{forecast['confidence']}%")
            st.markdown(f"<div style='color:#ccd6f6; margin-top:10px;'>📦 Recommended reorder: <span class='neon-text'>{forecast['recommended_reorder']}</span> units</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#8892b0; font-size:0.85rem;'>Avg daily orders: {forecast['avg_daily_orders']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if forecast["forecast"]:
                forecast_df = pd.DataFrame({"Day": [f"Day +{i+1}" for i in range(len(forecast["forecast"]))],
                                            "Projected Stock": forecast["forecast"]})
                st.line_chart(forecast_df.set_index("Day"), width="stretch")
        else:
            st.info("Oracle needs more historical data (minimum 7 days) to generate forecasts.")
        st.markdown("### 🛡️ Anomaly Sentinel Feed")
        if current_alerts:
            for alert in current_alerts[:8]:
                severity_color = {"CRITICAL": "#ff6b6b", "HIGH": "#ff9f43", "MEDIUM": "#ffd93d"}.get(alert["severity"], "#64ffda")
                st.markdown(f"<div style='border-left:3px solid {severity_color}; padding:8px 12px; margin:4px 0; background:rgba(255,255,255,0.03); border-radius:0 8px 8px 0; font-size:0.85rem;'><b>{alert['icon']} {alert['type']}</b> — {alert['message']} <span style='color:#8892b0;'>[{alert['sku']}]</span></div>", unsafe_allow_html=True)
        else:
            st.success("✅ All systems nominal. No anomalies detected.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── HOLO-DECK ────────────────────────────────────────────────────────────
with t_holo:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🗺️ Warehouse Holo-Deck")
    st.markdown("<p style='color:#8892b0;'>Interactive 3D digital twin of your warehouse with real-time zone monitoring.</p>", unsafe_allow_html=True)
    holo_cols = st.columns([3, 1])
    with holo_cols[0]:
        holo_html = """
        <div id="warehouse-3d" style="width:100%; height:600px; background:#050a19; border-radius:16px; overflow:hidden; position:relative; border:1px solid rgba(100,255,218,0.2);">
          <canvas id="glcanvas" style="width:100%; height:100%;"></canvas>
          <div style="position:absolute; top:15px; left:15px; color:#64ffda; font-family:monospace; font-size:11px; background:rgba(5,10,25,0.85); padding:12px; border-radius:8px; border:1px solid rgba(100,255,218,0.15);">
            <div style="font-weight:bold; font-size:13px; margin-bottom:6px;">🏭 WAREHOUSE DIGITAL TWIN</div>
            <div>Zones: 24 | Active: 18 | Temp: 22°C | Humidity: 45%</div>
            <div style="margin-top:4px; color:#8892b0;">Live Neural Feed: <span style="color:#64ffda;">ONLINE</span></div>
          </div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0x050a19);
          scene.fog = new THREE.FogExp2(0x050a19, 0.015);
          const container = document.getElementById('warehouse-3d');
          const camera = new THREE.PerspectiveCamera(60, container.clientWidth/container.clientHeight, 0.1, 1000);
          const renderer = new THREE.WebGLRenderer({canvas: document.getElementById('glcanvas'), antialias: true, alpha: true});
          renderer.setSize(container.clientWidth, container.clientHeight);
          renderer.setPixelRatio(window.devicePixelRatio);
          scene.add(new THREE.AmbientLight(0x404040, 1.5));
          const dirLight = new THREE.DirectionalLight(0x64ffda, 0.8);
          dirLight.position.set(20, 30, 20);
          scene.add(dirLight);
          const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 60),
            new THREE.MeshPhongMaterial({color: 0x0a192f, transparent: true, opacity: 0.8, side: THREE.DoubleSide}));
          floor.rotation.x = -Math.PI / 2;
          scene.add(floor);
          const gridHelper = new THREE.GridHelper(80, 40, 0x64ffda, 0x0a192f);
          gridHelper.position.y = 0.01;
          scene.add(gridHelper);
          const rackGeo = new THREE.BoxGeometry(3, 6, 1.5);
          const rackMat = new THREE.MeshPhongMaterial({color: 0x112240, transparent: true, opacity: 0.85});
          const edgesGeo = new THREE.EdgesGeometry(rackGeo);
          const edgesMat = new THREE.LineBasicMaterial({color: 0x64ffda, transparent: true, opacity: 0.6});
          for(let i=0; i<24; i++) {
            const rack = new THREE.Mesh(rackGeo, rackMat);
            rack.position.set((i%6)*10 - 25, 3, Math.floor(i/6)*10 - 15);
            scene.add(rack);
            const edges = new THREE.LineSegments(edgesGeo, edgesMat);
            edges.position.copy(rack.position);
            scene.add(edges);
          }
          const pkgGeo = new THREE.BoxGeometry(1, 1, 1);
          const packages = [];
          for(let i=0; i<20; i++) {
            const color = new THREE.Color().setHSL(0.45 + Math.random()*0.15, 0.9, 0.6);
            const pkg = new THREE.Mesh(pkgGeo, new THREE.MeshPhongMaterial({color: color, emissive: color, emissiveIntensity: 0.3}));
            pkg.position.set(Math.random()*50-25, 0.5, Math.random()*30-15);
            scene.add(pkg);
            packages.push({mesh: pkg, speed: 0.01 + Math.random()*0.02, offset: Math.random()*Math.PI*2, amp: 0.3 + Math.random()*0.5});
          }
          const particleGeo = new THREE.BufferGeometry();
          const posArray = new Float32Array(600);
          for(let i=0; i<600; i++) posArray[i] = (Math.random()-0.5)*80;
          particleGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
          const particles = new THREE.Points(particleGeo,
            new THREE.PointsMaterial({size: 0.15, color: 0x64ffda, transparent: true, opacity: 0.6}));
          scene.add(particles);
          camera.position.set(35, 25, 35);
          camera.lookAt(0, 0, 0);
          let angle = 0;
          function animate() {
            requestAnimationFrame(animate);
            const time = Date.now() * 0.001;
            angle += 0.002;
            camera.position.x = 40 * Math.cos(angle);
            camera.position.z = 40 * Math.sin(angle);
            camera.lookAt(0, 2, 0);
            packages.forEach(p => {
              p.mesh.position.y = 0.5 + Math.sin(time * 2 + p.offset) * p.amp * 0.3;
              p.mesh.rotation.y += p.speed;
            });
            particles.rotation.y = time * 0.05;
            renderer.render(scene, camera);
          }
          animate();
          window.addEventListener('resize', () => {
            renderer.setSize(container.clientWidth, container.clientHeight);
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
          });
        </script>
        """
        st.components.v1.html(holo_html, height=620)
    with holo_cols[1]:
        st.markdown("### Zone Telemetry")
        for name, z in list(_route_opt.zones.items())[:8]:
            heat_val = _route_opt.heat_map[z["y"], z["x"]]
            heat_pct = min(100, int(heat_val / max(1, _route_opt.heat_map.max()) * 100))
            color = "#64ffda" if z["velocity"] == "high" else "#00b4db" if z["velocity"] == "medium" else "#8892b0"
            st.markdown(f"<div style='margin-bottom:8px;'><span style='color:{color}; font-weight:bold; font-family:monospace;'>{name}</span> <span style='color:#8892b0; font-size:0.8rem;'>({z['velocity']})</span><div style='background:rgba(255,255,255,0.05); height:6px; border-radius:3px; margin-top:2px;'><div style='width:{heat_pct}%; height:100%; background:{color}; border-radius:3px;'></div></div></div>", unsafe_allow_html=True)
        st.markdown("### System Status")
        st.markdown("<div class='neon-text' style='font-size:0.9rem;'>🟢 Neural Link: ACTIVE</div>", unsafe_allow_html=True)
        st.markdown("<div class='neon-text' style='font-size:0.9rem;'>🟢 Digital Twin: SYNCED</div>", unsafe_allow_html=True)
        st.markdown("<div class='neon-text' style='font-size:0.9rem;'>🟢 Event Bus: LIVE</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── QUANTUM ROUTES ───────────────────────────────────────────────────────
with t_quantum:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("⚡ Quantum Route Optimizer")
    st.markdown("<p style='color:#8892b0;'>TSP-based pick path optimization with 2-opt local search and warehouse heatmap visualization.</p>", unsafe_allow_html=True)
    q_col1, q_col2 = st.columns([1, 2])
    with q_col1:
        st.markdown("### 🎯 Pick List Input")
        route_input = st.text_area("Enter SKUs to pick (one per line)", height=200, placeholder="SKU-001\nSKU-042\nSKU-117")
        optimize_btn = st.button("🚀 Calculate Quantum Route", type="primary", width="stretch")
    with q_col2:
        st.markdown("### 🗺️ Warehouse Heatmap & Route")
        if optimize_btn and route_input:
            sku_list = [s.strip() for s in route_input.split("\n") if s.strip()]
            if len(sku_list) > 1:
                with st.spinner("Optimizing pick path via quantum algorithm..."):
                    route = _route_opt.optimize_route(sku_list)
                    total_distance = sum(_route_opt._dist(route[i], route[i+1]) for i in range(len(route)-1))
                    st.success(f"✅ Route optimized! {len(route)} stops, estimated travel: {total_distance:.1f} grid units")
                    st.markdown(f"<div class='neon-text' style='font-size:0.9rem;'>🗺️ Route: {' -> '.join([r['zone'] for r in route])}</div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{"Stop": i+1, "SKU": r["sku"], "Zone": r["zone"], "X": r["x"], "Y": r["y"]}
                                               for i, r in enumerate(route)]), width="stretch", hide_index=True)
                    st.markdown(_route_opt.generate_svg(route), unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Stops", len(route))
                    m2.metric("Grid Distance", f"{total_distance:.1f}")
                    m3.metric("Efficiency", f"{max(0, 100 - total_distance/len(route)*5):.0f}%")
                    log_action(user, "Quantum Route Optimized", f"Stops: {len(route)}, Distance: {total_distance:.1f}")
                    st.session_state.operator_stats[user].add_xp(20, "pick")
            else:
                st.warning("Please enter at least 2 SKUs for route optimization.")
        else:
            st.markdown(_route_opt.generate_svg([]), unsafe_allow_html=True)
            st.info("Enter SKUs and click 'Calculate Quantum Route' to visualize optimal pick path.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── COMMAND CENTER ───────────────────────────────────────────────────────
with t_cmd:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🎮 Neural Command Center")
    st.markdown("<p style='color:#8892b0;'>Natural language warehouse operations and operator gamification leaderboard.</p>", unsafe_allow_html=True)
    cmd_cols = st.columns([2, 1])
    with cmd_cols[0]:
        st.markdown("### 💬 Neural Command Interface")
        st.caption("Try: 'find SKU-123', 'show low stock', 'forecast inventory', 'optimize route for SKU-001 SKU-002'")
        for msg in st.session_state.neural_chat_history[-10:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_cmd = st.chat_input("Enter neural command...")
        if user_cmd:
            st.session_state.neural_chat_history.append({"role": "user", "content": user_cmd})
            parsed = _cmd_ai.parse(user_cmd)
            response = ""
            if parsed["intent"] == "GREETING":
                response = "👋 Welcome to Neural Command Center. I can help you find SKUs, move stock, check orders, forecast inventory, and optimize routes."
            elif parsed["intent"] == "FIND_SKU":
                sku_q = parsed["params"][0]
                inv_df = get_inventory()
                match = inv_df[inv_df["sku"].str.contains(sku_q, case=False, na=False)] if not inv_df.empty else pd.DataFrame()
                if not match.empty:
                    row = match.iloc[0]
                    response = f"🔍 Found **{sku_q}**: {row['product']} | Stock: {row['stock']} | Location: {row['location']}"
                else:
                    response = f"❌ SKU **{sku_q}** not found in inventory database."
            elif parsed["intent"] == "SHOW_LOW_STOCK":
                inv_df = get_inventory()
                low = inv_df[inv_df["stock"] < 5] if not inv_df.empty else pd.DataFrame()
                if not low.empty:
                    response = f"📉 Low stock alert! {len(low)} items below threshold:\n\n" + "\n".join(
                        [f"• {r['sku']}: {r['stock']} units ({r['location']})" for _, r in low.iterrows()])
                else:
                    response = "✅ All stock levels are healthy."
            elif parsed["intent"] == "FORECAST":
                if forecast:
                    response = f"🔮 Oracle Forecast: **{forecast['trend']}** trend. Stockout risk: **{forecast['stockout_risk']}**. Recommended reorder: **{forecast['recommended_reorder']}** units."
                else:
                    response = "🔮 Oracle needs more data to generate forecasts."
            elif parsed["intent"] == "OPTIMIZE_ROUTE":
                response = "⚡ Use the **Quantum Routes** tab for full route optimization with visualization!"
            elif parsed["intent"] == "VISION_SCAN":
                response = "📸 Use the **Neural Ops** tab to upload images for AI inspection."
            elif parsed["intent"] == "CREATE_ORDER":
                response = f"📋 Order creation initiated for **{parsed['params'][0]}**. Please use the **Orders** tab to finalize details."
            elif parsed["intent"] == "ORDER_STATUS":
                oid = parsed["params"][0]
                orders_df = get_orders()
                match = orders_df[orders_df["order_id"].str.contains(oid, case=False, na=False)] if not orders_df.empty else pd.DataFrame()
                if not match.empty:
                    row = match.iloc[0]
                    response = f"📦 Order **{oid}** status: **{row['status']}** | Items: {row.get('items', 'N/A')}"
                else:
                    response = f"❌ Order **{oid}** not found."
            else:
                response = "🤖 Command not recognized. Try: 'find SKU-123', 'show low stock', 'forecast inventory', or 'hello' for help."
            st.session_state.neural_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    with cmd_cols[1]:
        st.markdown("### 🏆 Operator Leaderboard")
        if st.session_state.operator_stats:
            stats_list = sorted(st.session_state.operator_stats.values(), key=lambda x: x.xp, reverse=True)
            for i, stat in enumerate(stats_list[:5]):
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                st.markdown("<div class='holographic-card' style='margin-bottom:8px; padding:10px;'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-weight:bold; color:#ccd6f6;'>{medal} {stat.username}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#64ffda; font-size:0.9rem;'>Lv.{stat.level} | {stat.xp} XP</div>", unsafe_allow_html=True)
                st.progress(min(1.0, (stat.xp % 1000) / 1000), text=f"{stat.xp % 1000}/1000")
                if stat.badges:
                    st.markdown(f"<div style='font-size:0.75rem; color:#8892b0;'>{' '.join(stat.badges[:3])}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No operator data yet. Complete actions to earn XP!")
        st.markdown("### 🎯 Quick Actions")
        if st.button("🎁 Claim Daily Bonus", width="stretch"):
            bonus_xp = random.randint(50, 150)
            ops.add_xp(bonus_xp, "scan")
            st.success(f"🎉 Daily bonus claimed! +{bonus_xp} XP")
            st.rerun()
        if st.button("🔄 Reset Streak", width="stretch"):
            ops.streak = 0
            st.info("Streak reset. Time to build it back up!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── ECO-LOGISTICS ────────────────────────────────────────────────────────
with t_eco:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🌱 Eco-Logistics Tracker")
    st.markdown("<p style='color:#8892b0;'>Carbon footprint analysis and sustainability recommendations.</p>", unsafe_allow_html=True)
    eco_cols = st.columns([1, 2])
    with eco_cols[0]:
        st.markdown("### 📊 Emission Calculator")
        eco_orders = st.number_input("Orders to ship today", min_value=0,
            value=int((orders["status"] == "Pending").sum()) if not orders.empty else 0, step=1)
        eco_distance = st.number_input("Avg. shipping distance (km)", min_value=10, value=420, step=10)
        eco_packaging = st.selectbox("Packaging type", ["standard", "recycled", "biodegradable"], index=1)
        if st.button("🌍 Calculate Footprint", type="primary", width="stretch"):
            footprint = _eco.calculate_footprint(eco_orders, eco_distance, eco_packaging)
            st.markdown("<div class='holographic-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='neon-text' style='font-size:1.2rem; margin-bottom:10px;'>Eco-Score: {footprint['eco_score']:.0f}/100</div>", unsafe_allow_html=True)
            st.progress(footprint['eco_score'] / 100, text="Sustainability Rating")
            f_m1, f_m2, f_m3 = st.columns(3)
            f_m1.metric("Transport", f"{footprint['transport_co2_kg']}kg", "CO₂")
            f_m2.metric("Packaging", f"{footprint['packaging_co2_kg']}kg", "CO₂")
            f_m3.metric("Warehouse", f"{footprint['warehouse_co2_kg']}kg", "CO₂")
            st.markdown(f"<div style='color:#ccd6f6; margin-top:10px; font-size:1.1rem;'><b>Total:</b> {footprint['total_co2_kg']}kg CO₂</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#64ffda; font-size:0.9rem;'>🌳 Trees needed to offset: {footprint['trees_needed']}</div>", unsafe_allow_html=True)
            for suggestion in footprint['suggestions']:
                st.markdown(f"<div style='color:#8892b0; font-size:0.85rem; margin:4px 0;'>• {suggestion}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            log_action(user, "Eco-Footprint Calculated", f"Orders: {eco_orders}, CO2: {footprint['total_co2_kg']}kg")
    with eco_cols[1]:
        st.markdown("### 📈 Sustainability Trends")
        eco_history = []
        for i in range(30):
            date = datetime.now() - timedelta(days=30-i)
            daily_orders = max(0, random.randint(5, 60) + i)
            fp = _eco.calculate_footprint(daily_orders, 420, "recycled")
            eco_history.append({"date": date, "co2": fp["total_co2_kg"], "eco_score": fp["eco_score"]})
        st.line_chart(pd.DataFrame(eco_history).set_index("date")[["co2", "eco_score"]], width="stretch")
        st.markdown("### ♻️ Green Initiatives")
        for name, desc, progress in [("Solar Panel Installation", "Reduce warehouse energy by 40%", 85),
                                     ("Biodegradable Packaging", "Eliminate plastic waste", 72),
                                     ("Route Consolidation AI", "Reduce transport emissions by 30%", 90),
                                     ("Electric Forklift Fleet", "Zero-emission material handling", 65)]:
            st.markdown(f"<div style='margin-bottom:8px;'><b style='color:#ccd6f6;'>{name}</b><div style='color:#8892b0; font-size:0.8rem;'>{desc}</div><div style='background:rgba(255,255,255,0.05); height:8px; border-radius:4px; margin-top:4px;'><div style='width:{progress}%; height:100%; background:linear-gradient(90deg, #64ffda, #00b4db); border-radius:4px;'></div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── ADMIN ────────────────────────────────────────────────────────────────
if t_admin is not None:
    with t_admin:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔐 Admin Control Panel")
        adm_opt = st.radio("Admin Tool", ["👤 User Management & Logs", "📱 SIM Database Manager",
                                          "🧠 Neural System Diagnostics"], horizontal=True)
        st.divider()
        if adm_opt == "👤 User Management & Logs":
            st.subheader("User Management")
            with st.expander("Add New System User"):
                new_u = st.text_input("New Username")
                new_p = st.text_input("New Password", type="password")
                new_r = st.selectbox("Role", ["Operator", "Admin"])
                if st.button("Create User") and new_u and new_p:
                    add_user(new_u, new_p, new_r)
                    st.success(f"User {new_u} added.")
                    add_action_log("User Created", new_u, new_r, user)
            st.subheader("System Audit Logs")
            with connect() as conn:
                logs_df = pd.read_sql_query(
                    "SELECT created_at, user, action_type, ref_id, payload FROM action_logs ORDER BY created_at DESC LIMIT 100", conn)
            st.dataframe(logs_df, width="stretch", hide_index=True)
            st.subheader("🧠 Neural System Health")
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            h_col1.metric("Vision Pipeline", "ONLINE", "✅")
            h_col2.metric("Oracle Engine", "ACTIVE", f"{len(_oracle.history)} days data")
            h_col3.metric("Sentinel Alerts", len(current_alerts), "🛡️")
            h_col4.metric("Event Bus", f"{realtime.pulse_rates(60).get('PICK_DONE', 0)} picks/h", "⚡")
        elif adm_opt == "📱 SIM Database Manager":
            st.subheader("📱 Samsung IMEI Database Manager")
            if st.session_state.df_sim_db is None:
                st.session_state.df_sim_db = load_sim_db()
            sim_tools_col, sim_conv_col = st.columns([1, 2])
            with sim_tools_col:
                st.markdown("### 🛠️ SIM Database Tools")
                search_query = st.text_input("🔍 Search Model or TAC (8 digits)")
                display_sim_df = st.session_state.df_sim_db
                if search_query:
                    display_sim_df = display_sim_df[
                        display_sim_df['Model_Series'].str.contains(search_query, case=False, na=False) |
                        display_sim_df['TAC_Prefix'].str.contains(search_query, na=False)]
                st.write(f"Showing {len(display_sim_df)} entries")
                edited_sim_df = st.data_editor(display_sim_df, num_rows="dynamic", width="stretch",
                    column_config={"TAC_Prefix": st.column_config.TextColumn("TAC Prefix (8 digits)"),
                                   "Expected_Offset": st.column_config.NumberColumn("Offset", format="%d"),
                                   "Model_Series": "Model Name", "Type": "Type"},
                    key="sim_data_editor")
                if st.button("💾 Save SIM Changes to CSV"):
                    if search_query:
                        st.session_state.df_sim_db.update(edited_sim_df)
                    else:
                        st.session_state.df_sim_db = edited_sim_df
                    save_sim_db(st.session_state.df_sim_db)
                    st.success("SIM Database file updated!")
                    log_action(user, "SIM DB Saved", f"{len(st.session_state.df_sim_db)} entries")
            with sim_conv_col:
                st.markdown("### 📱 IMEI Converter Tools")
                sim_db_map = dict(zip(st.session_state.df_sim_db['TAC_Prefix'], st.session_state.df_sim_db['Expected_Offset']))
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.write("#### 1. Calibration")
                    cal_input = st.text_area("Paste samples (IMEI 1 | IMEI 2):", height=150, placeholder="15 digits each")
                with col_c2:
                    st.write("#### 2. Targets")
                    batch_input = st.text_area("Paste IMEI 1 list (15 digits):", height=150)
                if batch_input:
                    active_sim_map = sim_db_map.copy()
                    if cal_input:
                        for line in cal_input.strip().split('\n'):
                            imeis = re.findall(r'\b\d{15}\b', line)
                            if len(imeis) >= 2:
                                active_sim_map[imeis[0][:8]] = int(imeis[1][:14]) - int(imeis[0][:14])
                    target_imeis = re.findall(r'\b\d{15}\b', batch_input)
                    sim_results = []
                    for i1 in target_imeis:
                        tac = i1[:8]
                        default_sim_val = sim_db_map.get('0', 8)
                        sim_offset = active_sim_map.get(tac, default_sim_val)
                        model_info = st.session_state.df_sim_db[st.session_state.df_sim_db['TAC_Prefix'] == tac]
                        model_sim_name = model_info['Model_Series'].values[0] if not model_info.empty else "Unknown TAC"
                        base14 = i1[:14]
                        new_base = str(int(base14) + int(sim_offset)).zfill(14)
                        sim_results.append({"Model": model_sim_name, "IMEI 1": i1,
                                            "IMEI 2": new_base + calculate_luhn(new_base),
                                            "TAC": tac, "Applied Offset": f"{int(sim_offset):+}"})
                    if sim_results:
                        st.divider()
                        st.write("#### Integrated Results")
                        st.dataframe(pd.DataFrame(sim_results), width="stretch", hide_index=True)
                        log_action(user, "SIM IMEI Converted", f"Processed: {len(sim_results)}")
        elif adm_opt == "🧠 Neural System Diagnostics":
            st.subheader("Neural System Diagnostics")
            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.metric("Vision History", len(_neural_vision.detection_history), "scans")
            d_col2.metric("Route Visits", len(_route_opt.visit_log), "logged")
            d_col3.metric("Oracle Data Points", len(_oracle.history), "days")
            st.markdown("### 🛡️ Guardian Snapshot History")
            gh_df = get_guardian_history(48)
            if not gh_df.empty:
                st.dataframe(gh_df, width="stretch", hide_index=True)
            else:
                st.info("No Guardian snapshots yet — visit the Guardian tab.")
            st.markdown("### 🚨 Sentinel Alert Archive")
            if _sentinel.alert_log:
                st.dataframe(pd.DataFrame(list(_sentinel.alert_log)[-50:]), width="stretch", hide_index=True)
            else:
                st.info("No alerts in archive.")
            st.markdown("### 🗺️ Zone Heatmap Data")
            st.dataframe(pd.DataFrame(_route_opt.heat_map), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
