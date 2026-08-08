"""Guardian Ops Center renderer for Streamlit — v4.2.
Adds a health-trend SVG sparkline (zero deps) and a live floor-pulse strip.
"""
import pandas as pd
import streamlit as st


def _sparkline(values, width=320, height=52, color="#64ffda", fill="rgba(100,255,218,0.12)"):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    pts = []
    for i, v in enumerate(vals):
        x = round(i * (width - 8) / (len(vals) - 1) + 4, 1)
        y = round(height - 6 - (v - lo) / span * (height - 14), 1)
        pts.append(f"{x},{y}")
    last_x, last_y = pts[-1].split(",")
    return (f'<svg width="{width}" height="{height}" style="overflow:visible">'
            f'<polygon points="4,{height-4} {" ".join(pts)} {width-4},{height-4}" fill="{fill}"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{last_x}" cy="{last_y}" r="3" fill="{color}"/></svg>')


def render_guardian_dashboard(g, history_df: pd.DataFrame = None, pulse: dict = None):
    """g: a Guardian instance after analyze()."""
    health = g.health
    health_color = "#00ff88" if health > 0.8 else "#ffd93d" if health > 0.5 else "#ff6b6b"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:rgba(10,20,40,0.6);
                    border-radius:16px; border:1px solid {health_color}40;">
          <div style="font-size:0.9rem; color:#8892b0; margin-bottom:8px;">SYSTEM HEALTH</div>
          <div style="font-size:3rem; font-weight:bold; color:{health_color};
                      text-shadow:0 0 20px {health_color}40;">{int(health * 100)}%</div>
          <div style="font-size:0.85rem; color:{health_color}; margin-top:4px;">{g.ozone.get('status', 'UNKNOWN')}</div>
        </div>""", unsafe_allow_html=True)

    # health trend sparkline
    if history_df is not None and not history_df.empty:
        trend = history_df.sort_values("captured_at")["health_score"].tolist()
        spark = _sparkline(trend, color=health_color)
        if spark:
            st.markdown(f"""
            <div style="text-align:center; margin-top:10px;">
              <div style="font-size:0.75rem; color:#8892b0;">HEALTH TREND — last {len(trend)} snapshots</div>
              {spark}
            </div>""", unsafe_allow_html=True)

    # live floor pulse
    if pulse:
        chips = "".join(
            f"<span style='background:rgba(100,255,218,0.08); border:1px solid rgba(100,255,218,0.25);"
            f" border-radius:16px; padding:4px 12px; font-size:0.78rem; color:#ccd6f6;'>"
            f"{icon} {pulse.get(kind, 0)}</span>"
            for kind, icon in [("PICK_DONE", "⚡ picks"), ("PACK_DONE", "📦 packs"),
                               ("PUTAWAY_DONE", "🏗️ putaways"), ("ANDON_RAISED", "🚨 andon"),
                               ("COUNT_DONE", "🔍 counts")])
        st.markdown(f"<div style='display:flex; gap:8px; justify-content:center; margin:12px 0; flex-wrap:wrap;'>"
                    f"<span style='font-size:0.75rem; color:#8892b0; align-self:center;'>FLOOR PULSE (60 min)</span>{chips}</div>",
                    unsafe_allow_html=True)

    st.divider()

    if g.tuner:
        t = g.tuner
        r1 = st.columns(4)
        r1[0].metric("Total SKUs", t.get("total_skus", 0))
        r1[1].metric("Total Stock", t.get("total_stock", 0))
        r1[2].metric("Pending Orders", t.get("pending_orders", 0))
        r1[3].metric("Queue Items", t.get("queued_items", 0))
        r2 = st.columns(4)
        r2[0].metric("🧯 Andon Open", t.get("andon_open", 0))
        r2[1].metric("🕐 SLA Breached", t.get("sla_breached", 0))
        r2[2].metric("📦 Pack Flagged", t.get("pack_flagged", 0))
        r2[3].metric("📉 Low-Stock SKUs", t.get("low_stock_count", 0))

    st.subheader("🚨 Active Alerts")
    if g.alerts:
        for a in g.alerts:
            color = {"CRITICAL": "#ff6b6b", "WARNING": "#ffd93d", "INFO": "#64ffda"}.get(a["level"], "#64ffda")
            st.markdown(f"""
            <div style="border-left:3px solid {color}; padding:10px 15px; margin:8px 0;
                        background:rgba(255,255,255,0.03); border-radius:0 8px 8px 0;">
              <span style="color:{color}; font-weight:bold; font-size:0.85rem;">{a['level']}</span>
              <span style="color:#ccd6f6; margin-left:10px;">{a['message']}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No active alerts — all systems nominal")

    if g.suggestions:
        st.subheader("💡 Suggestions")
        for s in g.suggestions:
            st.markdown(f"<div style='color:#8892b0; font-size:0.9rem; margin:4px 0;'>• {s}</div>",
                        unsafe_allow_html=True)
    if g.recovery:
        st.subheader("🔧 Recovery Actions")
        for a in g.recovery:
            st.markdown(f"<div style='color:#ff6b6b; font-size:0.9rem; margin:4px 0;'>⚠️ {a}</div>",
                        unsafe_allow_html=True)

    st.divider()
    st.subheader("🌐 Ozone Status")
    o = st.columns(3)
    o[0].metric("Uptime", g.ozone.get("uptime", "N/A"))
    o[1].metric("Last Check", g.ozone.get("last_check", "N/A"))
    o[2].metric("Next Check", g.ozone.get("next_check", "N/A"))
