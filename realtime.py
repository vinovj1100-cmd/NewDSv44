"""Realtime Event Bus — v4.2 floor<->tower nervous system.
Every floor action publishes into event_log; the Flask PWA streams via SSE,
the Streamlit tower polls via @st.fragment. Doubles as the audit trail.
"""
import json
import threading
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Dict, Any

from db import connect

KINDS = {
    "PICK_DONE":       ("pick",     "#00ff88"),
    "PACK_OPENED":     ("pack",     "#8892b0"),
    "PACK_DONE":       ("pack",     "#64ffda"),
    "PACK_FLAG":       ("pack",     "#ffd93d"),
    "PUTAWAY_DONE":    ("putaway",  "#00b4db"),
    "ANDON_RAISED":    ("andon",    "#ff6b6b"),
    "ANDON_RESOLVED":  ("andon",    "#00ff88"),
    "GUARDIAN_ALERT":  ("guardian", "#ff6b6b"),
    "WAVE_CLOSED":     ("wave",     "#64ffda"),
    "ASN_RECEIVED":    ("inbound",  "#00b4db"),
    "COUNT_DONE":      ("count",    "#ffd93d"),
    "REPLEN_DONE":     ("replen",   "#64ffda"),
    "LABEL_JOB":       ("labels",   "#00b4db"),
}

_cv = threading.Condition()


def publish(kind, actor, message, ref_id=None, payload=None, severity="info"):
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO event_log (kind, actor, message, ref_id, payload, severity, created_at) VALUES (?,?,?,?,?,?,?)",
            (kind, actor, message, ref_id, json.dumps(payload or {}), severity, datetime.now().isoformat()))
        event_id = cur.lastrowid
    with _cv:
        _cv.notify_all()
    return event_id


def since(last_id=0, limit=100, kinds=None):
    with connect() as conn:
        q, params = "SELECT * FROM event_log WHERE id > ?", [last_id]
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += list(kinds)
        params.append(limit)
        rows = conn.execute(q + " ORDER BY id ASC LIMIT ?", params).fetchall()
    return [dict(r) for r in rows]


def recent(seconds=600, limit=6):
    cutoff = (datetime.now() - timedelta(seconds=seconds)).isoformat()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM event_log WHERE created_at >= ? ORDER BY id DESC LIMIT ?",
                            (cutoff, limit)).fetchall()
    return [dict(r) for r in rows]


def pulse_rates(minutes=60):
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    with connect() as conn:
        rows = conn.execute("SELECT kind, COUNT(*) n FROM event_log WHERE created_at >= ? GROUP BY kind",
                            (cutoff,)).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def seconds_since_last_event():
    """None if the bus has never fired — Guardian uses this for 'silent floor' detection."""
    with connect() as conn:
        row = conn.execute("SELECT MAX(created_at) m FROM event_log").fetchone()
    if not row or not row["m"]:
        return None
    return (datetime.now() - datetime.fromisoformat(row["m"])).total_seconds()


def shift_digest(hours=8):
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with connect() as conn:
        by_kind = conn.execute("SELECT kind, COUNT(*) n FROM event_log WHERE created_at >= ? GROUP BY kind",
                               (cutoff,)).fetchall()
        top = conn.execute("""SELECT actor, COUNT(*) n FROM event_log
            WHERE created_at >= ? AND actor IS NOT NULL GROUP BY actor ORDER BY n DESC LIMIT 5""",
            (cutoff,)).fetchall()
    return {"by_kind": {r["kind"]: r["n"] for r in by_kind},
            "top_actors": [(r["actor"], r["n"]) for r in top]}


def sse_stream(last_id=0):
    """SSE generator for the Flask PWA. Heartbeat every 15 s keeps mobiles alive."""
    cursor = last_id
    yield "retry: 3000\n\n"
    while True:
        for e in since(cursor, limit=50):
            cursor = e["id"]
            yield f"id: {e['id']}\ndata: {json.dumps(e, default=str)}\n\n"
        yield ": heartbeat\n\n"
        with _cv:
            _cv.wait(timeout=15)
