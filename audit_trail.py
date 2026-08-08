"""Immutable Audit Trail & Data Versioning v4.3
Tracks every CRUD operation with old/new values, actor, IP, session.
Uses a hash chain for tamper detection. GDPR-ready export.
"""
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from db import connect

class AuditTrail:
    def __init__(self):
        self._ensure_table()

    def _ensure_table(self):
        with connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_prev TEXT, hash_current TEXT,
                timestamp TEXT, actor TEXT, action TEXT,
                table_name TEXT, record_id TEXT,
                old_values TEXT, new_values TEXT,
                ip_address TEXT, session_id TEXT)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor)")

    def log(self, actor: str, action: str, table: str, record_id: str,
            old_vals: Optional[Dict] = None, new_vals: Optional[Dict] = None,
            ip: str = "127.0.0.1", session: str = "SYS") -> int:
        with connect() as conn:
            prev = conn.execute("SELECT hash_current FROM audit_logs ORDER BY id DESC LIMIT 1").fetchone()
            prev_hash = prev["hash_current"] if prev else "GENESIS"
            payload = {
                "actor": actor, "action": action, "table": table,
                "record_id": record_id, "old": old_vals, "new": new_vals,
                "ip": ip, "session": session, "ts": datetime.now().isoformat()
            }
            body = json.dumps(payload, default=str)
            curr_hash = hashlib.sha256(f"{prev_hash}|{body}".encode()).hexdigest()
            conn.execute("""INSERT INTO audit_logs (hash_prev, hash_current, timestamp, actor, action,
                table_name, record_id, old_values, new_values, ip_address, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (prev_hash, curr_hash, payload["ts"], actor, action, table, record_id,
                 json.dumps(old_vals or {}), json.dumps(new_vals or {}), ip, session))
            return payload["ts"]

    def verify_chain(self, since_id: int = 1) -> bool:
        with connect() as conn:
            rows = conn.execute("SELECT * FROM audit_logs WHERE id >= ? ORDER BY id ASC", (since_id,)).fetchall()
        prev = "GENESIS"
        for r in rows:
            # FIX: parse stored JSON strings back to objects first
            old_vals = json.loads(r["old_values"] or "{}")
            new_vals = json.loads(r["new_values"] or "{}")
            payload = {
                "actor": r["actor"], "action": r["action"], "table": r["table_name"],
                "record_id": r["record_id"], "old": old_vals, "new": new_vals,
                "ip": r["ip_address"], "session": r["session_id"], "ts": r["timestamp"]
            }
            expected = hashlib.sha256(f"{prev}|{json.dumps(payload, default=str)}".encode()).hexdigest()
            if r["hash_current"] != expected:
                return False
            prev = r["hash_current"]
        return True

    def export_compliance(self, start: str, end: str, actor: Optional[str] = None) -> List[Dict]:
        with connect() as conn:
            q = "SELECT * FROM audit_logs WHERE timestamp BETWEEN ? AND ?"
            params = [start, end]
            if actor:
                q += " AND actor = ?"; params.append(actor)
            q += " ORDER BY timestamp ASC"
            return [dict(r) for r in conn.execute(q, params).fetchall()]
