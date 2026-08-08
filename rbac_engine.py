"""Role-Based Access Control & Session Management v4.3
Handles permission matrices, idle timeouts, concurrent session limits,
and Streamlit tab visibility gating.
"""
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from db import connect, get_memory, save_memory

DEFAULT_MATRIX = {
    "super_admin": ["all"],
    "manager": ["view_dashboard", "manage_orders", "approve_returns", "run_reports", "manage_rules"],
    "operator": ["view_dashboard", "scan_items", "pack_orders", "raise_andon", "submit_counts"],
    "viewer": ["view_dashboard", "view_reports"],
    "finance": ["view_dashboard", "view_reports", "manage_inventory_policy", "export_financial"],
}

class RBACEngine:
    def __init__(self, config: dict):
        self.config = config
        self.permissions = {**DEFAULT_MATRIX, **config.get("roles", {})}
        self.sessions: Dict[str, Dict] = {}

    def check_permission(self, username: str, role: str, permission: str) -> bool:
        if permission == "all" or role == "super_admin":
            return True
        allowed = self.permissions.get(role, [])
        return permission in allowed

    def get_visible_tabs(self, role: str) -> List[str]:
        """Returns list of allowed tab names for Streamlit rendering."""
        matrix = {
            "view_dashboard": ["Dashboard", "🛡️ Guardian", "🎛️ Control Tower"],
            "manage_orders": ["Orders", "📦 PDF Sequencer v4.2"],
            "scan_items": ["Inventory", "Auditor"],
            "pack_orders": [],  # handled in Control Tower
            "raise_andon": [],
            "submit_counts": [],
            "view_reports": ["📊 Reports"],
            "run_reports": ["📊 Reports", "🧠 Neural Ops"],
            "manage_inventory_policy": ["Inventory"],
            "approve_returns": ["Orders"],
            "manage_rules": ["Admin 🔐"],
            "export_financial": ["Admin 🔐"],
        }
        allowed = set(self.permissions.get(role, []))
        tabs = []
        for perm, tab_list in matrix.items():
            if perm in allowed:
                tabs.extend(tab_list)
        # Always allow Dashboard & Settings
        return list(dict.fromkeys(tabs + ["Dashboard", "Admin 🔐"] if role == "super_admin" else ["Dashboard"]))

    def create_session(self, username: str, role: str) -> str:
        session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"
        self.sessions[session_id] = {
            "user": username, "role": role,
            "created": time.time(), "last_active": time.time()
        }
        self._prune_sessions(username)
        save_memory(f"session:{session_id}", str(self.sessions[session_id]))
        return session_id

    def validate_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        if session_id not in self.sessions:
            return False, "Session expired or invalid"
        sess = self.sessions[session_id]
        idle_min = (time.time() - sess["last_active"]) / 60
        timeout_min = self.config.get("session_timeout_min", 45)
        if idle_min > timeout_min:
            del self.sessions[session_id]
            return False, "Session timed out due to inactivity"
        sess["last_active"] = time.time()
        return True, None

    def _prune_sessions(self, username: str):
        max_concurrent = self.config.get("max_concurrent_sessions", 3)
        user_sessions = [s for s in self.sessions.values() if s["user"] == username]
        if len(user_sessions) > max_concurrent:
            sorted_sess = sorted(user_sessions, key=lambda x: x["last_active"])
            for old in sorted_sess[:-max_concurrent]:
                key = f"SES-{old.get('_id', '')}"
                for k in list(self.sessions.keys()):
                    if self.sessions[k] is old:
                        del self.sessions[k]; break
