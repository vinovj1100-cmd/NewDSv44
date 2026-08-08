"""Database module for Neural Fulfillment Platform v4.4
SQLite with WAL + busy timeout. v4.4 adds workflow transition tables and audit log.
"""
import sqlite3
import os
import pandas as pd
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = os.environ.get(
    "WMS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "warehouse_neural.db"),
)

@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _now():
    return datetime.now().isoformat()

def init_db():
    """Create all tables (idempotent) and seed defaults."""
    with connect() as conn:
        c = conn.cursor()
        # ── core ────────────────────────────────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT UNIQUE NOT NULL,
            product TEXT, stock INTEGER DEFAULT 0,
            location TEXT DEFAULT 'UNASSIGNED', note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'Pending', items TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, raw_title TEXT UNIQUE NOT NULL,
            standard_title TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL, value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS action_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, action_type TEXT NOT NULL,
            ref_id TEXT, payload TEXT, user TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, role TEXT DEFAULT 'Operator',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS sync_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT, action_type TEXT NOT NULL,
            payload TEXT NOT NULL, status TEXT DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, processed_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pref_key TEXT UNIQUE NOT NULL,
            pref_value TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        # ── v4.0 fulfillment / efficiency ───────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS inventory_policy (
            sku TEXT PRIMARY KEY, reorder_point INTEGER DEFAULT 0,
            min_stock INTEGER DEFAULT 0, max_stock INTEGER DEFAULT 0,
            abc_class TEXT DEFAULT 'C', velocity REAL DEFAULT 0,
            last_count_at TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS waves (
            id INTEGER PRIMARY KEY AUTOINCREMENT, wave_id TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'Open', strategy TEXT DEFAULT 'batch', order_ids TEXT,
            assigned_to TEXT, priority INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS pick_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT UNIQUE NOT NULL,
            wave_id TEXT, sku TEXT NOT NULL, order_id TEXT, location TEXT,
            qty INTEGER DEFAULT 1, status TEXT DEFAULT 'Pending', assigned_to TEXT,
            picked_qty INTEGER DEFAULT 0, picked_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (wave_id) REFERENCES waves(wave_id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS slotting_abc (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT, abc_class TEXT,
            velocity_score REAL DEFAULT 0, current_location TEXT,
            recommended_location TEXT, rationale TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS cycle_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, count_id TEXT UNIQUE NOT NULL,
            sku TEXT, location TEXT, expected_qty INTEGER, counted_qty INTEGER,
            variance INTEGER, abc_class TEXT, status TEXT DEFAULT 'Scheduled',
            assigned_to TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS asns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, asn_id TEXT UNIQUE NOT NULL,
            supplier TEXT, expected_items TEXT, status TEXT DEFAULT 'Expected',
            dock_door TEXT, eta TIMESTAMP, received_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS dock_appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, appointment_id TEXT UNIQUE NOT NULL,
            door TEXT, carrier TEXT, direction TEXT DEFAULT 'Inbound',
            appointment_time TIMESTAMP, status TEXT DEFAULT 'Scheduled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS returns_rma (
            id INTEGER PRIMARY KEY AUTOINCREMENT, rma_id TEXT UNIQUE NOT NULL,
            order_id TEXT, sku TEXT, reason TEXT,
            condition TEXT DEFAULT 'Uninspected', disposition TEXT DEFAULT 'Pending',
            status TEXT DEFAULT 'Received', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS labor_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, operator TEXT, activity TEXT,
            ref_id TEXT, units INTEGER DEFAULT 0, duration_sec INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS kpi_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_skus INTEGER, total_stock INTEGER, pending_orders INTEGER,
            open_picks INTEGER, completed_picks INTEGER, backlog_age_min INTEGER,
            stockout_risk_skus INTEGER, pick_completion_pct REAL,
            cycle_accuracy_pct REAL, queue_items INTEGER, oee_pct REAL, health_score REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS replenishments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, repl_id TEXT UNIQUE NOT NULL, sku TEXT,
            current_stock INTEGER, reorder_point INTEGER, suggested_qty INTEGER,
            status TEXT DEFAULT 'Open', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP)""")

        # ── v4.2 realtime floor bridge ──────────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, actor TEXT,
            message TEXT, ref_id TEXT, payload TEXT, severity TEXT DEFAULT 'info',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS bins (
            location TEXT PRIMARY KEY, zone TEXT DEFAULT 'A',
            capacity_units INTEGER DEFAULT 120, current_units INTEGER DEFAULT 0,
            abc_affinity TEXT DEFAULT 'C', active INTEGER DEFAULT 1)""")
        c.execute("""CREATE TABLE IF NOT EXISTS pack_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, pack_id TEXT UNIQUE NOT NULL,
            order_id TEXT NOT NULL, station TEXT DEFAULT 'PACK-1',
            status TEXT DEFAULT 'Open', expected_items TEXT,
            scanned_items TEXT DEFAULT '{}', expected_weight_g INTEGER DEFAULT 0,
            actual_weight_g INTEGER DEFAULT 0, weight_tolerance_pct REAL DEFAULT 8.0,
            carton_type TEXT, packed_by TEXT, flag_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS putaway_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT UNIQUE NOT NULL,
            asn_id TEXT, sku TEXT NOT NULL, qty INTEGER DEFAULT 1,
            from_location TEXT DEFAULT 'DOCK', suggested_location TEXT,
            actual_location TEXT, status TEXT DEFAULT 'Pending', assigned_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS andon_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT UNIQUE NOT NULL,
            zone TEXT, kind TEXT DEFAULT 'other', message TEXT, raised_by TEXT,
            severity TEXT DEFAULT 'high', status TEXT DEFAULT 'open', resolved_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            acked_at TIMESTAMP, resolved_at TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS label_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE NOT NULL,
            marketplace TEXT DEFAULT 'WB', mode TEXT, targets_count INTEGER DEFAULT 0,
            matched_count INTEGER DEFAULT 0, missing_count INTEGER DEFAULT 0,
            extra_count INTEGER DEFAULT 0, avg_confidence REAL DEFAULT 0,
            operator TEXT, stats TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS guardian_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            health_score REAL, status TEXT, alerts_count INTEGER DEFAULT 0,
            critical_count INTEGER DEFAULT 0, pending_orders INTEGER DEFAULT 0,
            queued_items INTEGER DEFAULT 0, low_stock_count INTEGER DEFAULT 0,
            andon_open INTEGER DEFAULT 0, sla_breached INTEGER DEFAULT 0,
            pack_flagged INTEGER DEFAULT 0)""")

        # ── v4.4 enterprise / workflow ──────────────────────────────────
        c.execute("""CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash_prev TEXT NOT NULL,
            hash_current TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            actor TEXT, action TEXT, table_name TEXT, record_id TEXT,
            old_values TEXT, new_values TEXT,
            ip_address TEXT, session_id TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            role TEXT,
            created_at TEXT,
            last_activity TEXT,
            ip_address TEXT,
            active INTEGER DEFAULT 1)""")
        c.execute("""CREATE TABLE IF NOT EXISTS workflow_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            from_status TEXT,
            to_status TEXT NOT NULL,
            actor TEXT,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS order_workflow (
            order_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'Pending',
            priority INTEGER DEFAULT 5,
            allocated_at TEXT,
            picking_started_at TEXT,
            packed_at TEXT,
            shipped_at TEXT,
            carrier TEXT,
            tracking TEXT,
            sla_deadline TEXT,
            updated_at TEXT)""")

        # ── default users ───────────────────────────────────────────────
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            import hashlib
            for u, p, r in [
                ("admin", "admin123", "super_admin"),
                ("manager", "manager123", "manager"),
                ("operator", "operator123", "operator"),
                ("viewer", "viewer123", "viewer"),
            ]:
                c.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                          (u, hashlib.sha256(p.encode()).hexdigest(), r))

        # ── seed a starter rack layout (A/B near dock = golden zone) ────
        if c.execute("SELECT COUNT(*) FROM bins").fetchone()[0] == 0:
            for z in "ABCD":
                for n in range(1, 9):
                    c.execute("INSERT INTO bins (location, zone, abc_affinity) VALUES (?,?,?)",
                              (f"{z}-{n:02d}", z, "A" if z in "AB" else "C"))

        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_pick_tasks_status ON pick_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_pick_tasks_wave ON pick_tasks(wave_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(sku)",
            "CREATE INDEX IF NOT EXISTS idx_cycle_status ON cycle_counts(status)",
            "CREATE INDEX IF NOT EXISTS idx_repl_status ON replenishments(status)",
            "CREATE INDEX IF NOT EXISTS idx_kpi_time ON kpi_snapshots(captured_at)",
            "CREATE INDEX IF NOT EXISTS idx_evt_time ON event_log(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_pack_status ON pack_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_put_status ON putaway_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_andon_status ON andon_alerts(status)",
            "CREATE INDEX IF NOT EXISTS idx_label_jobs_time ON label_jobs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_guardian_time ON guardian_snapshots(captured_at)",
        ]:
            c.execute(idx)


# ═══════════════════════ CORE HELPERS (signatures preserved) ═════════════

def get_inventory():
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT sku, product, stock, location, note FROM inventory ORDER BY updated_at DESC", conn)


def upsert_inventory(sku, product, stock, location, note=""):
    with connect() as conn:
        conn.execute("""INSERT INTO inventory (sku, product, stock, location, note, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(sku) DO UPDATE SET product=excluded.product, stock=excluded.stock,
            location=excluded.location, note=excluded.note, updated_at=excluded.updated_at""",
            (sku, product, stock, location, note, _now()))


def get_orders():
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT order_id, status, items, created_at, updated_at FROM orders ORDER BY created_at DESC", conn)


def create_order(order_id, status, items_list):
    with connect() as conn:
        items_str = ", ".join(items_list) if isinstance(items_list, list) else str(items_list)
        conn.execute("""INSERT INTO orders (order_id, status, items, updated_at) VALUES (?,?,?,?)
            ON CONFLICT(order_id) DO UPDATE SET status=excluded.status, items=excluded.items,
            updated_at=excluded.updated_at""", (order_id, status, items_str, _now()))


def update_order_status(order_id, new_status):
    with connect() as conn:
        conn.execute("UPDATE orders SET status=?, updated_at=? WHERE order_id=?",
                     (new_status, _now(), order_id))


def get_templates():
    with connect() as conn:
        return pd.read_sql_query("SELECT raw_title, standard_title, created_at FROM templates ORDER BY created_at DESC", conn)


def save_template(raw, standard):
    with connect() as conn:
        conn.execute("""INSERT INTO templates (raw_title, standard_title) VALUES (?,?)
            ON CONFLICT(raw_title) DO UPDATE SET standard_title=excluded.standard_title""", (raw, standard))


def save_memory(key, value):
    with connect() as conn:
        conn.execute("""INSERT INTO memory (key, value, updated_at) VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, _now()))


def get_memory(key):
    with connect() as conn:
        row = conn.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


def get_recent_preferences(limit=50):
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT key, value, created_at FROM memory ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,))


def add_action_log(action_type, ref_id=None, payload=None, user=None):
    with connect() as conn:
        conn.execute("INSERT INTO action_logs (action_type, ref_id, payload, user, created_at) VALUES (?,?,?,?,?)",
                     (action_type, ref_id, payload, user, _now()))


def record_preference(key, value):
    with connect() as conn:
        conn.execute("""INSERT INTO preferences (pref_key, pref_value) VALUES (?,?)
            ON CONFLICT(pref_key) DO UPDATE SET pref_value=excluded.pref_value""", (key, value))


def auth_login(username, password):
    import hashlib
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    with connect() as conn:
        row = conn.execute("SELECT username, role FROM users WHERE username=? AND password_hash=?",
                           (username, pwd_hash)).fetchone()
    return {"username": row["username"], "role": row["role"]} if row else None


def add_user(username, password, role="Operator"):
    import hashlib
    with connect() as conn:
        conn.execute("""INSERT INTO users (username, password_hash, role) VALUES (?,?,?)
            ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash, role=excluded.role""",
            (username, hashlib.sha256(password.encode()).hexdigest(), role))


def load_sim_db():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_database.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = pd.DataFrame({
        "TAC_Prefix": ["35089080", "35155810", "35460108", "35693803", "35824005"],
        "Expected_Offset": [8, 8, 8, 8, 8],
        "Model_Series": ["Galaxy S21", "Galaxy S22", "Galaxy S23", "Galaxy S24", "Galaxy Z Flip"],
        "Type": ["Smartphone", "Smartphone", "Smartphone", "Smartphone", "Foldable"],
    })
    df.to_csv(csv_path, index=False)
    return df


def save_sim_db(df):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_database.csv")
    df.to_csv(csv_path, index=False)


def enqueue_action(action_type, payload):
    with connect() as conn:
        conn.execute("INSERT INTO sync_queue (action_type, payload, status, created_at) VALUES (?,?,'pending',?)",
                     (action_type, str(payload), _now()))


def process_queue():
    with connect() as conn:
        pending = conn.execute(
            "SELECT id FROM sync_queue WHERE status='pending' ORDER BY created_at ASC").fetchall()
        synced = failed = 0
        for row in pending:
            try:
                conn.execute("UPDATE sync_queue SET status='synced', processed_at=? WHERE id=?", (_now(), row["id"]))
                synced += 1
            except Exception:
                conn.execute("UPDATE sync_queue SET status='failed', retry_count=retry_count+1 WHERE id=?", (row["id"],))
                failed += 1
    return synced, failed


def queue_status():
    with connect() as conn:
        queued = conn.execute("SELECT COUNT(*) q FROM sync_queue WHERE status='pending'").fetchone()["q"]
        row = conn.execute("SELECT MAX(processed_at) ls FROM sync_queue WHERE status='synced'").fetchone()
    return {"queued": queued, "last_sync": row["ls"] if row and row["ls"] else None}


def can_sync_now():
    online = get_memory("online_access")
    return online.lower() == "true" if online else True


# ═══════════════════════ v4.0 FULFILLMENT HELPERS ════════════════════════

def upsert_inventory_policy(sku, reorder_point=0, min_stock=0, max_stock=0, abc_class="C", velocity=0.0):
    with connect() as conn:
        conn.execute("""INSERT INTO inventory_policy (sku, reorder_point, min_stock, max_stock, abc_class, velocity, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(sku) DO UPDATE SET reorder_point=excluded.reorder_point, min_stock=excluded.min_stock,
            max_stock=excluded.max_stock, abc_class=excluded.abc_class, velocity=excluded.velocity,
            updated_at=excluded.updated_at""",
            (sku, reorder_point, min_stock, max_stock, abc_class, velocity, _now()))


def get_inventory_policy():
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM inventory_policy ORDER BY velocity DESC", conn)


def get_inventory_full():
    with connect() as conn:
        return pd.read_sql_query("""
            SELECT i.sku, i.product, i.stock, i.location, i.note,
                   COALESCE(p.reorder_point,0) reorder_point, COALESCE(p.min_stock,0) min_stock,
                   COALESCE(p.max_stock,0) max_stock, COALESCE(p.abc_class,'C') abc_class,
                   COALESCE(p.velocity,0) velocity
            FROM inventory i LEFT JOIN inventory_policy p ON i.sku=p.sku
            ORDER BY p.velocity DESC, i.sku""", conn)


def create_wave(wave_id, strategy, order_ids, assigned_to=None, priority=5):
    with connect() as conn:
        conn.execute("""INSERT INTO waves (wave_id, status, strategy, order_ids, assigned_to, priority, created_at)
            VALUES (?,'Open',?,?,?,?,?)
            ON CONFLICT(wave_id) DO UPDATE SET strategy=excluded.strategy, order_ids=excluded.order_ids,
            assigned_to=excluded.assigned_to, priority=excluded.priority""",
            (wave_id, strategy, ",".join(order_ids) if isinstance(order_ids, list) else order_ids,
             assigned_to, priority, _now()))


def get_waves():
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM waves ORDER BY created_at DESC", conn)


def add_pick_task(task_id, wave_id, sku, order_id, location, qty, assigned_to=None):
    with connect() as conn:
        conn.execute("""INSERT INTO pick_tasks (task_id, wave_id, sku, order_id, location, qty, status, assigned_to, created_at)
            VALUES (?,?,?,?,?,?,'Pending',?,?)
            ON CONFLICT(task_id) DO UPDATE SET qty=excluded.qty, assigned_to=excluded.assigned_to""",
            (task_id, wave_id, sku, order_id, location, int(qty), assigned_to, _now()))


def get_pick_tasks(status=None, assigned_to=None):
    with connect() as conn:
        q, clauses, params = "SELECT * FROM pick_tasks", [], []
        if status:
            clauses.append("status=?"); params.append(status)
        if assigned_to:
            clauses.append("assigned_to=?"); params.append(assigned_to)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        return pd.read_sql_query(q + " ORDER BY created_at ASC", conn, params=params)


def complete_pick_task(task_id, picked_qty, operator):
    with connect() as conn:
        conn.execute("UPDATE pick_tasks SET status='Completed', picked_qty=?, picked_at=? WHERE task_id=?",
                     (int(picked_qty), _now(), task_id))
        conn.execute("INSERT INTO labor_logs (operator, activity, ref_id, units, duration_sec, created_at) VALUES (?,'pick',?,?,0,?)",
                     (operator, task_id, int(picked_qty), _now()))


def close_wave(wave_id):
    with connect() as conn:
        conn.execute("UPDATE waves SET status='Closed', completed_at=? WHERE wave_id=?", (_now(), wave_id))


def save_slotting_analysis(rows):
    with connect() as conn:
        for r in rows:
            conn.execute("""INSERT INTO slotting_abc (sku, abc_class, velocity_score, current_location, recommended_location, rationale, analyzed_at)
                VALUES (?,?,?,?,?,?,?)""",
                (r.get("sku"), r.get("abc_class", "C"), r.get("velocity_score", 0),
                 r.get("current_location"), r.get("recommended_location"), r.get("rationale", ""), _now()))


def get_slotting():
    with connect() as conn:
        return pd.read_sql_query("""
            SELECT sku, abc_class, velocity_score, current_location, recommended_location, rationale, analyzed_at
            FROM slotting_abc WHERE id IN (SELECT MAX(id) FROM slotting_abc GROUP BY sku)
            ORDER BY velocity_score DESC""", conn)


def create_cycle_count(count_id, sku, location, expected_qty, abc_class="C", assigned_to=None):
    with connect() as conn:
        conn.execute("""INSERT INTO cycle_counts (count_id, sku, location, expected_qty, counted_qty, variance, abc_class, status, assigned_to, created_at)
            VALUES (?,?,?,?,0,0,?,'Scheduled',?,?)
            ON CONFLICT(count_id) DO UPDATE SET expected_qty=excluded.expected_qty""",
            (count_id, sku, location, int(expected_qty), abc_class, assigned_to, _now()))


def get_cycle_counts(status=None):
    with connect() as conn:
        if status:
            return pd.read_sql_query("SELECT * FROM cycle_counts WHERE status=? ORDER BY created_at DESC", conn, params=(status,))
        return pd.read_sql_query("SELECT * FROM cycle_counts ORDER BY created_at DESC", conn)


def complete_cycle_count(count_id, counted_qty, operator):
    with connect() as conn:
        row = conn.execute("SELECT expected_qty FROM cycle_counts WHERE count_id=?", (count_id,)).fetchone()
        variance = int(counted_qty) - int(row["expected_qty"] if row else 0)
        conn.execute("""UPDATE cycle_counts SET counted_qty=?, variance=?, status='Completed', assigned_to=?, completed_at=?
            WHERE count_id=?""", (int(counted_qty), variance, operator, _now(), count_id))
        conn.execute("INSERT INTO labor_logs (operator, activity, ref_id, units, duration_sec, created_at) VALUES (?,'cycle_count',?,?,0,?)",
                     (operator, count_id, abs(variance), _now()))


def create_asn(asn_id, supplier, expected_items, dock_door=None, eta=None):
    with connect() as conn:
        conn.execute("""INSERT INTO asns (asn_id, supplier, expected_items, status, dock_door, eta, created_at)
            VALUES (?,?,?,'Expected',?,?,?)
            ON CONFLICT(asn_id) DO UPDATE SET supplier=excluded.supplier, expected_items=excluded.expected_items,
            dock_door=excluded.dock_door, eta=excluded.eta""",
            (asn_id, supplier, ",".join(expected_items) if isinstance(expected_items, list) else expected_items,
             dock_door, eta, _now()))


def receive_asn(asn_id, received_items=None):
    with connect() as conn:
        conn.execute("UPDATE asns SET status='Received', received_at=? WHERE asn_id=?", (_now(), asn_id))


def get_asns(status=None):
    with connect() as conn:
        if status:
            return pd.read_sql_query("SELECT * FROM asns WHERE status=? ORDER BY created_at DESC", conn, params=(status,))
        return pd.read_sql_query("SELECT * FROM asns ORDER BY created_at DESC", conn)


def create_dock_appointment(appointment_id, door, carrier, direction, appointment_time):
    with connect() as conn:
        conn.execute("""INSERT INTO dock_appointments (appointment_id, door, carrier, direction, appointment_time, status, created_at)
            VALUES (?,?,?,?,?,'Scheduled',?)
            ON CONFLICT(appointment_id) DO UPDATE SET door=excluded.door, carrier=excluded.carrier,
            direction=excluded.direction, appointment_time=excluded.appointment_time""",
            (appointment_id, door, carrier, direction, appointment_time, _now()))


def get_dock_appointments():
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM dock_appointments ORDER BY appointment_time ASC", conn)


def create_rma(rma_id, order_id, sku, reason, condition="Uninspected"):
    with connect() as conn:
        conn.execute("""INSERT INTO returns_rma (rma_id, order_id, sku, reason, condition, disposition, status, created_at)
            VALUES (?,?,?,?,?,'Pending','Received',?)
            ON CONFLICT(rma_id) DO UPDATE SET reason=excluded.reason""",
            (rma_id, order_id, sku, reason, condition, _now()))


def update_rma_disposition(rma_id, disposition, status="Processed"):
    with connect() as conn:
        conn.execute("UPDATE returns_rma SET disposition=?, status=?, processed_at=? WHERE rma_id=?",
                     (disposition, status, _now(), rma_id))


def get_rmas(status=None):
    with connect() as conn:
        if status:
            return pd.read_sql_query("SELECT * FROM returns_rma WHERE status=? ORDER BY created_at DESC", conn, params=(status,))
        return pd.read_sql_query("SELECT * FROM returns_rma ORDER BY created_at DESC", conn)


def create_replenishment(repl_id, sku, current_stock, reorder_point, suggested_qty):
    with connect() as conn:
        conn.execute("""INSERT INTO replenishments (repl_id, sku, current_stock, reorder_point, suggested_qty, status, created_at)
            VALUES (?,?,?,?,?,'Open',?)
            ON CONFLICT(repl_id) DO UPDATE SET current_stock=excluded.current_stock, suggested_qty=excluded.suggested_qty""",
            (repl_id, sku, int(current_stock), int(reorder_point), int(suggested_qty), _now()))


def complete_replenishment(repl_id, operator):
    with connect() as conn:
        conn.execute("UPDATE replenishments SET status='Completed', completed_at=? WHERE repl_id=?", (_now(), repl_id))
        conn.execute("INSERT INTO labor_logs (operator, activity, ref_id, units, duration_sec, created_at) VALUES (?,'replenish',?,0,0,?)",
                     (operator, repl_id, _now()))


def get_replenishments(status=None):
    with connect() as conn:
        if status:
            return pd.read_sql_query("SELECT * FROM replenishments WHERE status=? ORDER BY created_at DESC", conn, params=(status,))
        return pd.read_sql_query("SELECT * FROM replenishments ORDER BY created_at DESC", conn)


def save_kpi_snapshot(metrics: dict):
    with connect() as conn:
        conn.execute("""INSERT INTO kpi_snapshots
            (captured_at, total_skus, total_stock, pending_orders, open_picks, completed_picks,
             backlog_age_min, stockout_risk_skus, pick_completion_pct, cycle_accuracy_pct,
             queue_items, oee_pct, health_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (_now(), metrics.get("total_skus", 0), metrics.get("total_stock", 0),
             metrics.get("pending_orders", 0), metrics.get("open_picks", 0),
             metrics.get("completed_picks", 0), metrics.get("backlog_age_min", 0),
             metrics.get("stockout_risk_skus", 0), metrics.get("pick_completion_pct", 0.0),
             metrics.get("cycle_accuracy_pct", 0.0), metrics.get("queue_items", 0),
             metrics.get("oee_pct", 0.0), metrics.get("health_score", 0.0)))


def get_kpi_history(limit=48):
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM kpi_snapshots ORDER BY captured_at DESC LIMIT ?", conn, params=(limit,))


def get_labor_logs(limit=200):
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT operator, activity, ref_id, units, duration_sec, created_at FROM labor_logs ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,))


def get_labor_summary():
    with connect() as conn:
        return pd.read_sql_query("""
            SELECT operator, activity, COUNT(*) events, SUM(units) units, SUM(duration_sec) seconds
            FROM labor_logs GROUP BY operator, activity ORDER BY events DESC""", conn)


# ═══════════════════════ v4.2 HELPERS ════════════════════════════════════

def save_label_job(job_id, marketplace, mode, targets_count, matched_count,
                   missing_count, extra_count, avg_confidence, operator, stats=None):
    import json
    with connect() as conn:
        conn.execute("""INSERT INTO label_jobs
            (job_id, marketplace, mode, targets_count, matched_count, missing_count,
             extra_count, avg_confidence, operator, stats, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET matched_count=excluded.matched_count""",
            (job_id, marketplace, mode, int(targets_count), int(matched_count),
             int(missing_count), int(extra_count), float(avg_confidence or 0),
             operator, json.dumps(stats or {}), _now()))


def get_label_jobs(limit=25):
    with connect() as conn:
        return pd.read_sql_query("""SELECT job_id, marketplace, mode, targets_count, matched_count,
                missing_count, extra_count, avg_confidence, operator, created_at
            FROM label_jobs ORDER BY created_at DESC LIMIT ?""", conn, params=(limit,))


def save_guardian_snapshot(metrics: dict):
    with connect() as conn:
        conn.execute("""INSERT INTO guardian_snapshots
            (captured_at, health_score, status, alerts_count, critical_count, pending_orders,
             queued_items, low_stock_count, andon_open, sla_breached, pack_flagged)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (_now(), metrics.get("health_score", 1.0), metrics.get("status", "HEALTHY"),
             metrics.get("alerts_count", 0), metrics.get("critical_count", 0),
             metrics.get("pending_orders", 0), metrics.get("queued_items", 0),
             metrics.get("low_stock_count", 0), metrics.get("andon_open", 0),
             metrics.get("sla_breached", 0), metrics.get("pack_flagged", 0)))


def get_guardian_history(limit=96):
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM guardian_snapshots ORDER BY captured_at DESC LIMIT ?", conn, params=(limit,))

# ═══════════════════════ v4.4 WORKFLOW & AUDIT HELPERS ═══════════════════

def transition_order(order_id: str, to_status: str, actor: str = None, reason: str = None):
    """Record workflow transition and update order + order_workflow tables."""
    with connect() as conn:
        row = conn.execute("SELECT status FROM orders WHERE order_id=?", (order_id,)).fetchone()
        from_status = row["status"] if row else None
        conn.execute(
            "UPDATE orders SET status=?, updated_at=? WHERE order_id=?",
            (to_status, _now(), order_id),
        )
        conn.execute(
            """INSERT INTO workflow_transitions (order_id, from_status, to_status, actor, reason, created_at)
               VALUES (?,?,?,?,?,?)""",
            (order_id, from_status, to_status, actor, reason, _now()),
        )
        # upsert order_workflow
        ts_field = {
            "Allocated": "allocated_at",
            "Picking": "picking_started_at",
            "Packed": "packed_at",
            "Shipped": "shipped_at",
        }.get(to_status)
        if ts_field:
            conn.execute(
                f"""INSERT INTO order_workflow (order_id, status, {ts_field}, updated_at)
                    VALUES (?,?,?,?)
                    ON CONFLICT(order_id) DO UPDATE SET
                        status=excluded.status, {ts_field}=excluded.{ts_field}, updated_at=excluded.updated_at""",
                (order_id, to_status, _now(), _now()),
            )
        else:
            conn.execute(
                """INSERT INTO order_workflow (order_id, status, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(order_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at""",
                (order_id, to_status, _now()),
            )
    return {"ok": True, "from": from_status, "to": to_status}


def get_workflow_history(order_id: str = None, limit: int = 100):
    with connect() as conn:
        if order_id:
            return pd.read_sql_query(
                "SELECT * FROM workflow_transitions WHERE order_id=? ORDER BY created_at DESC LIMIT ?",
                conn, params=(order_id, limit),
            )
        return pd.read_sql_query(
            "SELECT * FROM workflow_transitions ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,),
        )


def get_order_workflow(order_id: str = None):
    with connect() as conn:
        if order_id:
            row = conn.execute("SELECT * FROM order_workflow WHERE order_id=?", (order_id,)).fetchone()
            return dict(row) if row else None
        return pd.read_sql_query("SELECT * FROM order_workflow ORDER BY updated_at DESC", conn)
