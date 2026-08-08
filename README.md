# Neural Fulfillment Platform v4.4

Quantum-Enhanced Unified Console for warehouse operations.

## Quick Start

```bash
pip install -r requirements_v43.txt
python seed_data.py
streamlit run app_v44.py
```

## Structure

- `app_v44.py` — Main Streamlit console
- `db.py` — SQLite data layer
- `workflow_engine.py` — Order lifecycle state machine
- `rule_engine.py` — YAML-driven automation
- `rbac_engine.py` — Role-based access control
- `audit_trail.py` — Immutable audit log with hash chain
- `advanced_forecasting.py` — ETS / SARIMA demand forecasting
- `report_generator.py` — PDF/Excel/CSV reports
- `quantum_ai_engine.py` — Quantum ensemble route optimizer
- `wb_label_processor.py` — WB/Ozon label detection & sequencing
- `floor_ops.py` — Pack stations, putaway, andon, SLA aging
- `efficiency.py` — Wave planner, ABC slotting, replenishment, cycle counts, KPI
- `copilot.py` — Natural language analytics
- `realtime.py` — Event bus with SSE
- `guardian.py` / `dashboard.py` — Health monitoring & rendering
- `mobile/` — Flask PWA for floor operators
- `structured_data/` — JSON schemas and master data
- `tests/` — Unit tests

## Docker

```bash
docker-compose up --build
```

## Default Logins

| User     | Password     | Role        |
|----------|--------------|-------------|
| admin    | admin123     | super_admin |
| manager  | manager123   | manager     |
| operator | operator123  | operator    |
| viewer   | viewer123    | viewer      |
