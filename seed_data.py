"""Rich structured seed data for Neural Fulfillment / WMS LITE v4.4.
Creates realistic multi-zone inventory, policies, orders across workflow
states, ASNs, and users with RBAC roles.
"""
import random
from datetime import datetime, timedelta

from db import (
    init_db,
    upsert_inventory,
    create_order,
    upsert_inventory_policy,
    add_user,
    transition_order,
)

# Structured catalog: SKU, name, stock, location, ROP, min, max, ABC, velocity
CATALOG = [
    ("SKU-1001", "Quantum Processor X1", 450, "A-01", 100, 50, 500, "A", 9.5),
    ("SKU-1002", "Thermal Paste Pro", 1200, "B-04", 300, 150, 1500, "B", 6.2),
    ("SKU-1003", "Server Chassis 2U", 45, "C-12", 20, 10, 80, "C", 2.1),
    ("SKU-1004", "Optical Cable 10m", 850, "A-02", 200, 100, 1000, "A", 8.8),
    ("SKU-1005", "NVMe SSD 2TB", 320, "A-03", 80, 40, 400, "A", 7.4),
    ("SKU-1006", "Rack PDU 16A", 90, "B-01", 30, 15, 120, "B", 4.1),
    ("SKU-1007", "Cat6 Patch 1m", 2400, "B-05", 500, 250, 3000, "B", 5.5),
    ("SKU-1008", "Cooling Fan 120mm", 600, "C-03", 100, 50, 800, "C", 3.0),
    ("SKU-1009", "GPU Accelerator H100", 28, "A-04", 10, 5, 40, "A", 9.9),
    ("SKU-1010", "Fiber Transceiver SFP+", 180, "B-02", 40, 20, 200, "B", 5.8),
    ("SKU-1011", "Label Roll 100x50", 55, "D-01", 20, 10, 100, "C", 1.8),
    ("SKU-1012", "Packing Tape Clear", 400, "D-02", 80, 40, 500, "C", 2.5),
]

USERS = [
    ("admin", "admin123", "super_admin"),
    ("manager", "manager123", "manager"),
    ("operator", "operator123", "operator"),
    ("picker1", "picker123", "operator"),
    ("packer1", "packer123", "operator"),
    ("viewer", "viewer123", "viewer"),
]


def seed_warehouse(n_orders: int = 180, force_users: bool = True):
    print("Initializing Database Schemas...")
    init_db()

    if force_users:
        print("Ensuring RBAC users...")
        for u, p, r in USERS:
            try:
                add_user(u, p, r)
            except Exception:
                pass

    print(f"Populating {len(CATALOG)} SKUs + ABC policies...")
    for sku, prod, stock, loc, rop, min_s, max_s, abc, vel in CATALOG:
        upsert_inventory(sku, prod, stock, loc, "Seed v4.4")
        upsert_inventory_policy(sku, rop, min_s, max_s, abc, vel)

    print("Generating orders across workflow states...")
    statuses = ["Pending", "Allocated", "Picking", "Picked", "Packed", "Shipped", "Cancelled"]
    weights = [25, 15, 12, 10, 12, 20, 6]
    sku_list = [s[0] for s in CATALOG]
    created = []
    for i in range(1, n_orders + 1):
        order_id = f"ORD-{10000 + i}"
        status = random.choices(statuses, weights=weights)[0]
        items = random.sample(sku_list, k=random.randint(1, min(4, len(sku_list))))
        create_order(order_id, "Pending", items)
        if status != "Pending":
            path = {
                "Allocated": ["Allocated"],
                "Picking": ["Allocated", "Picking"],
                "Picked": ["Allocated", "Picking", "Picked"],
                "Packed": ["Allocated", "Picking", "Picked", "Packing", "Packed"],
                "Shipped": ["Allocated", "Picking", "Picked", "Packing", "Packed", "Shipped"],
                "Cancelled": ["Cancelled"],
            }.get(status, [])
            for st in path:
                try:
                    transition_order(order_id, st, actor="seed", reason="seed_data")
                except Exception:
                    pass
        created.append(order_id)

    print(f"Seed complete: {len(CATALOG)} SKUs, {len(created)} orders, {len(USERS)} users.")
    print("Logins: admin/admin123 | manager/manager123 | operator/operator123 | viewer/viewer123")


if __name__ == "__main__":
    seed_warehouse()
