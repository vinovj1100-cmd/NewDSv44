"""Rule Engine v4.3 — Configurable triggers & automated actions.
Evaluates conditions against live KPIs and executes actions (auto-wave,
notify, raise andon, pause receiving). Runs via APScheduler in background.
"""
import yaml
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

from db import get_orders, queue_status, get_cycle_counts
from floor_ops import SLAAging, AndonBoard
from efficiency import WavePlanner

class RuleEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.enabled = cfg.get("rule_engine", {}).get("enabled", False)
        self.rules = cfg.get("rule_engine", {}).get("rules", [])
        self.webhooks = cfg.get("rule_engine", {}).get("webhooks", {})
        self._sla = SLAAging()
        self._andon = AndonBoard()
        self._wave = WavePlanner()
        self._last_run: Dict[str, float] = {}

    def evaluate(self) -> List[Dict]:
        """Run all rules against current state. Returns triggered actions."""
        if not self.enabled:
            return []
        triggered = []
        state = self._get_state()
        for rule in self.rules:
            rule_id = rule["id"]
            cooldown = self._cooldown(rule_id, 60)  # 60s default cooldown
            if not self._match(rule["condition"], state) or cooldown:
                continue
            action = rule["action"]
            result = self._execute(action, rule, state)
            if result:
                triggered.append({"rule": rule_id, "action": action, "result": result})
        return triggered

    def _get_state(self) -> Dict[str, Any]:
        orders = get_orders()
        pending = int((orders["status"] == "Pending").sum()) if not orders.empty else 0
        backlog = len([r for r in get_cycle_counts(status="Completed") if r["variance"] != 0]) if not get_cycle_counts().empty else 0
        sla = self._sla.band()
        qs = queue_status()
        return {
            "pending_orders": pending, "backlog": backlog,
            "sla_breached": sla["breached"],
            "cycle_accuracy": 94.5,  # placeholder; wire to real KPI
            "queue_items": qs["queued"]
        }

    def _match(self, condition: str, state: Dict) -> bool:
        """Safe condition evaluator. Supports: > < >= <= == != and or ( )"""
        import ast, operator
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
            ast.Gt: operator.gt, ast.Lt: operator.lt,
            ast.GtE: operator.ge, ast.LtE: operator.le,
            ast.Eq: operator.eq, ast.NotEq: operator.ne,
            ast.And: operator.and_, ast.Or: operator.or_,
        }
        try:
            node = ast.parse(condition.strip(), mode='eval')
        except Exception:
            return False

        def _eval(n):
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            if isinstance(n, ast.Constant):
                return n.value
            if isinstance(n, ast.Name):
                return state.get(n.id, 0)
            if isinstance(n, ast.BinOp):
                return ops[type(n.op)](_eval(n.left), _eval(n.right))
            if isinstance(n, ast.UnaryOp):
                return ops[type(n.op)](_eval(n.operand))
            if isinstance(n, ast.BoolOp):
                vals = [_eval(v) for v in n.values]
                return ops[type(n.op)](*vals) if len(vals) == 2 else vals[0]
            if isinstance(n, ast.Compare):
                left = _eval(n.left)
                for op, comp in zip(n.ops, n.comparators):
                    if not ops[type(op)](left, _eval(comp)):
                        return False
                    left = _eval(comp)
                return True
            raise ValueError("Unsupported expression")
        try:
            return _eval(node)
        except Exception:
            return False
