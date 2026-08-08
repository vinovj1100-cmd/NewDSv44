"""v4.4 integration facade — wire advanced modules into the console.
Import this from app_v43 (or app_v44) to enable RuleEngine, Workflow,
Audit, Forecasting, and Report generator without scattering imports.
"""
from __future__ import annotations
from typing import Optional

_rule = None
_wf = None
_audit = None
_forecast = None
_reports = None
_rbac = None

def get_rule_engine(config_path: str = "config.yaml"):
    global _rule
    if _rule is None:
        from rule_engine import RuleEngine
        _rule = RuleEngine(config_path)
    return _rule

def get_workflow():
    global _wf
    if _wf is None:
        from workflow_engine import WorkflowEngine
        _wf = WorkflowEngine()
    return _wf

def get_audit():
    global _audit
    if _audit is None:
        from audit_trail import AuditTrail
        _audit = AuditTrail()
    return _audit

def get_forecast():
    global _forecast
    if _forecast is None:
        from advanced_forecasting import AdvancedForecaster
        _forecast = AdvancedForecaster()
    return _forecast

def get_reports():
    global _reports
    if _reports is None:
        from report_generator import ReportGenerator
        _reports = ReportGenerator()
    return _reports

def get_rbac(config: Optional[dict] = None):
    global _rbac
    if _rbac is None:
        from rbac_engine import RBACEngine
        _rbac = RBACEngine(config or {})
    return _rbac

def run_rules_once():
    """Evaluate automation rules; return list of triggered actions."""
    return get_rule_engine().evaluate()
