"""Tests for Workflow Engine v4.4"""
import pytest
from workflow_engine import WorkflowEngine, TRANSITIONS

class TestWorkflowEngine:
    def test_can_transition_valid(self):
        wf = WorkflowEngine()
        assert wf.can_transition("Pending", "Allocated") is True
        assert wf.can_transition("Packing", "Packed") is True

    def test_can_transition_invalid(self):
        wf = WorkflowEngine()
        assert wf.can_transition("Pending", "Shipped") is False
        assert wf.can_transition("Shipped", "Pending") is False

    def test_transition_matrix_completeness(self):
        for state, targets in TRANSITIONS.items():
            assert isinstance(targets, set)
            for t in targets:
                assert t in TRANSITIONS
