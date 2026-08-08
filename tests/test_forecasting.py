"""Tests for Advanced Forecasting Engine v4.4"""
import pytest
from advanced_forecasting import AdvancedForecaster

class TestAdvancedForecaster:
    def test_fallback_forecast(self):
        fc = AdvancedForecaster()
        result = fc._fallback_forecast()
        assert "forecast" in result
        assert len(result["forecast"]) == 14
        assert result["model"] == "fallback_heuristic"

    def test_config_defaults(self):
        fc = AdvancedForecaster()
        assert fc.config["horizon_days"] == 21
        assert fc.config["confidence_level"] == 0.95
