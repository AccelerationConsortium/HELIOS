"""Tests for the Sensing/QA Agent."""
import asyncio
import pytest
from app.agents.sensing_agent import (
    SensingAgent,
    SensingInput,
    SensingOutput,
    QCCheck,
    _check_volume_accuracy,
    _check_temperature_stability,
    _check_current_range,
    _detect_anomalies,
)


class TestVolumeAccuracy:
    def test_accurate_volume(self):
        result = _check_volume_accuracy(
            {"volume_ul": 100.0},
            {"actual_volume_ul": 99.0},
        )
        assert result is not None
        assert result.passed

    def test_inaccurate_volume(self):
        result = _check_volume_accuracy(
            {"volume_ul": 100.0},
            {"actual_volume_ul": 80.0},
            tolerance_pct=10.0,
        )
        assert result is not None
        assert not result.passed

    def test_no_actual_volume(self):
        result = _check_volume_accuracy(
            {"volume_ul": 100.0},
            {},
        )
        assert result is None


class TestTemperatureStability:
    def test_stable_temp(self):
        result = _check_temperature_stability(
            {"temp_c": 50.0},
            {"actual_temp_c": 50.5},
        )
        assert result is not None
        assert result.passed

    def test_unstable_temp(self):
        result = _check_temperature_stability(
            {"temp_c": 50.0},
            {"actual_temp_c": 55.0},
            tolerance_c=2.0,
        )
        assert result is not None
        assert not result.passed
        assert result.severity == "critical"


class TestCurrentRange:
    def test_within_range(self):
        result = _check_current_range(
            {"current_ma": 50.0},
            max_current_ma=100.0,
        )
        assert result is not None
        assert result.passed

    def test_over_range(self):
        result = _check_current_range(
            {"current_ma": 150.0},
            max_current_ma=100.0,
        )
        assert result is not None
        assert not result.passed
        assert result.severity == "critical"


class TestAnomalyDetection:
    def test_no_anomaly(self):
        history = [
            {"value": 10.0},
            {"value": 11.0},
            {"value": 10.5},
        ]
        anomalies = _detect_anomalies({"value": 10.8}, history)
        assert len(anomalies) == 0

    def test_detects_anomaly(self):
        history = [
            {"value": 10.0},
            {"value": 10.0},
            {"value": 10.0},
            {"value": 10.0},
        ]
        # Value of 100 is way outside normal range
        anomalies = _detect_anomalies({"value": 100.0}, history)
        assert len(anomalies) > 0

    def test_insufficient_history(self):
        history = [{"value": 10.0}]
        anomalies = _detect_anomalies({"value": 100.0}, history)
        assert len(anomalies) == 0


class TestSensingAgent:
    def test_good_quality_step(self):
        agent = SensingAgent()
        inp = SensingInput(
            step_key="s1",
            primitive="robot.aspirate",
            params={"volume_ul": 100.0},
            step_result={"actual_volume_ul": 99.0, "duration_s": 5.0},
        )
        result = asyncio.run(agent.run(inp))
        assert result.success
        assert result.output.overall_quality == "good"
        assert result.output.recommendation == "continue"

    def test_critical_failure(self):
        agent = SensingAgent()
        inp = SensingInput(
            step_key="s1",
            primitive="heat",
            params={"temp_c": 50.0},
            step_result={"actual_temp_c": 80.0},  # way off
        )
        result = asyncio.run(agent.run(inp))
        assert result.success
        assert result.output.overall_quality == "failed"
        assert result.output.recommendation == "abort"

    def test_custom_qc_check(self):
        agent = SensingAgent()
        inp = SensingInput(
            step_key="s1",
            primitive="squidstat.run_experiment",
            params={},
            step_result={"peak_voltage_v": 1.8},
            qc_checks=[
                QCCheck(name="voltage_check", metric="peak_voltage_v", max_value=2.0),
            ],
        )
        result = asyncio.run(agent.run(inp))
        assert result.success
        assert all(c.passed for c in result.output.checks)

    def test_validation_error(self):
        agent = SensingAgent()
        inp = SensingInput(step_key="", primitive="", params={}, step_result={})
        result = asyncio.run(agent.run(inp))
        assert not result.success
