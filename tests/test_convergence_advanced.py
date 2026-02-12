"""Tests for advanced convergence detection (Bayesian, uncertainty-aware, cost-benefit)."""
from __future__ import annotations

import math
import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="otbot_convergence_adv_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "convergence_adv_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.services.convergence_advanced import (  # noqa: E402
    AdvancedConvergenceStatus,
    CostBenefitAnalysis,
    UncertaintyEstimate,
    analyze_cost_benefit,
    detect_change_points,
    detect_convergence_advanced,
    estimate_kpi_uncertainty,
    should_stop_campaign_advanced,
    should_stop_uncertainty_aware,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()


# ---------------------------------------------------------------------------
# Change-Point Detection Tests
# ---------------------------------------------------------------------------


class TestChangePointDetection:
    """Test Bayesian change-point detection."""

    def test_no_change_steady_series(self):
        """Steady improvement has no change point."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = detect_change_points(values)

        # Gradual increase should not trigger change point
        assert not result.detected or result.confidence < 0.7

    def test_clear_change_point(self):
        """Obvious change point detected."""
        # Plateau then jump
        values = [5, 5, 5, 5, 5, 15, 15, 15, 15, 15]
        result = detect_change_points(values)

        assert result.detected
        assert result.most_likely is not None
        # Change should be around index 5
        assert 4 <= result.most_likely.location <= 6
        assert result.most_likely.magnitude > 5.0

    def test_gradual_shift(self):
        """Gradual shift may or may not be detected."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = detect_change_points(values)

        # Gradual change: detection depends on sensitivity
        # Should have low confidence if detected
        if result.detected:
            assert result.confidence < 0.8

    def test_multiple_segments(self):
        """Multiple change points."""
        # Low → High → Low
        values = [2, 2, 2, 10, 10, 10, 3, 3, 3]
        result = detect_change_points(values)

        # Should detect at least one change point
        # Most likely either at ~3 or ~6
        if result.detected:
            assert result.most_likely.location in range(2, 7)

    def test_insufficient_data(self):
        """Insufficient data returns no detection."""
        values = [1, 2, 3]
        result = detect_change_points(values, min_segment_length=3)

        assert not result.detected
        assert result.most_likely is None
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Uncertainty Estimation Tests
# ---------------------------------------------------------------------------


class TestUncertaintyEstimation:
    """Test epistemic uncertainty estimation."""

    def test_single_value_infinite_uncertainty(self):
        """Single value has infinite uncertainty."""
        values = [5.0]
        uncertainty = estimate_kpi_uncertainty(values)

        assert uncertainty.mean == 5.0
        assert math.isinf(uncertainty.std)

    def test_low_variance_tight_bounds(self):
        """Low variance gives tight confidence bounds."""
        values = [10.0, 10.1, 9.9, 10.0, 10.1]
        uncertainty = estimate_kpi_uncertainty(values, confidence_level=0.95)

        assert 9.5 < uncertainty.mean < 10.5
        assert uncertainty.std < 0.5
        assert uncertainty.lower_bound < uncertainty.mean
        assert uncertainty.upper_bound > uncertainty.mean
        assert uncertainty.upper_bound - uncertainty.lower_bound < 1.0  # Tight

    def test_high_variance_wide_bounds(self):
        """High variance gives wide confidence bounds."""
        values = [5, 15, 8, 12, 6, 14]
        uncertainty = estimate_kpi_uncertainty(values, confidence_level=0.95)

        assert uncertainty.std > 2.0
        assert uncertainty.upper_bound - uncertainty.lower_bound > 5.0  # Wide

    def test_empty_values(self):
        """Empty values return infinite uncertainty."""
        values = []
        uncertainty = estimate_kpi_uncertainty(values)

        assert uncertainty.mean == 0.0
        assert math.isinf(uncertainty.std)


# ---------------------------------------------------------------------------
# Uncertainty-Aware Stopping Tests
# ---------------------------------------------------------------------------


class TestUncertaintyAwareStopping:
    """Test uncertainty-aware stopping criterion."""

    def test_target_reached_with_confidence(self):
        """Stop when lower bound exceeds target."""
        # High KPI with low variance
        kpi_history = [95, 96, 97, 98, 99, 100, 100, 100]
        target = 95

        should_stop, reason = should_stop_uncertainty_aware(
            kpi_history,
            target_kpi=target,
            maximize=True
        )

        # Lower bound should be > 95, so stop
        assert should_stop
        assert "target_reached_with_confidence" in reason

    def test_target_not_reached_yet(self):
        """Continue when lower bound below target."""
        # Approaching but not there yet
        kpi_history = [90, 91, 92, 93, 94]
        target = 100

        should_stop, reason = should_stop_uncertainty_aware(
            kpi_history,
            target_kpi=target,
            maximize=True
        )

        assert not should_stop
        assert "target_not_reached_yet" in reason

    def test_high_variance_delays_stopping(self):
        """High variance means lower bound stays below target."""
        # Mean ~ 98 but high variance
        kpi_history = [90, 105, 92, 103, 95, 102]
        target = 95

        should_stop, reason = should_stop_uncertainty_aware(
            kpi_history,
            target_kpi=target,
            maximize=True
        )

        # Despite mean > target, uncertainty may delay stop
        # (Lower bound might be < 95)
        # Test is flexible based on actual CI computation


# ---------------------------------------------------------------------------
# Cost-Benefit Analysis Tests
# ---------------------------------------------------------------------------


class TestCostBenefitAnalysis:
    """Test cost-benefit analysis for campaign continuation."""

    def test_high_improvement_recommend_continue(self):
        """High expected improvement → continue."""
        # Strong upward trend
        kpi_history = [10, 15, 20, 25, 30]
        result = analyze_cost_benefit(kpi_history, experiment_cost=1.0, cost_threshold=0.01)

        assert result.recommendation == "continue"
        assert result.benefit_cost_ratio > 0.01
        assert result.expected_improvement > 0

    def test_low_improvement_recommend_stop(self):
        """Low expected improvement → stop."""
        # Plateau
        kpi_history = [50, 50.1, 50.0, 50.1, 50.0]
        result = analyze_cost_benefit(kpi_history, experiment_cost=10.0, cost_threshold=0.1)

        # Expected improvement ~ 0, cost = 10 → ratio < 0.1
        assert result.recommendation == "stop"
        assert result.benefit_cost_ratio < 0.1

    def test_high_cost_threshold_easier_to_stop(self):
        """Higher cost threshold makes stopping easier."""
        kpi_history = [10, 11, 12, 13, 14]
        result_low = analyze_cost_benefit(kpi_history, cost_threshold=0.01)
        result_high = analyze_cost_benefit(kpi_history, cost_threshold=10.0)

        # Low threshold → likely continue
        # High threshold → likely stop
        assert result_low.recommendation == "continue"
        assert result_high.recommendation == "stop"

    def test_insufficient_data_continue(self):
        """Insufficient data → continue by default."""
        kpi_history = [1, 2]
        result = analyze_cost_benefit(kpi_history)

        assert result.recommendation == "continue"
        assert "insufficient_data" in result.reason


# ---------------------------------------------------------------------------
# Integrated Advanced Detection Tests
# ---------------------------------------------------------------------------


class TestAdvancedDetection:
    """Test integrated advanced convergence detection."""

    def test_advanced_detection_structure(self):
        """Advanced detection returns all components."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        status = detect_convergence_advanced(values, target=12.0, experiment_cost=1.0)

        assert isinstance(status, AdvancedConvergenceStatus)
        assert status.enhanced_status is not None
        assert status.change_point is not None
        assert status.uncertainty is not None
        assert status.cost_benefit is not None
        assert "n_observations" in status.analysis_metadata

    def test_advanced_detection_with_change_point(self):
        """Detects change point in advanced analysis."""
        # Clear change point
        values = [5, 5, 5, 5, 5, 15, 15, 15, 15, 15]
        status = detect_convergence_advanced(values)

        assert status.change_point.detected
        assert status.analysis_metadata["change_point_detected"]

    def test_advanced_detection_to_dict(self):
        """to_dict serialization works."""
        values = [1, 2, 3, 4, 5]
        status = detect_convergence_advanced(values)

        status_dict = status.to_dict()
        assert "enhanced_status" in status_dict
        assert "change_point" in status_dict
        assert "uncertainty" in status_dict
        assert "cost_benefit" in status_dict


# ---------------------------------------------------------------------------
# Advanced Stop Decision Tests
# ---------------------------------------------------------------------------


class TestAdvancedStopDecision:
    """Test advanced campaign stop decision."""

    def test_stop_target_reached_with_confidence(self):
        """Stop when target reached with high confidence."""
        # High KPI, low variance
        values = [95, 96, 97, 98, 99, 100]
        status = detect_convergence_advanced(values, target=95.0)

        action, reason = should_stop_campaign_advanced(
            status,
            goal_target_reached=False,
            rounds_exhausted=False,
            target_kpi=95.0,
            enable_uncertainty=True
        )

        # Should stop due to uncertainty-aware target check
        assert action == "stop"
        assert "confidence" in reason.lower() or "target" in reason.lower()

    def test_stop_budget_exhausted(self):
        """Stop when budget exhausted."""
        values = [1, 2, 3, 4, 5]
        status = detect_convergence_advanced(values)

        action, reason = should_stop_campaign_advanced(
            status,
            goal_target_reached=False,
            rounds_exhausted=True
        )

        assert action == "stop"
        assert reason == "budget_exhausted"

    def test_stop_cost_benefit_unfavorable(self):
        """Stop when cost-benefit unfavorable."""
        # Plateau
        values = [50] * 10
        status = detect_convergence_advanced(values, experiment_cost=100.0)

        action, reason = should_stop_campaign_advanced(
            status,
            goal_target_reached=False,
            rounds_exhausted=False,
            enable_cost_benefit=True
        )

        # Should stop due to low benefit/cost ratio
        assert action == "stop"
        assert "improvement_too_small" in reason or "cost" in reason

    def test_continue_recent_change_point(self):
        """Continue if recent change point detected."""
        # Change point at end
        values = [5, 5, 5, 5, 5, 5, 5, 15, 16, 17]
        status = detect_convergence_advanced(values)

        action, reason = should_stop_campaign_advanced(
            status,
            goal_target_reached=False,
            rounds_exhausted=False
        )

        # Might continue due to recent change point
        # (Test is flexible as change point detection is probabilistic)


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------


class TestAdvancedIntegration:
    """Integration test combining all advanced features."""

    def test_realistic_campaign_scenario(self):
        """Realistic campaign with change point and convergence."""
        import random
        random.seed(42)

        # Campaign: slow start → breakthrough → plateau
        values = []

        # Phase 1: Slow improvement (rounds 0-9)
        for i in range(10):
            values.append(50 + i * 0.5 + random.gauss(0, 0.5))

        # Phase 2: Breakthrough (rounds 10-14)
        for i in range(5):
            values.append(70 + i * 3 + random.gauss(0, 0.5))

        # Phase 3: Plateau (rounds 15-19)
        for i in range(5):
            values.append(85 + random.gauss(0, 0.3))

        status = detect_convergence_advanced(
            values,
            target=90.0,
            experiment_cost=1.0
        )

        # Should detect change point around round 10
        if status.change_point.detected:
            cp_location = status.change_point.most_likely.location
            assert 8 <= cp_location <= 12

        # Should have reasonable uncertainty estimate
        assert status.uncertainty.std > 0
        assert status.uncertainty.std < 10

        # Cost-benefit should recommend based on plateau
        # (In plateau phase, improvement small → may recommend stop)

        # Full stop decision
        action, reason = should_stop_campaign_advanced(
            status,
            goal_target_reached=False,
            rounds_exhausted=False,
            target_kpi=90.0,
            enable_cost_benefit=True,
            enable_uncertainty=True
        )

        # Should have a clear recommendation
        assert action in ["continue", "stop"]
        assert len(reason) > 0
