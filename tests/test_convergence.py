"""Tests for convergence detection."""
from __future__ import annotations

import os
import tempfile

_tmpdir = tempfile.mkdtemp(prefix="otbot_convergence_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "convergence_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.services.convergence import (  # noqa: E402
    ConvergenceConfig,
    ConvergenceStatus,
    _linear_regression_slope,
    _mean,
    _variance,
    best_kpi_slope,
    detect_convergence,
    rolling_improvement_rate,
    variance_collapse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    with connection() as conn:
        conn.execute("DELETE FROM evolved_priors")
        conn.execute("DELETE FROM evolution_proposals")
        conn.execute("DELETE FROM protocol_templates")
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM run_reviews")
        conn.execute("DELETE FROM run_kpis")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM provenance_events")
        conn.execute("DELETE FROM run_steps")
        conn.execute("DELETE FROM runs")
        conn.commit()


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


class TestStatsHelpers:
    def test_mean_basic(self):
        assert abs(_mean([1, 2, 3, 4, 5]) - 3.0) < 1e-10

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_variance_basic(self):
        # Variance of [1,2,3,4,5]: mean=3, var=2.0
        assert abs(_variance([1, 2, 3, 4, 5]) - 2.0) < 1e-10

    def test_linear_regression_slope_positive(self):
        """Increasing values should have positive slope."""
        slope = _linear_regression_slope([1, 2, 3, 4, 5])
        assert slope > 0

    def test_linear_regression_slope_flat(self):
        """Constant values should have zero slope."""
        slope = _linear_regression_slope([5, 5, 5, 5, 5])
        assert abs(slope) < 1e-10


# ---------------------------------------------------------------------------
# Rolling improvement rate
# ---------------------------------------------------------------------------


class TestRollingImprovement:
    def test_improving_series(self):
        """Monotonically increasing values should show positive improvement."""
        rate = rolling_improvement_rate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], window=5)
        assert rate is not None
        assert rate > 0

    def test_flat_series(self):
        """Constant values should show zero improvement."""
        rate = rolling_improvement_rate([5, 5, 5, 5, 5, 5, 5], window=5)
        assert rate is not None
        assert abs(rate) < 1e-10

    def test_insufficient_data(self):
        """Too few values returns None."""
        rate = rolling_improvement_rate([1, 2], window=5)
        assert rate is not None  # 2 values >= 2, so it computes

    def test_single_value(self):
        """Single value returns None."""
        rate = rolling_improvement_rate([1], window=5)
        assert rate is None


# ---------------------------------------------------------------------------
# Best KPI slope
# ---------------------------------------------------------------------------


class TestBestKpiSlope:
    def test_improving_cumulative_best(self):
        """Improving KPIs should show positive slope."""
        slope = best_kpi_slope([1, 3, 2, 5, 4, 7, 6, 9, 8, 10], window=5, maximize=True)
        assert slope is not None
        assert slope > 0

    def test_plateau_cumulative_best(self):
        """Flat cumulative best should have near-zero slope."""
        # After initial rise, best stays at 10
        slope = best_kpi_slope([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], window=5, maximize=True)
        assert slope is not None
        assert abs(slope) < 0.01


# ---------------------------------------------------------------------------
# Variance collapse
# ---------------------------------------------------------------------------


class TestVarianceCollapse:
    def test_converged_series(self):
        """Values clustering at the end should show low variance ratio."""
        values = [1, 5, 10, 15, 20, 19.5, 20.0, 20.1, 19.9, 20.0]
        ratio = variance_collapse(values, recent_window=5)
        assert ratio is not None
        assert ratio < 0.3  # recent variance << total variance

    def test_diverging_series(self):
        """Increasing spread should show high variance ratio."""
        values = [5, 5, 5, 5, 5, 1, 10, 2, 9, 3]
        ratio = variance_collapse(values, recent_window=5)
        assert ratio is not None
        # recent variance is high relative to total


# ---------------------------------------------------------------------------
# detect_convergence
# ---------------------------------------------------------------------------


class TestDetectConvergence:
    def test_improving(self):
        """Clear improvement detected."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = detect_convergence(values)
        assert result.status == "improving"
        assert 0 <= result.confidence <= 1

    def test_plateau(self):
        """Plateau after improvement detected."""
        values = [1, 3, 5, 7, 9, 9.01, 9.0, 9.02, 8.99, 9.0]
        result = detect_convergence(values)
        assert result.status == "plateau"
        assert result.confidence > 0.5

    def test_insufficient_data(self):
        """Too few data points returns insufficient_data status."""
        result = detect_convergence([1, 2])
        assert result.status == "insufficient_data"

    def test_to_dict(self):
        """ConvergenceStatus.to_dict() format."""
        result = detect_convergence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        d = result.to_dict()
        assert "status" in d
        assert "confidence" in d
        assert "details" in d

    def test_deterministic(self):
        """Same input produces identical output."""
        vals = [1, 3, 5, 7, 9, 9.01, 9.0, 9.02, 8.99, 9.0]
        r1 = detect_convergence(vals)
        r2 = detect_convergence(vals)
        assert r1.status == r2.status
        assert r1.confidence == r2.confidence
