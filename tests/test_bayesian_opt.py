"""Tests for Bayesian optimization (GP/KNN surrogate + EI/UCB)."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_bo_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "bo_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.services.bayesian_opt import (  # noqa: E402
    Observation,
    SurrogateModel,
    _Phi,
    _phi,
    benchmark_strategies,
    denormalize_point,
    expected_improvement,
    normalize_params,
    sample_bo,
    upper_confidence_bound,
)
from app.services.candidate_gen import (  # noqa: E402
    ParameterSpace,
    SearchDimension,
)


@pytest.fixture(autouse=True)
def _setup_db():
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
# Helper
# ---------------------------------------------------------------------------


def _make_space(n_dims: int = 2) -> ParameterSpace:
    dims = tuple(
        SearchDimension(
            param_name=f"x{i}",
            param_type="number",
            min_value=0.0,
            max_value=10.0,
        )
        for i in range(n_dims)
    )
    return ParameterSpace(dimensions=dims, protocol_template={"steps": []})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalDistribution:
    def test_phi_at_zero(self):
        """phi(0) ≈ 0.3989."""
        assert abs(_phi(0) - 0.3989422804) < 1e-6

    def test_Phi_at_zero(self):
        """Phi(0) = 0.5."""
        assert abs(_Phi(0) - 0.5) < 1e-6

    def test_Phi_symmetry(self):
        """Phi(-x) ≈ 1 - Phi(x)."""
        assert abs(_Phi(-1.0) + _Phi(1.0) - 1.0) < 1e-10


class TestAcquisitionFunctions:
    def test_ei_positive_when_better(self):
        """EI > 0 when mean > best_so_far."""
        ei = expected_improvement(mean=5.0, std=1.0, best_so_far=4.0)
        assert ei > 0

    def test_ei_zero_when_no_uncertainty(self):
        """EI = 0 when std = 0."""
        ei = expected_improvement(mean=5.0, std=0.0, best_so_far=4.0)
        assert ei == 0.0

    def test_ucb_increases_with_kappa(self):
        """Higher kappa -> higher UCB."""
        ucb_low = upper_confidence_bound(mean=3.0, std=1.0, kappa=1.0)
        ucb_high = upper_confidence_bound(mean=3.0, std=1.0, kappa=3.0)
        assert ucb_high > ucb_low


class TestSurrogateModel:
    def test_predict_with_observations(self):
        """Surrogate returns mean and std for a query point."""
        obs = [
            Observation(params=(0.1,), objective=1.0),
            Observation(params=(0.5,), objective=5.0),
            Observation(params=(0.9,), objective=2.0),
        ]
        model = SurrogateModel(obs, k=3)
        mean, std = model.predict((0.5,))
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0.0

    def test_predict_near_known_point(self):
        """Prediction near a known point should be close to its value."""
        obs = [
            Observation(params=(0.0,), objective=0.0),
            Observation(params=(0.5,), objective=10.0),
            Observation(params=(1.0,), objective=0.0),
        ]
        model = SurrogateModel(obs, k=3)
        mean, _std = model.predict((0.5,))
        # Mean should be closer to 10 than to 0
        assert mean > 3.0


class TestNormalization:
    def test_normalize_denormalize_roundtrip(self):
        """normalize then denormalize returns original values."""
        space = _make_space(2)
        params = {"x0": 3.0, "x1": 7.0}
        normalized = normalize_params(params, space)
        assert all(0 <= v <= 1 for v in normalized)
        recovered = denormalize_point(list(normalized), space)
        assert abs(recovered["x0"] - 3.0) < 0.01
        assert abs(recovered["x1"] - 7.0) < 0.01


class TestSampleBO:
    def test_cold_start_falls_back_to_lhs(self):
        """With no observations, sample_bo falls back to LHS."""
        space = _make_space(2)
        results = sample_bo(space, 5, observations=[], seed=42)
        assert len(results) == 5
        # All params should be within bounds
        for r in results:
            assert 0.0 <= r["x0"] <= 10.0
            assert 0.0 <= r["x1"] <= 10.0

    def test_with_observations(self):
        """With sufficient observations, uses BO."""
        space = _make_space(2)
        obs = [
            Observation(
                params=normalize_params(
                    {"x0": float(i), "x1": float(i)}, space
                ),
                objective=float(10 - i),
            )
            for i in range(10)
        ]
        results = sample_bo(space, 5, observations=obs, seed=42)
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r["x0"] <= 10.0
            assert 0.0 <= r["x1"] <= 10.0

    def test_seed_reproducibility(self):
        """Same seed produces same candidates."""
        space = _make_space(2)
        obs = [
            Observation(
                params=normalize_params(
                    {"x0": float(i), "x1": float(i)}, space
                ),
                objective=float(i),
            )
            for i in range(10)
        ]
        r1 = sample_bo(space, 3, observations=obs, seed=99)
        r2 = sample_bo(space, 3, observations=obs, seed=99)
        assert r1 == r2


class TestBenchmarkStrategies:
    def test_benchmark_runs(self):
        """benchmark_strategies() returns convergence data for each strategy."""
        space = _make_space(2)

        def obj(params):
            return -(params["x0"] - 5) ** 2 - (params["x1"] - 5) ** 2

        results = benchmark_strategies(
            space,
            obj,
            strategies=["random", "lhs", "bo"],
            n_rounds=3,
            batch_size=3,
            seed=42,
        )
        assert "random" in results
        assert "lhs" in results
        assert "bo" in results
        assert len(results["random"]) == 3
