"""Tests for batch candidate generation (Phase C4: Parameter Space Exploration).

Covers:
- Data structures (SearchDimension, ParameterSpace, Candidate frozen checks)
- Random sampling (bounds, integer type, categorical)
- LHS sampling (correct count, bounds, space-filling, seed reproducibility)
- Grid search (exhaustive count, categorical, integer dedup)
- Prior-guided sampling (uses memory, fallback, bounds)
- Scoring (with priors, without priors)
- generate_batch main entry (E2E, invalid strategy, sort by score)
- DB storage (persistence, read path, list by campaign)
- Schema table existence
"""
from __future__ import annotations

import os
import tempfile
import uuid

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_candidate_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "candidate_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, utcnow_iso  # noqa: E402
from app.services.candidate_gen import (  # noqa: E402
    BATCH_SCHEMA_VERSION,
    BatchResult,
    Candidate,
    ParameterSpace,
    SearchDimension,
    SimplexConstraint,
    _apply_simplex_constraints,
    _gamma_sample,
    _score_candidate,
    generate_batch,
    get_batch,
    list_batches,
    list_candidates,
    sample_dirichlet,
    sample_grid,
    sample_lhs,
    sample_prior_guided,
    sample_random,
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
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM memory_semantic")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_space(n_dims: int = 2) -> ParameterSpace:
    """Create a simple numeric test ParameterSpace."""
    dims = []
    for i in range(n_dims):
        dims.append(
            SearchDimension(
                param_name=f"param_{i}",
                param_type="number",
                min_value=0.0,
                max_value=100.0,
                primitive="heat",
            )
        )
    return ParameterSpace(
        dimensions=tuple(dims),
        protocol_template={"steps": [{"id": "s1", "primitive": "heat", "params": {}}]},
    )


def _insert_memory_semantic(
    primitive: str, param_name: str, mean: float, stddev: float, sample_count: int
) -> None:
    """Insert a memory_semantic row directly for prior-guided tests."""
    with connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO memory_semantic "
            "(primitive, param_name, mean, stddev, sample_count, "
            "success_rate, success_count, total_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, ?)",
            (
                primitive,
                param_name,
                mean,
                stddev,
                sample_count,
                sample_count,
                sample_count,
                utcnow_iso(),
            ),
        )
        conn.commit()


# ===========================================================================
# 1. Data Structures
# ===========================================================================


class TestDataStructures:
    """Verify core dataclasses are correctly defined."""

    def test_search_dimension_frozen(self):
        dim = SearchDimension(param_name="vol", param_type="number", min_value=0, max_value=100)
        with pytest.raises(AttributeError):
            dim.param_name = "changed"  # type: ignore[misc]

    def test_parameter_space_n_dims(self):
        space = _make_space(3)
        assert space.n_dims == 3

    def test_candidate_frozen(self):
        c = Candidate(index=0, params={"x": 1}, origin="lhs")
        with pytest.raises(AttributeError):
            c.index = 5  # type: ignore[misc]


# ===========================================================================
# 2. Random Sampling
# ===========================================================================


class TestRandomSampling:
    """Verify uniform random sampling within bounds."""

    def test_within_bounds(self):
        space = _make_space(2)
        results = sample_random(space, 50, seed=42)
        assert len(results) == 50
        for point in results:
            for dim in space.dimensions:
                assert dim.min_value <= point[dim.param_name] <= dim.max_value

    def test_integer_type(self):
        space = ParameterSpace(
            dimensions=(
                SearchDimension(param_name="count", param_type="integer", min_value=1, max_value=10),
            ),
            protocol_template={},
        )
        results = sample_random(space, 20, seed=42)
        for point in results:
            assert isinstance(point["count"], int)
            assert 1 <= point["count"] <= 10

    def test_categorical_choices(self):
        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="color",
                    param_type="categorical",
                    choices=("red", "green", "blue"),
                ),
            ),
            protocol_template={},
        )
        results = sample_random(space, 30, seed=42)
        for point in results:
            assert point["color"] in ("red", "green", "blue")


# ===========================================================================
# 3. LHS Sampling
# ===========================================================================


class TestLHSSampling:
    """Verify Latin Hypercube Sampling properties."""

    def test_correct_count(self):
        space = _make_space(2)
        results = sample_lhs(space, 10, seed=42)
        assert len(results) == 10

    def test_within_bounds(self):
        space = _make_space(3)
        results = sample_lhs(space, 20, seed=42)
        for point in results:
            for dim in space.dimensions:
                assert dim.min_value <= point[dim.param_name] <= dim.max_value

    def test_space_filling_stratification(self):
        """Each stratum should be covered exactly once (LHS property)."""
        space = ParameterSpace(
            dimensions=(
                SearchDimension(param_name="x", param_type="number", min_value=0, max_value=100),
            ),
            protocol_template={},
        )
        n = 10
        results = sample_lhs(space, n, seed=42)
        values = [p["x"] for p in results]
        # Check each of the N strata [0,10), [10,20), ... has exactly one value
        strata_hits = [0] * n
        for v in values:
            stratum = min(int(v / (100.0 / n)), n - 1)
            strata_hits[stratum] += 1
        assert all(h == 1 for h in strata_hits), f"LHS strata coverage: {strata_hits}"

    def test_reproducible_with_seed(self):
        space = _make_space(2)
        r1 = sample_lhs(space, 5, seed=123)
        r2 = sample_lhs(space, 5, seed=123)
        assert r1 == r2


# ===========================================================================
# 4. Grid Search
# ===========================================================================


class TestGridSearch:
    """Verify exhaustive grid search."""

    def test_exhaustive_count(self):
        space = _make_space(2)
        results = sample_grid(space, n_per_dim=5)
        assert len(results) == 25  # 5 x 5

    def test_categorical_enumeration(self):
        space = ParameterSpace(
            dimensions=(
                SearchDimension(param_name="a", param_type="categorical", choices=("x", "y")),
                SearchDimension(param_name="b", param_type="categorical", choices=("1", "2", "3")),
            ),
            protocol_template={},
        )
        results = sample_grid(space)
        assert len(results) == 6  # 2 x 3
        combos = {(p["a"], p["b"]) for p in results}
        assert len(combos) == 6

    def test_integer_dedup(self):
        """Integer grid with tight range should deduplicate close values."""
        space = ParameterSpace(
            dimensions=(
                SearchDimension(param_name="n", param_type="integer", min_value=1, max_value=3),
            ),
            protocol_template={},
        )
        results = sample_grid(space, n_per_dim=5)
        values = {p["n"] for p in results}
        assert values == {1, 2, 3}  # deduped from 5 levels to 3 unique ints


# ===========================================================================
# 5. Prior-Guided Sampling
# ===========================================================================


class TestPriorGuided:
    """Verify prior-guided sampling uses memory and respects bounds."""

    def test_uses_memory_priors(self):
        """Samples should cluster around the prior mean when priors exist."""
        _insert_memory_semantic("heat", "param_0", mean=50.0, stddev=5.0, sample_count=10)
        _insert_memory_semantic("heat", "param_1", mean=75.0, stddev=3.0, sample_count=10)
        space = _make_space(2)

        results = sample_prior_guided(space, 100, seed=42)
        # Most param_0 values should be near 50, param_1 near 75
        p0_values = [p["param_0"] for p in results]
        p1_values = [p["param_1"] for p in results]
        p0_mean = sum(p0_values) / len(p0_values)
        p1_mean = sum(p1_values) / len(p1_values)
        assert abs(p0_mean - 50.0) < 10.0, f"param_0 mean={p0_mean}, expected ~50"
        assert abs(p1_mean - 75.0) < 10.0, f"param_1 mean={p1_mean}, expected ~75"

    def test_fallback_when_no_priors(self):
        """Without priors, should fall back to uniform random (no crash)."""
        space = _make_space(2)
        results = sample_prior_guided(space, 10, seed=42)
        assert len(results) == 10
        for point in results:
            assert 0.0 <= point["param_0"] <= 100.0
            assert 0.0 <= point["param_1"] <= 100.0

    def test_respects_bounds(self):
        """Even with wide stddev, samples must be clamped to bounds."""
        _insert_memory_semantic("heat", "param_0", mean=95.0, stddev=50.0, sample_count=10)
        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="param_0",
                    param_type="number",
                    min_value=0.0,
                    max_value=100.0,
                    primitive="heat",
                ),
            ),
            protocol_template={},
        )
        results = sample_prior_guided(space, 100, seed=42)
        for point in results:
            assert 0.0 <= point["param_0"] <= 100.0


# ===========================================================================
# 6. Scoring
# ===========================================================================


class TestScoring:
    """Verify candidate scoring against memory priors."""

    def test_score_with_priors(self):
        _insert_memory_semantic("heat", "param_0", mean=50.0, stddev=10.0, sample_count=10)
        _insert_memory_semantic("heat", "param_1", mean=50.0, stddev=10.0, sample_count=10)
        space = _make_space(2)
        # Point at the mean should have score ~0
        score_at_mean = _score_candidate({"param_0": 50.0, "param_1": 50.0}, space)
        assert score_at_mean is not None
        assert score_at_mean < 0.1
        # Point far from mean should have higher score
        score_far = _score_candidate({"param_0": 90.0, "param_1": 10.0}, space)
        assert score_far is not None
        assert score_far > score_at_mean

    def test_score_no_priors(self):
        space = _make_space(2)
        score = _score_candidate({"param_0": 50.0, "param_1": 50.0}, space)
        assert score is None


# ===========================================================================
# 7. generate_batch Main Entry
# ===========================================================================


class TestGenerateBatch:
    """Verify the end-to-end generate_batch workflow."""

    def test_lhs_default(self):
        space = _make_space(2)
        result = generate_batch(space, strategy="lhs", n_candidates=10, seed=42)
        assert isinstance(result, BatchResult)
        assert len(result.candidates) == 10
        assert result.strategy == "lhs"
        assert result.batch_id  # non-empty UUID

    def test_invalid_strategy_raises(self):
        space = _make_space(2)
        with pytest.raises(ValueError, match="Unknown strategy"):
            generate_batch(space, strategy="nonexistent_strategy", n_candidates=5)

    def test_sorted_by_score(self):
        """When priors exist, candidates should be sorted by score (lower first)."""
        _insert_memory_semantic("heat", "param_0", mean=50.0, stddev=10.0, sample_count=10)
        _insert_memory_semantic("heat", "param_1", mean=50.0, stddev=10.0, sample_count=10)
        space = _make_space(2)
        result = generate_batch(space, strategy="random", n_candidates=20, seed=42)
        scored = [c for c in result.candidates if c.score is not None]
        if len(scored) >= 2:
            for i in range(len(scored) - 1):
                assert scored[i].score <= scored[i + 1].score  # type: ignore[operator]


# ===========================================================================
# 8. DB Storage
# ===========================================================================


class TestStorage:
    """Verify batch persistence and read path."""

    def test_batch_persisted_to_db(self):
        space = _make_space(2)
        result = generate_batch(space, strategy="lhs", n_candidates=5, seed=42)

        with connection() as conn:
            row = conn.execute(
                "SELECT * FROM batch_requests WHERE id = ?", (result.batch_id,)
            ).fetchone()
        assert row is not None
        assert row["strategy"] == "lhs"
        assert row["n_candidates"] == 5
        assert row["status"] == "generated"

        with connection() as conn:
            cands = conn.execute(
                "SELECT * FROM batch_candidates WHERE batch_id = ? ORDER BY candidate_index",
                (result.batch_id,),
            ).fetchall()
        assert len(cands) == 5

    def test_get_batch_returns_candidates(self):
        space = _make_space(2)
        result = generate_batch(space, strategy="random", n_candidates=3, seed=42)
        batch = get_batch(result.batch_id)

        assert batch is not None
        assert batch["id"] == result.batch_id
        assert len(batch["candidates"]) == 3
        assert batch["candidates"][0]["origin"] == "random"
        assert "params" in batch["candidates"][0]

    def test_list_batches_by_campaign(self):
        space = _make_space(1)
        # Create a campaign first
        campaign_id = str(uuid.uuid4())
        with connection() as conn:
            conn.execute(
                "INSERT INTO campaigns "
                "(id, name, cadence_seconds, protocol_json, inputs_json, "
                "policy_json, next_fire_at, is_active, created_at, updated_at) "
                "VALUES (?, 'test', 60, '{}', '{}', '{}', ?, 1, ?, ?)",
                (campaign_id, utcnow_iso(), utcnow_iso(), utcnow_iso()),
            )
            conn.commit()

        generate_batch(space, n_candidates=2, seed=1, campaign_id=campaign_id)
        generate_batch(space, n_candidates=2, seed=2, campaign_id=campaign_id)
        generate_batch(space, n_candidates=2, seed=3)  # no campaign

        all_batches = list_batches()
        campaign_batches = list_batches(campaign_id=campaign_id)
        assert len(all_batches) == 3
        assert len(campaign_batches) == 2


# ===========================================================================
# 9. Schema
# ===========================================================================


class TestSchema:
    """Verify DB tables exist."""

    def test_batch_tables_exist(self):
        with connection() as conn:
            tables = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "batch_requests" in tables
        assert "batch_candidates" in tables


# ===========================================================================
# 10. Simplex Constraints
# ===========================================================================


class TestSimplexConstraints:
    """Verify simplex constraint enforcement."""

    def test_normalize_to_sum_one(self):
        """Simplex params should be normalized to sum=1.0."""
        constraint = SimplexConstraint(param_names=("x_A", "x_B", "x_C"))
        candidates = [{"x_A": 3.0, "x_B": 2.0, "x_C": 5.0, "temp": 50.0}]
        result = _apply_simplex_constraints(candidates, (constraint,))
        assert len(result) == 1
        total = result[0]["x_A"] + result[0]["x_B"] + result[0]["x_C"]
        assert abs(total - 1.0) < 1e-10
        # Proportions preserved: 3:2:5 → 0.3:0.2:0.5
        assert abs(result[0]["x_A"] - 0.3) < 1e-10
        assert abs(result[0]["x_B"] - 0.2) < 1e-10
        assert abs(result[0]["x_C"] - 0.5) < 1e-10
        # Non-simplex param untouched
        assert result[0]["temp"] == 50.0

    def test_custom_target_sum(self):
        """Simplex can target sums other than 1.0 (e.g., 100%)."""
        constraint = SimplexConstraint(param_names=("a", "b"), target_sum=100.0)
        candidates = [{"a": 1.0, "b": 3.0}]
        result = _apply_simplex_constraints(candidates, (constraint,))
        assert abs(result[0]["a"] + result[0]["b"] - 100.0) < 1e-10
        assert abs(result[0]["a"] - 25.0) < 1e-10

    def test_all_zeros_gives_equal_fractions(self):
        """When all simplex values are zero, assign equal fractions."""
        constraint = SimplexConstraint(param_names=("x", "y", "z"))
        candidates = [{"x": 0.0, "y": 0.0, "z": 0.0}]
        result = _apply_simplex_constraints(candidates, (constraint,))
        for name in ("x", "y", "z"):
            assert abs(result[0][name] - 1.0 / 3.0) < 1e-10

    def test_negative_values_clamped(self):
        """Negative values are clamped to zero before normalization."""
        constraint = SimplexConstraint(param_names=("a", "b", "c"))
        candidates = [{"a": -1.0, "b": 3.0, "c": 7.0}]
        result = _apply_simplex_constraints(candidates, (constraint,))
        assert result[0]["a"] == 0.0  # was negative → clamped to 0
        assert abs(result[0]["b"] + result[0]["c"] - 1.0) < 1e-10

    def test_multiple_constraints(self):
        """Multiple simplex groups can coexist."""
        c1 = SimplexConstraint(param_names=("x1", "x2"))
        c2 = SimplexConstraint(param_names=("y1", "y2", "y3"), target_sum=2.0)
        candidates = [{"x1": 3.0, "x2": 7.0, "y1": 1.0, "y2": 2.0, "y3": 3.0}]
        result = _apply_simplex_constraints(candidates, (c1, c2))
        assert abs(result[0]["x1"] + result[0]["x2"] - 1.0) < 1e-10
        assert abs(result[0]["y1"] + result[0]["y2"] + result[0]["y3"] - 2.0) < 1e-10

    def test_lhs_with_simplex(self):
        """LHS sampling + simplex should produce normalized results."""
        space = ParameterSpace(
            dimensions=(
                SearchDimension(param_name="x_A", param_type="number", min_value=0, max_value=1),
                SearchDimension(param_name="x_B", param_type="number", min_value=0, max_value=1),
                SearchDimension(param_name="x_C", param_type="number", min_value=0, max_value=1),
                SearchDimension(param_name="temp", param_type="number", min_value=20, max_value=80),
            ),
            protocol_template={"steps": []},
            simplex_constraints=(
                SimplexConstraint(param_names=("x_A", "x_B", "x_C")),
            ),
        )
        result = generate_batch(space, strategy="lhs", n_candidates=20, seed=42, store=False)
        for c in result.candidates:
            total = c.params["x_A"] + c.params["x_B"] + c.params["x_C"]
            assert abs(total - 1.0) < 1e-10, f"Simplex sum={total}"
            # Non-simplex param should still be in range
            assert 20.0 <= c.params["temp"] <= 80.0

    def test_random_with_simplex(self):
        """Random sampling + simplex should also normalize."""
        space = ParameterSpace(
            dimensions=(
                SearchDimension(param_name="a", param_type="number", min_value=0, max_value=1),
                SearchDimension(param_name="b", param_type="number", min_value=0, max_value=1),
            ),
            protocol_template={},
            simplex_constraints=(
                SimplexConstraint(param_names=("a", "b")),
            ),
        )
        result = generate_batch(space, strategy="random", n_candidates=10, seed=42, store=False)
        for c in result.candidates:
            total = c.params["a"] + c.params["b"]
            assert abs(total - 1.0) < 1e-10

    def test_ten_component_simplex(self):
        """10-component simplex (HER catalyst use case)."""
        names = tuple(f"x_{i}" for i in range(10))
        constraint = SimplexConstraint(param_names=names)
        candidates = [{n: float(i + 1) for i, n in enumerate(names)}]
        result = _apply_simplex_constraints(candidates, (constraint,))
        total = sum(result[0][n] for n in names)
        assert abs(total - 1.0) < 1e-10
        # Proportions: 1:2:3:...:10 → sum=55, so x_0=1/55, x_9=10/55
        assert abs(result[0]["x_0"] - 1.0 / 55.0) < 1e-10
        assert abs(result[0]["x_9"] - 10.0 / 55.0) < 1e-10

    def test_no_constraints_passthrough(self):
        """Empty constraints list → candidates unchanged."""
        candidates = [{"a": 1.0, "b": 2.0}]
        result = _apply_simplex_constraints(candidates, ())
        assert result == candidates

    def test_missing_param_treated_as_zero(self):
        """If a simplex param is missing from a candidate, treat as 0."""
        constraint = SimplexConstraint(param_names=("a", "b", "c"))
        candidates = [{"a": 3.0, "b": 7.0}]  # "c" missing
        result = _apply_simplex_constraints(candidates, (constraint,))
        total = result[0]["a"] + result[0]["b"] + result[0].get("c", 0.0)
        assert abs(total - 1.0) < 1e-10


# ===========================================================================
# 11. Dirichlet (Compositional) Sampling
# ===========================================================================


class TestDirichletSampling:
    """Verify Dirichlet simplex-native sampling."""

    def _make_simplex_space(self, n_components: int = 3) -> ParameterSpace:
        """Create a space with simplex + one non-simplex dim."""
        names = tuple(f"frac_{i}" for i in range(n_components))
        dims = [
            SearchDimension(param_name=n, param_type="number", min_value=0, max_value=1)
            for n in names
        ] + [
            SearchDimension(param_name="temp", param_type="number", min_value=20, max_value=80),
        ]
        return ParameterSpace(
            dimensions=tuple(dims),
            protocol_template={},
            simplex_constraints=(SimplexConstraint(param_names=names),),
        )

    def test_sum_to_one(self):
        """Dirichlet samples should sum to target (1.0)."""
        space = self._make_simplex_space(5)
        results = sample_dirichlet(space, 50, seed=42)
        assert len(results) == 50
        names = [f"frac_{i}" for i in range(5)]
        for pt in results:
            total = sum(pt[n] for n in names)
            assert abs(total - 1.0) < 1e-10, f"Dirichlet sum={total}"

    def test_all_positive(self):
        """All Dirichlet fractions should be non-negative."""
        space = self._make_simplex_space(10)
        results = sample_dirichlet(space, 100, seed=42)
        names = [f"frac_{i}" for i in range(10)]
        for pt in results:
            for n in names:
                assert pt[n] >= 0.0, f"{n}={pt[n]} is negative"

    def test_non_simplex_dim_sampled(self):
        """Non-simplex dimensions should still be uniformly sampled."""
        space = self._make_simplex_space(3)
        results = sample_dirichlet(space, 50, seed=42)
        for pt in results:
            assert 20.0 <= pt["temp"] <= 80.0

    def test_symmetric_alpha(self):
        """With alpha=1.0 (uniform), samples should be spread across simplex."""
        space = self._make_simplex_space(3)
        results = sample_dirichlet(space, 200, alpha=1.0, seed=42)
        means = {f"frac_{i}": 0.0 for i in range(3)}
        for pt in results:
            for n in means:
                means[n] += pt[n]
        for n in means:
            means[n] /= 200
            # Mean of Dir(1,1,1) = (1/3, 1/3, 1/3)
            assert abs(means[n] - 1.0 / 3.0) < 0.1, f"mean {n}={means[n]}"

    def test_concentrated_alpha(self):
        """With high alpha, samples should cluster near equal fractions."""
        space = self._make_simplex_space(3)
        results = sample_dirichlet(space, 100, alpha=100.0, seed=42)
        for pt in results:
            for i in range(3):
                assert abs(pt[f"frac_{i}"] - 1.0 / 3.0) < 0.1

    def test_ten_component_her(self):
        """10-component Dirichlet for HER catalyst compositions."""
        names = tuple(f"stock_{i}" for i in range(10))
        dims = [
            SearchDimension(param_name=n, param_type="number", min_value=0, max_value=1)
            for n in names
        ]
        space = ParameterSpace(
            dimensions=tuple(dims),
            protocol_template={},
            simplex_constraints=(SimplexConstraint(param_names=names),),
        )
        results = sample_dirichlet(space, 50, seed=42)
        assert len(results) == 50
        for pt in results:
            total = sum(pt[n] for n in names)
            assert abs(total - 1.0) < 1e-10

    def test_reproducible_with_seed(self):
        """Same seed → same results."""
        space = self._make_simplex_space(3)
        r1 = sample_dirichlet(space, 5, seed=123)
        r2 = sample_dirichlet(space, 5, seed=123)
        assert r1 == r2

    def test_generate_batch_dirichlet_strategy(self):
        """Dirichlet works through generate_batch entry point."""
        space = self._make_simplex_space(4)
        result = generate_batch(space, strategy="dirichlet", n_candidates=10, seed=42, store=False)
        assert len(result.candidates) == 10
        assert result.strategy == "dirichlet"
        names = [f"frac_{i}" for i in range(4)]
        for c in result.candidates:
            total = sum(c.params[n] for n in names)
            assert abs(total - 1.0) < 1e-10

    def test_gamma_sample_alpha_one(self):
        """Gamma(1,1) should produce exponential-like samples."""
        import random
        rng = random.Random(42)
        samples = [_gamma_sample(1.0, rng) for _ in range(1000)]
        assert all(s >= 0 for s in samples)
        mean = sum(samples) / len(samples)
        # E[Gamma(1,1)] = 1.0
        assert abs(mean - 1.0) < 0.2

    def test_gamma_sample_small_alpha(self):
        """Gamma(0.1,1) should produce many near-zero samples."""
        import random
        rng = random.Random(42)
        samples = [_gamma_sample(0.1, rng) for _ in range(500)]
        assert all(s >= 0 for s in samples)
        near_zero = sum(1 for s in samples if s < 0.01)
        assert near_zero > 100  # many samples near 0 for small alpha


# ===========================================================================
# 12. Top-K Ranking
# ===========================================================================


class TestTopKRanking:
    """Test the OrchestratorAgent._compute_top_k_ranking static method."""

    def test_basic_ranking_minimize(self):
        """Minimize: lowest KPI first."""
        from app.agents.orchestrator import OrchestratorAgent
        params = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        kpis = [300.0, 100.0, 200.0]
        rounds = [1, 2, 3]
        ranking = OrchestratorAgent._compute_top_k_ranking(
            params, kpis, rounds, "minimize", k=3,
        )
        assert len(ranking) == 3
        assert ranking[0].kpi_value == 100.0
        assert ranking[0].params == {"x": 2.0}
        assert ranking[0].rank == 1

    def test_basic_ranking_maximize(self):
        """Maximize: highest KPI first."""
        from app.agents.orchestrator import OrchestratorAgent
        params = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        kpis = [300.0, 100.0, 200.0]
        rounds = [1, 2, 3]
        ranking = OrchestratorAgent._compute_top_k_ranking(
            params, kpis, rounds, "maximize", k=3,
        )
        assert ranking[0].kpi_value == 300.0
        assert ranking[0].rank == 1

    def test_replicate_grouping_and_uncertainty(self):
        """Identical params should be grouped with mean + std."""
        from app.agents.orchestrator import OrchestratorAgent
        params = [{"x": 1.0}, {"x": 1.0}, {"x": 1.0}, {"x": 2.0}]
        kpis = [100.0, 110.0, 90.0, 200.0]
        rounds = [1, 2, 3, 4]
        ranking = OrchestratorAgent._compute_top_k_ranking(
            params, kpis, rounds, "minimize", k=5,
        )
        # Two unique recipes
        assert len(ranking) == 2
        # x=1.0 group: mean=100, std=10
        x1_recipe = [r for r in ranking if r.params == {"x": 1.0}][0]
        assert abs(x1_recipe.kpi_value - 100.0) < 0.01
        assert x1_recipe.n_observations == 3
        assert x1_recipe.kpi_uncertainty is not None
        assert abs(x1_recipe.kpi_uncertainty - 10.0) < 0.01
        assert x1_recipe.round_numbers == [1, 2, 3]

    def test_single_observation_no_uncertainty(self):
        """Single observation → uncertainty is None."""
        from app.agents.orchestrator import OrchestratorAgent
        params = [{"x": 5.0}]
        kpis = [50.0]
        rounds = [1]
        ranking = OrchestratorAgent._compute_top_k_ranking(
            params, kpis, rounds, "minimize", k=3,
        )
        assert len(ranking) == 1
        assert ranking[0].kpi_uncertainty is None
        assert ranking[0].n_observations == 1

    def test_k_limits_output(self):
        """Only top-k results returned."""
        from app.agents.orchestrator import OrchestratorAgent
        params = [{"x": float(i)} for i in range(20)]
        kpis = [float(i * 10) for i in range(20)]
        rounds = list(range(1, 21))
        ranking = OrchestratorAgent._compute_top_k_ranking(
            params, kpis, rounds, "minimize", k=5,
        )
        assert len(ranking) == 5
        assert ranking[0].kpi_value == 0.0

    def test_empty_history(self):
        """No data → empty ranking."""
        from app.agents.orchestrator import OrchestratorAgent
        ranking = OrchestratorAgent._compute_top_k_ranking(
            [], [], [], "minimize", k=5,
        )
        assert len(ranking) == 0
