"""Tests for protocol pattern library."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_patterns_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "patterns_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.services.protocol_patterns import (  # noqa: E402
    PatternParam,
    PatternStep,
    ProtocolPattern,
    build_protocol_from_pattern,
    get_pattern,
    list_patterns,
    register_pattern,
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
        conn.execute("DELETE FROM snapshot_runs")
        conn.execute("DELETE FROM qc_flags")
        conn.execute("DELETE FROM run_failure_signatures")
        conn.execute("DELETE FROM experiment_index")
        conn.execute("DELETE FROM runs")
        conn.commit()


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


class TestPatternDataclasses:
    def test_pattern_param_frozen(self):
        """PatternParam is immutable."""
        param = PatternParam(name="x", param_type="number", default=1.0)
        with pytest.raises(AttributeError):
            param.name = "y"  # type: ignore[misc]

    def test_pattern_step_frozen(self):
        """PatternStep is immutable."""
        step = PatternStep(
            name="s1",
            primitive="heat",
            params=(),
            order=1,
        )
        with pytest.raises(AttributeError):
            step.name = "s2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OER Screening built-in pattern
# ---------------------------------------------------------------------------


class TestOERScreeningPattern:
    def test_oer_pattern_registered(self):
        """OER_SCREENING pattern is available in registry."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        assert pattern.domain == "oer"

    def test_oer_has_four_steps(self):
        """OER pattern has synthesis -> deposition -> annealing -> electrochem_test."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        assert len(pattern.steps) == 4
        names = [s.name for s in pattern.steps]
        assert names == ["synthesis", "deposition", "annealing", "electrochem_test"]

    def test_oer_optimizable_params(self):
        """Optimizable params exclude safety-locked ones."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        opt = pattern.get_optimizable_params()
        names = {p.name for p in opt}
        # max_temp_c and potential_range_v are safety-locked
        assert "max_temp_c" not in names
        assert "potential_range_v" not in names
        # synthesis(4) + deposition(2) + annealing(2, excl max_temp_c) +
        # electrochem(3, excl potential_range_v) = 11
        assert len(opt) >= 8

    def test_oer_safety_locked_params(self):
        """Safety-locked params identified correctly."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        locked = pattern.get_safety_locked_params()
        names = {p.name for p in locked}
        assert "max_temp_c" in names
        assert "potential_range_v" in names
        assert len(locked) == 2


# ---------------------------------------------------------------------------
# ParameterSpace conversion
# ---------------------------------------------------------------------------


class TestParameterSpaceConversion:
    def test_to_parameter_space_returns_valid_space(self):
        """Pattern converts to ParameterSpace with correct dimensions."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        space = pattern.to_parameter_space()
        assert space.n_dims >= 8
        # Safety-locked params are NOT in the space
        dim_names = {d.param_name for d in space.dimensions}
        assert "max_temp_c" not in dim_names
        assert "potential_range_v" not in dim_names

    def test_parameter_space_has_bounds(self):
        """All numeric dimensions have min/max bounds."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        space = pattern.to_parameter_space()
        for dim in space.dimensions:
            if dim.param_type in ("number", "integer"):
                assert dim.min_value is not None, f"{dim.param_name} missing min_value"
                assert dim.max_value is not None, f"{dim.param_name} missing max_value"


# ---------------------------------------------------------------------------
# Protocol JSON generation
# ---------------------------------------------------------------------------


class TestProtocolJsonGeneration:
    def test_to_protocol_json_with_defaults(self):
        """Generates valid protocol JSON using defaults."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        proto = pattern.to_protocol_json({})
        assert "metadata" in proto
        assert "steps" in proto
        assert len(proto["steps"]) == 4
        # Verify metadata fields
        assert proto["metadata"]["pattern_id"] == "oer_screening"
        assert proto["metadata"]["domain"] == "oer"

    def test_to_protocol_json_with_overrides(self):
        """Overridden params appear in protocol JSON."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        proto = pattern.to_protocol_json({"annealing_temp_c": 400.0})
        # Find the annealing step (primitive="heat")
        annealing = [s for s in proto["steps"] if s["primitive"] == "heat"]
        assert len(annealing) == 1
        assert annealing[0]["params"]["annealing_temp_c"] == 400.0

    def test_safety_locked_params_not_overridable(self):
        """Safety-locked params use their default value even if overridden."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        # Override a safety-locked param -- to_protocol_json silently ignores it
        proto = pattern.to_protocol_json({"max_temp_c": 9999})
        annealing = [s for s in proto["steps"] if s["primitive"] == "heat"]
        assert len(annealing) == 1
        # Safety param should remain at its declared default (700.0)
        assert annealing[0]["params"]["max_temp_c"] == 700.0


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_in_range(self):
        """Params within bounds produce no errors."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        errors = pattern.validate_params({"annealing_temp_c": 350.0})
        assert len(errors) == 0

    def test_validate_out_of_range(self):
        """Params outside bounds produce errors."""
        pattern = get_pattern("oer_screening")
        assert pattern is not None
        errors = pattern.validate_params({"annealing_temp_c": 99999.0})
        assert len(errors) > 0
        assert any("above max" in e for e in errors)


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_patterns_all(self):
        """list_patterns() returns at least the built-in OER pattern."""
        patterns = list_patterns()
        assert len(patterns) >= 1
        ids = {p.id for p in patterns}
        assert "oer_screening" in ids

    def test_list_patterns_by_domain(self):
        """list_patterns(domain='oer') filters correctly."""
        patterns = list_patterns(domain="oer")
        assert len(patterns) >= 1
        assert all(p.domain == "oer" for p in patterns)

    def test_get_nonexistent_pattern(self):
        """get_pattern() returns None for unknown ID."""
        assert get_pattern("nonexistent_pattern_xyz") is None

    def test_register_custom_pattern(self):
        """A custom pattern can be registered and retrieved."""
        custom = ProtocolPattern(
            id="test_custom",
            name="Test Custom Pattern",
            domain="test",
            description="A test pattern",
            steps=(
                PatternStep(
                    name="step1",
                    primitive="noop",
                    params=(
                        PatternParam(name="p1", param_type="number", min_value=0, max_value=10, default=5),
                    ),
                    order=1,
                ),
            ),
        )
        register_pattern(custom)
        retrieved = get_pattern("test_custom")
        assert retrieved is not None
        assert retrieved.name == "Test Custom Pattern"
        assert retrieved.domain == "test"


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


class TestBuildProtocolFromPattern:
    def test_build_protocol_success(self):
        """build_protocol_from_pattern() produces valid protocol JSON."""
        proto = build_protocol_from_pattern("oer_screening", {})
        assert "steps" in proto
        assert len(proto["steps"]) == 4

    def test_build_protocol_unknown_pattern(self):
        """build_protocol_from_pattern() raises ValueError for unknown pattern."""
        with pytest.raises(ValueError, match="unknown pattern"):
            build_protocol_from_pattern("no_such_pattern", {})

    def test_build_protocol_invalid_params(self):
        """build_protocol_from_pattern() raises ValueError for out-of-range params."""
        with pytest.raises(ValueError, match="param validation failed"):
            build_protocol_from_pattern("oer_screening", {"annealing_temp_c": 99999.0})
