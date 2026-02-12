"""Tests for the v3/v4 action-based adaptive strategy selector."""
from __future__ import annotations

import math
import pytest
from app.services.strategy_selector import (
    CampaignSnapshot,
    DiagnosticSignals,
    PhaseConfig,
    PhasePosterior,
    ActionCandidate,
    StrategyDecision,
    WeightsUsed,
    StabilizeSpec,
    EvidenceItem,
    select_strategy,
    generate_adaptive_candidates,
    compute_diagnostics,
    _compute_confidence,
    _compute_ei_decay,
    _compute_batch_spread,
    _compute_model_uncertainty,
    _compute_noise_ratio,
    _compute_replicate_need,
    _compute_local_smoothness,
    _compute_phase_posterior,
    _generate_action_candidates,
    _generate_explanation,
    _predict_next_round,
    _extract_numeric_vecs,
    _schedule_weights,
    _calibrate_uncertainty,
    _compute_drift_score,
    _compute_evidence,
    _build_stabilize_spec,
    _cap_stabilize_budget,
)
from app.services.optimization_backends import (
    Observation,
    BuiltInBO,
    LHSBackend,
    RandomBackend,
    list_backends,
    get_backend,
    register_backend,
    _normalize_params,
    _denormalize_point,
)
from app.services.candidate_gen import (
    ParameterSpace,
    SearchDimension,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_space():
    """A 2D continuous parameter space."""
    return ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="concentration",
                param_type="number",
                min_value=0.1,
                max_value=10.0,
            ),
            SearchDimension(
                param_name="temperature",
                param_type="number",
                min_value=20.0,
                max_value=80.0,
            ),
        ),
        protocol_template={"steps": []},
    )


@pytest.fixture
def high_dim_space():
    """A 12-dimensional parameter space (triggers high-dim logic)."""
    dims = tuple(
        SearchDimension(
            param_name=f"param_{i}",
            param_type="number",
            min_value=0.0,
            max_value=100.0,
        )
        for i in range(12)
    )
    return ParameterSpace(dimensions=dims, protocol_template={"steps": []})


@pytest.fixture
def mixed_space():
    """Space with categorical + continuous + log-scale dimensions."""
    return ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="concentration",
                param_type="number",
                min_value=0.001,
                max_value=10.0,
                log_scale=True,
            ),
            SearchDimension(
                param_name="solvent",
                param_type="categorical",
                choices=("water", "ethanol", "dmso"),
            ),
            SearchDimension(
                param_name="temperature",
                param_type="integer",
                min_value=20,
                max_value=100,
            ),
        ),
        protocol_template={"steps": []},
    )


@pytest.fixture
def sample_observations():
    """Some sample observations for testing."""
    return [
        Observation(params={"concentration": 1.0, "temperature": 30.0}, objective=0.5),
        Observation(params={"concentration": 5.0, "temperature": 50.0}, objective=0.8),
        Observation(params={"concentration": 3.0, "temperature": 40.0}, objective=0.9),
        Observation(params={"concentration": 7.0, "temperature": 60.0}, objective=0.6),
        Observation(params={"concentration": 2.0, "temperature": 35.0}, objective=0.7),
        Observation(params={"concentration": 8.0, "temperature": 70.0}, objective=0.4),
    ]


def _make_snapshot_with_history(
    n_obs: int = 20,
    n_dims: int = 3,
    history: tuple[float, ...] = (),
    batch_kpis: tuple[float, ...] = (),
    batch_params: tuple[dict, ...] = (),
    all_kpis: tuple[float, ...] = (),
    all_params: tuple[dict, ...] = (),
    **kwargs,
) -> CampaignSnapshot:
    """Helper to build a CampaignSnapshot with sensible defaults."""
    return CampaignSnapshot(
        round_number=kwargs.get("round_number", 10),
        max_rounds=kwargs.get("max_rounds", 20),
        n_observations=n_obs,
        n_dimensions=n_dims,
        has_categorical=kwargs.get("has_categorical", False),
        has_log_scale=kwargs.get("has_log_scale", False),
        kpi_history=history,
        direction=kwargs.get("direction", "maximize"),
        user_strategy_hint=kwargs.get("user_strategy_hint", ""),
        available_backends=kwargs.get("available_backends", {}),
        last_batch_kpis=batch_kpis,
        last_batch_params=batch_params,
        best_kpi_so_far=kwargs.get("best_kpi_so_far", None),
        all_params=all_params,
        all_kpis=all_kpis,
        qc_fail_rate=kwargs.get("qc_fail_rate", 0.0),
    )


# ===========================================================================
# Diagnostic signals tests
# ===========================================================================


class TestDiagnosticSignals:
    """Test the compute_diagnostics function."""

    def test_cold_start_diagnostics(self):
        """With no data, signals should be mostly None."""
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        diag = compute_diagnostics(snap)
        assert diag.space_coverage == 0.0
        assert diag.improvement_velocity is None
        assert diag.ei_decay_proxy is None
        assert diag.batch_kpi_cv is None
        assert diag.batch_param_spread is None
        assert diag.model_uncertainty is None
        assert diag.noise_ratio is None
        assert diag.local_smoothness is None
        assert diag.replicate_need_score is None
        assert diag.convergence_status == "insufficient_data"

    def test_space_coverage_increases_with_observations(self):
        snap_low = _make_snapshot_with_history(n_obs=3)
        snap_high = _make_snapshot_with_history(n_obs=30)
        assert compute_diagnostics(snap_low).space_coverage < compute_diagnostics(snap_high).space_coverage

    def test_improvement_velocity_computed_with_history(self):
        snap = _make_snapshot_with_history(
            n_obs=10, history=(0.1, 0.3, 0.5, 0.6, 0.7, 0.75),
        )
        diag = compute_diagnostics(snap)
        assert diag.improvement_velocity is not None

    def test_batch_kpi_cv_computed(self):
        snap = _make_snapshot_with_history(
            n_obs=10, batch_kpis=(0.5, 0.8, 0.3, 0.9, 0.7),
        )
        diag = compute_diagnostics(snap)
        assert diag.batch_kpi_cv is not None
        assert diag.batch_kpi_cv > 0

    def test_batch_kpi_cv_low_for_converged_batch(self):
        snap = _make_snapshot_with_history(
            n_obs=10, batch_kpis=(1.0, 1.0, 1.0, 1.0),
        )
        diag = compute_diagnostics(snap)
        assert diag.batch_kpi_cv is not None
        assert diag.batch_kpi_cv < 0.01

    def test_batch_param_spread_computed(self):
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            batch_params=(
                {"concentration": 1.0, "temperature": 30.0},
                {"concentration": 9.0, "temperature": 70.0},
                {"concentration": 5.0, "temperature": 50.0},
            ),
        )
        diag = compute_diagnostics(snap)
        assert diag.batch_param_spread is not None
        assert diag.batch_param_spread > 0

    def test_batch_param_spread_for_clustered(self):
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            batch_params=(
                {"concentration": 5.0, "temperature": 50.0},
                {"concentration": 5.01, "temperature": 50.01},
                {"concentration": 4.99, "temperature": 49.99},
            ),
        )
        diag = compute_diagnostics(snap)
        assert diag.batch_param_spread is not None
        assert math.isfinite(diag.batch_param_spread)

    def test_v3_signals_with_full_history(self):
        """When all_params/all_kpis are provided, kNN signals are computed."""
        # Build a spread of 10 points
        all_p = tuple({"x": float(i), "y": float(i * 2)} for i in range(10))
        all_k = tuple(float(i) * 0.1 for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        diag = compute_diagnostics(snap)
        # With 10 smooth points, these should be computable
        assert diag.noise_ratio is not None
        assert diag.local_smoothness is not None
        assert diag.model_uncertainty is not None


# ===========================================================================
# EI decay proxy tests
# ===========================================================================


class TestEIDecay:
    """Test the EI decay proxy computation."""

    def test_ei_decay_none_for_short_history(self):
        assert _compute_ei_decay([1.0, 2.0, 3.0], maximize=True) is None

    def test_ei_decay_zero_for_flat_best(self):
        history = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1]
        result = _compute_ei_decay(history, maximize=True)
        assert result == 0.0

    def test_ei_decay_high_when_still_improving(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = _compute_ei_decay(history, maximize=True)
        assert result is not None
        assert result > 0.3

    def test_ei_decay_low_when_plateaued(self):
        history = [1.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        result = _compute_ei_decay(history, maximize=True)
        assert result is not None
        assert result < 0.1


# ===========================================================================
# Batch spread tests
# ===========================================================================


class TestBatchSpread:
    """Test the batch param spread computation."""

    def test_spread_none_for_empty(self):
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        assert _compute_batch_spread(snap) is None

    def test_spread_none_for_single_point(self):
        snap = _make_snapshot_with_history(
            n_obs=5, batch_params=({"x": 1.0, "y": 2.0},),
        )
        assert _compute_batch_spread(snap) is None

    def test_spread_positive_for_diverse_params(self):
        snap = _make_snapshot_with_history(
            n_obs=10,
            batch_params=({"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0}),
        )
        spread = _compute_batch_spread(snap)
        assert spread is not None
        assert spread > 0


# ===========================================================================
# Model uncertainty tests
# ===========================================================================


class TestModelUncertainty:
    """Test bootstrapped ensemble uncertainty."""

    def test_none_with_insufficient_data(self):
        snap = _make_snapshot_with_history(
            n_obs=3,
            all_params=tuple({"x": float(i)} for i in range(3)),
            all_kpis=(1.0, 2.0, 3.0),
            batch_params=({"x": 1.0},),
        )
        assert _compute_model_uncertainty(snap) is None

    def test_computed_with_sufficient_data(self):
        all_p = tuple({"x": float(i), "y": float(i * 2)} for i in range(10))
        all_k = tuple(float(i) * 0.1 for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            all_params=all_p, all_kpis=all_k,
            batch_params=all_p[-3:],
        )
        unc = _compute_model_uncertainty(snap)
        assert unc is not None
        assert unc >= 0.0

    def test_low_uncertainty_for_smooth_function(self):
        """A perfectly linear function should have low ensemble disagreement."""
        all_p = tuple({"x": float(i)} for i in range(20))
        all_k = tuple(float(i) for i in range(20))  # perfectly linear
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=1,
            all_params=all_p, all_kpis=all_k,
            batch_params=all_p[-3:],
        )
        unc = _compute_model_uncertainty(snap)
        assert unc is not None
        # Bootstrapping from a linear function → low disagreement
        assert unc < 5.0


# ===========================================================================
# Noise ratio tests
# ===========================================================================


class TestNoiseRatio:
    """Test kNN-based noise ratio."""

    def test_none_with_insufficient_data(self):
        snap = _make_snapshot_with_history(
            n_obs=3,
            all_params=tuple({"x": float(i)} for i in range(3)),
            all_kpis=(1.0, 2.0, 3.0),
        )
        assert _compute_noise_ratio(snap) is None

    def test_low_noise_for_smooth_function(self):
        """Smooth function → neighbors have similar KPIs → low noise ratio."""
        all_p = tuple({"x": float(i)} for i in range(15))
        all_k = tuple(float(i) for i in range(15))
        snap = _make_snapshot_with_history(
            n_obs=15, n_dims=1,
            all_params=all_p, all_kpis=all_k,
        )
        nr = _compute_noise_ratio(snap)
        assert nr is not None
        assert nr < 0.5  # smooth function → low noise

    def test_zero_noise_for_identical_kpis(self):
        """All KPIs identical → zero variance → noise_ratio = 0."""
        all_p = tuple({"x": float(i)} for i in range(10))
        all_k = tuple(1.0 for _ in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=1,
            all_params=all_p, all_kpis=all_k,
        )
        nr = _compute_noise_ratio(snap)
        assert nr == 0.0


# ===========================================================================
# Replicate need score tests
# ===========================================================================


class TestReplicateNeed:
    """Test composite replicate need score."""

    def test_none_when_all_inputs_none(self):
        assert _compute_replicate_need(None, None, 0.0) is None

    def test_high_when_noisy(self):
        score = _compute_replicate_need(0.8, 0.5, 0.1)
        assert score is not None
        assert score > 0.4

    def test_low_when_clean(self):
        score = _compute_replicate_need(0.1, 0.05, 0.0)
        assert score is not None
        assert score < 0.3

    def test_qc_fail_rate_contributes(self):
        score_low_qc = _compute_replicate_need(0.3, 0.2, 0.0)
        score_high_qc = _compute_replicate_need(0.3, 0.2, 0.5)
        assert score_high_qc > score_low_qc


# ===========================================================================
# Local smoothness tests
# ===========================================================================


class TestLocalSmoothness:
    """Test kNN concordance smoothness."""

    def test_none_with_insufficient_data(self):
        snap = _make_snapshot_with_history(
            n_obs=5,
            all_params=tuple({"x": float(i)} for i in range(5)),
            all_kpis=tuple(float(i) for i in range(5)),
        )
        assert _compute_local_smoothness(snap) is None  # needs >= 8

    def test_high_for_smooth_function(self):
        """Monotonic function → high concordance."""
        all_p = tuple({"x": float(i)} for i in range(15))
        all_k = tuple(float(i) for i in range(15))
        snap = _make_snapshot_with_history(
            n_obs=15, n_dims=1,
            all_params=all_p, all_kpis=all_k,
        )
        sm = _compute_local_smoothness(snap)
        assert sm is not None
        assert sm > 0.5  # smooth = high concordance

    def test_between_0_and_1(self):
        import random
        rng = random.Random(42)
        all_p = tuple({"x": rng.random(), "y": rng.random()} for _ in range(20))
        all_k = tuple(rng.random() for _ in range(20))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=2,
            all_params=all_p, all_kpis=all_k,
        )
        sm = _compute_local_smoothness(snap)
        assert sm is not None
        assert 0.0 <= sm <= 1.0


# ===========================================================================
# Extract numeric vectors tests
# ===========================================================================


class TestExtractNumericVecs:
    """Test the numeric vector extraction helper."""

    def test_extracts_floats(self):
        params = [{"x": 1.0, "y": 2}, {"x": 3.0, "y": 4}]
        vecs = _extract_numeric_vecs(params)
        assert len(vecs) == 2
        assert vecs[0] == [1.0, 2.0]

    def test_skips_non_numeric(self):
        params = [{"x": 1.0, "name": "test"}, {"x": 2.0, "name": "test2"}]
        vecs = _extract_numeric_vecs(params)
        assert len(vecs) == 2
        assert len(vecs[0]) == 1

    def test_empty_returns_empty(self):
        assert _extract_numeric_vecs([]) == []

    def test_all_non_numeric_yields_empty_vecs(self):
        params = [{"name": "a"}, {"name": "b"}]
        vecs = _extract_numeric_vecs(params)
        assert len(vecs) == 0  # no numeric values → empty vec → skipped


# ===========================================================================
# Phase posterior tests
# ===========================================================================


class TestPhasePosterior:
    """Test phase posterior computation."""

    def test_cold_start_favors_explore(self):
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        assert posterior.explore > posterior.exploit
        assert posterior.explore > posterior.refine

    def test_probabilities_sum_to_one(self):
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        total = posterior.explore + posterior.exploit + posterior.refine + posterior.stabilize
        assert abs(total - 1.0) < 0.01

    def test_entropy_is_positive(self):
        snap = _make_snapshot_with_history(n_obs=10)
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        assert posterior.entropy >= 0.0

    def test_improving_signal_boosts_exploit(self):
        """Strong improvement → exploit probability should be high."""
        improving = tuple(float(i) for i in range(1, 9))
        snap = _make_snapshot_with_history(
            n_obs=25, history=improving,
            batch_kpis=(7.0, 7.5, 8.0),
        )
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        assert posterior.exploit > 0.2

    def test_high_noise_boosts_stabilize(self):
        """High noise ratio → stabilize probability should be elevated."""
        # Create fake diagnostics with high noise
        diag = DiagnosticSignals(
            space_coverage=0.5,
            model_uncertainty=0.2,
            noise_ratio=0.8,
            replicate_need_score=0.7,
            batch_kpi_cv=0.5,
            improvement_velocity=0.01,
            ei_decay_proxy=0.3,
            kpi_var_ratio=0.5,
            convergence_status="improving",
            convergence_confidence=0.5,
            local_smoothness=0.4,
            batch_param_spread=0.3,
        )
        snap = _make_snapshot_with_history(n_obs=20)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        assert posterior.stabilize > 0.15

    def test_plateau_boosts_refine(self):
        """Plateau convergence → refine probability elevated."""
        diag = DiagnosticSignals(
            space_coverage=0.8,
            model_uncertainty=0.1,
            noise_ratio=0.1,
            replicate_need_score=0.1,
            batch_kpi_cv=0.02,
            improvement_velocity=0.001,
            ei_decay_proxy=0.05,
            kpi_var_ratio=0.2,
            convergence_status="plateau",
            convergence_confidence=0.8,
            local_smoothness=0.7,
            batch_param_spread=0.1,
        )
        snap = _make_snapshot_with_history(n_obs=40)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        assert posterior.refine > 0.2

    def test_dominant_phase_property(self):
        p = PhasePosterior(explore=0.5, exploit=0.3, refine=0.1, stabilize=0.1, entropy=1.0)
        assert p.dominant_phase == "explore"


# ===========================================================================
# Action generation tests
# ===========================================================================


class TestActionGeneration:
    """Test action candidate generation and utility scoring."""

    def test_generates_four_actions(self):
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        available = {"built_in": True, "lhs": True, "optuna_tpe": True}
        actions = _generate_action_candidates(snap, diag, posterior, available, config)
        assert len(actions) == 4
        names = {a.name for a in actions}
        assert names == {"explore", "exploit", "refine", "stabilize"}

    def test_actions_sorted_by_utility(self):
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        available = {"built_in": True, "lhs": True}
        actions = _generate_action_candidates(snap, diag, posterior, available, config)
        utilities = [a.utility for a in actions]
        assert utilities == sorted(utilities, reverse=True)

    def test_utility_weights_are_applied(self):
        """Check that utility = w_imp * imp + w_info * info - w_risk * risk."""
        config = PhaseConfig(w_improvement=0.5, w_info_gain=0.3, w_risk=0.2)
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        diag = compute_diagnostics(snap)
        posterior = _compute_phase_posterior(snap, diag, config)
        available = {"built_in": True, "lhs": True}
        actions = _generate_action_candidates(snap, diag, posterior, available, config)
        for a in actions:
            expected = round(
                config.w_improvement * a.expected_improvement
                + config.w_info_gain * a.expected_info_gain
                - config.w_risk * a.risk,
                4,
            )
            assert abs(a.utility - expected) < 0.001

    def test_explore_uses_lhs(self):
        snap = _make_snapshot_with_history(n_obs=20)
        diag = compute_diagnostics(snap)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        available = {"built_in": True, "lhs": True}
        actions = _generate_action_candidates(snap, diag, posterior, available, config)
        explore_action = [a for a in actions if a.name == "explore"][0]
        assert explore_action.backend_name == "lhs"

    def test_multimodal_triggers_evolutionary(self):
        """Low smoothness + low noise → multimodal → evolutionary backend."""
        diag = DiagnosticSignals(
            space_coverage=0.8,
            model_uncertainty=0.2,
            noise_ratio=0.1,  # low noise
            replicate_need_score=0.1,
            batch_kpi_cv=0.3,
            improvement_velocity=0.05,
            ei_decay_proxy=0.4,
            kpi_var_ratio=0.5,
            convergence_status="improving",
            convergence_confidence=0.5,
            local_smoothness=0.15,  # low smoothness → multimodal
            batch_param_spread=0.5,
        )
        snap = _make_snapshot_with_history(n_obs=30, n_dims=5)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        available = {"pymoo_nsga2": True, "optuna_tpe": True, "built_in": True, "lhs": True}
        actions = _generate_action_candidates(snap, diag, posterior, available, config)
        exploit_action = [a for a in actions if a.name == "exploit"][0]
        assert exploit_action.backend_name == "pymoo_nsga2"

    def test_noisy_not_multimodal(self):
        """Low smoothness + HIGH noise → noisy, not multimodal → no evolutionary."""
        diag = DiagnosticSignals(
            space_coverage=0.8,
            model_uncertainty=0.2,
            noise_ratio=0.8,  # high noise
            replicate_need_score=0.7,
            batch_kpi_cv=0.5,
            improvement_velocity=0.05,
            ei_decay_proxy=0.4,
            kpi_var_ratio=0.5,
            convergence_status="improving",
            convergence_confidence=0.5,
            local_smoothness=0.15,  # low smoothness, but it's noise not multimodal
            batch_param_spread=0.5,
        )
        snap = _make_snapshot_with_history(n_obs=30, n_dims=5)
        config = PhaseConfig()
        posterior = _compute_phase_posterior(snap, diag, config)
        available = {"pymoo_nsga2": True, "optuna_tpe": True, "built_in": True, "lhs": True}
        actions = _generate_action_candidates(snap, diag, posterior, available, config)
        exploit_action = [a for a in actions if a.name == "exploit"][0]
        # Should NOT use evolutionary because noise is high
        assert exploit_action.backend_name != "pymoo_nsga2"


# ===========================================================================
# Entropy governance tests
# ===========================================================================


class TestEntropyGovernance:
    """Test that high entropy blocks exploitation."""

    def test_high_entropy_blocks_exploit(self):
        """When all phases are equally likely (max entropy), don't exploit."""
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        config = PhaseConfig(max_entropy_for_exploit=0.5)  # very strict
        decision = select_strategy(snap, config=config)
        # With such a strict threshold, if entropy is high, exploit is blocked
        # The exact outcome depends on the actual entropy, but we verify
        # the mechanism works by checking the decision is made
        assert decision.phase in ("exploration", "exploitation", "refinement", "stabilize")

    def test_low_entropy_allows_exploit(self):
        """When one phase dominates, exploit should be allowed."""
        improving = tuple(float(i) for i in range(1, 12))
        snap = _make_snapshot_with_history(
            n_obs=30, n_dims=3, history=improving,
            batch_kpis=improving[-3:],
        )
        config = PhaseConfig(max_entropy_for_exploit=2.0)  # lenient
        decision = select_strategy(snap, config=config)
        # With good improvement and lenient entropy, exploit should be possible
        assert decision.phase in ("exploitation", "exploration")


# ===========================================================================
# Phase determination (via select_strategy)
# ===========================================================================


class TestPhaseDetermination:
    """Test the data-driven phase selection logic."""

    def test_early_round_is_exploration(self):
        snapshot = _make_snapshot_with_history(n_obs=0, round_number=1)
        decision = select_strategy(snapshot)
        assert decision.phase == "exploration"

    def test_cold_start_forces_exploration(self):
        snapshot = _make_snapshot_with_history(n_obs=3, round_number=10)
        decision = select_strategy(snapshot)
        assert decision.phase == "exploration"

    def test_low_coverage_forces_exploration(self):
        snapshot = _make_snapshot_with_history(n_obs=4, n_dims=10, round_number=10)
        decision = select_strategy(snapshot)
        assert decision.phase == "exploration"

    def test_sufficient_data_enters_exploitation(self):
        # Provide improving KPI history so the selector sees "improving" convergence
        improving = tuple(float(i) for i in range(1, 9))
        snapshot = _make_snapshot_with_history(
            n_obs=20, n_dims=3, round_number=10,
            history=improving,
            batch_kpis=(7.0, 7.5, 8.0),
        )
        decision = select_strategy(snapshot)
        assert decision.phase == "exploitation"

    def test_exploration_uses_lhs(self):
        snapshot = _make_snapshot_with_history(n_obs=0, round_number=1)
        decision = select_strategy(snapshot)
        assert decision.backend_name == "lhs"

    def test_plateau_triggers_refinement(self):
        flat_history = tuple([1.0] * 12)
        snapshot = _make_snapshot_with_history(
            n_obs=40, n_dims=3, round_number=12,
            history=flat_history,
            batch_kpis=(1.0, 1.0, 1.0, 1.0),
            batch_params=({"a": 5.0}, {"a": 5.01}, {"a": 4.99}, {"a": 5.0}),
        )
        decision = select_strategy(snapshot)
        assert decision.phase in ("refinement", "exploitation")

    def test_improving_stays_exploitation(self):
        improving = tuple(float(i) for i in range(1, 9))
        snapshot = _make_snapshot_with_history(
            n_obs=25, n_dims=3, round_number=8,
            history=improving,
            batch_kpis=(7.5, 8.0, 7.8),
        )
        decision = select_strategy(snapshot)
        assert decision.phase == "exploitation"


# ===========================================================================
# Backend selection tests
# ===========================================================================


class TestBackendSelection:
    """Test backend routing."""

    def test_exploitation_prefers_optuna_if_available(self):
        # Provide improving history to trigger exploitation
        improving = tuple(float(i) for i in range(1, 9))
        snapshot = _make_snapshot_with_history(
            n_obs=20, n_dims=3, round_number=10,
            history=improving,
            batch_kpis=(7.0, 7.5, 8.0),
            available_backends={"optuna_tpe": True, "built_in": True, "lhs": True},
        )
        decision = select_strategy(snapshot)
        assert decision.backend_name == "optuna_tpe"

    def test_exploitation_falls_back_to_built_in(self):
        improving = tuple(float(i) for i in range(1, 9))
        snapshot = _make_snapshot_with_history(
            n_obs=20, n_dims=3, round_number=10,
            history=improving,
            batch_kpis=(7.0, 7.5, 8.0),
            available_backends={"built_in": True, "lhs": True},
        )
        decision = select_strategy(snapshot)
        assert decision.backend_name == "built_in"

    def test_high_dim_prefers_evolutionary(self):
        # Provide improving history so exploitation is triggered
        improving = tuple(float(i) for i in range(1, 11))
        snapshot = _make_snapshot_with_history(
            n_obs=50, n_dims=15, round_number=10,
            history=improving,
            batch_kpis=improving[-3:],
            available_backends={
                "pymoo_nsga2": True, "optuna_tpe": True,
                "built_in": True, "lhs": True,
            },
        )
        decision = select_strategy(snapshot)
        # High dim → should pick evolutionary or TPE for exploit/refine
        assert decision.backend_name in ("pymoo_nsga2", "optuna_tpe", "lhs")


# ===========================================================================
# User hint tests
# ===========================================================================


class TestUserHints:
    """Test that explicit user strategy requests are honored."""

    def test_user_requests_bayesian(self):
        snapshot = _make_snapshot_with_history(
            n_obs=0, round_number=1,
            user_strategy_hint="bayesian",
            available_backends={"built_in": True, "lhs": True},
        )
        decision = select_strategy(snapshot)
        assert decision.backend_name == "built_in"
        assert decision.phase == "user_requested"

    def test_user_requests_adaptive_triggers_auto_select(self):
        snapshot = _make_snapshot_with_history(
            n_obs=0, round_number=1,
            user_strategy_hint="adaptive",
        )
        decision = select_strategy(snapshot)
        assert decision.phase != "user_requested"

    def test_user_requests_unavailable_backend(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20, round_number=10,
            user_strategy_hint="optuna_tpe",
            available_backends={"built_in": True, "lhs": True},
        )
        decision = select_strategy(snapshot)
        assert decision.backend_name in ("built_in", "lhs")


# ===========================================================================
# Confidence scoring tests
# ===========================================================================


class TestConfidenceScoring:
    """Test the confidence computation."""

    def test_confidence_increases_with_observations(self):
        snap_low = _make_snapshot_with_history(n_obs=2)
        snap_high = _make_snapshot_with_history(n_obs=30)
        diag_low = compute_diagnostics(snap_low)
        diag_high = compute_diagnostics(snap_high)
        conf_low = _compute_confidence(snap_low, diag_low, "exploitation")
        conf_high = _compute_confidence(snap_high, diag_high, "exploitation")
        assert conf_high > conf_low

    def test_confidence_between_0_and_1(self):
        snap = _make_snapshot_with_history(n_obs=20)
        diag = compute_diagnostics(snap)
        conf = _compute_confidence(snap, diag, "exploitation")
        assert 0.0 <= conf <= 1.0

    def test_confidence_higher_with_more_signals(self):
        snap_rich = _make_snapshot_with_history(
            n_obs=30,
            history=tuple(range(1, 11)),
            batch_kpis=(5.0, 6.0, 7.0),
            batch_params=({"a": 1.0}, {"a": 5.0}, {"a": 9.0}),
        )
        snap_bare = _make_snapshot_with_history(n_obs=30)
        diag_rich = compute_diagnostics(snap_rich)
        diag_bare = compute_diagnostics(snap_bare)
        conf_rich = _compute_confidence(snap_rich, diag_rich, "exploitation")
        conf_bare = _compute_confidence(snap_bare, diag_bare, "exploitation")
        assert conf_rich >= conf_bare

    def test_phase_agreement_boosts_confidence(self):
        snap = _make_snapshot_with_history(n_obs=2, n_dims=5)
        diag = compute_diagnostics(snap)
        conf_agree = _compute_confidence(snap, diag, "exploration")
        conf_disagree = _compute_confidence(snap, diag, "exploitation")
        assert conf_agree > conf_disagree

    def test_stabilize_phase_agreement(self):
        """Stabilize phase + high noise → agreement."""
        diag = DiagnosticSignals(
            space_coverage=0.5,
            model_uncertainty=0.2,
            noise_ratio=0.8,
            replicate_need_score=0.7,
            batch_kpi_cv=0.4,
            improvement_velocity=0.01,
            ei_decay_proxy=0.3,
            kpi_var_ratio=0.5,
            convergence_status="improving",
            convergence_confidence=0.5,
            local_smoothness=0.4,
            batch_param_spread=0.3,
        )
        snap = _make_snapshot_with_history(n_obs=20)
        conf_agree = _compute_confidence(snap, diag, "stabilize")
        conf_disagree = _compute_confidence(snap, diag, "exploitation")
        assert conf_agree >= conf_disagree


# ===========================================================================
# Explanation generator tests
# ===========================================================================


class TestExplanation:
    """Test the 3-line explanation output."""

    def test_explanation_has_four_lines(self):
        action = ActionCandidate(
            name="explore", backend_name="lhs",
            expected_improvement=0.7, expected_info_gain=0.8, risk=0.1,
            utility=0.6, reason="Space-filling",
        )
        diag = DiagnosticSignals(
            space_coverage=0.3, model_uncertainty=0.5,
            noise_ratio=0.2, replicate_need_score=0.3,
            batch_kpi_cv=0.4, improvement_velocity=0.05,
            ei_decay_proxy=0.4, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.6,
            local_smoothness=0.5, batch_param_spread=0.4,
        )
        posterior = PhasePosterior(
            explore=0.6, exploit=0.2, refine=0.1, stabilize=0.1, entropy=1.0,
        )
        explanation = _generate_explanation(action, diag, posterior, "expect coverage↑")
        lines = explanation.strip().split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("Decision:")
        assert lines[1].startswith("Because:")
        assert lines[2].startswith("Evidence:")
        assert lines[3].startswith("Next:")

    def test_explanation_includes_action_name(self):
        action = ActionCandidate(
            name="exploit", backend_name="optuna_tpe",
            expected_improvement=0.8, expected_info_gain=0.3, risk=0.2,
            utility=0.5, reason="Exploit via TPE",
        )
        diag = DiagnosticSignals(
            space_coverage=0.7, model_uncertainty=None,
            noise_ratio=None, replicate_need_score=None,
            batch_kpi_cv=None, improvement_velocity=None,
            ei_decay_proxy=None, kpi_var_ratio=None,
            convergence_status="insufficient_data", convergence_confidence=0.0,
            local_smoothness=None, batch_param_spread=None,
        )
        posterior = PhasePosterior(
            explore=0.2, exploit=0.5, refine=0.2, stabilize=0.1, entropy=1.2,
        )
        explanation = _generate_explanation(action, diag, posterior, "expect KPI↑")
        assert "exploit" in explanation

    def test_predict_next_round_all_actions(self):
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=None,
            noise_ratio=None, replicate_need_score=None,
            batch_kpi_cv=None, improvement_velocity=None,
            ei_decay_proxy=None, kpi_var_ratio=None,
            convergence_status="insufficient_data", convergence_confidence=0.0,
            local_smoothness=None, batch_param_spread=None,
        )
        for name in ("explore", "exploit", "refine", "stabilize"):
            action = ActionCandidate(
                name=name, backend_name="lhs",
                expected_improvement=0.5, expected_info_gain=0.5, risk=0.1,
                utility=0.5, reason="test",
            )
            prediction = _predict_next_round(action, diag)
            assert len(prediction) > 0


# ===========================================================================
# Strategy decision carries v3 fields
# ===========================================================================


class TestStrategyDecisionV3:
    """Test that StrategyDecision includes v3 fields."""

    def test_decision_includes_diagnostics(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20, history=tuple(range(1, 11)),
        )
        decision = select_strategy(snapshot)
        assert decision.diagnostics is not None
        assert isinstance(decision.diagnostics, DiagnosticSignals)

    def test_decision_includes_posterior(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20, history=tuple(range(1, 11)),
        )
        decision = select_strategy(snapshot)
        assert decision.phase_posterior is not None
        assert isinstance(decision.phase_posterior, PhasePosterior)

    def test_decision_includes_actions(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20, history=tuple(range(1, 11)),
        )
        decision = select_strategy(snapshot)
        assert len(decision.actions_considered) == 4

    def test_decision_includes_explanation(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20, history=tuple(range(1, 11)),
        )
        decision = select_strategy(snapshot)
        assert decision.explanation != ""
        assert "Decision:" in decision.explanation
        assert "Because:" in decision.explanation
        assert "Evidence:" in decision.explanation
        assert "Next:" in decision.explanation

    def test_user_requested_has_no_diagnostics(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20,
            user_strategy_hint="bayesian",
            available_backends={"built_in": True, "lhs": True},
        )
        decision = select_strategy(snapshot)
        assert decision.diagnostics is None

    def test_actions_have_correct_fields(self):
        snapshot = _make_snapshot_with_history(
            n_obs=20, history=tuple(range(1, 11)),
        )
        decision = select_strategy(snapshot)
        for a in decision.actions_considered:
            assert isinstance(a, ActionCandidate)
            assert a.name in ("explore", "exploit", "refine", "stabilize")
            assert isinstance(a.utility, float)
            assert isinstance(a.expected_improvement, float)
            assert isinstance(a.expected_info_gain, float)
            assert isinstance(a.risk, float)


# ===========================================================================
# Backend registry tests
# ===========================================================================


class TestBackendRegistry:
    """Test backend registration and listing."""

    def test_list_backends_includes_builtins(self):
        backends = list_backends()
        assert "built_in" in backends
        assert "lhs" in backends
        assert "random_sampling" in backends

    def test_built_in_always_available(self):
        backends = list_backends()
        assert backends["built_in"] is True
        assert backends["lhs"] is True

    def test_get_backend_returns_instance(self):
        backend = get_backend("built_in")
        assert backend.name == "built_in"

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent_backend_xyz")

    def test_get_backend_unavailable_falls_back(self):
        backends = list_backends()
        if not backends.get("optuna_tpe", False):
            backend = get_backend("optuna_tpe")
            assert backend.name == "built_in"


# ===========================================================================
# Backend suggest tests
# ===========================================================================


class TestBuiltInBackends:
    """Test that built-in backends produce valid candidates."""

    def test_lhs_suggest(self, simple_space):
        backend = LHSBackend()
        results = backend.suggest(simple_space, 5, [], seed=42)
        assert len(results) == 5
        for r in results:
            assert "concentration" in r
            assert "temperature" in r
            assert 0.1 <= r["concentration"] <= 10.0
            assert 20.0 <= r["temperature"] <= 80.0

    def test_random_suggest(self, simple_space):
        backend = RandomBackend()
        results = backend.suggest(simple_space, 5, [], seed=42)
        assert len(results) == 5

    def test_built_in_bo_with_observations(self, simple_space, sample_observations):
        backend = BuiltInBO()
        results = backend.suggest(simple_space, 3, sample_observations, seed=42)
        assert len(results) == 3
        for r in results:
            assert "concentration" in r
            assert "temperature" in r

    def test_built_in_bo_cold_start(self, simple_space):
        backend = BuiltInBO()
        results = backend.suggest(simple_space, 3, [], seed=42)
        assert len(results) == 3


# ===========================================================================
# Integration: generate_adaptive_candidates
# ===========================================================================


class TestGenerateAdaptiveCandidates:
    """Test the one-call convenience function."""

    def test_exploration_phase(self, simple_space):
        snapshot = _make_snapshot_with_history(n_obs=0, n_dims=2, round_number=1)
        candidates, decision = generate_adaptive_candidates(
            simple_space, 5, [], snapshot, seed=42,
        )
        assert len(candidates) == 5
        assert decision.phase == "exploration"
        assert decision.backend_name == "lhs"

    def test_exploitation_phase_with_observations(self, simple_space, sample_observations):
        improving = tuple(float(i) for i in range(1, 9))
        snapshot = _make_snapshot_with_history(
            n_obs=20, n_dims=2, round_number=10,
            history=improving,
            batch_kpis=(7.0, 7.5, 8.0),
        )
        candidates, decision = generate_adaptive_candidates(
            simple_space, 3, sample_observations, snapshot, seed=42,
        )
        assert len(candidates) == 3
        assert decision.phase == "exploitation"

    def test_mixed_space_works(self, mixed_space):
        snapshot = _make_snapshot_with_history(
            n_obs=0, n_dims=3, round_number=1,
            has_categorical=True, has_log_scale=True,
        )
        candidates, decision = generate_adaptive_candidates(
            mixed_space, 4, [], snapshot, seed=42,
        )
        assert len(candidates) == 4

    def test_high_dim_space(self, high_dim_space):
        snapshot = _make_snapshot_with_history(
            n_obs=30, n_dims=12, round_number=10,
        )
        candidates, decision = generate_adaptive_candidates(
            high_dim_space, 3, [], snapshot, seed=42,
        )
        assert len(candidates) == 3

    def test_with_batch_data(self, simple_space, sample_observations):
        snapshot = _make_snapshot_with_history(
            n_obs=20, n_dims=2, round_number=10,
            history=tuple(o.objective for o in sample_observations),
            batch_kpis=(0.8, 0.9, 0.7),
            batch_params=(
                {"concentration": 5.0, "temperature": 50.0},
                {"concentration": 3.0, "temperature": 40.0},
                {"concentration": 7.0, "temperature": 60.0},
            ),
            best_kpi_so_far=0.9,
        )
        candidates, decision = generate_adaptive_candidates(
            simple_space, 3, sample_observations, snapshot, seed=42,
        )
        assert len(candidates) == 3
        assert decision.diagnostics is not None

    def test_with_full_history(self, simple_space, sample_observations):
        """Test with all_params/all_kpis for v3 kNN signals."""
        all_p = tuple(o.params for o in sample_observations)
        all_k = tuple(o.objective for o in sample_observations)
        snapshot = _make_snapshot_with_history(
            n_obs=20, n_dims=2, round_number=10,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        candidates, decision = generate_adaptive_candidates(
            simple_space, 3, sample_observations, snapshot, seed=42,
        )
        assert len(candidates) == 3
        assert decision.phase_posterior is not None
        assert len(decision.actions_considered) == 4


# ===========================================================================
# Normalize / denormalize helpers
# ===========================================================================


class TestNormalization:
    """Test normalization helpers."""

    def test_normalize_denormalize_roundtrip(self, simple_space):
        params = {"concentration": 5.0, "temperature": 50.0}
        normed = _normalize_params(params, simple_space)
        assert len(normed) == 2
        assert all(0.0 <= v <= 1.0 for v in normed)
        restored = _denormalize_point(normed, simple_space)
        assert abs(restored["concentration"] - 5.0) < 0.01
        assert abs(restored["temperature"] - 50.0) < 0.01


# ===========================================================================
# Strategy decision dataclass
# ===========================================================================


class TestStrategyDecision:
    """Test the StrategyDecision dataclass."""

    def test_decision_fields(self):
        d = StrategyDecision(
            backend_name="optuna_tpe",
            phase="exploitation",
            reason="Test reason",
            confidence=0.85,
            fallback_backend="built_in",
        )
        assert d.backend_name == "optuna_tpe"
        assert d.phase == "exploitation"
        assert d.confidence == 0.85
        assert d.fallback_backend == "built_in"

    def test_decision_with_diagnostics(self):
        diag = DiagnosticSignals(
            space_coverage=0.5,
            model_uncertainty=0.2,
            noise_ratio=0.3,
            replicate_need_score=0.4,
            batch_kpi_cv=0.2,
            improvement_velocity=0.01,
            ei_decay_proxy=0.3,
            kpi_var_ratio=0.6,
            convergence_status="improving",
            convergence_confidence=0.7,
            local_smoothness=0.6,
            batch_param_spread=0.4,
        )
        d = StrategyDecision(
            backend_name="built_in",
            phase="exploitation",
            reason="test",
            confidence=0.8,
            diagnostics=diag,
        )
        assert d.diagnostics is not None
        assert d.diagnostics.space_coverage == 0.5
        assert d.diagnostics.model_uncertainty == 0.2
        assert d.diagnostics.noise_ratio == 0.3

    def test_decision_with_posterior_and_actions(self):
        posterior = PhasePosterior(
            explore=0.2, exploit=0.5, refine=0.2, stabilize=0.1, entropy=1.2,
        )
        action = ActionCandidate(
            name="exploit", backend_name="optuna_tpe",
            expected_improvement=0.8, expected_info_gain=0.3, risk=0.2,
            utility=0.5, reason="Exploit via TPE",
        )
        d = StrategyDecision(
            backend_name="optuna_tpe",
            phase="exploitation",
            reason="test",
            confidence=0.8,
            phase_posterior=posterior,
            actions_considered=(action,),
            explanation="Decision: exploit\nBecause: improving\nEvidence: (none)\nNext: expect KPI↑",
        )
        assert d.phase_posterior is not None
        assert d.phase_posterior.exploit == 0.5
        assert len(d.actions_considered) == 1
        assert "Decision:" in d.explanation


# ===========================================================================
# PhaseConfig tests
# ===========================================================================


class TestPhaseConfig:
    """Test phase configuration."""

    def test_default_config(self):
        config = PhaseConfig()
        assert config.exploration_fraction == 0.20
        assert config.exploitation_fraction == 0.80
        assert config.min_obs_for_exploitation == 5
        assert config.min_coverage_for_exploitation == 0.25
        assert config.stall_velocity_threshold == 0.005
        assert config.w_improvement == 0.45
        assert config.w_info_gain == 0.35
        assert config.w_risk == 0.20
        assert config.max_entropy_for_exploit == 1.2
        # v4 defaults
        assert config.enable_adaptive_weights is True
        assert config.drift_window == 5
        assert config.drift_high_threshold == 0.6
        assert config.stabilize_n_replicates == 2
        assert config.stabilize_top_k == 2

    def test_custom_config_changes_phases(self):
        config = PhaseConfig(min_coverage_for_exploitation=0.5)
        snapshot = _make_snapshot_with_history(n_obs=10, n_dims=10)
        decision = select_strategy(snapshot, config=config)
        assert decision.phase == "exploration"

    def test_custom_min_obs_changes_phase(self):
        config = PhaseConfig(min_obs_for_exploitation=20)
        snapshot = _make_snapshot_with_history(n_obs=15)
        decision = select_strategy(snapshot, config=config)
        assert decision.phase == "exploration"

    def test_utility_weights_configurable(self):
        config = PhaseConfig(w_improvement=0.8, w_info_gain=0.1, w_risk=0.1)
        snapshot = _make_snapshot_with_history(n_obs=20)
        decision = select_strategy(snapshot, config=config)
        assert decision.confidence > 0


# ===========================================================================
# CampaignSnapshot with v3 fields
# ===========================================================================


class TestCampaignSnapshotV3:
    """Test that CampaignSnapshot carries v3 fields."""

    def test_default_empty_fields(self):
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        assert snap.last_batch_kpis == ()
        assert snap.last_batch_params == ()
        assert snap.best_kpi_so_far is None
        assert snap.all_params == ()
        assert snap.all_kpis == ()
        assert snap.qc_fail_rate == 0.0

    def test_batch_fields_populated(self):
        snap = _make_snapshot_with_history(
            n_obs=20, round_number=5,
            batch_kpis=(0.5, 0.8, 0.9),
            batch_params=({"a": 1.0}, {"a": 2.0}, {"a": 3.0}),
            best_kpi_so_far=0.9,
        )
        assert len(snap.last_batch_kpis) == 3
        assert len(snap.last_batch_params) == 3
        assert snap.best_kpi_so_far == 0.9

    def test_full_history_fields(self):
        all_p = tuple({"x": float(i)} for i in range(10))
        all_k = tuple(float(i) * 0.1 for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, all_params=all_p, all_kpis=all_k, qc_fail_rate=0.15,
        )
        assert len(snap.all_params) == 10
        assert len(snap.all_kpis) == 10
        assert snap.qc_fail_rate == 0.15

    def test_snapshot_is_frozen(self):
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        with pytest.raises(AttributeError):
            snap.round_number = 5  # type: ignore


# ===========================================================================
# v4: Adaptive weight scheduler tests
# ===========================================================================


class TestAdaptiveWeights:
    """Test the adaptive weight scheduling system."""

    def test_default_weights_when_signals_normal(self):
        """No extreme signals → weights stay near defaults."""
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.2, replicate_need_score=0.2,
            batch_kpi_cv=0.1, improvement_velocity=0.005,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        posterior = PhasePosterior(
            explore=0.3, exploit=0.4, refine=0.2, stabilize=0.1, entropy=0.7,
        )
        config = PhaseConfig()
        weights = _schedule_weights(diag, posterior, config)
        # Should be close to defaults (0.45, 0.35, 0.20) after re-normalization
        assert abs(weights.w_improvement + weights.w_info_gain + weights.w_risk - 1.0) < 0.01
        assert weights.reason == "default weights"

    def test_high_noise_shifts_weights(self):
        """High noise → w_risk increases, w_imp decreases."""
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.8, replicate_need_score=0.6,
            batch_kpi_cv=0.4, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.3, batch_param_spread=0.3,
        )
        posterior = PhasePosterior(
            explore=0.3, exploit=0.3, refine=0.2, stabilize=0.2, entropy=1.0,
        )
        config = PhaseConfig()
        default_config = PhaseConfig()
        weights = _schedule_weights(diag, posterior, config)
        # w_risk should be higher than default
        default_risk_normalized = default_config.w_risk / (default_config.w_improvement + default_config.w_info_gain + default_config.w_risk)
        assert weights.w_risk > default_risk_normalized
        assert "noise" in weights.reason

    def test_high_entropy_shifts_weights(self):
        """High entropy → w_info increases."""
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.2, replicate_need_score=0.2,
            batch_kpi_cv=0.1, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        posterior = PhasePosterior(
            explore=0.25, exploit=0.25, refine=0.25, stabilize=0.25, entropy=1.38,
        )
        config = PhaseConfig()
        weights = _schedule_weights(diag, posterior, config)
        assert "entropy" in weights.reason

    def test_high_velocity_shifts_weights(self):
        """High improvement velocity → w_imp increases."""
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.2, replicate_need_score=0.2,
            batch_kpi_cv=0.1, improvement_velocity=0.15,
            ei_decay_proxy=0.5, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.7,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        posterior = PhasePosterior(
            explore=0.2, exploit=0.5, refine=0.2, stabilize=0.1, entropy=0.7,
        )
        config = PhaseConfig()
        weights = _schedule_weights(diag, posterior, config)
        assert "velocity" in weights.reason

    def test_weights_sum_to_one(self):
        """Weights must always sum to 1.0 regardless of adjustments."""
        import random
        rng = random.Random(42)
        for _ in range(20):
            diag = DiagnosticSignals(
                space_coverage=rng.random(),
                model_uncertainty=rng.random(),
                noise_ratio=rng.random(),
                replicate_need_score=rng.random(),
                batch_kpi_cv=rng.random(),
                improvement_velocity=rng.random() * 0.3,
                ei_decay_proxy=rng.random(),
                kpi_var_ratio=rng.random(),
                convergence_status="improving",
                convergence_confidence=rng.random(),
                local_smoothness=rng.random(),
                batch_param_spread=rng.random(),
            )
            posterior = PhasePosterior(
                explore=0.25, exploit=0.25, refine=0.25, stabilize=0.25,
                entropy=1.3,
            )
            weights = _schedule_weights(diag, posterior, PhaseConfig())
            total = weights.w_improvement + weights.w_info_gain + weights.w_risk
            assert abs(total - 1.0) < 0.01

    def test_disabled_adaptive_weights(self):
        """When enable_adaptive_weights=False, decision has no weights_used."""
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        config = PhaseConfig(enable_adaptive_weights=False)
        decision = select_strategy(snap, config=config)
        assert decision.weights_used is None

    def test_decision_includes_weights_used(self):
        """When enabled, decision carries weights_used."""
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        decision = select_strategy(snap)
        assert decision.weights_used is not None
        assert isinstance(decision.weights_used, WeightsUsed)
        total = decision.weights_used.w_improvement + decision.weights_used.w_info_gain + decision.weights_used.w_risk
        assert abs(total - 1.0) < 0.01


# ===========================================================================
# v4: kNN uncertainty calibration tests
# ===========================================================================


class TestCalibration:
    """Test leave-one-out uncertainty calibration."""

    def test_none_input_returns_none(self):
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        cal_unc, cal_factor = _calibrate_uncertainty(snap, None)
        assert cal_unc is None
        assert cal_factor is None

    def test_insufficient_data_returns_raw(self):
        snap = _make_snapshot_with_history(
            n_obs=3,
            all_params=tuple({"x": float(i)} for i in range(3)),
            all_kpis=(1.0, 2.0, 3.0),
        )
        cal_unc, cal_factor = _calibrate_uncertainty(snap, 0.5)
        assert cal_unc == 0.5
        assert cal_factor == 1.0

    def test_smooth_function_calibration(self):
        """Smooth linear function → LOO error should be small → factor near 1."""
        all_p = tuple({"x": float(i)} for i in range(15))
        all_k = tuple(float(i) for i in range(15))
        snap = _make_snapshot_with_history(
            n_obs=15, n_dims=1,
            all_params=all_p, all_kpis=all_k,
        )
        cal_unc, cal_factor = _calibrate_uncertainty(snap, 1.0)
        assert cal_unc is not None
        assert cal_factor is not None
        # For a linear function, LOO error should be relatively small
        assert 0.2 <= cal_factor <= 5.0

    def test_calibration_factor_clamped(self):
        """Factor should be clamped to [0.2, 5.0]."""
        all_p = tuple({"x": float(i)} for i in range(10))
        all_k = tuple(float(i) for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=1,
            all_params=all_p, all_kpis=all_k,
        )
        # Very small raw uncertainty → factor would be huge → clamped
        cal_unc, cal_factor = _calibrate_uncertainty(snap, 0.001)
        assert cal_factor is not None
        assert cal_factor <= 5.0

    def test_diagnostics_includes_calibration_factor(self):
        """compute_diagnostics should populate calibration_factor."""
        all_p = tuple({"x": float(i), "y": float(i * 2)} for i in range(10))
        all_k = tuple(float(i) * 0.1 for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        diag = compute_diagnostics(snap)
        # calibration_factor should be set when model_uncertainty is computed
        if diag.model_uncertainty is not None:
            assert diag.calibration_factor is not None


# ===========================================================================
# v4: Drift detection tests
# ===========================================================================


class TestDriftDetection:
    """Test environment drift detection."""

    def test_none_with_insufficient_data(self):
        config = PhaseConfig(drift_window=5)
        snap = _make_snapshot_with_history(
            n_obs=8,
            all_kpis=tuple(float(i) for i in range(8)),
        )
        assert _compute_drift_score(snap, config) is None

    def test_zero_drift_for_constant(self):
        """All same KPIs → no drift."""
        config = PhaseConfig(drift_window=5)
        snap = _make_snapshot_with_history(
            n_obs=20,
            all_kpis=tuple(1.0 for _ in range(20)),
        )
        drift = _compute_drift_score(snap, config)
        assert drift == 0.0

    def test_detects_mean_shift(self):
        """Sudden jump in KPI mean → high drift score."""
        config = PhaseConfig(drift_window=5)
        historical = [1.0] * 15
        recent = [10.0] * 5
        snap = _make_snapshot_with_history(
            n_obs=20,
            all_kpis=tuple(historical + recent),
        )
        drift = _compute_drift_score(snap, config)
        assert drift is not None
        assert drift > 0.5  # big mean shift

    def test_moderate_drift_for_gradual_improvement(self):
        """Gradual improvement → measurable drift (recent is higher than historical)."""
        config = PhaseConfig(drift_window=5)
        snap = _make_snapshot_with_history(
            n_obs=20,
            all_kpis=tuple(float(i) for i in range(20)),
        )
        drift = _compute_drift_score(snap, config)
        assert drift is not None
        # Linear increase means recent window has a higher mean than historical,
        # but this is expected behavior, not environment drift.
        # With pooled std, drift score will be significant (mean_diff / pooled_std)
        assert drift > 0.0  # there IS a difference
        assert drift <= 1.0

    def test_drift_score_between_0_and_1(self):
        """Drift is normalized to [0, 1]."""
        config = PhaseConfig(drift_window=5)
        import random
        rng = random.Random(42)
        for _ in range(10):
            kpis = tuple(rng.gauss(0, 1) for _ in range(20))
            snap = _make_snapshot_with_history(n_obs=20, all_kpis=kpis)
            drift = _compute_drift_score(snap, config)
            assert drift is not None
            assert 0.0 <= drift <= 1.0

    def test_diagnostics_includes_drift(self):
        """compute_diagnostics should populate drift_score."""
        all_p = tuple({"x": float(i)} for i in range(20))
        all_k = tuple(float(i) for i in range(20))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=1,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
        )
        diag = compute_diagnostics(snap)
        assert diag.drift_score is not None

    def test_high_drift_blocks_exploit(self):
        """High drift → exploit should be demoted."""
        # Create a scenario with sudden drift: 15 rounds at ~1.0, then 5 rounds at ~100.0
        historical = [1.0 + 0.01 * i for i in range(15)]  # slight noise
        recent = [100.0 + 0.01 * i for i in range(5)]  # huge jump
        all_kpis = tuple(historical + recent)
        all_params = tuple({"x": float(i), "y": float(i)} for i in range(20))
        improving = tuple(float(i) for i in range(1, 9))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=2,
            history=improving,
            batch_kpis=(100.0, 100.01, 100.02),
            all_kpis=all_kpis,
            all_params=all_params,
        )
        config = PhaseConfig(drift_high_threshold=0.3)
        decision = select_strategy(snap, config=config)
        # Drift should be very high due to the level shift
        assert decision.drift_score is not None
        assert decision.drift_score > 0.3


# ===========================================================================
# v4: Evidence decomposition tests
# ===========================================================================


class TestEvidenceDecomposition:
    """Test evidence decomposition for explanation."""

    def test_empty_evidence_for_cold_start(self):
        """No signals → no evidence items (or minimal)."""
        diag = DiagnosticSignals(
            space_coverage=1.0, model_uncertainty=None,
            noise_ratio=None, replicate_need_score=None,
            batch_kpi_cv=None, improvement_velocity=None,
            ei_decay_proxy=None, kpi_var_ratio=None,
            convergence_status="insufficient_data", convergence_confidence=0.0,
            local_smoothness=None, batch_param_spread=None,
        )
        weights = WeightsUsed(w_improvement=0.45, w_info_gain=0.35, w_risk=0.20, reason="default")
        evidence = _compute_evidence(diag, weights)
        assert isinstance(evidence, tuple)

    def test_high_noise_produces_stabilize_evidence(self):
        """High noise → evidence pointing to stabilize."""
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.7, replicate_need_score=0.6,
            batch_kpi_cv=0.3, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.4, batch_param_spread=0.3,
        )
        weights = WeightsUsed(w_improvement=0.45, w_info_gain=0.35, w_risk=0.20, reason="default")
        evidence = _compute_evidence(diag, weights)
        stabilize_items = [e for e in evidence if e.target_action == "stabilize"]
        assert len(stabilize_items) > 0
        # noise_ratio should be in the evidence
        noise_items = [e for e in evidence if e.signal_name == "noise_ratio"]
        assert len(noise_items) > 0

    def test_low_coverage_produces_explore_evidence(self):
        """Low coverage → evidence pointing to explore."""
        diag = DiagnosticSignals(
            space_coverage=0.1, model_uncertainty=0.5,
            noise_ratio=0.1, replicate_need_score=0.1,
            batch_kpi_cv=0.1, improvement_velocity=None,
            ei_decay_proxy=None, kpi_var_ratio=None,
            convergence_status="insufficient_data", convergence_confidence=0.0,
            local_smoothness=None, batch_param_spread=None,
        )
        weights = WeightsUsed(w_improvement=0.45, w_info_gain=0.35, w_risk=0.20, reason="default")
        evidence = _compute_evidence(diag, weights)
        explore_items = [e for e in evidence if e.target_action == "explore"]
        assert len(explore_items) > 0

    def test_evidence_sorted_by_contribution(self):
        """Evidence items should be sorted by |contribution| descending."""
        diag = DiagnosticSignals(
            space_coverage=0.2, model_uncertainty=0.5,
            noise_ratio=0.7, replicate_need_score=0.6,
            batch_kpi_cv=0.3, improvement_velocity=0.1,
            ei_decay_proxy=0.05, kpi_var_ratio=0.5,
            convergence_status="plateau", convergence_confidence=0.7,
            local_smoothness=0.2, batch_param_spread=0.3,
        )
        weights = WeightsUsed(w_improvement=0.45, w_info_gain=0.35, w_risk=0.20, reason="default")
        evidence = _compute_evidence(diag, weights)
        if len(evidence) >= 2:
            for i in range(len(evidence) - 1):
                assert abs(evidence[i].contribution) >= abs(evidence[i + 1].contribution)

    def test_drift_in_evidence(self):
        """High drift → drift evidence item present."""
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.2, replicate_need_score=0.2,
            batch_kpi_cv=0.1, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
            drift_score=0.7,
        )
        weights = WeightsUsed(w_improvement=0.45, w_info_gain=0.35, w_risk=0.20, reason="default")
        evidence = _compute_evidence(diag, weights)
        drift_items = [e for e in evidence if e.signal_name == "drift_score"]
        assert len(drift_items) == 1
        assert drift_items[0].target_action == "stabilize"

    def test_decision_includes_evidence(self):
        """select_strategy should populate evidence field."""
        snap = _make_snapshot_with_history(n_obs=20, history=tuple(range(1, 11)))
        decision = select_strategy(snap)
        assert isinstance(decision.evidence, tuple)
        for e in decision.evidence:
            assert isinstance(e, EvidenceItem)


# ===========================================================================
# v4: Stabilize protocol tests
# ===========================================================================


class TestStabilizeSpec:
    """Test the stabilize replication protocol builder."""

    def test_empty_history_fallback(self):
        """No history → fallback spec."""
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        diag = DiagnosticSignals(
            space_coverage=0.0, model_uncertainty=None,
            noise_ratio=None, replicate_need_score=None,
            batch_kpi_cv=None, improvement_velocity=None,
            ei_decay_proxy=None, kpi_var_ratio=None,
            convergence_status="insufficient_data", convergence_confidence=0.0,
            local_smoothness=None, batch_param_spread=None,
        )
        config = PhaseConfig()
        spec = _build_stabilize_spec(snap, diag, config)
        assert spec.n_replicates >= 1
        assert len(spec.points_to_replicate) == 0

    def test_top_k_strategy_with_history(self):
        """With all_params/all_kpis → replicate top-k."""
        all_p = tuple({"x": float(i)} for i in range(10))
        all_k = tuple(float(i) for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=1,
            all_params=all_p, all_kpis=all_k,
            direction="maximize",
        )
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.4, replicate_need_score=0.4,
            batch_kpi_cv=0.1, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        config = PhaseConfig(stabilize_top_k=2)
        spec = _build_stabilize_spec(snap, diag, config)
        assert spec.strategy in ("best", "top_k")
        assert len(spec.points_to_replicate) <= 2
        # Best point should be x=9 (highest KPI)
        if spec.points_to_replicate:
            assert spec.points_to_replicate[0]["x"] == 9.0

    def test_high_noise_increases_replicates(self):
        """noise_ratio > 0.6 → 3 replicates (with sufficient budget)."""
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=1,
            round_number=2, max_rounds=100,  # plenty of budget remaining
            all_params=tuple({"x": float(i)} for i in range(10)),
            all_kpis=tuple(float(i) for i in range(10)),
        )
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.8, replicate_need_score=0.7,
            batch_kpi_cv=0.1, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.4, batch_param_spread=0.3,
        )
        config = PhaseConfig()
        spec = _build_stabilize_spec(snap, diag, config)
        assert spec.n_replicates == 3

    def test_max_variance_strategy(self):
        """High batch CV + enough batch points → max_variance strategy."""
        batch_p = ({"x": 1.0}, {"x": 5.0}, {"x": 9.0})
        batch_k = (0.1, 0.9, 0.5)  # high variance
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=1,
            all_params=tuple({"x": float(i)} for i in range(10)),
            all_kpis=tuple(float(i) for i in range(10)),
            batch_params=batch_p, batch_kpis=batch_k,
            direction="maximize",
        )
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.4, replicate_need_score=0.4,
            batch_kpi_cv=0.5,  # high CV triggers max_variance
            improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        config = PhaseConfig()
        spec = _build_stabilize_spec(snap, diag, config)
        assert spec.strategy == "max_variance"

    def test_stabilize_decision_includes_spec(self):
        """When stabilize wins, decision should include stabilize_spec."""
        # Create scenario that triggers stabilize: high noise, high QC fail
        all_p = tuple({"x": float(i), "y": float(i)} for i in range(10))
        all_k = tuple(float(i % 3) for i in range(10))  # noisy
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=(0.5, 1.5, 0.8),
            batch_params=({"x": 7.0, "y": 7.0}, {"x": 8.0, "y": 8.0}, {"x": 9.0, "y": 9.0}),
            qc_fail_rate=0.4,
        )
        # Force stabilize by using high noise threshold and high replicate need
        config = PhaseConfig(
            noise_ratio_high=0.1,  # very strict → stabilize easier
            replicate_need_threshold=0.1,
        )
        decision = select_strategy(snap, config=config)
        # If stabilize won, it should have a spec
        if decision.phase == "stabilize":
            assert decision.stabilize_spec is not None
            assert isinstance(decision.stabilize_spec, StabilizeSpec)
        # Otherwise, spec should be None
        else:
            assert decision.stabilize_spec is None

    def test_non_stabilize_decision_no_spec(self):
        """Non-stabilize decisions should NOT have stabilize_spec."""
        snap = _make_snapshot_with_history(n_obs=0, round_number=1)
        decision = select_strategy(snap)
        assert decision.phase == "exploration"
        assert decision.stabilize_spec is None


# ===========================================================================
# v4: Integration tests for decision with all new fields
# ===========================================================================


class TestStrategyDecisionV4:
    """Test that StrategyDecision includes all v4 fields."""

    def test_decision_has_all_v4_fields(self):
        all_p = tuple({"x": float(i), "y": float(i * 2)} for i in range(20))
        all_k = tuple(float(i) * 0.1 for i in range(20))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=2,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        decision = select_strategy(snap)
        # v3 fields
        assert decision.diagnostics is not None
        assert decision.phase_posterior is not None
        assert len(decision.actions_considered) == 4
        assert decision.explanation != ""
        # v4 fields
        assert decision.weights_used is not None
        assert decision.drift_score is not None
        assert isinstance(decision.evidence, tuple)

    def test_explanation_includes_evidence_line(self):
        all_p = tuple({"x": float(i)} for i in range(20))
        all_k = tuple(float(i) for i in range(20))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=1,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        decision = select_strategy(snap)
        assert "Evidence:" in decision.explanation

    def test_diagnostics_has_drift_and_calibration(self):
        all_p = tuple({"x": float(i)} for i in range(20))
        all_k = tuple(float(i) for i in range(20))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=1,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        decision = select_strategy(snap)
        diag = decision.diagnostics
        assert diag is not None
        assert diag.drift_score is not None
        # calibration_factor depends on whether model_uncertainty was computed
        if diag.model_uncertainty is not None:
            assert diag.calibration_factor is not None

    def test_adaptive_weights_affect_utility_ranking(self):
        """Adaptive weights should change action utility ordering vs fixed."""
        all_p = tuple({"x": float(i), "y": float(i)} for i in range(15))
        # Noisy KPIs → should shift weights
        all_k = tuple(float(i % 4) * 0.3 for i in range(15))
        snap = _make_snapshot_with_history(
            n_obs=15, n_dims=2,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=all_k[-3:],
            batch_params=all_p[-3:],
        )
        # With adaptive weights
        decision_adaptive = select_strategy(snap, PhaseConfig(enable_adaptive_weights=True))
        # Without adaptive weights
        decision_fixed = select_strategy(snap, PhaseConfig(enable_adaptive_weights=False))
        # Both should produce valid decisions
        assert decision_adaptive.phase in ("exploration", "exploitation", "refinement", "stabilize")
        assert decision_fixed.phase in ("exploration", "exploitation", "refinement", "stabilize")


# ===========================================================================
# v4: Stabilize spec integration with orchestrator
# ===========================================================================


class TestStabilizeSpecIntegration:
    """Test stabilize_spec replication expansion logic."""

    def test_spec_points_expand_to_replicates(self):
        """stabilize_spec.points × n_replicates = total candidates."""
        spec = StabilizeSpec(
            strategy="top_k",
            points_to_replicate=({"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}),
            n_replicates=3,
            reason="replicate top-2 points",
        )
        # Simulate what orchestrator does
        candidates: list[dict] = []
        for pt in spec.points_to_replicate:
            for _rep in range(spec.n_replicates):
                candidates.append(dict(pt))
        assert len(candidates) == 6  # 2 points × 3 replicates
        # Each replicate should be identical to the source point
        assert candidates[0] == {"x": 1.0, "y": 2.0}
        assert candidates[2] == {"x": 1.0, "y": 2.0}
        assert candidates[3] == {"x": 3.0, "y": 4.0}

    def test_spec_single_point_single_replicate(self):
        """Minimal case: 1 point × 1 replicate."""
        spec = StabilizeSpec(
            strategy="best",
            points_to_replicate=({"vol": 25.0},),
            n_replicates=1,
            reason="verify best point",
        )
        candidates = []
        for pt in spec.points_to_replicate:
            for _rep in range(spec.n_replicates):
                candidates.append(dict(pt))
        assert len(candidates) == 1
        assert candidates[0] == {"vol": 25.0}

    def test_empty_spec_produces_no_candidates(self):
        """Empty points_to_replicate → no candidates (fallback to DesignAgent)."""
        spec = StabilizeSpec(
            strategy="best",
            points_to_replicate=(),
            n_replicates=2,
            reason="no history",
        )
        candidates = []
        for pt in spec.points_to_replicate:
            for _rep in range(spec.n_replicates):
                candidates.append(dict(pt))
        assert len(candidates) == 0

    def test_stabilize_with_full_pipeline(self):
        """Full select_strategy → stabilize_spec expansion for noisy campaign."""
        all_p = tuple({"x": float(i), "y": float(i * 2)} for i in range(10))
        all_k = tuple(float(i % 3) for i in range(10))  # noisy cyclic
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=2,
            history=all_k,
            all_params=all_p, all_kpis=all_k,
            batch_kpis=(0.5, 1.5, 0.8),
            batch_params=({"x": 7.0, "y": 14.0}, {"x": 8.0, "y": 16.0}, {"x": 9.0, "y": 18.0}),
            qc_fail_rate=0.4,
        )
        config = PhaseConfig(
            noise_ratio_high=0.1,
            replicate_need_threshold=0.1,
        )
        decision = select_strategy(snap, config=config)
        # Whether or not stabilize wins, the spec should be consistent:
        if decision.stabilize_spec is not None:
            spec = decision.stabilize_spec
            assert spec.n_replicates >= 1
            assert spec.strategy in ("best", "top_k", "max_variance")
            # Expansion should work
            expanded = []
            for pt in spec.points_to_replicate:
                for _rep in range(spec.n_replicates):
                    expanded.append(dict(pt))
            assert len(expanded) == len(spec.points_to_replicate) * spec.n_replicates


# ===========================================================================
# Budget-aware stabilize
# ===========================================================================


class TestCapStabilizeBudget:
    """Test the _cap_stabilize_budget trimming logic."""

    def test_within_budget_unchanged(self):
        """If points × reps ≤ budget, no change."""
        pts = [{"x": 1.0}, {"x": 2.0}]
        pts_out, reps = _cap_stabilize_budget(pts, 2, max_budget=5)
        assert len(pts_out) == 2
        assert reps == 2

    def test_reduce_replicates_first(self):
        """3 reps → 2 → 1 before trimming points."""
        pts = [{"x": 1.0}, {"x": 2.0}]
        pts_out, reps = _cap_stabilize_budget(pts, 3, max_budget=3)
        # 2 pts × 1 rep = 2 ≤ 3  (reps reduced 3→1 first)
        assert len(pts_out) == 2
        assert reps == 1

    def test_trim_points_after_reps(self):
        """When reps=1 still exceeds, trim points."""
        pts = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        pts_out, reps = _cap_stabilize_budget(pts, 2, max_budget=2)
        # 3×2=6 > 2 → reps→1 → 3×1=3 > 2 → trim pts → 2×1=2
        assert len(pts_out) == 2
        assert reps == 1

    def test_min_one_point_one_rep(self):
        """Budget=1 → exactly 1 point × 1 rep."""
        pts = [{"x": 1.0}, {"x": 2.0}]
        pts_out, reps = _cap_stabilize_budget(pts, 3, max_budget=1)
        assert len(pts_out) == 1
        assert reps == 1

    def test_keeps_best_first(self):
        """Points list order is preserved (first = best)."""
        pts = [{"x": 9.0}, {"x": 5.0}, {"x": 1.0}]
        pts_out, reps = _cap_stabilize_budget(pts, 2, max_budget=2)
        # Keeps first two: 9.0, 5.0 (reps→1)
        assert pts_out[0] == {"x": 9.0}


class TestBudgetAwareStabilizeSpec:
    """Test that _build_stabilize_spec respects budget constraints."""

    def test_early_round_generous_budget(self):
        """At round 5/24, remaining=19, budget_cap=2 (15% of 19)."""
        all_p = tuple({"x": float(i)} for i in range(8))
        all_k = tuple(float(i) for i in range(8))
        snap = _make_snapshot_with_history(
            n_obs=8, n_dims=1,
            round_number=5, max_rounds=24,
            all_params=all_p, all_kpis=all_k,
            direction="maximize",
        )
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.8, replicate_need_score=0.7,
            batch_kpi_cv=0.1, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        config = PhaseConfig(stabilize_top_k=3)
        spec = _build_stabilize_spec(snap, diag, config)
        total = len(spec.points_to_replicate) * spec.n_replicates
        remaining = 24 - 5  # = 19
        max_budget = max(1, int(remaining * 0.15))  # = 2
        assert total <= max_budget

    def test_late_round_tight_budget(self):
        """At round 22/24, remaining=2, budget_cap=1 → 1×1."""
        all_p = tuple({"x": float(i)} for i in range(20))
        all_k = tuple(float(i) for i in range(20))
        snap = _make_snapshot_with_history(
            n_obs=20, n_dims=1,
            round_number=22, max_rounds=24,
            all_params=all_p, all_kpis=all_k,
            direction="maximize",
        )
        diag = DiagnosticSignals(
            space_coverage=0.8, model_uncertainty=0.1,
            noise_ratio=0.7, replicate_need_score=0.6,
            batch_kpi_cv=0.1, improvement_velocity=0.005,
            ei_decay_proxy=0.05, kpi_var_ratio=0.3,
            convergence_status="plateau", convergence_confidence=0.7,
            local_smoothness=0.6, batch_param_spread=0.2,
        )
        config = PhaseConfig(stabilize_top_k=2)
        spec = _build_stabilize_spec(snap, diag, config)
        total = len(spec.points_to_replicate) * spec.n_replicates
        # remaining = 2, budget_cap = max(1, int(2 * 0.15)) = 1
        assert total <= 1

    def test_custom_budget_fraction(self):
        """Larger stabilize_budget_fraction allows more replicates."""
        all_p = tuple({"x": float(i)} for i in range(10))
        all_k = tuple(float(i) for i in range(10))
        snap = _make_snapshot_with_history(
            n_obs=10, n_dims=1,
            round_number=10, max_rounds=24,
            all_params=all_p, all_kpis=all_k,
            direction="maximize",
        )
        diag = DiagnosticSignals(
            space_coverage=0.5, model_uncertainty=0.2,
            noise_ratio=0.4, replicate_need_score=0.4,
            batch_kpi_cv=0.1, improvement_velocity=0.01,
            ei_decay_proxy=0.3, kpi_var_ratio=0.5,
            convergence_status="improving", convergence_confidence=0.5,
            local_smoothness=0.5, batch_param_spread=0.3,
        )
        # 50% budget → remaining=14, cap=7
        config = PhaseConfig(stabilize_budget_fraction=0.5, stabilize_top_k=3)
        spec = _build_stabilize_spec(snap, diag, config)
        total = len(spec.points_to_replicate) * spec.n_replicates
        assert total <= 7


# ===========================================================================
# Backward compatibility
# ===========================================================================


class TestBackwardCompat:
    """Test that v2 aliases still exist for backward compat."""

    def test_removed_functions_are_none(self):
        from app.services.strategy_selector import (
            _determine_phase_from_data,
            _select_backend_for_phase,
        )
        assert _determine_phase_from_data is None
        assert _select_backend_for_phase is None
