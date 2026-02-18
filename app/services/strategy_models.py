"""Data models for the Adaptive Strategy Selector.

All frozen dataclasses used across strategy sub-modules live here
so they can be imported without circular dependencies.

Public types:
    CampaignSnapshot, DiagnosticSignals, WeightsUsed, StabilizeSpec,
    EvidenceItem, ActionCandidate, PhasePosterior, StrategyDecision,
    PhaseConfig
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Campaign snapshot — enriched with batch-level data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignSnapshot:
    """Immutable view of campaign state for strategy selection.

    The ``last_batch_kpis`` and ``last_batch_params`` fields carry the
    results from the most recent round so the selector can react to
    *what actually happened* rather than just the round counter.

    ``all_params`` and ``all_kpis`` carry the *full* observation history
    (not just last batch) for computing kNN-based signals.  They are
    optional — if absent, local_smoothness and noise_ratio are unavailable.
    """

    round_number: int  # current round (1-based)
    max_rounds: int
    n_observations: int  # total evaluations so far
    n_dimensions: int
    has_categorical: bool  # any categorical/boolean dims?
    has_log_scale: bool  # any log-scale dims?
    kpi_history: tuple[float, ...] = ()
    direction: str = "maximize"  # "minimize" | "maximize"
    user_strategy_hint: str = ""  # user-requested strategy (can override)
    available_backends: dict[str, bool] = field(default_factory=dict)

    # --- Batch-level data from the last round ---
    last_batch_kpis: tuple[float, ...] = ()
    last_batch_params: tuple[dict[str, Any], ...] = ()
    best_kpi_so_far: float | None = None

    # --- Full observation history (for kNN signals) ---
    all_params: tuple[dict[str, Any], ...] = ()
    all_kpis: tuple[float, ...] = ()

    # --- QC data ---
    qc_fail_rate: float = 0.0  # fraction of candidates that failed QC


# ---------------------------------------------------------------------------
# Diagnostic signals — v3: three failure modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticSignals:
    """All the signals the selector uses to make a decision.

    Grouped by failure mode:
      - Epistemic: space_coverage, model_uncertainty
      - Aleatoric: noise_ratio, replicate_need_score, batch_kpi_cv
      - Saturation: improvement_velocity, ei_decay_proxy, convergence_*
      - Landscape: local_smoothness, batch_param_spread
    """

    # --- Epistemic (model doesn't know enough) ---
    space_coverage: float  # 0.0–1.0; 1.0 = well-covered
    model_uncertainty: float | None  # mean surrogate std at batch points; None if unavailable

    # --- Aleatoric (noise dominates) ---
    noise_ratio: float | None  # within-neighbour variance / between-candidate variance
    replicate_need_score: float | None  # composite: noise + batch_cv + qc_fail
    batch_kpi_cv: float | None  # CV of last batch KPIs

    # --- Saturation (true convergence) ---
    improvement_velocity: float | None  # rolling relative improvement
    ei_decay_proxy: float | None  # ratio of recent_improvement / overall_improvement
    kpi_var_ratio: float | None  # from convergence.variance_collapse
    convergence_status: str  # "improving" | "plateau" | "diverging" | "insufficient_data"
    convergence_confidence: float

    # --- Landscape shape ---
    local_smoothness: float | None  # kNN consistency; high = smooth, low = rugged/multimodal
    batch_param_spread: float | None  # mean pairwise distance of last batch params

    # --- Calibration (v4) ---
    calibration_factor: float | None = None  # LOO calibration factor for model_uncertainty
    drift_score: float | None = None  # distribution shift between recent and historical windows


# ---------------------------------------------------------------------------
# Action candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightsUsed:
    """Record of utility weights actually used (after adaptive scheduling)."""

    w_improvement: float
    w_info_gain: float
    w_risk: float
    reason: str  # why weights were adjusted


@dataclass(frozen=True)
class StabilizeSpec:
    """Concrete replication protocol for the stabilize action.

    Answers: *what* to replicate, *how many* replicates, and *why*.
    """

    strategy: str  # "best" | "top_k" | "max_variance"
    points_to_replicate: tuple[dict[str, Any], ...]  # param dicts to re-run
    n_replicates: int  # how many times to run each point (1–3)
    reason: str


@dataclass(frozen=True)
class EvidenceItem:
    """One signal's contribution to an action's utility."""

    signal_name: str  # e.g. "noise_ratio"
    signal_value: float | None
    target_action: str  # which action it pushes toward
    contribution: float  # signed contribution to utility
    description: str  # e.g. "noise_ratio=0.62 → stabilize (+0.18)"


@dataclass(frozen=True)
class ActionCandidate:
    """A candidate action the selector can recommend."""

    name: str  # "explore" | "exploit" | "refine" | "stabilize" | "expand"
    backend_name: str  # which optimization backend to use
    expected_improvement: float  # 0–1 proxy for how much KPI gain to expect
    expected_info_gain: float  # 0–1 proxy for how much uncertainty reduction
    risk: float  # 0–1 proxy for QC fail / noise / wasted round
    utility: float  # = w_improve * improvement + w_info * info_gain - w_risk * risk
    reason: str  # human-readable explanation


# ---------------------------------------------------------------------------
# Phase posterior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhasePosterior:
    """Soft probability over phases, plus entropy for governance."""

    explore: float  # P(should explore)
    exploit: float  # P(should exploit)
    refine: float  # P(should refine)
    stabilize: float  # P(should stabilize — replicate / reduce noise)
    entropy: float  # Shannon entropy; high = uncertain about what to do

    @property
    def dominant_phase(self) -> str:
        phases = {
            "explore": self.explore,
            "exploit": self.exploit,
            "refine": self.refine,
            "stabilize": self.stabilize,
        }
        return max(phases, key=phases.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Selection result — v3: carries actions + posterior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyDecision:
    """The selector's recommendation."""

    backend_name: str  # which backend to use
    phase: str  # dominant phase label for backward compat
    reason: str  # human-readable multi-line explanation
    confidence: float  # 0.0–1.0
    fallback_backend: str = "built_in"
    diagnostics: DiagnosticSignals | None = None
    phase_posterior: PhasePosterior | None = None
    actions_considered: tuple[ActionCandidate, ...] = ()
    # 3-line explanation for SSE
    explanation: str = ""
    # v4 additions
    weights_used: WeightsUsed | None = None
    drift_score: float | None = None
    evidence: tuple[EvidenceItem, ...] = ()
    stabilize_spec: StabilizeSpec | None = None


# ---------------------------------------------------------------------------
# Phase config
# ---------------------------------------------------------------------------


@dataclass
class PhaseConfig:
    """Thresholds for data-driven phase transitions."""

    # --- Epistemic thresholds ---
    min_coverage_for_exploitation: float = 0.25
    min_obs_for_exploitation: int = 5

    # --- Aleatoric thresholds ---
    noise_ratio_high: float = 0.5  # above this → noise dominates
    replicate_need_threshold: float = 0.6  # above this → should stabilize

    # --- Saturation thresholds ---
    stall_velocity_threshold: float = 0.005
    ei_decay_threshold: float = 0.10
    batch_cv_convergence: float = 0.05
    batch_spread_convergence: float = 0.15
    convergence_confidence_threshold: float = 0.6

    # --- Landscape thresholds ---
    local_smoothness_multimodal: float = 0.3  # below → rugged/multimodal
    local_smoothness_noisy: float = 0.15  # below + high noise → noisy, not multimodal

    # --- Round-based safety net ---
    exploration_fraction: float = 0.20
    exploitation_fraction: float = 0.80

    # --- Dimensionality ---
    high_dim_threshold: int = 10
    low_dim_threshold: int = 3

    # --- Utility weights ---
    w_improvement: float = 0.45
    w_info_gain: float = 0.35
    w_risk: float = 0.20

    # --- Phase entropy governance ---
    max_entropy_for_exploit: float = 1.2  # above this, don't exploit (too uncertain)

    # --- Adaptive weight scheduling (v4) ---
    enable_adaptive_weights: bool = True
    weight_noise_sensitivity: float = 0.3  # how much noise_ratio shifts weights
    weight_entropy_sensitivity: float = 0.2  # how much phase_entropy shifts weights
    weight_velocity_sensitivity: float = 0.2  # how much improvement_velocity shifts weights

    # --- Drift detection (v4) ---
    drift_window: int = 5  # recent window size for drift detection
    drift_high_threshold: float = 0.6  # above this → force stabilize/explore
    drift_exploit_penalty: float = 0.5  # multiply exploit posterior by this when drift high

    # --- Stabilize protocol (v4) ---
    stabilize_n_replicates: int = 2  # default replicates per point
    stabilize_top_k: int = 2  # how many top points to consider
    stabilize_budget_fraction: float = 0.15  # max fraction of remaining rounds for stabilization

    # --- Backend preferences ---
    exploitation_backends: tuple[str, ...] = (
        "optuna_tpe",
        "built_in",
    )
    refinement_backends: tuple[str, ...] = (
        "optuna_cmaes",
        "scipy_de",
        "built_in",
    )
    high_dim_backends: tuple[str, ...] = (
        "pymoo_nsga2",
        "optuna_tpe",
        "built_in",
    )
