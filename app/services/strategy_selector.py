"""Adaptive Strategy Selector v3 — action-based optimization agent.

Replaces the v2 "label a phase then pick a backend" approach with an
**action-candidate architecture**:

1. Compute diagnostic signals (epistemic, aleatoric, saturation)
2. Generate candidate *actions* (explore / exploit / refine / stabilize / expand)
3. Score each action with expected utility = improvement + info_gain − risk
4. Govern the decision with phase_posterior + phase_entropy

Three failure modes are now first-class citizens:
  A. **Epistemic** — model doesn't know enough (high surrogate uncertainty)
  B. **Aleatoric** — noise dominates (high within-replicate variance)
  C. **Saturation** — true convergence (low uncertainty + low EI)

The selector is still stateless — all inputs come from ``CampaignSnapshot``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.services.convergence import (
    ConvergenceConfig,
    ConvergenceStatus,
    detect_convergence,
    rolling_improvement_rate,
    variance_collapse,
    _mean,
    _variance,
)
from app.services.optimization_backends import (
    BackendProtocol,
    Observation,
    get_backend,
    list_backends,
)

logger = logging.getLogger(__name__)


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
        phases = {"explore": self.explore, "exploit": self.exploit,
                  "refine": self.refine, "stabilize": self.stabilize}
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


# ---------------------------------------------------------------------------
# Compute diagnostic signals
# ---------------------------------------------------------------------------


def compute_diagnostics(
    snapshot: CampaignSnapshot,
    config: PhaseConfig | None = None,
) -> DiagnosticSignals:
    """Compute all diagnostic signals from the campaign snapshot."""
    if config is None:
        config = PhaseConfig()

    history = list(snapshot.kpi_history)
    maximize = snapshot.direction == "maximize"

    # === Epistemic signals ===

    # 1. Space coverage
    obs_per_dim = snapshot.n_observations / max(snapshot.n_dimensions, 1)
    space_coverage = min(1.0, obs_per_dim / 10.0)

    # 2. Model uncertainty — bootstrapped ensemble approximation
    raw_uncertainty = _compute_model_uncertainty(snapshot)

    # 2b. LOO calibration (v4)
    model_uncertainty, calibration_factor = _calibrate_uncertainty(
        snapshot, raw_uncertainty,
    )

    # === Aleatoric signals ===

    # 3. Noise ratio — kNN-based within-neighbor vs between-candidate variance
    noise_ratio = _compute_noise_ratio(snapshot)

    # 4. Batch KPI diversity (CV of last batch)
    batch_kpi_cv = None
    if len(snapshot.last_batch_kpis) >= 2:
        bm = _mean(list(snapshot.last_batch_kpis))
        bv = _variance(list(snapshot.last_batch_kpis))
        if abs(bm) > 1e-10:
            batch_kpi_cv = math.sqrt(max(bv, 0.0)) / abs(bm)

    # 5. Replicate need score — composite
    replicate_need_score = _compute_replicate_need(
        noise_ratio, batch_kpi_cv, snapshot.qc_fail_rate,
    )

    # === Saturation signals ===

    # 6. Improvement velocity
    improvement_velocity = None
    if len(history) >= 3:
        oriented = history if maximize else [-v for v in history]
        improvement_velocity = rolling_improvement_rate(oriented, window=5)

    # 7. EI decay proxy
    ei_decay_proxy = _compute_ei_decay(history, maximize)

    # 8. KPI variance ratio
    kpi_var_ratio = None
    if len(history) >= 5:
        kpi_var_ratio = variance_collapse(history, recent_window=min(5, len(history)))

    # 9. Convergence detector
    conv_status = "insufficient_data"
    conv_confidence = 0.0
    if len(history) >= 5:
        try:
            conv = detect_convergence(history, maximize=maximize)
            conv_status = conv.status
            conv_confidence = conv.confidence
        except Exception:
            pass

    # === Landscape signals ===

    # 10. Local smoothness — kNN consistency
    local_smoothness = _compute_local_smoothness(snapshot)

    # 11. Batch param spread
    batch_param_spread = _compute_batch_spread(snapshot)

    # === Drift detection (v4) ===
    drift_score = _compute_drift_score(snapshot, config)

    return DiagnosticSignals(
        space_coverage=space_coverage,
        model_uncertainty=model_uncertainty,
        noise_ratio=noise_ratio,
        replicate_need_score=replicate_need_score,
        batch_kpi_cv=batch_kpi_cv,
        improvement_velocity=improvement_velocity,
        ei_decay_proxy=ei_decay_proxy,
        kpi_var_ratio=kpi_var_ratio,
        convergence_status=conv_status,
        convergence_confidence=conv_confidence,
        local_smoothness=local_smoothness,
        batch_param_spread=batch_param_spread,
        calibration_factor=calibration_factor,
        drift_score=drift_score,
    )


# ---------------------------------------------------------------------------
# New signal computations
# ---------------------------------------------------------------------------


def _compute_model_uncertainty(snapshot: CampaignSnapshot) -> float | None:
    """Bootstrapped ensemble uncertainty at last-batch points.

    Builds K bootstrap subsets of all observations, fits a mean
    predictor (just mean of k-nearest), and measures disagreement
    across bootstraps at last-batch param locations.

    Returns mean std across batch points.  None if insufficient data.
    """
    all_params = list(snapshot.all_params)
    all_kpis = list(snapshot.all_kpis)
    batch_params = list(snapshot.last_batch_params)

    if len(all_params) < 5 or len(batch_params) < 1:
        return None

    # Extract numeric vectors
    all_vecs = _extract_numeric_vecs(all_params)
    batch_vecs = _extract_numeric_vecs(batch_params)
    if len(all_vecs) < 5 or len(batch_vecs) < 1:
        return None

    import random as rng_mod
    rng = rng_mod.Random(42)
    n_boot = 5
    k = min(3, len(all_vecs) - 1)
    n_feats = len(all_vecs[0])

    ensemble_preds: list[list[float]] = [[] for _ in batch_vecs]

    for _ in range(n_boot):
        # Bootstrap sample
        indices = [rng.randrange(len(all_vecs)) for _ in range(len(all_vecs))]
        boot_vecs = [all_vecs[i] for i in indices]
        boot_kpis = [all_kpis[i] for i in indices]

        # Predict at each batch point
        for bi, bvec in enumerate(batch_vecs):
            dists = []
            for j, bv in enumerate(boot_vecs):
                d = math.sqrt(sum((bvec[f] - bv[f]) ** 2 for f in range(min(n_feats, len(bvec), len(bv)))))
                dists.append((d, boot_kpis[j]))
            dists.sort(key=lambda x: x[0])
            knn_kpis = [d[1] for d in dists[:k]]
            pred = _mean(knn_kpis) if knn_kpis else 0.0
            ensemble_preds[bi].append(pred)

    # Mean std across batch points
    stds = []
    for preds in ensemble_preds:
        if len(preds) >= 2:
            m = _mean(preds)
            v = sum((p - m) ** 2 for p in preds) / len(preds)
            stds.append(math.sqrt(max(v, 0.0)))
    if not stds:
        return None
    return _mean(stds)


def _compute_noise_ratio(snapshot: CampaignSnapshot) -> float | None:
    """Noise ratio: within-neighbour variance / between-candidate variance.

    For each observation, find its k=3 nearest neighbours.  The variance
    of neighbour KPIs (for nearby params) estimates aleatoric noise.
    Compare to overall KPI variance.

    High ratio → noise dominates → should replicate, not explore further.
    """
    all_params = list(snapshot.all_params)
    all_kpis = list(snapshot.all_kpis)
    if len(all_params) < 6:
        return None

    vecs = _extract_numeric_vecs(all_params)
    if len(vecs) < 6:
        return None

    n_feats = len(vecs[0])
    k = min(3, len(vecs) - 1)

    # Overall KPI variance
    total_var = _variance(all_kpis)
    if total_var < 1e-15:
        return 0.0

    # For each point, compute variance of k-nearest-neighbour KPIs
    within_vars: list[float] = []
    for i in range(len(vecs)):
        dists = []
        for j in range(len(vecs)):
            if i == j:
                continue
            d = math.sqrt(sum((vecs[i][f] - vecs[j][f]) ** 2 for f in range(n_feats)))
            dists.append((d, all_kpis[j]))
        dists.sort(key=lambda x: x[0])
        nn_kpis = [all_kpis[i]] + [d[1] for d in dists[:k]]
        within_vars.append(_variance(nn_kpis))

    mean_within_var = _mean(within_vars)
    return mean_within_var / total_var


def _compute_replicate_need(
    noise_ratio: float | None,
    batch_kpi_cv: float | None,
    qc_fail_rate: float,
) -> float | None:
    """Composite replicate need score: higher → should stabilize.

    Weighted sum of noise_ratio, batch_kpi_cv, and QC fail rate.
    """
    components: list[float] = []
    weights: list[float] = []

    if noise_ratio is not None:
        components.append(min(1.0, noise_ratio))
        weights.append(0.4)

    if batch_kpi_cv is not None:
        # CV > 0.3 is quite noisy for optimization
        components.append(min(1.0, batch_kpi_cv / 0.3))
        weights.append(0.3)

    if qc_fail_rate > 0:
        components.append(min(1.0, qc_fail_rate / 0.3))
        weights.append(0.3)

    if not components:
        return None

    total_w = sum(weights)
    return sum(c * w for c, w in zip(components, weights)) / total_w


def _compute_local_smoothness(snapshot: CampaignSnapshot) -> float | None:
    """Local smoothness: kNN rank consistency.

    For each observation, check if nearby points (in param space) have
    similar KPI rankings.  High smoothness = uni-modal landscape.
    Low smoothness = multimodal or noisy.

    To distinguish multimodal from noisy: check noise_ratio separately.
    """
    all_params = list(snapshot.all_params)
    all_kpis = list(snapshot.all_kpis)
    if len(all_params) < 8:
        return None

    vecs = _extract_numeric_vecs(all_params)
    if len(vecs) < 8:
        return None

    n_feats = len(vecs[0])
    k = min(5, len(vecs) - 1)

    # For each point, compute rank correlation between distance and |ΔKPI|
    concordance_scores: list[float] = []
    for i in range(len(vecs)):
        neighbors = []
        for j in range(len(vecs)):
            if i == j:
                continue
            d = math.sqrt(sum((vecs[i][f] - vecs[j][f]) ** 2 for f in range(n_feats)))
            delta_kpi = abs(all_kpis[i] - all_kpis[j])
            neighbors.append((d, delta_kpi))
        neighbors.sort(key=lambda x: x[0])
        nn = neighbors[:k]

        # Concordance: close points should have small ΔKPI
        # Count concordant pairs among k neighbors
        n_conc = 0
        n_total = 0
        for a in range(len(nn)):
            for b in range(a + 1, len(nn)):
                d_a, dk_a = nn[a]
                d_b, dk_b = nn[b]
                # Concordant if closer point has smaller |ΔKPI|
                if (d_a < d_b and dk_a <= dk_b) or (d_a > d_b and dk_a >= dk_b):
                    n_conc += 1
                n_total += 1

        if n_total > 0:
            concordance_scores.append(n_conc / n_total)

    if not concordance_scores:
        return None
    return _mean(concordance_scores)


# ---------------------------------------------------------------------------
# Helper: extract numeric vectors from param dicts
# ---------------------------------------------------------------------------


def _extract_numeric_vecs(params_list: list[dict[str, Any]]) -> list[list[float]]:
    """Extract numeric values from param dicts, returning vectors."""
    vecs: list[list[float]] = []
    for p in params_list:
        vec = []
        for v in p.values():
            if isinstance(v, (int, float)):
                vec.append(float(v))
        if vec:
            vecs.append(vec)
    return vecs


# ---------------------------------------------------------------------------
# Existing helper functions (kept from v2)
# ---------------------------------------------------------------------------


def _compute_ei_decay(history: list[float], maximize: bool) -> float | None:
    """Proxy for Expected Improvement decay."""
    if len(history) < 6:
        return None

    cum_best: list[float] = []
    best = history[0]
    for v in history:
        best = max(best, v) if maximize else min(best, v)
        cum_best.append(best)

    overall_improvement = abs(cum_best[-1] - cum_best[0])
    if overall_improvement < 1e-12:
        return 0.0

    recent_window = min(5, len(cum_best) // 2)
    recent_improvement = abs(cum_best[-1] - cum_best[-recent_window])
    return recent_improvement / overall_improvement


def _compute_batch_spread(snapshot: CampaignSnapshot) -> float | None:
    """Mean pairwise L2 distance of last batch params (batch-normed)."""
    params_list = list(snapshot.last_batch_params)
    if len(params_list) < 2:
        return None

    numeric_vecs = _extract_numeric_vecs(params_list)
    if len(numeric_vecs) < 2:
        return None

    n_feats = len(numeric_vecs[0])
    mins = [min(v[j] for v in numeric_vecs) for j in range(n_feats)]
    maxs = [max(v[j] for v in numeric_vecs) for j in range(n_feats)]
    ranges = [maxs[j] - mins[j] if maxs[j] - mins[j] > 1e-12 else 1.0 for j in range(n_feats)]

    normed = [
        [(v[j] - mins[j]) / ranges[j] for j in range(n_feats)]
        for v in numeric_vecs
    ]

    total_dist = 0.0
    n_pairs = 0
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            d = math.sqrt(sum((normed[i][k] - normed[j][k]) ** 2 for k in range(n_feats)))
            total_dist += d
            n_pairs += 1

    return total_dist / max(n_pairs, 1)


# ---------------------------------------------------------------------------
# v4: Adaptive weight scheduler
# ---------------------------------------------------------------------------


def _schedule_weights(
    diag: DiagnosticSignals,
    posterior: PhasePosterior,
    config: PhaseConfig,
) -> WeightsUsed:
    """Adapt utility weights based on current signals.

    Rules:
    - High noise_ratio → ↑w_risk, ↑w_info, ↓w_imp  (protect against noise)
    - High phase_entropy → ↑w_info  (need more information to decide)
    - High improvement_velocity → ↑w_imp  (ride the wave)

    Weights are re-normalized to sum to 1.0 after adjustment.
    """
    w_imp = config.w_improvement
    w_info = config.w_info_gain
    w_risk = config.w_risk
    reasons: list[str] = []

    # --- noise_ratio adjustment ---
    nr = diag.noise_ratio
    if nr is not None and nr > 0.3:
        shift = min(0.25, (nr - 0.3) * config.weight_noise_sensitivity)
        w_risk += shift
        w_info += shift * 0.5
        w_imp -= shift
        reasons.append(f"noise={nr:.2f}→↑risk,↑info,↓imp")

    # --- phase_entropy adjustment ---
    # max entropy for 4 categories = ln(4) ≈ 1.386
    if posterior.entropy > 0.8:
        shift = min(0.15, (posterior.entropy - 0.8) * config.weight_entropy_sensitivity)
        w_info += shift
        w_imp -= shift * 0.5
        reasons.append(f"entropy={posterior.entropy:.2f}→↑info")

    # --- improvement_velocity adjustment ---
    vel = diag.improvement_velocity
    if vel is not None and vel > 0.02:
        shift = min(0.15, (vel - 0.02) * config.weight_velocity_sensitivity * 10)
        w_imp += shift
        w_info -= shift * 0.3
        reasons.append(f"velocity={vel:.3f}→↑imp")

    # --- Clamp and re-normalize ---
    w_imp = max(0.1, w_imp)
    w_info = max(0.1, w_info)
    w_risk = max(0.05, w_risk)
    total = w_imp + w_info + w_risk
    w_imp /= total
    w_info /= total
    w_risk /= total

    reason = "; ".join(reasons) if reasons else "default weights"

    return WeightsUsed(
        w_improvement=round(w_imp, 4),
        w_info_gain=round(w_info, 4),
        w_risk=round(w_risk, 4),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# v4: kNN uncertainty calibration (leave-one-out)
# ---------------------------------------------------------------------------


def _calibrate_uncertainty(
    snapshot: CampaignSnapshot,
    raw_uncertainty: float | None,
) -> tuple[float | None, float | None]:
    """Leave-one-out calibration for bootstrapped kNN uncertainty.

    For each observation, predict its KPI using k-nearest neighbors
    (excluding itself) and compute |pred - true|.  The mean absolute
    error becomes the calibration reference.

    Returns (calibrated_uncertainty, calibration_factor).
    calibration_factor = 1.0 when raw LOO error ≈ raw_uncertainty,
    > 1.0 when model is over-confident, < 1.0 when over-uncertain.
    """
    if raw_uncertainty is None:
        return None, None

    all_params = list(snapshot.all_params)
    all_kpis = list(snapshot.all_kpis)
    if len(all_params) < 6:
        return raw_uncertainty, 1.0  # not enough data to calibrate

    vecs = _extract_numeric_vecs(all_params)
    if len(vecs) < 6:
        return raw_uncertainty, 1.0

    n = len(vecs)
    n_feats = len(vecs[0])
    k = min(3, n - 2)  # k neighbors, leaving 1 out
    if k < 1:
        return raw_uncertainty, 1.0

    loo_errors: list[float] = []
    for i in range(n):
        # Find k nearest neighbors excluding i
        dists: list[tuple[float, float]] = []
        for j in range(n):
            if i == j:
                continue
            d = math.sqrt(sum(
                (vecs[i][f] - vecs[j][f]) ** 2
                for f in range(min(n_feats, len(vecs[i]), len(vecs[j])))
            ))
            dists.append((d, all_kpis[j]))
        dists.sort(key=lambda x: x[0])
        knn_kpis = [d[1] for d in dists[:k]]
        pred = _mean(knn_kpis) if knn_kpis else 0.0
        loo_errors.append(abs(pred - all_kpis[i]))

    mean_loo_error = _mean(loo_errors) if loo_errors else 0.0

    # calibration_factor: ratio of empirical error to raw uncertainty
    # > 1 means model is over-confident (underestimates uncertainty)
    # < 1 means model is over-uncertain (overestimates uncertainty)
    if raw_uncertainty > 1e-12:
        calibration_factor = mean_loo_error / raw_uncertainty
    else:
        calibration_factor = 1.0

    # Clamp to avoid extreme corrections
    calibration_factor = max(0.2, min(5.0, calibration_factor))

    calibrated = raw_uncertainty * calibration_factor
    return calibrated, round(calibration_factor, 4)


# ---------------------------------------------------------------------------
# v4: Drift detector
# ---------------------------------------------------------------------------


def _compute_drift_score(
    snapshot: CampaignSnapshot,
    config: PhaseConfig,
) -> float | None:
    """Detect environment drift by comparing recent vs historical KPI residuals.

    Computes a normalized score:
      drift_score = |mean_recent - mean_historical| / pooled_std

    This is essentially a simplified Welch's t-statistic magnitude.
    High drift → environment changed → should stabilize/explore.
    """
    all_kpis = list(snapshot.all_kpis)
    w = config.drift_window
    if len(all_kpis) < w * 2:
        return None

    recent = all_kpis[-w:]
    historical = all_kpis[:-w]

    mean_recent = _mean(recent)
    mean_hist = _mean(historical)

    var_recent = _variance(recent)
    var_hist = _variance(historical)

    # Pooled std (approximation)
    pooled_var = (var_recent + var_hist) / 2.0
    mean_diff = abs(mean_recent - mean_hist)

    if pooled_var < 1e-15:
        # Both windows have near-zero variance
        # If means differ, that's a clear level shift
        overall_var = _variance(all_kpis)
        if overall_var < 1e-15:
            # Truly constant across everything → check raw mean diff
            return 1.0 if mean_diff > 1e-10 else 0.0
        # Use overall std as denominator
        drift = mean_diff / math.sqrt(overall_var)
        return min(1.0, drift / 3.0)

    pooled_std = math.sqrt(pooled_var)
    drift = mean_diff / pooled_std

    # Normalize to 0-1 range (saturates at ~3 std deviations)
    return min(1.0, drift / 3.0)


# ---------------------------------------------------------------------------
# v4: Evidence decomposition
# ---------------------------------------------------------------------------


def _compute_evidence(
    diag: DiagnosticSignals,
    weights: WeightsUsed,
) -> tuple[EvidenceItem, ...]:
    """Decompose each signal's contribution to action utility.

    For each diagnostic signal, compute which action it pushes toward
    and by how much.  Returns sorted by |contribution| descending.
    """
    evidence: list[EvidenceItem] = []

    # --- space_coverage → explore ---
    cov_gap = max(0.0, 1.0 - diag.space_coverage)
    if cov_gap > 0.1:
        contrib = cov_gap * weights.w_info_gain
        evidence.append(EvidenceItem(
            signal_name="space_coverage",
            signal_value=diag.space_coverage,
            target_action="explore",
            contribution=round(contrib, 4),
            description=f"coverage={diag.space_coverage:.2f} → explore (+{contrib:.2f})",
        ))

    # --- model_uncertainty → explore ---
    if diag.model_uncertainty is not None and diag.model_uncertainty > 0.2:
        contrib = min(0.3, diag.model_uncertainty) * weights.w_info_gain
        evidence.append(EvidenceItem(
            signal_name="model_uncertainty",
            signal_value=diag.model_uncertainty,
            target_action="explore",
            contribution=round(contrib, 4),
            description=f"uncertainty={diag.model_uncertainty:.2f} → explore (+{contrib:.2f})",
        ))

    # --- noise_ratio → stabilize ---
    if diag.noise_ratio is not None and diag.noise_ratio > 0.3:
        contrib = diag.noise_ratio * weights.w_risk
        evidence.append(EvidenceItem(
            signal_name="noise_ratio",
            signal_value=diag.noise_ratio,
            target_action="stabilize",
            contribution=round(contrib, 4),
            description=f"noise_ratio={diag.noise_ratio:.2f} → stabilize (+{contrib:.2f})",
        ))

    # --- noise_ratio → exploit penalty ---
    if diag.noise_ratio is not None and diag.noise_ratio > 0.5:
        penalty = -diag.noise_ratio * weights.w_risk
        evidence.append(EvidenceItem(
            signal_name="noise_ratio",
            signal_value=diag.noise_ratio,
            target_action="exploit",
            contribution=round(penalty, 4),
            description=f"noise_ratio={diag.noise_ratio:.2f} → exploit ({penalty:.2f})",
        ))

    # --- improvement_velocity → exploit ---
    if diag.improvement_velocity is not None and diag.improvement_velocity > 0.01:
        contrib = min(0.3, diag.improvement_velocity * 5) * weights.w_improvement
        evidence.append(EvidenceItem(
            signal_name="improvement_velocity",
            signal_value=diag.improvement_velocity,
            target_action="exploit",
            contribution=round(contrib, 4),
            description=f"velocity={diag.improvement_velocity:.3f} → exploit (+{contrib:.2f})",
        ))

    # --- convergence plateau → refine ---
    if diag.convergence_status == "plateau" and diag.convergence_confidence > 0.4:
        contrib = diag.convergence_confidence * 0.3 * weights.w_improvement
        evidence.append(EvidenceItem(
            signal_name="convergence_plateau",
            signal_value=diag.convergence_confidence,
            target_action="refine",
            contribution=round(contrib, 4),
            description=f"plateau(conf={diag.convergence_confidence:.2f}) → refine (+{contrib:.2f})",
        ))

    # --- ei_decay → refine ---
    if diag.ei_decay_proxy is not None and diag.ei_decay_proxy < 0.1:
        contrib = (0.1 - diag.ei_decay_proxy) * 2.0 * weights.w_improvement
        evidence.append(EvidenceItem(
            signal_name="ei_decay",
            signal_value=diag.ei_decay_proxy,
            target_action="refine",
            contribution=round(contrib, 4),
            description=f"EI_decay={diag.ei_decay_proxy:.2f} → refine (+{contrib:.2f})",
        ))

    # --- local_smoothness low → explore (multimodal) ---
    if diag.local_smoothness is not None and diag.local_smoothness < 0.3:
        if diag.noise_ratio is None or diag.noise_ratio < 0.5:
            contrib = (0.3 - diag.local_smoothness) * weights.w_info_gain
            evidence.append(EvidenceItem(
                signal_name="local_smoothness",
                signal_value=diag.local_smoothness,
                target_action="explore",
                contribution=round(contrib, 4),
                description=f"smooth={diag.local_smoothness:.2f} (multimodal) → explore (+{contrib:.2f})",
            ))

    # --- drift_score → stabilize ---
    if diag.drift_score is not None and diag.drift_score > 0.3:
        contrib = diag.drift_score * weights.w_risk
        evidence.append(EvidenceItem(
            signal_name="drift_score",
            signal_value=diag.drift_score,
            target_action="stabilize",
            contribution=round(contrib, 4),
            description=f"drift={diag.drift_score:.2f} → stabilize (+{contrib:.2f})",
        ))

    # --- replicate_need → stabilize ---
    if diag.replicate_need_score is not None and diag.replicate_need_score > 0.4:
        contrib = diag.replicate_need_score * 0.5 * weights.w_risk
        evidence.append(EvidenceItem(
            signal_name="replicate_need",
            signal_value=diag.replicate_need_score,
            target_action="stabilize",
            contribution=round(contrib, 4),
            description=f"replicate_need={diag.replicate_need_score:.2f} → stabilize (+{contrib:.2f})",
        ))

    # Sort by |contribution| descending
    evidence.sort(key=lambda e: abs(e.contribution), reverse=True)
    return tuple(evidence)


# ---------------------------------------------------------------------------
# v4: Stabilize protocol builder
# ---------------------------------------------------------------------------


def _build_stabilize_spec(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    config: PhaseConfig,
) -> StabilizeSpec:
    """Build a concrete replication plan for the stabilize action.

    **Budget-aware**: total replications (n_points × n_reps) is capped at
    ``stabilize_budget_fraction`` of remaining rounds so we never blow the
    experiment budget on verification alone.

    Strategy selection:
    1. If we have a clear best point and noise is high → replicate best
    2. If noise is moderate → replicate top-k to get better estimates
    3. If batch variance is high → replicate max-variance point

    Replicate count (before budget cap):
    - noise_ratio > 0.6 → 3 replicates (high noise needs more)
    - noise_ratio > 0.3 → 2 replicates
    - else → 1 replicate (just verify)
    """
    all_params = list(snapshot.all_params)
    all_kpis = list(snapshot.all_kpis)
    batch_params = list(snapshot.last_batch_params)
    batch_kpis = list(snapshot.last_batch_kpis)
    maximize = snapshot.direction == "maximize"

    # --- Budget awareness ---
    remaining_rounds = max(0, snapshot.max_rounds - snapshot.round_number)
    max_stabilize_budget = max(
        1,
        int(remaining_rounds * config.stabilize_budget_fraction),
    )

    # Determine number of replicates (raw, before budget cap)
    nr = diag.noise_ratio
    if nr is not None and nr > 0.6:
        n_reps = 3
    elif nr is not None and nr > 0.3:
        n_reps = config.stabilize_n_replicates
    else:
        n_reps = 1

    # Determine which points to replicate
    top_k = config.stabilize_top_k

    # Check if batch has high variance → replicate max-variance point
    if (
        len(batch_params) >= 3
        and diag.batch_kpi_cv is not None
        and diag.batch_kpi_cv > 0.3
    ):
        # Find the point with the most extreme KPI (could be outlier)
        if maximize:
            best_batch_idx = max(range(len(batch_kpis)), key=lambda i: batch_kpis[i])
        else:
            best_batch_idx = min(range(len(batch_kpis)), key=lambda i: batch_kpis[i])
        # Also include the point furthest from the best (most uncertain)
        worst_batch_idx = min(range(len(batch_kpis)), key=lambda i: batch_kpis[i]) if maximize else max(range(len(batch_kpis)), key=lambda i: batch_kpis[i])

        points = [batch_params[best_batch_idx]]
        if worst_batch_idx != best_batch_idx and top_k > 1:
            points.append(batch_params[worst_batch_idx])

        # Budget cap: trim points × reps to fit budget
        points, n_reps = _cap_stabilize_budget(points, n_reps, max_stabilize_budget)

        return StabilizeSpec(
            strategy="max_variance",
            points_to_replicate=tuple(points),
            n_replicates=n_reps,
            reason=f"batch CV={diag.batch_kpi_cv:.2f} high → replicate extreme points ({n_reps}x, budget_cap={max_stabilize_budget})",
        )

    # Default: replicate global top-k points
    if all_params and all_kpis:
        paired = list(zip(all_kpis, all_params))
        paired.sort(key=lambda x: x[0], reverse=maximize)
        top_points = [p for _, p in paired[:top_k]]

        # Budget cap
        top_points, n_reps = _cap_stabilize_budget(top_points, n_reps, max_stabilize_budget)

        return StabilizeSpec(
            strategy="top_k" if len(top_points) > 1 else "best",
            points_to_replicate=tuple(top_points),
            n_replicates=n_reps,
            reason=f"replicate top-{len(top_points)} points ({n_reps}x each, budget_cap={max_stabilize_budget})",
        )

    # Fallback: nothing to replicate
    return StabilizeSpec(
        strategy="best",
        points_to_replicate=(),
        n_replicates=n_reps,
        reason="no history — will explore instead",
    )


def _cap_stabilize_budget(
    points: list[dict[str, Any]],
    n_reps: int,
    max_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    """Trim points and replicates so total runs ≤ max_budget.

    Priority: keep more distinct points over more replicates.
    Reduces replicates first, then trims points.
    """
    if len(points) * n_reps <= max_budget:
        return points, n_reps

    # Step 1: reduce replicates while keeping all points
    while n_reps > 1 and len(points) * n_reps > max_budget:
        n_reps -= 1

    # Step 2: if still over budget, trim points (keep best first)
    while len(points) > 1 and len(points) * n_reps > max_budget:
        points = points[:-1]

    # Step 3: if single point × 1 rep still > budget (shouldn't happen with max_budget≥1)
    if len(points) * n_reps > max_budget:
        n_reps = 1
        points = points[:1]

    return points, n_reps


# ---------------------------------------------------------------------------
# Phase posterior computation
# ---------------------------------------------------------------------------


def _compute_phase_posterior(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    config: PhaseConfig,
) -> PhasePosterior:
    """Compute soft probability over phases from diagnostic signals.

    Uses a simple score → softmax approach.  Each signal contributes
    evidence toward one or more phases.
    """
    scores = {"explore": 0.0, "exploit": 0.0, "refine": 0.0, "stabilize": 0.0}

    # --- Epistemic → explore ---
    if snapshot.n_observations < config.min_obs_for_exploitation:
        scores["explore"] += 3.0  # strong prior
    if diag.space_coverage < config.min_coverage_for_exploitation:
        scores["explore"] += 2.0
    if diag.model_uncertainty is not None and diag.model_uncertainty > 0.3:
        scores["explore"] += 1.5

    # --- Aleatoric → stabilize ---
    if diag.noise_ratio is not None and diag.noise_ratio > config.noise_ratio_high:
        scores["stabilize"] += 2.0
    if diag.replicate_need_score is not None and diag.replicate_need_score > config.replicate_need_threshold:
        scores["stabilize"] += 1.5
    if snapshot.qc_fail_rate > 0.2:
        scores["stabilize"] += 1.0

    # --- Saturation → refine ---
    if diag.convergence_status == "plateau" and diag.convergence_confidence > 0.5:
        scores["refine"] += diag.convergence_confidence * 2.0
    if diag.ei_decay_proxy is not None and diag.ei_decay_proxy < config.ei_decay_threshold:
        scores["refine"] += 1.5
    if diag.improvement_velocity is not None and abs(diag.improvement_velocity) < config.stall_velocity_threshold:
        scores["refine"] += 1.0
    if diag.batch_kpi_cv is not None and diag.batch_kpi_cv < config.batch_cv_convergence:
        scores["refine"] += 0.8

    # --- Improving → exploit ---
    if diag.convergence_status == "improving" and diag.convergence_confidence > 0.4:
        scores["exploit"] += diag.convergence_confidence * 2.5
    if diag.ei_decay_proxy is not None and diag.ei_decay_proxy > 0.3:
        scores["exploit"] += 1.0
    if diag.batch_kpi_cv is not None and diag.batch_kpi_cv > 0.2:
        scores["exploit"] += 0.8

    # --- Diverging → explore (reset) ---
    if diag.convergence_status == "diverging" and diag.convergence_confidence > 0.5:
        scores["explore"] += 2.5

    # Baseline: if we have enough data and nothing else is screaming, mild exploit
    if snapshot.n_observations >= config.min_obs_for_exploitation and max(scores.values()) < 1.0:
        scores["exploit"] += 1.0

    # Softmax
    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    probs = {k: v / total for k, v in exp_scores.items()}

    # Entropy
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())

    return PhasePosterior(
        explore=round(probs["explore"], 4),
        exploit=round(probs["exploit"], 4),
        refine=round(probs["refine"], 4),
        stabilize=round(probs["stabilize"], 4),
        entropy=round(entropy, 4),
    )


# ---------------------------------------------------------------------------
# Action generation + utility scoring
# ---------------------------------------------------------------------------


def _generate_action_candidates(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    posterior: PhasePosterior,
    available: dict[str, bool],
    config: PhaseConfig,
    weights: WeightsUsed | None = None,
) -> list[ActionCandidate]:
    """Generate candidate actions and score them."""
    # Use adaptive weights if provided, else config defaults
    w_imp = weights.w_improvement if weights else config.w_improvement
    w_info = weights.w_info_gain if weights else config.w_info_gain
    w_risk = weights.w_risk if weights else config.w_risk

    actions: list[ActionCandidate] = []

    # === Action 1: Explore — max uncertainty / space-filling ===
    actions.append(ActionCandidate(
        name="explore",
        backend_name="lhs",
        expected_improvement=_explore_improvement_proxy(diag),
        expected_info_gain=_explore_info_gain_proxy(diag),
        risk=_explore_risk_proxy(diag),
        utility=0.0,  # computed below
        reason="Space-filling exploration (LHS)",
    ))

    # === Action 2: Exploit — BO on current best region ===
    exploit_backend = _pick_first_available(config.exploitation_backends, available)
    # If multi-modal (smooth low + stability high), use evolutionary
    is_multimodal = (
        diag.local_smoothness is not None
        and diag.local_smoothness < config.local_smoothness_multimodal
        and (diag.noise_ratio is None or diag.noise_ratio < config.noise_ratio_high)
    )
    if is_multimodal and available.get("pymoo_nsga2", False):
        exploit_backend = "pymoo_nsga2"
        exploit_reason = f"Exploit via evolutionary (multimodal landscape, smoothness={diag.local_smoothness:.2f})"
    else:
        exploit_reason = f"Exploit via {exploit_backend}"

    actions.append(ActionCandidate(
        name="exploit",
        backend_name=exploit_backend,
        expected_improvement=_exploit_improvement_proxy(diag),
        expected_info_gain=_exploit_info_gain_proxy(diag),
        risk=_exploit_risk_proxy(diag, config),
        utility=0.0,
        reason=exploit_reason,
    ))

    # === Action 3: Refine — local search around best ===
    refine_backend = _pick_refine_backend(snapshot, diag, available, config)
    actions.append(ActionCandidate(
        name="refine",
        backend_name=refine_backend,
        expected_improvement=_refine_improvement_proxy(diag),
        expected_info_gain=_refine_info_gain_proxy(diag),
        risk=_refine_risk_proxy(diag),
        utility=0.0,
        reason=f"Local refinement via {refine_backend}",
    ))

    # === Action 4: Stabilize — replicate top candidates ===
    actions.append(ActionCandidate(
        name="stabilize",
        backend_name="built_in",  # re-evaluate same region
        expected_improvement=_stabilize_improvement_proxy(diag),
        expected_info_gain=_stabilize_info_gain_proxy(diag),
        risk=_stabilize_risk_proxy(diag),
        utility=0.0,
        reason="Stabilize: replicate near-best to reduce noise",
    ))

    # === Compute utilities with (possibly adaptive) weights ===
    scored: list[ActionCandidate] = []
    for a in actions:
        utility = (
            w_imp * a.expected_improvement
            + w_info * a.expected_info_gain
            - w_risk * a.risk
        )
        scored.append(ActionCandidate(
            name=a.name,
            backend_name=a.backend_name,
            expected_improvement=round(a.expected_improvement, 4),
            expected_info_gain=round(a.expected_info_gain, 4),
            risk=round(a.risk, 4),
            utility=round(utility, 4),
            reason=a.reason,
        ))

    return sorted(scored, key=lambda a: a.utility, reverse=True)


# ---------------------------------------------------------------------------
# Utility proxy functions
# ---------------------------------------------------------------------------

# -- Explore --

def _explore_improvement_proxy(diag: DiagnosticSignals) -> float:
    """Exploration improves most when coverage is low."""
    return max(0.0, 1.0 - diag.space_coverage)


def _explore_info_gain_proxy(diag: DiagnosticSignals) -> float:
    """Exploration always gains info, especially with low coverage."""
    base = max(0.3, 1.0 - diag.space_coverage)
    if diag.model_uncertainty is not None and diag.model_uncertainty > 0.2:
        base = min(1.0, base + 0.2)
    return base


def _explore_risk_proxy(diag: DiagnosticSignals) -> float:
    """Exploration risk is low (space-filling is always valid)."""
    risk = 0.1
    if diag.noise_ratio is not None and diag.noise_ratio > 0.5:
        risk += 0.2  # high noise makes any strategy risky
    return min(1.0, risk)


# -- Exploit --

def _exploit_improvement_proxy(diag: DiagnosticSignals) -> float:
    """Exploitation improves when model is good and still improving."""
    base = 0.5
    if diag.convergence_status == "improving":
        base += 0.3 * diag.convergence_confidence
    if diag.ei_decay_proxy is not None and diag.ei_decay_proxy > 0.3:
        base += 0.2
    if diag.improvement_velocity is not None and diag.improvement_velocity > 0.01:
        base += 0.1
    return min(1.0, base)


def _exploit_info_gain_proxy(diag: DiagnosticSignals) -> float:
    """Exploitation gains moderate info by probing the best region."""
    return 0.3


def _exploit_risk_proxy(diag: DiagnosticSignals, config: PhaseConfig) -> float:
    """Exploitation is risky when model is uncertain or noisy."""
    risk = 0.2
    if diag.model_uncertainty is not None and diag.model_uncertainty > 0.3:
        risk += 0.3
    if diag.noise_ratio is not None and diag.noise_ratio > config.noise_ratio_high:
        risk += 0.3
    return min(1.0, risk)


# -- Refine --

def _refine_improvement_proxy(diag: DiagnosticSignals) -> float:
    """Refinement helps when we're near optimum but haven't squeezed it."""
    base = 0.3
    if diag.convergence_status == "plateau":
        base += 0.3  # plateau = worth trying a different method
    if diag.ei_decay_proxy is not None and diag.ei_decay_proxy < 0.1:
        base += 0.2  # EI exhausted → refinement method may find more
    return min(1.0, base)


def _refine_info_gain_proxy(diag: DiagnosticSignals) -> float:
    """Refinement gives modest info gain."""
    return 0.2


def _refine_risk_proxy(diag: DiagnosticSignals) -> float:
    """Refinement risk is moderate — might over-fit to local optimum."""
    risk = 0.3
    if diag.noise_ratio is not None and diag.noise_ratio > 0.5:
        risk += 0.2
    return min(1.0, risk)


# -- Stabilize --

def _stabilize_improvement_proxy(diag: DiagnosticSignals) -> float:
    """Stabilization doesn't directly improve KPI much."""
    return 0.1


def _stabilize_info_gain_proxy(diag: DiagnosticSignals) -> float:
    """Stabilization gains high info when noise is high."""
    base = 0.2
    if diag.noise_ratio is not None and diag.noise_ratio > 0.4:
        base += 0.5
    if diag.replicate_need_score is not None and diag.replicate_need_score > 0.5:
        base += 0.3
    return min(1.0, base)


def _stabilize_risk_proxy(diag: DiagnosticSignals) -> float:
    """Stabilization has low risk — it's a safe play."""
    return 0.1


# ---------------------------------------------------------------------------
# Backend selection helpers
# ---------------------------------------------------------------------------


def _pick_first_available(
    preference: tuple[str, ...], available: dict[str, bool],
) -> str:
    for b in preference:
        if available.get(b, False):
            return b
    return "built_in"


def _pick_refine_backend(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    available: dict[str, bool],
    config: PhaseConfig,
) -> str:
    """Pick refinement backend considering dimensionality + batch data."""
    # High-dim → evolutionary
    if snapshot.n_dimensions >= config.high_dim_threshold:
        return _pick_first_available(config.high_dim_backends, available)

    # Low-dim → CMA-ES
    if snapshot.n_dimensions <= config.low_dim_threshold:
        for b in ("optuna_cmaes", "scipy_de", "built_in"):
            if available.get(b, False):
                return b

    # High spread in refinement → DE
    if (
        diag.batch_param_spread is not None
        and diag.batch_param_spread > 0.4
        and available.get("scipy_de", False)
    ):
        return "scipy_de"

    return _pick_first_available(config.refinement_backends, available)


# ---------------------------------------------------------------------------
# Explanation generator
# ---------------------------------------------------------------------------


def _generate_explanation(
    decision_action: ActionCandidate,
    diag: DiagnosticSignals,
    posterior: PhasePosterior,
    next_expectation: str,
    evidence: tuple[EvidenceItem, ...] = (),
) -> str:
    """Generate 3+ line explanation for SSE display.

    Line 1: Decision (what + confidence)
    Line 2: Because (which signals triggered)
    Line 3: Evidence pointers (top signal→action contributions)
    Line 4: Next (what we expect to see next round)
    """
    # Line 1: Decision
    line1 = f"Decision: {decision_action.name} (P={getattr(posterior, decision_action.name, 0):.2f}), backend={decision_action.backend_name}"

    # Line 2: Because — collect active signals
    because_parts: list[str] = []
    if diag.convergence_status != "insufficient_data":
        because_parts.append(f"{diag.convergence_status}(conf={diag.convergence_confidence:.2f})")
    if diag.ei_decay_proxy is not None:
        label = "EI_low" if diag.ei_decay_proxy < 0.1 else "EI_ok"
        because_parts.append(f"{label}={diag.ei_decay_proxy:.2f}")
    if diag.noise_ratio is not None:
        because_parts.append(f"noise={diag.noise_ratio:.2f}")
    if diag.local_smoothness is not None:
        because_parts.append(f"smooth={diag.local_smoothness:.2f}")
    if diag.model_uncertainty is not None:
        because_parts.append(f"unc={diag.model_uncertainty:.2f}")
    if diag.drift_score is not None and diag.drift_score > 0.2:
        because_parts.append(f"drift={diag.drift_score:.2f}")
    line2 = "Because: " + " + ".join(because_parts) if because_parts else "Because: insufficient signals"

    # Line 3: Evidence pointers (top 3)
    if evidence:
        top_ev = evidence[:3]
        ev_parts = [e.description for e in top_ev]
        line3 = "Evidence: " + " | ".join(ev_parts)
    else:
        line3 = "Evidence: (no signal decomposition available)"

    # Line 4: Next
    line4 = f"Next: {next_expectation}"

    return f"{line1}\n{line2}\n{line3}\n{line4}"


def _predict_next_round(
    decision_action: ActionCandidate,
    diag: DiagnosticSignals,
) -> str:
    """Predict what we expect to see next round."""
    if decision_action.name == "explore":
        return "expect coverage↑, uncertainty↓; else stuck in unexplored region"
    if decision_action.name == "exploit":
        return "expect KPI↑, EI maintains; else switch to refine"
    if decision_action.name == "refine":
        return "expect local improvement or confirm optimum; else expand search"
    if decision_action.name == "stabilize":
        return "expect noise↓, confidence↑; then resume optimization"
    return "monitoring for improvement"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _compute_confidence(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    phase: str,
) -> float:
    """Confidence is higher when diagnostic signals agree and are strong."""
    signal_count = sum(1 for s in [
        diag.improvement_velocity,
        diag.ei_decay_proxy,
        diag.batch_kpi_cv,
        diag.batch_param_spread,
        diag.kpi_var_ratio,
        diag.model_uncertainty,
        diag.noise_ratio,
        diag.local_smoothness,
    ] if s is not None)
    signal_richness = min(1.0, signal_count / 8.0)

    obs_conf = min(1.0, snapshot.n_observations / 20.0)

    conv_conf = diag.convergence_confidence if diag.convergence_status != "insufficient_data" else 0.0

    phase_agree = 0.5
    if phase in ("exploit", "exploitation") and diag.convergence_status == "improving":
        phase_agree = 0.9
    elif phase in ("refine", "refinement") and diag.convergence_status == "plateau":
        phase_agree = 0.9
    elif phase in ("explore", "exploration") and diag.space_coverage < 0.3:
        phase_agree = 0.9
    elif phase == "stabilize" and diag.noise_ratio is not None and diag.noise_ratio > 0.5:
        phase_agree = 0.9

    weighted = (
        obs_conf * 0.15
        + signal_richness * 0.25
        + conv_conf * 0.25
        + phase_agree * 0.35
    )
    return round(max(0.0, min(1.0, weighted)), 3)


# ---------------------------------------------------------------------------
# Core selector — v3: action-based
# ---------------------------------------------------------------------------


def select_strategy(
    snapshot: CampaignSnapshot,
    config: PhaseConfig | None = None,
) -> StrategyDecision:
    """Select the best optimization strategy using action-candidate ranking.

    Decision flow:
    1. User override → honor it
    2. Compute diagnostic signals (epistemic / aleatoric / saturation / drift)
    3. Compute phase posterior (soft probabilities + entropy)
    4. Adaptive weight scheduling based on signals
    5. Generate candidate actions with adaptive utility scores
    6. Govern: entropy gate + drift gate
    7. Evidence decomposition
    8. Build stabilize spec if needed
    9. Generate explanation with evidence pointers
    """
    if config is None:
        config = PhaseConfig()

    available = snapshot.available_backends or list_backends()

    # ----- User override -----
    if snapshot.user_strategy_hint:
        return _handle_user_hint(snapshot, available, config)

    # ----- Compute diagnostics (now includes calibration + drift) -----
    diag = compute_diagnostics(snapshot, config)

    # ----- Phase posterior -----
    posterior = _compute_phase_posterior(snapshot, diag, config)

    # ----- Adaptive weight scheduling (v4) -----
    weights: WeightsUsed | None = None
    if config.enable_adaptive_weights:
        weights = _schedule_weights(diag, posterior, config)

    # ----- Generate action candidates with adaptive weights -----
    actions = _generate_action_candidates(
        snapshot, diag, posterior, available, config, weights=weights,
    )

    # ----- Governance: if entropy too high, block exploit -----
    if posterior.entropy > config.max_entropy_for_exploit:
        governed_actions = [a for a in actions if a.name != "exploit"]
        if governed_actions:
            actions = governed_actions + [a for a in actions if a.name == "exploit"]

    # ----- Governance: if drift high, block exploit + boost stabilize/explore -----
    if diag.drift_score is not None and diag.drift_score > config.drift_high_threshold:
        # Demote exploit, promote stabilize/explore
        governed_actions = [a for a in actions if a.name != "exploit"]
        if governed_actions:
            actions = governed_actions + [a for a in actions if a.name == "exploit"]
        logger.info(
            "Drift governance: drift_score=%.2f > %.2f, demoting exploit",
            diag.drift_score, config.drift_high_threshold,
        )

    # ----- Pick best action -----
    best_action = actions[0] if actions else ActionCandidate(
        name="explore", backend_name="lhs",
        expected_improvement=0.5, expected_info_gain=0.5, risk=0.1,
        utility=0.5, reason="Fallback explore",
    )

    # Map action name to phase label for backward compat
    phase_map = {
        "explore": "exploration",
        "exploit": "exploitation",
        "refine": "refinement",
        "stabilize": "stabilize",
    }
    phase = phase_map.get(best_action.name, best_action.name)

    confidence = _compute_confidence(snapshot, diag, phase)

    backend = best_action.backend_name
    fallback = "built_in" if backend != "built_in" else "lhs"

    # ----- Evidence decomposition (v4) -----
    eff_weights = weights or WeightsUsed(
        w_improvement=config.w_improvement,
        w_info_gain=config.w_info_gain,
        w_risk=config.w_risk,
        reason="default weights",
    )
    evidence = _compute_evidence(diag, eff_weights)

    # ----- Stabilize spec (v4) -----
    stabilize_spec = None
    if best_action.name == "stabilize":
        stabilize_spec = _build_stabilize_spec(snapshot, diag, config)

    # ----- Explanation with evidence pointers -----
    next_expect = _predict_next_round(best_action, diag)
    explanation = _generate_explanation(
        best_action, diag, posterior, next_expect, evidence=evidence,
    )

    reason = f"{best_action.reason} (utility={best_action.utility:.3f}, P({best_action.name})={getattr(posterior, best_action.name, 0):.2f})"

    logger.info(
        "Strategy v4 [round %d/%d, obs=%d]: action=%s, backend=%s, "
        "utility=%.3f | weights=[imp=%.2f info=%.2f risk=%.2f] | "
        "posterior=[E=%.2f X=%.2f R=%.2f S=%.2f] H=%.2f | "
        "coverage=%.2f, noise=%s, smooth=%s, unc=%s, drift=%s, conv=%s(%.2f)",
        snapshot.round_number, snapshot.max_rounds, snapshot.n_observations,
        best_action.name, backend, best_action.utility,
        eff_weights.w_improvement, eff_weights.w_info_gain, eff_weights.w_risk,
        posterior.explore, posterior.exploit, posterior.refine, posterior.stabilize,
        posterior.entropy,
        diag.space_coverage,
        f"{diag.noise_ratio:.3f}" if diag.noise_ratio is not None else "N/A",
        f"{diag.local_smoothness:.3f}" if diag.local_smoothness is not None else "N/A",
        f"{diag.model_uncertainty:.3f}" if diag.model_uncertainty is not None else "N/A",
        f"{diag.drift_score:.3f}" if diag.drift_score is not None else "N/A",
        diag.convergence_status, diag.convergence_confidence,
    )

    return StrategyDecision(
        backend_name=backend,
        phase=phase,
        reason=reason,
        confidence=confidence,
        fallback_backend=fallback,
        diagnostics=diag,
        phase_posterior=posterior,
        actions_considered=tuple(actions),
        explanation=explanation,
        weights_used=weights,
        drift_score=diag.drift_score,
        evidence=evidence,
        stabilize_spec=stabilize_spec,
    )


# ---------------------------------------------------------------------------
# User hint handling
# ---------------------------------------------------------------------------


def _handle_user_hint(
    snapshot: CampaignSnapshot,
    available: dict[str, bool],
    config: PhaseConfig,
) -> StrategyDecision:
    """Handle explicit user strategy request."""
    hint = snapshot.user_strategy_hint.lower().strip()

    _STRATEGY_MAP = {
        "lhs": "lhs",
        "random": "random_sampling",
        "bayesian": "built_in",
        "bo": "built_in",
        "tpe": "optuna_tpe",
        "optuna": "optuna_tpe",
        "cmaes": "optuna_cmaes",
        "cma-es": "optuna_cmaes",
        "de": "scipy_de",
        "differential_evolution": "scipy_de",
        "evolutionary": "pymoo_nsga2",
        "nsga2": "pymoo_nsga2",
        "nsga-ii": "pymoo_nsga2",
        "adaptive": "",
    }

    backend = _STRATEGY_MAP.get(hint, hint)

    if not backend or backend == "adaptive":
        new_snapshot = CampaignSnapshot(
            round_number=snapshot.round_number,
            max_rounds=snapshot.max_rounds,
            n_observations=snapshot.n_observations,
            n_dimensions=snapshot.n_dimensions,
            has_categorical=snapshot.has_categorical,
            has_log_scale=snapshot.has_log_scale,
            kpi_history=snapshot.kpi_history,
            direction=snapshot.direction,
            user_strategy_hint="",
            available_backends=snapshot.available_backends,
            last_batch_kpis=snapshot.last_batch_kpis,
            last_batch_params=snapshot.last_batch_params,
            best_kpi_so_far=snapshot.best_kpi_so_far,
            all_params=snapshot.all_params,
            all_kpis=snapshot.all_kpis,
            qc_fail_rate=snapshot.qc_fail_rate,
        )
        return select_strategy(new_snapshot, config)

    if available.get(backend, False):
        return StrategyDecision(
            backend_name=backend,
            phase="user_requested",
            reason=f"User requested '{hint}' → {backend}",
            confidence=1.0,
            fallback_backend="built_in",
        )

    logger.warning(
        "User requested backend '%s' but it's not available. Auto-selecting.",
        hint,
    )
    new_snapshot = CampaignSnapshot(
        round_number=snapshot.round_number,
        max_rounds=snapshot.max_rounds,
        n_observations=snapshot.n_observations,
        n_dimensions=snapshot.n_dimensions,
        has_categorical=snapshot.has_categorical,
        has_log_scale=snapshot.has_log_scale,
        kpi_history=snapshot.kpi_history,
        direction=snapshot.direction,
        user_strategy_hint="",
        available_backends=snapshot.available_backends,
        last_batch_kpis=snapshot.last_batch_kpis,
        last_batch_params=snapshot.last_batch_params,
        best_kpi_so_far=snapshot.best_kpi_so_far,
        all_params=snapshot.all_params,
        all_kpis=snapshot.all_kpis,
        qc_fail_rate=snapshot.qc_fail_rate,
    )
    decision = select_strategy(new_snapshot, config)
    return StrategyDecision(
        backend_name=decision.backend_name,
        phase=decision.phase,
        reason=f"User requested '{hint}' (unavailable) → auto-selected {decision.backend_name}",
        confidence=decision.confidence * 0.8,
        fallback_backend=decision.fallback_backend,
        diagnostics=decision.diagnostics,
        phase_posterior=decision.phase_posterior,
        actions_considered=decision.actions_considered,
        explanation=decision.explanation,
    )


# ---------------------------------------------------------------------------
# Convenience: generate candidates via strategy selector
# ---------------------------------------------------------------------------


def generate_adaptive_candidates(
    space: Any,  # ParameterSpace
    n: int,
    observations: list[Observation],
    snapshot: CampaignSnapshot,
    *,
    seed: int | None = None,
    phase_config: PhaseConfig | None = None,
) -> tuple[list[dict[str, Any]], StrategyDecision]:
    """One-call convenience: select strategy + generate candidates.

    Returns (candidates, decision) so the caller can log the strategy choice.
    """
    decision = select_strategy(snapshot, config=phase_config)

    try:
        backend = get_backend(decision.backend_name)
        candidates = backend.suggest(space, n, observations, seed=seed)
    except Exception:
        logger.warning(
            "Backend '%s' failed, trying fallback '%s'",
            decision.backend_name,
            decision.fallback_backend,
            exc_info=True,
        )
        try:
            fallback = get_backend(decision.fallback_backend)
            candidates = fallback.suggest(space, n, observations, seed=seed)
        except Exception:
            logger.error("Fallback backend also failed, using LHS", exc_info=True)
            from app.services.candidate_gen import sample_lhs
            candidates = sample_lhs(space, n, seed=seed)

    return candidates, decision


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# v2 callers may reference these — keep them importable
_determine_phase_from_data = None  # removed in v3
_select_backend_for_phase = None  # removed in v3
