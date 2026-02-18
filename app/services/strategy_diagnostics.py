"""Diagnostic signal computation for the Adaptive Strategy Selector.

Computes epistemic, aleatoric, saturation, and landscape signals from
a ``CampaignSnapshot``.  The main entry-point is :func:`compute_diagnostics`.

All helpers are module-private (prefixed with ``_``) except
``compute_diagnostics`` and ``_extract_numeric_vecs`` (used by scoring).

Public API:
    compute_diagnostics
"""
from __future__ import annotations

import logging
import math
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
from app.services.strategy_models import (
    CampaignSnapshot,
    DiagnosticSignals,
    PhaseConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry-point
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
# Epistemic signal helpers
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


# ---------------------------------------------------------------------------
# Aleatoric signal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Landscape signal helpers
# ---------------------------------------------------------------------------


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
# Shared helper: extract numeric vectors from param dicts
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
# Saturation signal helpers
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
