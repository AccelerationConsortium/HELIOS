"""Scoring, weighting, evidence, and phase computation for the Adaptive Strategy Selector.

Contains:
- Adaptive weight scheduler (v4)
- Evidence decomposition (v4)
- Stabilize protocol builder (v4)
- Phase posterior computation (softmax over phase scores)
- Confidence scoring

Public API:
    schedule_weights, compute_evidence, build_stabilize_spec,
    compute_phase_posterior, compute_confidence
"""
from __future__ import annotations

import logging
import math
from typing import Any

from app.services.convergence import _mean
from app.services.strategy_models import (
    CampaignSnapshot,
    DiagnosticSignals,
    EvidenceItem,
    PhasePosterior,
    PhaseConfig,
    StabilizeSpec,
    WeightsUsed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v4: Adaptive weight scheduler
# ---------------------------------------------------------------------------


def schedule_weights(
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
# v4: Evidence decomposition
# ---------------------------------------------------------------------------


def compute_evidence(
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


def build_stabilize_spec(
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


def compute_phase_posterior(
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
# Confidence scoring
# ---------------------------------------------------------------------------


def compute_confidence(
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
