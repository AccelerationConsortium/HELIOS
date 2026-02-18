"""Action generation and utility scoring for the Adaptive Strategy Selector.

Contains:
- Action candidate generation (explore / exploit / refine / stabilize)
- 12 utility proxy functions (4 actions × 3 dimensions)
- Backend selection helpers
- Explanation generator
- Next-round prediction

Public API:
    generate_action_candidates, generate_explanation, predict_next_round
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.strategy_models import (
    ActionCandidate,
    CampaignSnapshot,
    DiagnosticSignals,
    EvidenceItem,
    PhaseConfig,
    PhasePosterior,
    WeightsUsed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action generation + utility scoring
# ---------------------------------------------------------------------------


def generate_action_candidates(
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    posterior: PhasePosterior,
    available: dict[str, bool],
    config: PhaseConfig,
    weights: WeightsUsed | None = None,
) -> list[ActionCandidate]:
    """Generate candidate actions and score them.

    Each action is scored with:
        utility = w_improvement × improvement + w_info_gain × info_gain − w_risk × risk

    Returns a list sorted by utility (descending).
    """
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
        exploit_reason = (
            f"Exploit via evolutionary (multimodal landscape, "
            f"smoothness={diag.local_smoothness:.2f})"
        )
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
# Utility proxy functions (4 actions × 3 dimensions)
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
    """Pick the first available backend from the preference list."""
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


def generate_explanation(
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
    line1 = (
        f"Decision: {decision_action.name} "
        f"(P={getattr(posterior, decision_action.name, 0):.2f}), "
        f"backend={decision_action.backend_name}"
    )

    # Line 2: Because — collect active signals
    because_parts: list[str] = []
    if diag.convergence_status != "insufficient_data":
        because_parts.append(
            f"{diag.convergence_status}(conf={diag.convergence_confidence:.2f})"
        )
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
    line2 = (
        "Because: " + " + ".join(because_parts)
        if because_parts
        else "Because: insufficient signals"
    )

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


def predict_next_round(
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
