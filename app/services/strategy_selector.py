"""Adaptive Strategy Selector v3 — action-based optimization agent.

Replaces the v2 "label a phase then pick a backend" approach with an
**action-candidate architecture**:

1. Compute diagnostic signals (epistemic, aleatoric, saturation)
2. Generate candidate *actions* (explore / exploit / refine / stabilize)
3. Score each action with expected utility = improvement + info_gain − risk
4. Govern the decision with phase_posterior + phase_entropy

Three failure modes are now first-class citizens:
  A. **Epistemic** — model doesn't know enough (high surrogate uncertainty)
  B. **Aleatoric** — noise dominates (high within-replicate variance)
  C. **Saturation** — true convergence (low uncertainty + low EI)

The selector is stateless — all inputs come from ``CampaignSnapshot``.

**Module layout** (v4 refactor):
    strategy_models.py      — frozen dataclasses (zero deps)
    strategy_diagnostics.py — diagnostic signal computation
    strategy_scoring.py     — weight scheduling, evidence, phase posterior
    strategy_actions.py     — action generation + utility proxies
    strategy_selector.py    — main API + backward-compatible re-exports (this file)
"""
from __future__ import annotations

import logging
from typing import Any

# --- Re-export all public types from sub-modules for backward compat ---
from app.services.strategy_models import (  # noqa: F401
    ActionCandidate,
    CampaignSnapshot,
    DiagnosticSignals,
    EvidenceItem,
    PhaseConfig,
    PhasePosterior,
    StabilizeSpec,
    StrategyDecision,
    WeightsUsed,
)
from app.services.strategy_diagnostics import (  # noqa: F401
    compute_diagnostics,
    _calibrate_uncertainty,
    _compute_batch_spread,
    _compute_drift_score,
    _compute_ei_decay,
    _compute_local_smoothness,
    _compute_model_uncertainty,
    _compute_noise_ratio,
    _compute_replicate_need,
    _extract_numeric_vecs,
)
from app.services.strategy_scoring import (
    build_stabilize_spec,
    compute_confidence,
    compute_evidence,
    compute_phase_posterior,
    schedule_weights,
    _cap_stabilize_budget,
)
from app.services.strategy_actions import (
    generate_action_candidates,
    generate_explanation,
    predict_next_round,
)
from app.services.optimization_backends import (
    Observation,
    get_backend,
    list_backends,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private-name aliases for backward compat
# ---------------------------------------------------------------------------
# Sub-modules export public names; callers of the original monolith used
# private names.  Keep them importable so nothing breaks.

_compute_phase_posterior = compute_phase_posterior  # noqa: F841
_schedule_weights = schedule_weights  # noqa: F841
_compute_evidence = compute_evidence  # noqa: F841
_build_stabilize_spec = build_stabilize_spec  # noqa: F841
_compute_confidence = compute_confidence  # noqa: F841
_generate_action_candidates = generate_action_candidates  # noqa: F841
_generate_explanation = generate_explanation  # noqa: F841
_predict_next_round = predict_next_round  # noqa: F841


# ---------------------------------------------------------------------------
# Core selector — v3/v4: action-based
# ---------------------------------------------------------------------------




def _compute_adaptive_entropy_threshold(
    n_observations: int,
    n_dimensions: int,
) -> float:
    """
    Compute adaptive entropy threshold based on training progress.

    Entropy measures uncertainty in phase assignment. High entropy means we're
    uncertain whether to explore, exploit, refine, or stabilize.

    However, the posterior entropy is a property of the phase distribution,
    not directly of the data. If the phase posterior assigns high probability
    to exploit (despite high entropy), we should trust that signal.

    This threshold blocks exploitation only when entropy is very high (>95% of max),
    which indicates true ambiguity about the optimization phase. We scale the
    threshold based on sample size relative to problem dimensionality.

    Args:
        n_observations: Total observations collected so far
        n_dimensions: Problem dimensionality

    Returns:
        Entropy threshold above which exploitation is blocked
    """
    # Maximum entropy for 4-way phase distribution: log(4) ≈ 1.3863
    # We block exploit only when entropy > 95% of max ≈ 1.317
    # But adjust based on training maturity:

    # Sample complexity: how many observations per dimension do we have?
    obs_per_dim = max(1, n_observations / max(1, n_dimensions))

    if obs_per_dim < 3:
        # Very early (< 3 obs per dim): block exploitation more aggressively
        base = 1.10
    elif obs_per_dim < 10:
        # Early-mid (3-10 obs per dim): moderate threshold
        base = 1.30
    else:
        # Late (> 10 obs per dim): use nominal threshold (95% of max)
        base = 1.32

    return base


def select_strategy(
    snapshot: CampaignSnapshot,
    config: PhaseConfig | None = None,
) -> StrategyDecision:
    """Select the best optimization strategy using action-candidate ranking.

    Decision flow:
    1. User override → honor it
    2. Compute diagnostic signals (epistemic / aleatoric / saturation / drift)
    3. Optional Nexus enrichment (v5): causal insights + meta-learning
    4. Compute phase posterior (soft probabilities + entropy)
    5. Adaptive weight scheduling based on signals
    6. Generate candidate actions with adaptive utility scores
    7. Govern: entropy gate + drift gate
    8. Evidence decomposition
    9. Build stabilize spec if needed
    10. Generate explanation with evidence pointers
    """
    if config is None:
        config = PhaseConfig()

    available = snapshot.available_backends or list_backends()

    # ----- User override -----
    if snapshot.user_strategy_hint:
        return _handle_user_hint(snapshot, available, config)

    # ----- Compute diagnostics (now includes calibration + drift) -----
    diag = compute_diagnostics(snapshot, config)

    # ----- Optional Nexus enrichment (v5) -----
    nexus_evidence: list[EvidenceItem] = []
    nexus_weight_adj: dict[str, float] = {}
    if config.enable_nexus:
        try:
            from app.services.nexus_advisor import NexusAdvisor
            _nexus = NexusAdvisor()

            # Build causal data from snapshot history
            _causal_data: list[list[float]] | None = None
            _var_names: list[str] | None = None
            if snapshot.all_params and snapshot.all_kpis:
                _sample_keys = sorted(snapshot.all_params[0].keys()) if snapshot.all_params else []
                _numeric_keys = [
                    k for k in _sample_keys
                    if isinstance(snapshot.all_params[0].get(k), (int, float))
                ]
                if _numeric_keys:
                    _var_names = _numeric_keys + ["kpi"]
                    _causal_data = [
                        [float(p.get(k, 0)) for k in _numeric_keys] + [kpi]
                        for p, kpi in zip(snapshot.all_params, snapshot.all_kpis)
                    ]

            # Try to get a campaign_id from snapshot metadata (best-effort)
            _campaign_id = getattr(snapshot, "nexus_campaign_id", None) or "default"

            insights = _nexus.get_enhanced_diagnostics(
                campaign_id=_campaign_id,
                causal_data=_causal_data,
                var_names=_var_names,
            )
            if insights is not None:
                # Inject causal edges as evidence items
                for edge in insights.causal_edges:
                    if edge.strength > 0.3:
                        nexus_evidence.append(EvidenceItem(
                            signal_name=f"nexus_causal_{edge.source}→{edge.target}",
                            signal_value=edge.strength,
                            target_action="exploit" if edge.strength > 0.7 else "explore",
                            contribution=round(edge.strength * 0.15, 4),
                            description=f"Nexus causal: {edge.source}→{edge.target} (str={edge.strength:.2f})",
                        ))

            meta = _nexus.get_meta_learning_advice(campaign_id=_campaign_id)
            if meta is not None and meta.weight_adjustments:
                nexus_weight_adj = meta.weight_adjustments
                logger.info("Nexus meta-learning advice: %s", nexus_weight_adj)
        except Exception:
            logger.debug("Nexus enrichment skipped (unavailable or error)", exc_info=True)

    # ----- Phase posterior -----
    posterior = compute_phase_posterior(snapshot, diag, config)

    # ----- Adaptive weight scheduling (v4) -----
    weights: WeightsUsed | None = None
    if config.enable_adaptive_weights:
        weights = schedule_weights(diag, posterior, config)

    # ----- Apply Nexus meta-learning weight adjustments (v5) -----
    if nexus_weight_adj and weights is not None:
        w_imp = weights.w_improvement + nexus_weight_adj.get("w_improvement", 0.0)
        w_info = weights.w_info_gain + nexus_weight_adj.get("w_info_gain", 0.0)
        w_risk = weights.w_risk + nexus_weight_adj.get("w_risk", 0.0)
        # Re-normalize
        w_imp = max(0.1, w_imp)
        w_info = max(0.1, w_info)
        w_risk = max(0.05, w_risk)
        total = w_imp + w_info + w_risk
        weights = WeightsUsed(
            w_improvement=round(w_imp / total, 4),
            w_info_gain=round(w_info / total, 4),
            w_risk=round(w_risk / total, 4),
            reason=weights.reason + "; nexus meta-learning adj",
        )

    # ----- Generate action candidates with adaptive weights -----
    actions = generate_action_candidates(
        snapshot, diag, posterior, available, config, weights=weights,
    )

    # ----- Governance: if entropy too high, block exploit (adaptive threshold) -----
    # Use adaptive threshold based on training progress
    adaptive_threshold = _compute_adaptive_entropy_threshold(
        snapshot.n_observations, snapshot.n_dimensions
    )
    if posterior.entropy > adaptive_threshold:
        governed_actions = [a for a in actions if a.name != "exploit"]
        if governed_actions:
            actions = governed_actions + [a for a in actions if a.name == "exploit"]

    # ----- Governance: if drift high, block exploit + boost stabilize/explore -----
    if diag.drift_score is not None and diag.drift_score > config.drift_high_threshold:
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

    confidence = compute_confidence(snapshot, diag, phase)

    backend = best_action.backend_name
    fallback = "built_in" if backend != "built_in" else "lhs"

    # ----- Evidence decomposition (v4) -----
    eff_weights = weights or WeightsUsed(
        w_improvement=config.w_improvement,
        w_info_gain=config.w_info_gain,
        w_risk=config.w_risk,
        reason="default weights",
    )
    evidence = compute_evidence(diag, eff_weights)

    # ----- Merge Nexus evidence (v5) -----
    if nexus_evidence:
        merged = list(evidence) + nexus_evidence
        merged.sort(key=lambda e: abs(e.contribution), reverse=True)
        evidence = tuple(merged)

    # ----- Stabilize spec (v4) -----
    stabilize_spec = None
    if best_action.name == "stabilize":
        stabilize_spec = build_stabilize_spec(snapshot, diag, config)

    # ----- Explanation with evidence pointers -----
    next_expect = predict_next_round(best_action, diag)
    explanation = generate_explanation(
        best_action, diag, posterior, next_expect, evidence=evidence,
    )

    reason = (
        f"{best_action.reason} "
        f"(utility={best_action.utility:.3f}, "
        f"P({best_action.name})={getattr(posterior, best_action.name, 0):.2f})"
    )

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
