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
from dataclasses import asdict, replace
from enum import Enum
from typing import Any

from app.services.optimization_backends import (
    Observation,
    get_backend,
    list_backends,
)
from app.services.strategy_actions import (
    generate_action_candidates,
    generate_explanation,
    predict_next_round,
)
from app.services.strategy_diagnostics import (  # noqa: F401
    _calibrate_uncertainty,
    _compute_batch_spread,
    _compute_drift_score,
    _compute_ei_decay,
    _compute_local_smoothness,
    _compute_model_uncertainty,
    _compute_noise_ratio,
    _compute_replicate_need,
    _extract_numeric_vecs,
    compute_diagnostics,
)

# --- Re-export all public types from sub-modules for backward compat ---
from app.services.strategy_models import (  # noqa: F401
    ActionCandidate,
    BanditInfluenceRecord,
    CampaignContext,
    CampaignIntent,
    CampaignSnapshot,
    ContextGateDecision,
    ContextualBanditDecision,
    DiagnosticSignals,
    EvidenceItem,
    FailureType,
    LearnedPolicyDeploymentMode,
    NexusRecommendationTrace,
    ObjectiveLevel,
    ObjectiveTransitionProposal,
    OnlineInfluenceMode,
    OnlineInfluenceOutcome,
    OptimizationMode,
    PhaseConfig,
    PhasePosterior,
    RankingInfluenceRecord,
    ShadowBanditEvaluationRecord,
    StabilizeSpec,
    StrategyDecision,
    StrategyEvidence,
    StrategyOutcome,
    StrategyReward,
    StrategyTrace,
    WeightsUsed,
)
from app.services.strategy_scoring import (
    build_stabilize_spec,
    compute_confidence,
    compute_evidence,
    compute_phase_posterior,
    schedule_weights,
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
    from app.services.context_policy import (
        evaluate_context_gate,
        propose_space_revision,
    )

    context_gate = evaluate_context_gate(snapshot)
    space_revision = propose_space_revision(snapshot)

    # ----- Optional optimization intelligence enrichment (v5) -----
    intelligence_evidence: list[EvidenceItem] = []
    intelligence_weight_adj: dict[str, float] = {}
    intelligence_recommended_backends: tuple[str, ...] = ()
    if config.enable_nexus or config.enable_optimization_intelligence:
        try:
            from app.services.optimization_intelligence import OptimizationIntelligenceAdvisor

            intelligence = OptimizationIntelligenceAdvisor().advise(snapshot)
            intelligence_evidence = list(intelligence.evidence)
            intelligence_weight_adj = intelligence.weight_adjustments
            intelligence_recommended_backends = intelligence.recommended_backends
            if intelligence.has_signal:
                logger.info(
                    "Optimization intelligence advice: sources=%s weights=%s phase=%s",
                    intelligence.sources,
                    intelligence_weight_adj,
                    intelligence.recommended_phase,
                )
        except Exception:
            logger.debug("Optimization intelligence skipped (unavailable or error)", exc_info=True)

    # ----- Method advisor (P3a): benchmark-derived problem-class bias -----
    method_advice: tuple[str, ...] = ()
    if config.enable_method_advisor:
        try:
            from app.services.method_advisor import recommend_backends

            method_advice = recommend_backends(snapshot, diag)
        except Exception:
            logger.debug("Method advisor skipped (unavailable or error)", exc_info=True)

    # ----- Phase posterior -----
    posterior = compute_phase_posterior(snapshot, diag, config)

    # ----- Adaptive weight scheduling (v4) -----
    weights: WeightsUsed | None = None
    if config.enable_adaptive_weights:
        weights = schedule_weights(diag, posterior, config)

    # ----- Apply cross-campaign meta-learning weight adjustments (v5) -----
    if intelligence_weight_adj and weights is not None:
        w_imp = weights.w_improvement + intelligence_weight_adj.get("w_improvement", 0.0)
        w_info = weights.w_info_gain + intelligence_weight_adj.get("w_info_gain", 0.0)
        w_risk = weights.w_risk + intelligence_weight_adj.get("w_risk", 0.0)
        # Re-normalize
        w_imp = max(0.1, w_imp)
        w_info = max(0.1, w_info)
        w_risk = max(0.05, w_risk)
        total = w_imp + w_info + w_risk
        weights = WeightsUsed(
            w_improvement=round(w_imp / total, 4),
            w_info_gain=round(w_info / total, 4),
            w_risk=round(w_risk / total, 4),
            reason=weights.reason + "; optimization intelligence meta-learning adj",
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

    # ----- Backend ratification: conservative fingerprint soft-bias (Δ2) -----
    # The phase-policy default (best_action.backend_name) is re-ranked against
    # the action's preference pool.  A fingerprint recommendation can promote an
    # *available, phase-compatible* alternative; it can never select an
    # unavailable or out-of-pool backend, nor change the campaign phase.  With no
    # recommendation and no recent failures this is a no-op.
    from app.services.backend_selection import rank_backends
    from app.services.online_influence import (
        build_online_influence_outcome,
        effective_policy_influence_config,
    )

    base_pool = _action_backend_pool(best_action.name, snapshot, config)
    if best_action.backend_name == "built_in":
        # Degraded default: keep the natural preference order (built_in last).
        backend_pool = base_pool if "built_in" in base_pool else (*base_pool, "built_in")
    else:
        # Deliberate phase pick (incl. multimodal/refine specials) leads the pool.
        backend_pool = (best_action.backend_name, *base_pool)
    backend_pool = tuple(dict.fromkeys(backend_pool))
    selected_intent_for_policy = _infer_campaign_intent(snapshot, best_action, diag)
    selected_mode_for_policy = _infer_optimization_mode(best_action, snapshot)
    from app.services.action_policy import ActionPolicyMatrix, ActionTransitionGuard
    from app.services.ranking_influence import bounded_ranking_influence

    action_policy = ActionPolicyMatrix().evaluate(
        snapshot,
        selected_intent=selected_intent_for_policy,
        selected_mode=selected_mode_for_policy,
        selected_backend=best_action.backend_name,
    )
    transition_guard = ActionTransitionGuard().evaluate(
        snapshot.previous_intent,
        selected_intent_for_policy,
        action_policy.evidence,
    )
    bandit_decision = _trace_bandit_decision(
        snapshot,
        best_action.name,
        best_action.backend_name,
        backend_pool,
    )
    effective_influence_config, _online_influence_reason = effective_policy_influence_config(
        config.policy_influence,
        config.online_influence_rollout,
        snapshot,
    )
    ranking_deltas, ranking_influences, ranking_evidence = bounded_ranking_influence(
        snapshot=snapshot,
        backend_pool=backend_pool,
        action_name=best_action.name,
        action_policy=action_policy,
        transition_guard=transition_guard,
        bandit_decision=None,
        config=effective_influence_config,
    )
    bandit_influence_record = None
    if effective_influence_config.enable_bandit_rerank and bandit_decision is not None:
        from app.services.bandit_influence import bandit_soft_influence

        bandit_deltas, bandit_records, bandit_evidence, bandit_influence_record = (
            bandit_soft_influence(
                snapshot=snapshot,
                decision=bandit_decision,
                backend_pool=backend_pool,
                policy_config=effective_influence_config,
                rollout=config.online_influence_rollout,
            )
        )
        for candidate_backend, delta in bandit_deltas.items():
            proposed = ranking_deltas.get(candidate_backend, 0.0) + delta
            ranking_deltas[candidate_backend] = round(
                max(
                    -effective_influence_config.max_total_score_delta,
                    min(effective_influence_config.max_total_score_delta, proposed),
                ),
                4,
            )
        ranking_influences = tuple((*ranking_influences, *bandit_records))
        ranking_evidence = tuple((*ranking_evidence, *bandit_evidence))

    # The phase already committed to best_action.backend_name, so it is always a
    # valid choice regardless of the caller-supplied availability map.
    available_for_rank = {**available, best_action.backend_name: True}

    # Merge recommendation channels: campaign-specific meta-learning first
    # (most specific), then the problem-class method advice (P3a).
    #
    # Method advice only biases exploit/refine actions -- "which optimizer wins
    # on this problem class" is meaningful when optimizing, not when the phase
    # has already committed to space-filling (explore) or replication
    # (stabilize).  Advice is further filtered to the current action's pool so it
    # never enlarges the candidate set or muddies the provenance.
    method_advice_in_pool: tuple[str, ...] = ()
    if best_action.name in ("exploit", "refine"):
        method_advice_in_pool = tuple(b for b in method_advice if b in backend_pool)
    recommended = tuple(
        dict.fromkeys((*intelligence_recommended_backends, *method_advice_in_pool))
    )

    baseline_backend_selection = rank_backends(
        phase=best_action.name,
        pool=backend_pool,
        available=available_for_rank,
        recommended=recommended,
        failure_counts=snapshot.backend_failure_counts,
        phase_weight=config.backend_phase_weight,
        fingerprint_weight=config.backend_fingerprint_weight,
        failure_penalty=config.backend_failure_penalty,
        failure_veto_threshold=config.backend_failure_veto_threshold,
        fallback_backend="built_in",
    )
    safe_backend_selection = rank_backends(
        phase=best_action.name,
        pool=backend_pool,
        available=available_for_rank,
        recommended=recommended,
        failure_counts=snapshot.backend_failure_counts,
        phase_weight=config.backend_phase_weight,
        fingerprint_weight=config.backend_fingerprint_weight,
        failure_penalty=config.backend_failure_penalty,
        failure_veto_threshold=config.backend_failure_veto_threshold,
        fallback_backend="built_in",
        influence_deltas=ranking_deltas,
    )
    learned_policy_shadow = None
    learned_policy_influence = None
    learned_safe_ranking_record = None
    safe_top_backend = safe_backend_selection.selected_backend
    preliminary_reason = (
        f"{best_action.reason} "
        f"(utility={best_action.utility:.3f}, "
        f"P({best_action.name})={getattr(posterior, best_action.name, 0):.2f})"
    )
    if _learned_live_canary_enabled(config, snapshot):
        preliminary_trace = _build_strategy_trace(
            snapshot=snapshot,
            diag=diag,
            action=best_action,
            backend=safe_backend_selection.selected_backend,
            backend_selection=safe_backend_selection,
            evidence=(),
            reason=preliminary_reason,
            context_gate=context_gate,
            space_revision=space_revision,
            bandit_decision=bandit_decision,
            action_policy=action_policy,
            transition_guard=transition_guard,
            ranking_influences=ranking_influences,
            ranking_evidence=ranking_evidence,
            influence_config=effective_influence_config,
            online_influence_outcome=None,
            bandit_influence=bandit_influence_record,
            available_actions=tuple(actions),
        )
        from app.services.learned_policy import (
            LearnedPolicyPromotionGate,
            LearnedPolicyShadowRunner,
        )

        learned_runner = LearnedPolicyShadowRunner(
            registry_entry=config.learned_policy_registry_entry,
            policy=config.learned_policy,
            mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
            promotion_gate=LearnedPolicyPromotionGate(
                min_shadow_rounds=config.learned_policy_min_shadow_rounds,
            ),
            shadow_summary=config.learned_policy_shadow_summary,
            max_safe_soft_delta=config.learned_policy_max_safe_soft_delta,
        )
        learned_trace = learned_runner.run(preliminary_trace)
        learned_policy_shadow = learned_trace.learned_policy_shadow
        learned_policy_influence = learned_trace.learned_policy_influence
        proposed_record = learned_runner.ranking_influence_record(
            learned_policy_influence,
            source="learned_policy",
        )
        if proposed_record is not None:
            current_delta = ranking_deltas.get(proposed_record.target, 0.0)
            total_cap = min(
                effective_influence_config.max_total_score_delta,
                config.online_influence_rollout.max_total_score_delta,
            )
            capped_total = max(
                -total_cap,
                min(total_cap, current_delta + proposed_record.score_delta),
            )
            actual_delta = round(capped_total - current_delta, 6)
            learned_safe_ranking_record = replace(
                proposed_record,
                score_delta=actual_delta,
                applied_weight=abs(actual_delta),
                capped=proposed_record.capped or actual_delta != proposed_record.score_delta,
            )
            learned_policy_influence = replace(
                learned_policy_influence,
                applied_delta=actual_delta,
                capped=learned_policy_influence.capped or learned_safe_ranking_record.capped,
                changed_top1=False,
            )
            ranking_influences = tuple((*ranking_influences, learned_safe_ranking_record))
            if abs(actual_delta) > 0:
                ranking_deltas[proposed_record.target] = round(capped_total, 6)
    backend_selection = rank_backends(
        phase=best_action.name,
        pool=backend_pool,
        available=available_for_rank,
        recommended=recommended,
        failure_counts=snapshot.backend_failure_counts,
        phase_weight=config.backend_phase_weight,
        fingerprint_weight=config.backend_fingerprint_weight,
        failure_penalty=config.backend_failure_penalty,
        failure_veto_threshold=config.backend_failure_veto_threshold,
        fallback_backend="built_in",
        influence_deltas=ranking_deltas,
    )
    if learned_policy_influence is not None:
        learned_policy_influence = replace(
            learned_policy_influence,
            changed_top1=safe_top_backend != backend_selection.selected_backend,
        )
    backend = backend_selection.selected_backend
    fallback = "built_in" if backend != "built_in" else "lhs"

    # ----- Evidence decomposition (v4) -----
    eff_weights = weights or WeightsUsed(
        w_improvement=config.w_improvement,
        w_info_gain=config.w_info_gain,
        w_risk=config.w_risk,
        reason="default weights",
    )
    evidence = compute_evidence(diag, eff_weights)

    # ----- Merge optimization intelligence evidence (v5) -----
    if intelligence_evidence:
        merged = list(evidence) + intelligence_evidence
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
    online_influence_outcome = None
    strategy_trace = _build_strategy_trace(
        snapshot=snapshot,
        diag=diag,
        action=best_action,
        backend=backend,
        backend_selection=backend_selection,
        evidence=evidence,
        reason=reason,
        context_gate=context_gate,
        space_revision=space_revision,
        bandit_decision=bandit_decision,
        action_policy=action_policy,
        transition_guard=transition_guard,
        ranking_influences=ranking_influences,
        ranking_evidence=ranking_evidence,
        influence_config=effective_influence_config,
        online_influence_outcome=online_influence_outcome,
        bandit_influence=bandit_influence_record,
        available_actions=tuple(actions),
    )
    if learned_policy_shadow is not None or learned_policy_influence is not None:
        strategy_trace = replace(
            strategy_trace,
            learned_policy_shadow=learned_policy_shadow,
            learned_policy_influence=learned_policy_influence,
        )
    trace_for_online = strategy_trace_to_dict(strategy_trace)
    online_influence_outcome = build_online_influence_outcome(
        rollout=config.online_influence_rollout,
        baseline_top_backend=baseline_backend_selection.selected_backend,
        influenced_top_backend=backend_selection.selected_backend,
        ranking_influences=ranking_influences,
        trace=trace_for_online,
        config=effective_influence_config,
        safe_influence_top_backend=safe_top_backend,
        learned_influenced_top_backend=backend_selection.selected_backend,
        learned_policy_influences=(
            (learned_policy_influence,) if learned_policy_influence is not None else ()
        ),
    )
    if online_influence_outcome is not None:
        strategy_trace = _replace_strategy_trace_online_outcome(
            strategy_trace,
            online_influence_outcome,
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
        backend_selection=backend_selection,
        strategy_trace=strategy_trace,
        recommended_backends=tuple(intelligence_recommended_backends),
    )


def _action_backend_pool(
    action_name: str,
    snapshot: CampaignSnapshot,
    config: PhaseConfig,
) -> tuple[str, ...]:
    """Return the ordered backend preference pool for a winning action.

    Mirrors the per-action preference used by ``generate_action_candidates`` so
    the Δ2 re-rank operates over the same phase-compatible candidates.
    """
    if action_name == "exploit":
        return config.exploitation_backends
    if action_name == "refine":
        if snapshot.n_dimensions >= config.high_dim_threshold:
            return config.high_dim_backends + config.refinement_backends
        return config.refinement_backends
    if action_name == "explore":
        return config.explore_backends
    return ("built_in",)  # stabilize re-evaluates with the built-in optimizer


def _build_strategy_trace(
    *,
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    action: ActionCandidate,
    backend: str,
    backend_selection: Any,
    evidence: tuple[EvidenceItem, ...],
    reason: str,
    context_gate: ContextGateDecision,
    space_revision: Any,
    bandit_decision: ContextualBanditDecision | None,
    action_policy: Any,
    transition_guard: Any,
    ranking_influences: tuple[RankingInfluenceRecord, ...],
    ranking_evidence: tuple[StrategyEvidence, ...],
    influence_config: Any,
    online_influence_outcome: OnlineInfluenceOutcome | None,
    bandit_influence: BanditInfluenceRecord | None,
    available_actions: tuple[ActionCandidate, ...],
) -> StrategyTrace:
    """Build JSON-safe strategy provenance without changing selection behavior."""
    context = snapshot.campaign_context or CampaignContext()
    strategy_evidence: list[StrategyEvidence] = [
        StrategyEvidence(
            source="diagnostics",
            target=item.target_action,
            effect="boost" if item.contribution >= 0 else "penalize",
            strength=abs(item.contribution),
            reason=item.description,
            metadata={"signal": item.signal_name, "value": item.signal_value},
        )
        for item in evidence
    ]

    if backend_selection is not None:
        for rec in backend_selection.fingerprint_recommendation:
            strategy_evidence.append(
                StrategyEvidence(
                    source="nexus",
                    target=rec,
                    effect="boost",
                    strength=0.3,
                    reason="Problem fingerprint recommended this backend",
                )
            )
        for score in backend_selection.score_components:
            if score.failure_penalty > 0:
                strategy_evidence.append(
                    StrategyEvidence(
                        source="failure_history",
                        target=score.backend,
                        effect="penalize",
                        strength=score.failure_penalty,
                        reason="Recent backend failures reduced its rank",
                    )
                )

    for failure in snapshot.failure_events:
        ftype = getattr(failure.failure_type, "value", failure.failure_type)
        strategy_evidence.append(
            StrategyEvidence(
                source=f"failure:{ftype}",
                target=failure.backend_name or str(ftype),
                effect="penalize" if failure.penalize_backend else "context",
                strength=1.0 if failure.penalize_backend else 0.0,
                reason=failure.reason,
                metadata={
                    "round_number": failure.round_number,
                    "candidate_index": failure.candidate_index,
                    "params": failure.params,
                },
            )
        )
    if not context_gate.ready_for_optimization:
        strategy_evidence.append(
            StrategyEvidence(
                source="context_gate",
                target=str(getattr(context_gate.recommended_intent, "value", context_gate.recommended_intent)),
                effect="veto",
                strength=1.0,
                reason=context_gate.reason,
                metadata={
                    "requires_space_revision": context_gate.requires_space_revision,
                    "requires_hypothesis_update": context_gate.requires_hypothesis_update,
                    "requires_route_switch": context_gate.requires_route_switch,
                    "requires_calibration": context_gate.requires_calibration,
                    "requires_human_review": context_gate.requires_human_review,
                },
            )
        )
    elif context_gate.requires_hypothesis_update or context_gate.requires_route_switch:
        strategy_evidence.append(
            StrategyEvidence(
                source="context_gate",
                target=str(getattr(context_gate.recommended_intent, "value", context_gate.recommended_intent)),
                effect="context",
                strength=0.5,
                reason=context_gate.reason,
            )
        )
    if space_revision is not None:
        strategy_evidence.append(
            StrategyEvidence(
                source="space_policy",
                target="parameter_space",
                effect="context",
                strength=0.5,
                reason=space_revision.reason,
            )
        )
    selected_intent = _infer_campaign_intent(snapshot, action, diag)
    selected_mode = _infer_optimization_mode(action, snapshot)
    strategy_evidence.extend(action_policy.evidence)
    strategy_evidence.extend(transition_guard.evidence)
    strategy_evidence.extend(ranking_evidence)
    influence_records = list(ranking_influences)
    has_bandit_record = any(
        record.source.startswith("bandit")
        for record in influence_records
    )
    if (
        influence_config.enable_bandit_rerank
        and bandit_decision is not None
        and not has_bandit_record
    ):
        influence_records.append(RankingInfluenceRecord(
            source="bandit_shadow",
            target=bandit_decision.suggested_backend,
            raw_signal=bandit_decision.confidence,
            applied_weight=0.0,
            score_delta=0.0,
            capped=False,
            reason="Contextual bandit remains shadow-only; no execution influence applied",
        ))
    strategy_reward = StrategyReward()
    shadow_record = _shadow_record_from_bandit(bandit_decision, strategy_reward)
    objective_transition = _objective_transition_proposal(context, context_gate, strategy_evidence)

    return StrategyTrace(
        round_number=snapshot.round_number,
        selected_intent=selected_intent,
        selected_mode=selected_mode,
        selected_backend=backend,
        campaign_id=snapshot.campaign_id,
        state_summary={
            "n_observations": snapshot.n_observations,
            "n_dimensions": snapshot.n_dimensions,
            "space_coverage": diag.space_coverage,
            "noise_ratio": diag.noise_ratio,
            "drift_score": diag.drift_score,
            "convergence_status": diag.convergence_status,
            "convergence_confidence": diag.convergence_confidence,
            "qc_fail_rate": snapshot.qc_fail_rate,
            "recent_backend_failures": dict(snapshot.backend_failure_counts),
            "n_failed_params": len(snapshot.failed_params),
        },
        context_summary=_context_trace_summary(context, snapshot, diag, context_gate),
        context_gate=context_gate,
        space_revision=space_revision,
        action_policy=action_policy,
        transition_guard=transition_guard,
        ranking_influences=tuple(influence_records),
        bandit_influence=bandit_influence,
        online_influence_outcome=online_influence_outcome,
        bandit_decision=bandit_decision,
        shadow_bandit_record=shadow_record,
        objective_transition=objective_transition,
        strategy_reward=strategy_reward,
        available_actions=_action_trace_table(available_actions),
        candidate_backends=_backend_score_table(backend_selection),
        nexus_recommendation=_nexus_recommendation_trace(backend_selection),
        outcome=StrategyOutcome(
            outcome=None,
            reward=strategy_reward,
            failure_events=snapshot.failure_events,
            observed=False,
        ),
        learned_policy_shadow=None,
        learned_policy_influence=None,
        evidence=tuple(strategy_evidence),
        reason=reason,
    )


def _infer_campaign_intent(
    snapshot: CampaignSnapshot,
    action: ActionCandidate,
    diag: DiagnosticSignals,
) -> CampaignIntent:
    context = snapshot.campaign_context
    level = (
        getattr(context.current_objective_level, "value", context.current_objective_level)
        if context is not None
        else ObjectiveLevel.PERFORMANCE.value
    )
    backend_failures = [
        f for f in snapshot.failure_events
        if getattr(f.failure_type, "value", f.failure_type) == FailureType.BACKEND.value
    ]
    hardware_failures = [
        f for f in snapshot.failure_events
        if getattr(f.failure_type, "value", f.failure_type) == FailureType.HARDWARE.value
    ]
    constraint_failures = [
        f for f in snapshot.failure_events
        if getattr(f.failure_type, "value", f.failure_type) == FailureType.CONSTRAINT.value
    ]
    measurement_failures = [
        f for f in snapshot.failure_events
        if getattr(f.failure_type, "value", f.failure_type) == FailureType.MEASUREMENT.value
    ]
    scientific_negative = [
        f for f in snapshot.failure_events
        if getattr(f.failure_type, "value", f.failure_type)
        == FailureType.SCIENTIFIC_NEGATIVE.value
    ]

    if hardware_failures or constraint_failures or (backend_failures and action.name != "stabilize"):
        return CampaignIntent.RECOVER
    if measurement_failures:
        return CampaignIntent.DIAGNOSE
    if scientific_negative:
        return CampaignIntent.VALIDATE
    if level == ObjectiveLevel.BASELINE.value:
        return CampaignIntent.DISCOVER
    if action.name == "stabilize" or level in (
        ObjectiveLevel.FEASIBILITY.value,
        ObjectiveLevel.DATA_QUALITY.value,
    ):
        return CampaignIntent.STABILIZE
    if level == ObjectiveLevel.MECHANISM.value:
        return CampaignIntent.VALIDATE
    if level == ObjectiveLevel.GENERALIZATION.value:
        return CampaignIntent.PIVOT
    if level == ObjectiveLevel.PERFORMANCE.value:
        return CampaignIntent.OPTIMIZE
    if diag.drift_score is not None and diag.drift_score > 0.6:
        return CampaignIntent.DIAGNOSE
    if action.name == "explore":
        return CampaignIntent.DISCOVER
    return CampaignIntent.OPTIMIZE


def _infer_optimization_mode(
    action: ActionCandidate,
    snapshot: CampaignSnapshot,
) -> OptimizationMode:
    if action.name == "stabilize":
        return OptimizationMode.REPLICATE
    if snapshot.failed_params:
        return OptimizationMode.FAILURE_AVOIDANCE
    if action.name == "explore":
        return OptimizationMode.EXPLORE
    if action.name == "exploit":
        return OptimizationMode.EXPLOIT
    if action.name == "refine":
        return OptimizationMode.REFINE
    return OptimizationMode.STABILIZE


def _backend_score_table(backend_selection: Any) -> tuple[dict[str, Any], ...]:
    if backend_selection is None:
        return ()
    return tuple(
        {
            "name": score.backend,
            "phase_score": score.phase_score,
            "fingerprint_boost": score.fingerprint_boost,
            "failure_penalty": score.failure_penalty,
            "influence_delta": getattr(score, "influence_delta", 0.0),
            "total": score.total,
        }
        for score in backend_selection.score_components
    )


def _context_trace_summary(
    context: CampaignContext,
    snapshot: CampaignSnapshot,
    diag: DiagnosticSignals,
    context_gate: ContextGateDecision,
) -> dict[str, Any]:
    summary = context.summary()
    failed_params = tuple(snapshot.failed_params)
    constraints = tuple(context.known_constraints)
    summary["parameter_space_health"] = {
        **dict(summary.get("parameter_space_health") or {}),
        "n_failed_params": len(failed_params),
        "failed_params": [dict(params) for params in failed_params],
        "known_constraints": [dict(item) for item in constraints],
        "infeasible_region_count": len(failed_params) + len(constraints),
        "requires_revision": bool(
            context_gate.requires_space_revision or failed_params
        ),
        "reason": (
            context_gate.reason
            if context_gate.requires_space_revision or failed_params
            else str((summary.get("parameter_space_health") or {}).get("reason") or "")
        ),
    }
    summary["route_context"] = {
        **dict(summary.get("route_context") or {}),
        "routes": list(context.synthesis_routes),
        "requires_route_switch": context_gate.requires_route_switch,
        "reason": (
            context_gate.reason
            if context_gate.requires_route_switch
            else str((summary.get("route_context") or {}).get("reason") or "")
        ),
    }
    summary["budget_context"] = {
        **dict(summary.get("budget_context") or {}),
        "remaining": dict(context.budget_remaining),
        "max_rounds": snapshot.max_rounds,
        "current_round": snapshot.round_number,
        "pressure": _budget_pressure(snapshot),
    }
    summary["data_quality_context"] = {
        **dict(summary.get("data_quality_context") or {}),
        "qc_fail_rate": snapshot.qc_fail_rate,
        "noise_ratio": diag.noise_ratio,
        "requires_calibration": context_gate.requires_calibration,
        "reason": (
            context_gate.reason
            if context_gate.requires_calibration
            else str((summary.get("data_quality_context") or {}).get("reason") or "")
        ),
    }
    summary["prior_campaign_context"] = {
        **dict(summary.get("prior_campaign_context") or {}),
        "n_prior_campaigns": len(context.prior_campaigns),
        "n_literature_priors": len(context.literature_priors),
        "warm_start_available": bool(context.prior_campaigns or context.literature_priors),
    }
    return summary


def _budget_pressure(snapshot: CampaignSnapshot) -> str:
    if snapshot.max_rounds <= 0:
        return "unknown"
    remaining_fraction = max(snapshot.max_rounds - snapshot.round_number, 0) / snapshot.max_rounds
    if remaining_fraction <= 0.2:
        return "high"
    if remaining_fraction <= 0.5:
        return "medium"
    return "low"


def _action_trace_table(actions: tuple[ActionCandidate, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": action.name,
            "backend_name": action.backend_name,
            "expected_improvement": action.expected_improvement,
            "expected_info_gain": action.expected_info_gain,
            "risk": action.risk,
            "utility": action.utility,
            "reason": action.reason,
        }
        for action in actions
    )


def _nexus_recommendation_trace(backend_selection: Any) -> NexusRecommendationTrace | None:
    if backend_selection is None:
        return None
    recommended = tuple(getattr(backend_selection, "fingerprint_recommendation", ()) or ())
    if not recommended:
        return None
    return NexusRecommendationTrace(
        recommended_backends=recommended,
        reason=getattr(backend_selection, "reason", ""),
        selected_backend=getattr(backend_selection, "selected_backend", None),
        score_weight=max(
            (
                float(getattr(score, "fingerprint_boost", 0.0) or 0.0)
                for score in getattr(backend_selection, "score_components", ())
            ),
            default=0.0,
        ),
    )


def _trace_bandit_decision(
    snapshot: CampaignSnapshot,
    action_name: str,
    backend: str,
    backend_pool: tuple[str, ...],
) -> ContextualBanditDecision | None:
    if not backend_pool:
        return None
    try:
        from app.services.backend_memory import (
            ContextualStrategyBandit,
            problem_context_key,
        )

        context_key = problem_context_key(snapshot)
        arms = tuple(f"{action_name}:{candidate}" for candidate in backend_pool)
        return ContextualStrategyBandit(snapshot.strategy_bandit_stats).select(
            context_key=context_key,
            arms=arms,
            actual_action=action_name,
            actual_backend=backend,
        )
    except Exception:
        logger.debug("Contextual bandit trace skipped", exc_info=True)
        return None


def _shadow_record_from_bandit(
    decision: ContextualBanditDecision | None,
    reward: StrategyReward,
) -> ShadowBanditEvaluationRecord | None:
    if decision is None:
        return None
    return ShadowBanditEvaluationRecord(
        actual_action=decision.actual_action,
        actual_backend=decision.actual_backend,
        suggested_action=decision.suggested_action,
        suggested_backend=decision.suggested_backend,
        agrees_with_actual=decision.agrees_with_actual,
        bandit_confidence=decision.confidence,
        actual_reward=reward.composite_reward,
        outcome=None,
    )


def _objective_transition_proposal(
    context: CampaignContext,
    gate: ContextGateDecision,
    evidence: list[StrategyEvidence],
) -> ObjectiveTransitionProposal | None:
    current = getattr(context.current_objective_level, "value", context.current_objective_level)
    target = None
    if gate.recommended_intent == CampaignIntent.DISCOVER and current != ObjectiveLevel.BASELINE.value:
        target = ObjectiveLevel.BASELINE.value
    elif gate.recommended_intent == CampaignIntent.OPTIMIZE and current != ObjectiveLevel.PERFORMANCE.value:
        target = ObjectiveLevel.PERFORMANCE.value
    elif gate.recommended_intent == CampaignIntent.VALIDATE and current != ObjectiveLevel.MECHANISM.value:
        target = ObjectiveLevel.MECHANISM.value
    elif gate.recommended_intent in (CampaignIntent.TRANSFER, CampaignIntent.PIVOT) and current != ObjectiveLevel.GENERALIZATION.value:
        target = ObjectiveLevel.GENERALIZATION.value
    if target is None:
        return None
    return ObjectiveTransitionProposal(
        from_level=current,
        to_level=target,
        reason=gate.reason,
        evidence=tuple(evidence[:5]),
        confidence=0.5,
        auto_applied=False,
    )


def strategy_trace_to_dict(trace: StrategyTrace | None) -> dict[str, Any] | None:
    """Convert a StrategyTrace dataclass into JSON-safe primitives."""
    if trace is None:
        return None
    return _json_safe(asdict(trace))


def _replace_strategy_trace_online_outcome(
    trace: StrategyTrace,
    outcome: OnlineInfluenceOutcome,
) -> StrategyTrace:
    return replace(trace, online_influence_outcome=outcome)


def _learned_live_canary_enabled(config: PhaseConfig, snapshot: CampaignSnapshot) -> bool:
    rollout = config.online_influence_rollout
    mode = str(getattr(rollout.mode, "value", rollout.mode))
    if not rollout.enabled or not rollout.enable_learned_safe_soft_live:
        return False
    if mode not in {OnlineInfluenceMode.SAFE_SOFT.value, OnlineInfluenceMode.EVALUATION.value}:
        return False
    if not rollout.allowed_campaign_ids or snapshot.campaign_id not in rollout.allowed_campaign_ids:
        return False
    context = snapshot.campaign_context
    level = str(getattr(
        context.current_objective_level if context is not None else ObjectiveLevel.PERFORMANCE,
        "value",
        context.current_objective_level if context is not None else ObjectiveLevel.PERFORMANCE,
    ))
    allowed_levels = tuple(str(getattr(item, "value", item)) for item in rollout.allowed_objective_levels)
    if not allowed_levels or level not in allowed_levels:
        return False
    entry = config.learned_policy_registry_entry
    if entry is None or config.learned_policy is None:
        return False
    return bool(getattr(entry, "approved_for_shadow", False) and getattr(entry, "approved_for_safe_soft", False))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
            backend_failure_counts=snapshot.backend_failure_counts,
            failed_params=snapshot.failed_params,
            campaign_context=snapshot.campaign_context,
            failure_events=snapshot.failure_events,
            campaign_id=snapshot.campaign_id,
            previous_intent=snapshot.previous_intent,
            backend_performance_records=snapshot.backend_performance_records,
            strategy_bandit_stats=snapshot.strategy_bandit_stats,
        )
        return select_strategy(new_snapshot, config)

    if available.get(backend, False):
        selected_intent = (
            CampaignIntent.DISCOVER
            if backend in ("lhs", "random_sampling", "nexus_lhs", "nexus_sobol")
            else CampaignIntent.OPTIMIZE
        )
        selected_mode = (
            OptimizationMode.EXPLORE
            if backend in ("lhs", "random_sampling", "nexus_lhs", "nexus_sobol")
            else OptimizationMode.EXPLOIT
        )
        trace = StrategyTrace(
            round_number=snapshot.round_number,
            selected_intent=selected_intent,
            selected_mode=selected_mode,
            selected_backend=backend,
            campaign_id=snapshot.campaign_id,
            state_summary={
                "n_observations": snapshot.n_observations,
                "n_dimensions": snapshot.n_dimensions,
                "qc_fail_rate": snapshot.qc_fail_rate,
            },
            context_summary=(snapshot.campaign_context or CampaignContext()).summary(),
            context_gate=ContextGateDecision(),
            space_revision=None,
            action_policy=None,
            transition_guard=None,
            ranking_influences=(),
            bandit_influence=None,
            online_influence_outcome=None,
            bandit_decision=None,
            shadow_bandit_record=None,
            objective_transition=None,
            strategy_reward=StrategyReward(),
            available_actions=(
                {
                    "name": "user_requested",
                    "backend_name": backend,
                    "expected_improvement": 0.0,
                    "expected_info_gain": 0.0,
                    "risk": 0.0,
                    "utility": 1.0,
                    "reason": f"User explicitly requested '{hint}'",
                },
            ),
            candidate_backends=({"name": backend, "total": 1.0},),
            nexus_recommendation=None,
            outcome=StrategyOutcome(outcome=None, reward=StrategyReward(), observed=False),
            learned_policy_shadow=None,
            learned_policy_influence=None,
            evidence=(
                StrategyEvidence(
                    source="human_prior",
                    target=backend,
                    effect="veto",
                    strength=1.0,
                    reason=f"User explicitly requested '{hint}'",
                ),
            ),
            reason=f"User requested '{hint}' → {backend}",
        )
        return StrategyDecision(
            backend_name=backend,
            phase="user_requested",
            reason=f"User requested '{hint}' → {backend}",
            confidence=1.0,
            fallback_backend="built_in",
            strategy_trace=trace,
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
        backend_failure_counts=snapshot.backend_failure_counts,
        failed_params=snapshot.failed_params,
        campaign_context=snapshot.campaign_context,
        failure_events=snapshot.failure_events,
        campaign_id=snapshot.campaign_id,
        previous_intent=snapshot.previous_intent,
        backend_performance_records=snapshot.backend_performance_records,
        strategy_bandit_stats=snapshot.strategy_bandit_stats,
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
    backend_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], StrategyDecision]:
    """One-call convenience: select strategy + generate candidates.

    Returns (candidates, decision); ``decision.backend_state`` carries any opaque
    state the chosen backend emitted (e.g. bomcp TuRBO trust region) so the
    caller can persist it and pass it back next round via ``backend_state``.
    """
    import dataclasses

    decision = select_strategy(snapshot, config=phase_config)

    new_state: dict[str, Any] | None = None
    try:
        backend = get_backend(decision.backend_name)
        candidates = backend.suggest(
            space, n, observations, seed=seed, backend_state=backend_state
        )
        new_state = getattr(backend, "last_backend_state", None)
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
            new_state = getattr(fallback, "last_backend_state", None)
        except Exception:
            logger.error("Fallback backend also failed, using LHS", exc_info=True)
            from app.services.candidate_gen import sample_lhs
            candidates = sample_lhs(space, n, seed=seed)

    decision = dataclasses.replace(decision, backend_state=new_state)

    # ----- Failure-region avoidance (Dim 9 / P3b) -----
    # Drop any candidate that falls in the learned failure region and top up
    # with feasible points, so *every* backend's output steers around coordinates
    # where past experiments failed.  No-op when no failures are recorded.
    if snapshot.failed_params:
        from app.services.failure_region import avoid_failure_region

        candidates = avoid_failure_region(
            candidates, space, n, list(snapshot.failed_params), seed
        )

    return candidates, decision


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# v2 callers may reference these — keep them importable
_determine_phase_from_data = None  # removed in v3
_select_backend_for_phase = None  # removed in v3
