"""Bounded, opt-in influence on backend ranking."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.services.backend_memory import BackendPerformanceMemory, problem_context_key
from app.services.bandit_influence import bandit_soft_influence
from app.services.strategy_models import (
    ActionPolicyDecision,
    ActionTransitionRecord,
    CampaignSnapshot,
    ContextualBanditDecision,
    FailureType,
    OnlineInfluenceRolloutConfig,
    PolicyInfluenceConfig,
    RankingInfluenceRecord,
    StrategyEvidence,
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _bounded_delta(raw_signal: float, weight: float, cap: float) -> tuple[float, bool]:
    proposed = raw_signal * weight
    bounded = max(-cap, min(cap, proposed))
    return round(bounded, 4), abs(bounded - proposed) > 1e-12


def _add_backend_delta(
    deltas: dict[str, float],
    backend: str,
    delta: float,
    total_cap: float,
) -> bool:
    proposed = deltas.get(backend, 0.0) + delta
    bounded = max(-total_cap, min(total_cap, proposed))
    deltas[backend] = round(bounded, 4)
    return abs(bounded - proposed) > 1e-12


def bounded_ranking_influence(
    *,
    snapshot: CampaignSnapshot,
    backend_pool: tuple[str, ...],
    action_name: str,
    action_policy: ActionPolicyDecision,
    transition_guard: ActionTransitionRecord,
    bandit_decision: ContextualBanditDecision | None,
    config: PolicyInfluenceConfig,
    rollout: OnlineInfluenceRolloutConfig | None = None,
) -> tuple[dict[str, float], tuple[RankingInfluenceRecord, ...], tuple[StrategyEvidence, ...]]:
    """Compute opt-in ranking deltas and trace records.

    Defaults produce empty deltas. Space revisions are intentionally not
    consulted here; they remain approval-only proposals.
    """
    deltas: dict[str, float] = defaultdict(float)
    records: list[RankingInfluenceRecord] = []
    evidence: list[StrategyEvidence] = []

    if config.enable_action_policy_rerank:
        _apply_action_policy(
            action_policy,
            backend_pool,
            config,
            deltas,
            records,
            evidence,
        )

    if config.enable_backend_memory_rerank:
        _apply_backend_memory(
            snapshot,
            backend_pool,
            action_name,
            config,
            deltas,
            records,
            evidence,
        )

    if config.enable_transition_guard_penalty:
        _apply_transition_guard(
            transition_guard,
            backend_pool,
            config,
            deltas,
            records,
            evidence,
        )

    if config.enable_bandit_rerank and bandit_decision is not None:
        if rollout is None:
            records.append(RankingInfluenceRecord(
                source="bandit_shadow",
                target=bandit_decision.suggested_backend,
                raw_signal=bandit_decision.confidence,
                applied_weight=0.0,
                score_delta=0.0,
                capped=False,
                reason="Contextual bandit remains shadow-only; no execution influence applied",
            ))
        else:
            bandit_deltas, bandit_records, bandit_evidence, _bandit_record = (
                bandit_soft_influence(
                    snapshot=snapshot,
                    decision=bandit_decision,
                    backend_pool=backend_pool,
                    policy_config=config,
                    rollout=rollout,
                )
            )
            for backend, delta in bandit_deltas.items():
                _add_backend_delta(deltas, backend, delta, config.max_total_score_delta)
            records.extend(bandit_records)
            evidence.extend(bandit_evidence)

    return dict(deltas), tuple(records), tuple(evidence)


def _apply_action_policy(
    action_policy: ActionPolicyDecision,
    backend_pool: tuple[str, ...],
    config: PolicyInfluenceConfig,
    deltas: dict[str, float],
    records: list[RankingInfluenceRecord],
    evidence: list[StrategyEvidence],
) -> None:
    for backend in backend_pool:
        raw_signal = action_policy.backend_priors.get(backend, 0.0)
        if raw_signal == 0:
            continue
        delta, capped = _bounded_delta(
            raw_signal,
            config.max_action_policy_weight,
            config.max_action_policy_weight,
        )
        capped = _add_backend_delta(deltas, backend, delta, config.max_total_score_delta) or capped
        records.append(RankingInfluenceRecord(
            source="action_policy",
            target=backend,
            raw_signal=raw_signal,
            applied_weight=config.max_action_policy_weight,
            score_delta=delta,
            capped=capped,
            reason="Action policy backend prior applied within configured cap",
        ))
        evidence.append(StrategyEvidence(
            source="ranking_influence:action_policy",
            target=backend,
            effect="boost" if delta >= 0 else "penalize",
            strength=abs(delta),
            reason="Bounded action-policy rerank influence applied",
            metadata={"raw_signal": raw_signal, "capped": capped},
        ))

    if action_policy.vetoes and not config.allow_action_policy_hard_veto:
        for target in action_policy.vetoes:
            records.append(RankingInfluenceRecord(
                source="action_policy_veto_shadow",
                target=target,
                raw_signal=1.0,
                applied_weight=0.0,
                score_delta=0.0,
                capped=True,
                reason="Action-policy veto recorded but hard veto is disabled",
            ))


def _apply_backend_memory(
    snapshot: CampaignSnapshot,
    backend_pool: tuple[str, ...],
    action_name: str,
    config: PolicyInfluenceConfig,
    deltas: dict[str, float],
    records: list[RankingInfluenceRecord],
    evidence: list[StrategyEvidence],
) -> None:
    context_key = problem_context_key(snapshot)
    memory = BackendPerformanceMemory(snapshot.backend_performance_records)
    for backend in backend_pool:
        raw_signal = min(0.0, memory.bias_for(context_key, backend, action_name))
        if raw_signal == 0.0:
            continue
        delta, capped = _bounded_delta(
            raw_signal,
            config.max_backend_memory_weight,
            config.max_backend_memory_weight,
        )
        capped = _add_backend_delta(deltas, backend, delta, config.max_total_score_delta) or capped
        records.append(RankingInfluenceRecord(
            source="backend_memory",
            target=backend,
            raw_signal=raw_signal,
            applied_weight=config.max_backend_memory_weight,
            score_delta=delta,
            capped=capped,
            reason="Attributed backend memory penalty applied within configured cap",
        ))
        evidence.append(StrategyEvidence(
            source="ranking_influence:backend_memory",
            target=backend,
            effect="penalize",
            strength=abs(delta),
            reason="Bounded backend-memory rerank penalty applied",
            metadata={"context_key": context_key, "raw_signal": raw_signal, "capped": capped},
        ))

    for failure in snapshot.failure_events:
        ftype = _value(failure.failure_type)
        if ftype not in (FailureType.CONSTRAINT.value, FailureType.BACKEND.value):
            continue
        backend = failure.backend_name
        if not backend or backend not in backend_pool:
            continue
        raw_signal = -1.0
        delta, capped = _bounded_delta(
            raw_signal,
            config.max_backend_memory_weight,
            config.max_backend_memory_weight,
        )
        capped = _add_backend_delta(deltas, backend, delta, config.max_total_score_delta) or capped
        records.append(RankingInfluenceRecord(
            source="backend_memory_failure_event",
            target=backend,
            raw_signal=raw_signal,
            applied_weight=config.max_backend_memory_weight,
            score_delta=delta,
            capped=capped,
            reason=f"Attributed {ftype} failure penalized backend within configured cap",
        ))
        evidence.append(StrategyEvidence(
            source="ranking_influence:backend_memory",
            target=backend,
            effect="penalize",
            strength=abs(delta),
            reason=f"Bounded {ftype} failure penalty applied",
            metadata={"failure_type": ftype, "capped": capped},
        ))


def _apply_transition_guard(
    transition_guard: ActionTransitionRecord,
    backend_pool: tuple[str, ...],
    config: PolicyInfluenceConfig,
    deltas: dict[str, float],
    records: list[RankingInfluenceRecord],
    evidence: list[StrategyEvidence],
) -> None:
    if not transition_guard.unstable:
        return
    raw_signal = -1.0
    for backend in backend_pool:
        delta, capped = _bounded_delta(
            raw_signal,
            config.max_transition_guard_weight,
            config.max_transition_guard_weight,
        )
        capped = _add_backend_delta(deltas, backend, delta, config.max_total_score_delta) or capped
        records.append(RankingInfluenceRecord(
            source="transition_guard",
            target=backend,
            raw_signal=raw_signal,
            applied_weight=config.max_transition_guard_weight,
            score_delta=delta,
            capped=capped,
            reason="Unstable intent transition applied as equal bounded backend penalty",
        ))
    evidence.append(StrategyEvidence(
        source="ranking_influence:transition_guard",
        target=f"{transition_guard.from_intent}->{transition_guard.to_intent}",
        effect="penalize",
        strength=config.max_transition_guard_weight,
        reason="Bounded unstable-transition penalty recorded",
        metadata={"unstable": transition_guard.unstable},
    ))
