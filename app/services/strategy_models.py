"""Data models for the Adaptive Strategy Selector.

All frozen dataclasses used across strategy sub-modules live here
so they can be imported without circular dependencies.

Public types:
    CampaignSnapshot, DiagnosticSignals, WeightsUsed, StabilizeSpec,
    EvidenceItem, ActionCandidate, PhasePosterior, StrategyDecision,
    PhaseConfig
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.backend_selection import BackendSelection


class ObjectiveLevel(StrEnum):
    """Scientific objective ladder used by the campaign controller."""

    FEASIBILITY = "feasibility"
    DATA_QUALITY = "data_quality"
    BASELINE = "baseline"
    PERFORMANCE = "performance"
    MECHANISM = "mechanism"
    GENERALIZATION = "generalization"


class FailureType(StrEnum):
    """Typed failure attribution for strategy learning."""

    HARDWARE = "hardware"
    PROTOCOL = "protocol"
    CONSTRAINT = "constraint"
    MEASUREMENT = "measurement"
    MODEL = "model"
    BACKEND = "backend"
    SCIENTIFIC_NEGATIVE = "scientific_negative"


class CampaignIntent(StrEnum):
    """Campaign-level action intent, above optimizer mode."""

    DISCOVER = "discover"
    OPTIMIZE = "optimize"
    VALIDATE = "validate"
    STABILIZE = "stabilize"
    RECOVER = "recover"
    DIAGNOSE = "diagnose"
    TRANSFER = "transfer"
    PIVOT = "pivot"
    REVISE_SPACE = "revise_space"
    HYPOTHESIS_GENERATE = "hypothesis_generate"
    HYPOTHESIS_TEST = "hypothesis_test"


class OptimizationMode(StrEnum):
    """Backend-facing optimization mode."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"
    REFINE = "refine"
    STABILIZE = "stabilize"
    CONSTRAINED_SEARCH = "constrained_search"
    MULTI_OBJECTIVE_TRADEOFF = "multi_objective_tradeoff"
    FAILURE_AVOIDANCE = "failure_avoidance"
    ROUTE_SWITCH = "route_switch"
    SPACE_REVISION = "revise_space"
    REPLICATE = "replicate"
    BASELINE_CALIBRATION = "baseline_calibration"
    MECHANISM_VALIDATION = "mechanism_validation"
    HYPOTHESIS_GENERATION = "hypothesis_generate"
    HYPOTHESIS_TEST = "hypothesis_test"
    CONTROL_PROBE = "control_probe"
    FAILURE_LOCALIZATION = "failure_localization"
    WARM_START = "warm_start"


class OnlineInfluenceMode(StrEnum):
    """Guarded online policy influence modes."""

    OFF = "off"
    SHADOW = "shadow"
    SAFE_SOFT = "safe_soft"
    EVALUATION = "evaluation"


class LearnedPolicyDeploymentMode(StrEnum):
    """Deployment mode for learned meta-policy shadowing."""

    OFF = "off"
    SHADOW = "shadow"
    SAFE_SOFT = "safe_soft"


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective in the campaign hierarchy."""

    level: ObjectiveLevel | str
    name: str
    metric: str | None = None
    direction: str = "maximize"
    target: float | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class ObjectiveHierarchy:
    """First-class objective ladder context for campaign-stage decisions."""

    objectives: tuple[ObjectiveSpec, ...] = ()
    current_level: ObjectiveLevel | str = ObjectiveLevel.PERFORMANCE
    active_objective: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class FailureEvent:
    """Typed failure/negative-result event available to strategy selection."""

    failure_type: FailureType | str
    reason: str
    backend_name: str | None = None
    round_number: int | None = None
    candidate_index: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    penalize_backend: bool = False


@dataclass(frozen=True)
class ParameterSpaceHealth:
    """Current health of the searchable parameter space."""

    n_failed_params: int = 0
    failed_params: tuple[dict[str, Any], ...] = ()
    known_constraints: tuple[dict[str, Any], ...] = ()
    infeasible_region_count: int = 0
    requires_revision: bool = False
    reason: str = ""


@dataclass(frozen=True)
class RouteContext:
    """Available synthesis or experiment routes and active route state."""

    routes: tuple[str, ...] = ()
    active_route: str | None = None
    suggested_route: str | None = None
    requires_route_switch: bool = False
    reason: str = ""


@dataclass(frozen=True)
class BudgetContext:
    """Remaining campaign budget for decision policy and trace review."""

    remaining: dict[str, Any] = field(default_factory=dict)
    max_rounds: int | None = None
    current_round: int | None = None
    pressure: str = "unknown"


@dataclass(frozen=True)
class DataQualityContext:
    """Measurement and data-quality state relevant to strategy selection."""

    measurement_protocols: tuple[dict[str, Any], ...] = ()
    instrument_state: dict[str, Any] = field(default_factory=dict)
    qc_fail_rate: float | None = None
    noise_ratio: float | None = None
    requires_calibration: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PriorCampaignContext:
    """Reusable prior campaign and literature context."""

    prior_campaigns: tuple[dict[str, Any], ...] = ()
    literature_priors: tuple[dict[str, Any], ...] = ()
    warm_start_available: bool = False
    transfer_reason: str = ""


@dataclass(frozen=True)
class CampaignContext:
    """Scientific context beyond the numerical optimization loop state."""

    scientific_goal: str = ""
    objective_hierarchy: tuple[ObjectiveSpec, ...] = ()
    objective_context: ObjectiveHierarchy | None = None
    current_objective_level: ObjectiveLevel | str = ObjectiveLevel.PERFORMANCE
    domain_hypotheses: tuple[str, ...] = ()
    known_constraints: tuple[dict[str, Any], ...] = ()
    parameter_space_health: ParameterSpaceHealth | None = None
    synthesis_routes: tuple[str, ...] = ()
    route_context: RouteContext | None = None
    measurement_protocols: tuple[dict[str, Any], ...] = ()
    instrument_state: dict[str, Any] = field(default_factory=dict)
    data_quality_context: DataQualityContext | None = None
    material_family: str | None = None
    prior_campaigns: tuple[dict[str, Any], ...] = ()
    literature_priors: tuple[dict[str, Any], ...] = ()
    prior_campaign_context: PriorCampaignContext | None = None
    budget_remaining: dict[str, Any] = field(default_factory=dict)
    budget_context: BudgetContext | None = None
    human_preferences: dict[str, Any] = field(default_factory=dict)
    human_observations: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """Compact JSON-safe summary for traces and event payloads."""
        objective_context = self.objective_context or ObjectiveHierarchy(
            objectives=self.objective_hierarchy,
            current_level=self.current_objective_level,
        )
        route_context = self.route_context or RouteContext(routes=self.synthesis_routes)
        data_quality_context = self.data_quality_context or DataQualityContext(
            measurement_protocols=self.measurement_protocols,
            instrument_state=self.instrument_state,
        )
        prior_campaign_context = self.prior_campaign_context or PriorCampaignContext(
            prior_campaigns=self.prior_campaigns,
            literature_priors=self.literature_priors,
            warm_start_available=bool(self.prior_campaigns or self.literature_priors),
        )
        budget_context = self.budget_context or BudgetContext(remaining=self.budget_remaining)
        parameter_space_health = self.parameter_space_health or ParameterSpaceHealth(
            known_constraints=self.known_constraints,
        )
        return {
            "scientific_goal": self.scientific_goal,
            "current_objective_level": getattr(
                self.current_objective_level, "value", self.current_objective_level
            ),
            "n_objectives": len(self.objective_hierarchy),
            "n_hypotheses": len(self.domain_hypotheses),
            "n_constraints": len(self.known_constraints),
            "synthesis_routes": list(self.synthesis_routes),
            "material_family": self.material_family,
            "budget_remaining": dict(self.budget_remaining),
            "n_literature_priors": len(self.literature_priors),
            "n_human_observations": len(self.human_observations),
            "objective_hierarchy": {
                "current_level": getattr(
                    objective_context.current_level,
                    "value",
                    objective_context.current_level,
                ),
                "active_objective": objective_context.active_objective,
                "n_objectives": len(objective_context.objectives),
                "rationale": objective_context.rationale,
            },
            "parameter_space_health": {
                "n_failed_params": parameter_space_health.n_failed_params,
                "infeasible_region_count": parameter_space_health.infeasible_region_count,
                "requires_revision": parameter_space_health.requires_revision,
                "reason": parameter_space_health.reason,
            },
            "route_context": {
                "routes": list(route_context.routes),
                "active_route": route_context.active_route,
                "suggested_route": route_context.suggested_route,
                "requires_route_switch": route_context.requires_route_switch,
                "reason": route_context.reason,
            },
            "budget_context": {
                "remaining": dict(budget_context.remaining),
                "max_rounds": budget_context.max_rounds,
                "current_round": budget_context.current_round,
                "pressure": budget_context.pressure,
            },
            "data_quality_context": {
                "n_measurement_protocols": len(data_quality_context.measurement_protocols),
                "instrument_state": dict(data_quality_context.instrument_state),
                "qc_fail_rate": data_quality_context.qc_fail_rate,
                "noise_ratio": data_quality_context.noise_ratio,
                "requires_calibration": data_quality_context.requires_calibration,
                "reason": data_quality_context.reason,
            },
            "prior_campaign_context": {
                "n_prior_campaigns": len(prior_campaign_context.prior_campaigns),
                "n_literature_priors": len(prior_campaign_context.literature_priors),
                "warm_start_available": prior_campaign_context.warm_start_available,
                "transfer_reason": prior_campaign_context.transfer_reason,
            },
        }


@dataclass(frozen=True)
class StrategyEvidence:
    """Evidence that boosts, penalizes, or vetoes an action/backend choice."""

    source: str
    target: str
    effect: str
    strength: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextGateDecision:
    """Reasoning gate result before backend optimization."""

    ready_for_optimization: bool = True
    requires_space_revision: bool = False
    requires_hypothesis_update: bool = False
    requires_route_switch: bool = False
    requires_calibration: bool = False
    requires_human_review: bool = False
    recommended_intent: CampaignIntent | str = CampaignIntent.OPTIMIZE
    reason: str = "ready for optimization"


@dataclass(frozen=True)
class SpaceRevision:
    """Proposed context-aware parameter-space or route revision."""

    revision_type: str = "constraint_update"
    lifecycle_status: str = "proposed"
    add_parameters: tuple[dict[str, Any], ...] = ()
    remove_parameters: tuple[str, ...] = ()
    narrow_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    expand_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    add_constraints: tuple[dict[str, Any], ...] = ()
    switch_route: str | None = None
    objective_level: ObjectiveLevel | str | None = None
    reason: str = ""
    risk_level: str = "medium"
    approval_required: bool = True
    affected_parameters: tuple[str, ...] = ()
    affected_routes: tuple[str, ...] = ()
    affected_objectives: tuple[str, ...] = ()
    auto_applied: bool = False


@dataclass(frozen=True)
class StrategyReward:
    """Logging-only reward decomposition for strategy evaluation."""

    objective_improvement: float = 0.0
    information_gain: float = 0.0
    constraint_satisfaction: float = 1.0
    data_quality_gain: float = 0.0
    novelty: float = 0.0
    failure_penalty: float = 0.0
    cost_penalty: float = 0.0
    time_penalty: float = 0.0
    composite_reward: float = 0.0
    reward_version: str = "strategy_reward_v1"


@dataclass(frozen=True)
class NexusRecommendationTrace:
    """First-class Nexus/fingerprint recommendation provenance."""

    recommended_backends: tuple[str, ...] = ()
    source: str = "nexus"
    reason: str = ""
    applied_as_evidence: bool = True
    selected_backend: str | None = None
    score_weight: float = 0.0


@dataclass(frozen=True)
class StrategyOutcome:
    """First-class per-round outcome container for later policy learning."""

    outcome: str | None = None
    reward: StrategyReward | None = None
    failure_events: tuple[FailureEvent, ...] = ()
    safety_flags: tuple[str, ...] = ()
    notes: str = ""
    observed: bool = False


@dataclass(frozen=True)
class LearnedPolicyRegistryEntry:
    """Registered learned policy metadata and deployment approvals."""

    policy_id: str
    policy_version: str
    trained_on_dataset_version: str
    feature_schema_version: str
    reward_version: str
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    approved_for_shadow: bool = False
    approved_for_safe_soft: bool = False


@dataclass(frozen=True)
class LearnedPolicyShadowRecord:
    """Live shadow-only learned policy output attached to StrategyTrace."""

    policy_id: str
    policy_version: str
    deployment_mode: LearnedPolicyDeploymentMode | str
    suggested_intent: str | None = None
    suggested_mode: str | None = None
    suggested_backend: str | None = None
    score_deltas: tuple[dict[str, Any], ...] = ()
    confidence: float = 0.0
    safety_mask_valid: bool = True
    invalid_suggestion_reasons: tuple[str, ...] = ()
    safety_warnings: tuple[str, ...] = ()
    actual_intent: str | None = None
    actual_mode: str | None = None
    actual_backend: str | None = None
    intent_agrees: bool = False
    mode_agrees: bool = False
    backend_agrees: bool = False
    would_change_top1: bool = False
    counterfactual_label: str = "unknown_counterfactual"
    reason: str = ""


@dataclass(frozen=True)
class LearnedPolicyPromotionGateResult:
    """Eligibility result for learned-policy tiny safe influence."""

    eligible: bool
    reasons: tuple[str, ...] = ()
    shadow_rounds: int = 0
    safety_warning_rate: float = 0.0
    invalid_suggestion_rate: float = 0.0
    confidence_calibration: float = 0.0
    top_k_agreement: float = 0.0
    offline_benchmark_passed: bool = False
    reward_sanity_passed: bool = False


@dataclass(frozen=True)
class LearnedPolicyInfluenceRecord:
    """Tiny learned-policy score influence audit record."""

    policy_id: str
    policy_version: str
    eligibility: LearnedPolicyPromotionGateResult
    suggested_backend: str | None = None
    target_backend: str | None = None
    raw_delta: float = 0.0
    applied_delta: float = 0.0
    capped: bool = False
    confidence: float = 0.0
    would_change_top1: bool = False
    changed_top1: bool = False
    safety_mask_valid: bool = True
    safety_warnings: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ShadowBanditEvaluationRecord:
    """Logging-only comparison between execution and shadow bandit advice."""

    actual_action: str
    actual_backend: str
    suggested_action: str
    suggested_backend: str
    agrees_with_actual: bool
    bandit_confidence: float
    actual_reward: float | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class ObjectiveTransitionProposal:
    """Proposal to move along the objective hierarchy."""

    from_level: ObjectiveLevel | str
    to_level: ObjectiveLevel | str
    reason: str
    evidence: tuple[StrategyEvidence, ...] = ()
    confidence: float = 0.0
    auto_applied: bool = False


@dataclass(frozen=True)
class ActionPolicyDecision:
    """Trace-only action policy priors for campaign intent and mode."""

    intent_priors: dict[str, float] = field(default_factory=dict)
    mode_priors: dict[str, float] = field(default_factory=dict)
    backend_priors: dict[str, float] = field(default_factory=dict)
    vetoes: tuple[str, ...] = ()
    evidence: tuple[StrategyEvidence, ...] = ()


@dataclass(frozen=True)
class ActionTransitionRecord:
    """Trace-only audit of campaign-intent transitions."""

    from_intent: CampaignIntent | str | None
    to_intent: CampaignIntent | str
    allowed: bool = True
    unstable: bool = False
    reason: str = ""
    evidence: tuple[StrategyEvidence, ...] = ()


@dataclass(frozen=True)
class PolicyInfluenceConfig:
    """Config gate for bounded, fully traced policy influence."""

    enable_action_policy_rerank: bool = False
    enable_backend_memory_rerank: bool = False
    enable_bandit_rerank: bool = False
    bandit_offline_eval_passed: bool = False
    bandit_calibration_score: float = 0.0
    enable_transition_guard_penalty: bool = False
    max_action_policy_weight: float = 0.08
    max_backend_memory_weight: float = 0.10
    max_bandit_weight: float = 0.05
    max_transition_guard_weight: float = 0.06
    max_total_score_delta: float = 0.15
    allow_action_policy_hard_veto: bool = False


@dataclass(frozen=True)
class RankingInfluenceRecord:
    """Applied or shadow ranking influence with capped score delta."""

    source: str
    target: str
    raw_signal: float
    applied_weight: float
    score_delta: float
    capped: bool
    reason: str


@dataclass(frozen=True)
class BanditEligibilityResult:
    """Eligibility result for contextual-bandit soft influence."""

    eligible: bool
    reasons: tuple[str, ...] = ()
    calibration_bucket: str = "unknown"


@dataclass(frozen=True)
class BanditInfluenceRecord:
    """Trace record for contextual-bandit soft influence."""

    suggested_action: str
    suggested_backend: str
    confidence: float
    eligibility: BanditEligibilityResult
    applied_weight: float
    score_delta: float
    capped: bool
    calibration_bucket: str
    reason: str


@dataclass(frozen=True)
class OnlineInfluenceRolloutConfig:
    """Rollout gate for guarded online soft influence."""

    enabled: bool = False
    mode: OnlineInfluenceMode | str = OnlineInfluenceMode.OFF
    allowed_campaign_ids: tuple[str, ...] = ()
    allowed_objective_levels: tuple[ObjectiveLevel | str, ...] = ()
    max_rounds: int | None = None
    auto_disable_on_safety_warning: bool = True
    auto_disable_on_cap_violation: bool = True
    auto_disable_on_unexplained_ranking_change: bool = True
    enable_bandit_soft_influence: bool = False
    enable_learned_safe_soft_live: bool = False
    learned_live_top1_change_rate_threshold: float = 0.25
    bandit_min_observations: int = 8
    bandit_min_confidence: float = 0.4
    bandit_min_calibration_score: float = 0.5
    bandit_allowed_objective_levels: tuple[ObjectiveLevel | str, ...] = ()
    bandit_disallowed_failure_types: tuple[FailureType | str, ...] = (
        FailureType.HARDWARE,
        FailureType.MEASUREMENT,
        FailureType.SCIENTIFIC_NEGATIVE,
    )
    max_action_policy_weight: float = 0.03
    max_backend_memory_weight: float = 0.02
    max_bandit_weight: float = 0.01
    max_total_score_delta: float = 0.04


@dataclass(frozen=True)
class OnlineInfluenceOutcome:
    """Per-round online influence audit record."""

    mode: OnlineInfluenceMode | str
    enabled: bool
    baseline_top_backend: str
    influenced_top_backend: str
    top1_changed: bool
    safe_influence_top_backend: str | None = None
    learned_influenced_top_backend: str | None = None
    learned_changed_top1: bool = False
    applied_influences: tuple[RankingInfluenceRecord, ...] = ()
    learned_policy_influences: tuple[LearnedPolicyInfluenceRecord, ...] = ()
    reward: float | None = None
    outcome: str | None = None
    failure_events: tuple[FailureEvent, ...] = ()
    safety_warnings: tuple[str, ...] = ()
    auto_disabled: bool = False
    reason: str = ""


@dataclass(frozen=True)
class BackendPerformance:
    """Credit assignment summary for a backend in a problem/context bucket."""

    backend_name: str
    action_type: str
    problem_fingerprint: str
    num_calls: int = 0
    success_rate: float = 0.0
    mean_improvement: float = 0.0
    mean_uncertainty_reduction: float = 0.0
    failure_rate: float = 0.0
    constraint_violation_rate: float = 0.0
    latency: float = 0.0
    cost: float = 0.0
    last_used_at: str | None = None


@dataclass(frozen=True)
class ContextualBanditDecision:
    """Bandit recommendation over action/backend arms."""

    selected_arm: str
    context_key: str
    arm_scores: tuple[dict[str, Any], ...]
    reason: str
    actual_action: str = ""
    actual_backend: str = ""
    suggested_action: str = ""
    suggested_backend: str = ""
    agrees_with_actual: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class StrategyTrace:
    """Structured decision provenance for one strategy-selection round."""

    round_number: int
    selected_intent: CampaignIntent | str
    selected_mode: OptimizationMode | str
    selected_backend: str
    campaign_id: str
    state_summary: dict[str, Any]
    context_summary: dict[str, Any]
    context_gate: ContextGateDecision | None
    space_revision: SpaceRevision | None
    action_policy: ActionPolicyDecision | None
    transition_guard: ActionTransitionRecord | None
    ranking_influences: tuple[RankingInfluenceRecord, ...]
    bandit_influence: BanditInfluenceRecord | None
    online_influence_outcome: OnlineInfluenceOutcome | None
    bandit_decision: ContextualBanditDecision | None
    shadow_bandit_record: ShadowBanditEvaluationRecord | None
    objective_transition: ObjectiveTransitionProposal | None
    strategy_reward: StrategyReward | None
    available_actions: tuple[dict[str, Any], ...]
    candidate_backends: tuple[dict[str, Any], ...]
    nexus_recommendation: NexusRecommendationTrace | None
    outcome: StrategyOutcome | None
    learned_policy_shadow: LearnedPolicyShadowRecord | None
    learned_policy_influence: LearnedPolicyInfluenceRecord | None
    evidence: tuple[StrategyEvidence, ...]
    reason: str


@dataclass(frozen=True)
class PolicyTrainingRecord:
    """Canonical offline learning row derived from a StrategyTrace."""

    campaign_id: str
    loop_id: str
    state_features: dict[str, Any]
    context_features: dict[str, Any]
    available_actions: tuple[dict[str, Any], ...]
    selected_intent: str
    selected_mode: str
    selected_backend: str
    candidate_backends: tuple[dict[str, Any], ...]
    applied_influences: tuple[dict[str, Any], ...]
    reward: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    safety_flags: tuple[str, ...] = ()
    record_version: str = "policy_training_record_v1"


@dataclass(frozen=True)
class PolicyReplayRecord:
    """Canonical replay row for counterfactual policy evaluation."""

    campaign_id: str
    loop_id: str
    state_features: dict[str, Any]
    context_features: dict[str, Any]
    available_actions: tuple[dict[str, Any], ...]
    selected_intent: str
    selected_mode: str
    selected_backend: str
    candidate_backends: tuple[dict[str, Any], ...]
    applied_influences: tuple[dict[str, Any], ...]
    reward: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    safety_flags: tuple[str, ...] = ()
    nexus_recommendation: dict[str, Any] | None = None
    record_version: str = "policy_replay_record_v1"


def policy_training_record_from_trace(
    trace: StrategyTrace,
    *,
    loop_id: str | None = None,
) -> PolicyTrainingRecord:
    """Convert a StrategyTrace into a stable offline training record."""
    outcome = _trace_outcome_dict(trace)
    return PolicyTrainingRecord(
        campaign_id=trace.campaign_id,
        loop_id=loop_id or f"round-{trace.round_number}",
        state_features=dict(trace.state_summary),
        context_features=dict(trace.context_summary),
        available_actions=tuple(dict(action) for action in trace.available_actions),
        selected_intent=_string_value(trace.selected_intent),
        selected_mode=_string_value(trace.selected_mode),
        selected_backend=trace.selected_backend,
        candidate_backends=tuple(dict(row) for row in trace.candidate_backends),
        applied_influences=tuple(_plain_dict(item) for item in trace.ranking_influences),
        reward=_plain_dict(trace.strategy_reward) if trace.strategy_reward is not None else None,
        outcome=outcome,
        safety_flags=_trace_safety_flags(trace),
    )


def policy_replay_record_from_trace(
    trace: StrategyTrace,
    *,
    loop_id: str | None = None,
) -> PolicyReplayRecord:
    """Convert a StrategyTrace into a stable replay/evaluation record."""
    training = policy_training_record_from_trace(trace, loop_id=loop_id)
    return PolicyReplayRecord(
        campaign_id=training.campaign_id,
        loop_id=training.loop_id,
        state_features=training.state_features,
        context_features=training.context_features,
        available_actions=training.available_actions,
        selected_intent=training.selected_intent,
        selected_mode=training.selected_mode,
        selected_backend=training.selected_backend,
        candidate_backends=training.candidate_backends,
        applied_influences=training.applied_influences,
        reward=training.reward,
        outcome=training.outcome,
        safety_flags=training.safety_flags,
        nexus_recommendation=(
            _plain_dict(trace.nexus_recommendation)
            if trace.nexus_recommendation is not None
            else None
        ),
    )


def _trace_outcome_dict(trace: StrategyTrace) -> dict[str, Any] | None:
    if trace.outcome is not None:
        return _plain_dict(trace.outcome)
    if trace.online_influence_outcome is not None:
        return _plain_dict(trace.online_influence_outcome)
    if trace.shadow_bandit_record is not None and trace.shadow_bandit_record.outcome:
        return _plain_dict(trace.shadow_bandit_record)
    return None


def _trace_safety_flags(trace: StrategyTrace) -> tuple[str, ...]:
    flags: list[str] = []
    if trace.online_influence_outcome is not None:
        flags.extend(trace.online_influence_outcome.safety_warnings)
        if trace.online_influence_outcome.auto_disabled:
            flags.append("online_influence_auto_disabled")
    if trace.space_revision is not None and trace.space_revision.auto_applied:
        flags.append("space_revision_auto_applied")
    if trace.transition_guard is not None and trace.transition_guard.unstable:
        flags.append("unstable_transition")
    return tuple(flags)


def _plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raw = {"value": value}
    return _plain_value(raw)


def _plain_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _string_value(value: Any) -> str:
    return str(getattr(value, "value", value))

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

    # --- Per-backend recent failure history (Δ2: penalize/veto in ranking) ---
    backend_failure_counts: dict[str, int] = field(default_factory=dict)

    # --- Failed experiment coordinates (Dim 9 / P3b: avoid in generation) ---
    # Param dicts of experiments that errored or failed QC; used to learn a
    # parameter-space failure region future candidates steer around.
    failed_params: tuple[dict[str, Any], ...] = ()

    # --- Scientific campaign context (beyond numerical loop state) ---
    campaign_context: CampaignContext | None = None
    failure_events: tuple[FailureEvent, ...] = ()
    campaign_id: str = ""
    previous_intent: CampaignIntent | str | None = None
    backend_performance_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy_bandit_stats: dict[str, dict[str, float]] = field(default_factory=dict)


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
    # Δ2: provenance for backend ranking (phase pool, fingerprint bias, scores)
    backend_selection: BackendSelection | None = None
    # Context-aware campaign decision trace.
    strategy_trace: StrategyTrace | None = None
    # (c) Opaque, JSON-safe backend state (e.g. bomcp TuRBO trust region) emitted
    # by the chosen backend; the caller persists it and passes it back next round.
    backend_state: dict[str, Any] | None = None
    # Δ2 backend-selection trace (audit/provenance): the fingerprint-soft-biased
    # backend ranking surfaced this round. Consumed by arbitration provenance;
    # advisory only.
    recommended_backends: tuple[str, ...] = ()


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

    # --- Optimization intelligence integration (v5, opt-in) ---
    enable_nexus: bool = False  # backward-compatible alias for Nexus-backed optimization intelligence
    enable_optimization_intelligence: bool = False  # causal + cross-campaign meta-learning advice

    # --- Method advisor (P3a): benchmark-derived problem-class -> method bias ---
    enable_method_advisor: bool = True  # soft-bias backend ranking by problem structure

    # --- Backend preferences ---
    # Order: existing optional backends first (no behaviour change when they are
    # installed), then Nexus equivalents (enrichment when optuna/scipy/pymoo are
    # absent), then ``bomcp`` (real GP-BO via bo-engine) just ahead of the
    # guaranteed ``built_in`` fallback -- so the KNN heuristic only runs when no
    # real engine is installed.  Whether ``bomcp`` should outrank optuna/nexus is
    # deferred to the method-comparison benchmark (dims 4/8).
    exploitation_backends: tuple[str, ...] = (
        "optuna_tpe",
        "nexus_tpe",
        "nexus_gp_bo",
        "bomcp",
        "built_in",
    )
    refinement_backends: tuple[str, ...] = (
        "optuna_cmaes",
        "nexus_cmaes",
        "scipy_de",
        "nexus_de",
        "bomcp",
        "built_in",
    )
    high_dim_backends: tuple[str, ...] = (
        "pymoo_nsga2",
        "nexus_nsga2",
        "optuna_tpe",
        "nexus_turbo",
        "bomcp",
        "built_in",
    )
    explore_backends: tuple[str, ...] = (
        "lhs",
        "nexus_lhs",
        "nexus_sobol",
    )

    # --- Backend ranking weights (Δ2: conservative fingerprint soft-bias) ---
    # Phase policy dominates; the fingerprint recommendation is a secondary
    # additive boost; recent per-backend failures penalize and (past the
    # threshold) veto.  Defaults are tuned so a recommendation can flip an
    # adjacent preference but cannot overturn a clearly-preferred backend.
    backend_phase_weight: float = 1.0
    backend_fingerprint_weight: float = 0.3
    backend_failure_penalty: float = 0.5
    backend_failure_veto_threshold: int = 3
    policy_influence: PolicyInfluenceConfig = field(default_factory=PolicyInfluenceConfig)
    online_influence_rollout: OnlineInfluenceRolloutConfig = field(
        default_factory=OnlineInfluenceRolloutConfig
    )
    learned_policy_registry_entry: Any | None = None
    learned_policy: Any | None = None
    learned_policy_shadow_summary: dict[str, Any] = field(default_factory=dict)
    learned_policy_min_shadow_rounds: int = 10
    learned_policy_max_safe_soft_delta: float = 0.005
