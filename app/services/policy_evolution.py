"""Offline policy-evolution lifecycle registry and guardrails.

This module is deliberately not imported by ``strategy_selector``.  It records
and evaluates policy-evolution plans, but it does not train policies, rank
backends, promote policies, or change live execution behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class PolicyEvolutionTriggerType(StrEnum):
    """Reasons a policy-evolution review may be proposed."""

    NEW_TRACES_AVAILABLE = "new_traces_available"
    DATASET_SIZE_THRESHOLD_MET = "dataset_size_threshold_met"
    REWARD_DRIFT_DETECTED = "reward_drift_detected"
    BACKEND_PERFORMANCE_SHIFT = "backend_performance_shift"
    CURRENT_POLICY_UNDERPERFORMANCE = "current_policy_underperformance"
    SHADOW_POLICY_OUTPERFORMED = "shadow_policy_outperformed"
    CANARY_POLICY_PASSED = "canary_policy_passed"
    MANUAL_REQUEST = "manual_request"


class PolicyEvolutionPlanStatus(StrEnum):
    """Lifecycle state for an evolution plan."""

    PROPOSED = "proposed"
    DATASET_READY = "dataset_ready"
    OFFLINE_EVALUATED = "offline_evaluated"
    SHADOW_ELIGIBLE = "shadow_eligible"
    CANARY_ELIGIBLE = "canary_eligible"
    PROMOTION_ELIGIBLE = "promotion_eligible"
    WEIGHT_TUNING_ELIGIBLE = "weight_tuning_eligible"
    STRUCTURE_REVIEW_ELIGIBLE = "structure_review_eligible"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class PolicyEvolutionRecommendation(StrEnum):
    """Recommended next review action for a plan."""

    KEEP_CURRENT = "keep_current"
    PREPARE_DATASET = "prepare_dataset"
    TRAIN_CANDIDATE = "train_candidate"
    RUN_OFFLINE_EVAL = "run_offline_eval"
    APPROVE_SHADOW = "approve_shadow"
    APPROVE_CANARY = "approve_canary"
    PROPOSE_PROMOTION = "propose_promotion"
    APPROVE_PROMOTION = "approve_promotion"
    APPROVE_WEIGHT_TUNING = "approve_weight_tuning"
    REVIEW_STRUCTURE_PROPOSAL = "review_structure_proposal"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REJECT = "reject"


class CandidatePolicyTrainingMode(StrEnum):
    """Offline candidate policy training modes."""

    IMITATION = "imitation"
    BACKEND_RERANKER = "backend_reranker"
    META_POLICY = "meta_policy"


class CandidatePolicyTrainingJobStatus(StrEnum):
    """Lifecycle state for an offline candidate-policy training job."""

    CREATED = "created"
    DATASET_BUILT = "dataset_built"
    AUDIT_PASSED = "audit_passed"
    REWARD_SANITY_PASSED = "reward_sanity_passed"
    TRAINED = "trained"
    OFFLINE_EVALUATED = "offline_evaluated"
    FAILED = "failed"


class ShadowPromotionProposalStatus(StrEnum):
    """Lifecycle state for a shadow deployment proposal."""

    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ShadowApprovalMode(StrEnum):
    """Explicit approval source for shadow-only policy runs."""

    MANUAL = "manual"
    CONFIG = "config"
    TEST = "test"


class ShadowRunScheduleStatus(StrEnum):
    """Lifecycle state for a shadow run schedule."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ShadowRunRecommendation(StrEnum):
    """Recommendation from a completed shadow run."""

    CONTINUE_SHADOW = "continue_shadow"
    PROPOSE_CANARY = "propose_canary"
    REDUCE_SCOPE = "reduce_scope"
    REJECT_POLICY = "reject_policy"


class CanaryPromotionProposalStatus(StrEnum):
    """Lifecycle state for a canary deployment proposal."""

    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class CanaryApprovalMode(StrEnum):
    """Explicit approval source for bounded canary runs."""

    MANUAL = "manual"
    CONFIG = "config"
    TEST = "test"


class CanaryRunScheduleStatus(StrEnum):
    """Lifecycle state for a bounded canary run schedule."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    AUTO_DISABLED = "auto_disabled"


class CanaryRunRecommendation(StrEnum):
    """Recommendation from a completed bounded canary run."""

    CONTINUE_CANARY = "continue_canary"
    REDUCE_WEIGHT = "reduce_weight"
    CONTINUE_SHADOW = "continue_shadow"
    ROLLBACK = "rollback"
    REJECT_POLICY = "reject_policy"
    PROPOSE_PROMOTION = "propose_promotion"


class FinalPromotionProposalStatus(StrEnum):
    """Lifecycle state for a final promotion proposal."""

    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class FinalApprovalMode(StrEnum):
    """Explicit approval source for final safe-soft eligibility."""

    MANUAL = "manual"
    CONFIG = "config"
    TEST = "test"


class PolicyWeightTuningTarget(StrEnum):
    """Policy influence weight knobs that can be proposed for review."""

    ACTION_POLICY_MAX_WEIGHT = "action_policy_max_weight"
    BACKEND_MEMORY_MAX_WEIGHT = "backend_memory_max_weight"
    BANDIT_MAX_WEIGHT = "bandit_max_weight"
    LEARNED_POLICY_MAX_WEIGHT = "learned_policy_max_weight"
    TRANSITION_GUARD_PENALTY = "transition_guard_penalty"
    TOTAL_INFLUENCE_CAP = "total_influence_cap"


class WeightTuningEvidenceSource(StrEnum):
    """Evidence sources for proposal-only weight tuning."""

    OFFLINE_EVAL = "offline_eval"
    SHADOW_RUN = "shadow_run"
    CANARY_RUN = "canary_run"
    FINAL_APPROVAL = "final_approval"
    SAFETY_REPORT = "safety_report"
    REWARD_REPORT = "reward_report"
    FAILURE_REPORT = "failure_report"


class PolicyWeightTuningRiskLevel(StrEnum):
    """Risk label for a proposed weight change."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyWeightTuningProposalStatus(StrEnum):
    """Lifecycle state for a policy weight tuning proposal."""

    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PolicyStructureProposalType(StrEnum):
    """Structural policy/controller changes that can be proposed for review."""

    NEW_ACTION_MODE = "new_action_mode"
    NEW_CAMPAIGN_INTENT = "new_campaign_intent"
    NEW_CONTEXT_FEATURE = "new_context_feature"
    NEW_FAILURE_SUBTYPE = "new_failure_subtype"
    NEW_POLICY_RULE = "new_policy_rule"
    MODIFY_POLICY_RULE = "modify_policy_rule"
    DEPRECATE_POLICY_RULE = "deprecate_policy_rule"
    NEW_TRANSITION_GUARD_RULE = "new_transition_guard_rule"
    MODIFY_BACKEND_PRIOR_RULE = "modify_backend_prior_rule"
    REWARD_FEATURE_CHANGE = "reward_feature_change"


class PolicyStructureEvidenceSource(StrEnum):
    """Evidence source for policy structure proposals."""

    TRACE_ANALYSIS = "trace_analysis"
    REPLAY_EVALUATION = "replay_evaluation"
    SHADOW_RUN = "shadow_run"
    CANARY_RUN = "canary_run"
    REWARD_REPORT = "reward_report"
    FAILURE_REPORT = "failure_report"
    ABLATION_REPORT = "ablation_report"
    HUMAN_OBSERVATION = "human_observation"


class PolicyStructureRiskLevel(StrEnum):
    """Risk label for structural policy proposals."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyStructureProposalStatus(StrEnum):
    """Lifecycle state for structural policy proposals."""

    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PolicyEvolutionStage(StrEnum):
    """Auditable stages in the human-approved policy evolution workflow."""

    TRIGGERED = "triggered"
    PLANNED = "planned"
    TRAINING_REQUESTED = "training_requested"
    CANDIDATE_TRAINED = "candidate_trained"
    OFFLINE_EVALUATED = "offline_evaluated"
    SHADOW_PROPOSED = "shadow_proposed"
    SHADOW_APPROVED = "shadow_approved"
    SHADOW_RUNNING = "shadow_running"
    SHADOW_COMPLETED = "shadow_completed"
    CANARY_PROPOSED = "canary_proposed"
    CANARY_APPROVED = "canary_approved"
    CANARY_RUNNING = "canary_running"
    CANARY_COMPLETED = "canary_completed"
    PROMOTION_PROPOSED = "promotion_proposed"
    FINAL_APPROVED = "final_approved"
    WEIGHT_TUNING_PROPOSED = "weight_tuning_proposed"
    STRUCTURE_REVIEW_PROPOSED = "structure_review_proposed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class PolicyEvolutionWorkflowStatus(StrEnum):
    """High-level workflow status."""

    ACTIVE = "active"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class PolicyEvolutionAuditActorType(StrEnum):
    """Actor types accepted in workflow audit logs."""

    SYSTEM = "system"
    HUMAN = "human"
    CONFIG = "config"
    TEST = "test"


class PolicyEvolutionWorkflowRecommendation(StrEnum):
    """Recommended next workflow action."""

    CONTINUE_WORKFLOW = "continue_workflow"
    WAIT_FOR_APPROVAL = "wait_for_approval"
    RUN_TRAINING = "run_training"
    RUN_SHADOW = "run_shadow"
    RUN_CANARY = "run_canary"
    APPROVE_PROMOTION = "approve_promotion"
    REVIEW_WEIGHT_TUNING = "review_weight_tuning"
    REVIEW_STRUCTURE_PROPOSAL = "review_structure_proposal"
    ROLLBACK = "rollback"
    REJECT = "reject"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PolicyEvolutionTrigger:
    """Structured reason to start a policy-evolution review."""

    trigger_type: PolicyEvolutionTriggerType | str
    trigger_reason: str
    campaign_ids: tuple[str, ...] = ()
    trace_count: int = 0
    dataset_version: str | None = None
    created_at: str = field(default_factory=lambda: _now_iso())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyEvolutionTrigger:
        return cls(
            trigger_type=raw.get("trigger_type", PolicyEvolutionTriggerType.MANUAL_REQUEST),
            trigger_reason=str(raw.get("trigger_reason", "")),
            campaign_ids=tuple(raw.get("campaign_ids") or ()),
            trace_count=int(raw.get("trace_count") or 0),
            dataset_version=raw.get("dataset_version"),
            created_at=str(raw.get("created_at") or _now_iso()),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PolicyEvolutionPlan:
    """Review plan for one candidate policy version."""

    plan_id: str
    source_policy_id: str
    source_policy_version: str
    candidate_policy_id: str
    candidate_policy_version: str
    trigger: PolicyEvolutionTrigger
    dataset_version: str | None
    feature_schema_version: str
    reward_version: str
    required_checks: tuple[str, ...] = (
        "dataset_audit",
        "reward_sanity",
        "offline_benchmark",
        "promotion_gate",
        "evolution_guard",
    )
    shadow_required: bool = True
    canary_required: bool = True
    promotion_allowed: bool = False
    rollback_policy_id: str | None = None
    rollback_policy_version: str | None = None
    status: PolicyEvolutionPlanStatus | str = PolicyEvolutionPlanStatus.PROPOSED
    reasons: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())
    proposed_changes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyEvolutionPlan:
        return cls(
            plan_id=str(raw.get("plan_id", "")),
            source_policy_id=str(raw.get("source_policy_id", "")),
            source_policy_version=str(raw.get("source_policy_version", "")),
            candidate_policy_id=str(raw.get("candidate_policy_id", "")),
            candidate_policy_version=str(raw.get("candidate_policy_version", "")),
            trigger=PolicyEvolutionTrigger.from_dict(dict(raw.get("trigger") or {})),
            dataset_version=raw.get("dataset_version"),
            feature_schema_version=str(raw.get("feature_schema_version", "")),
            reward_version=str(raw.get("reward_version", "")),
            required_checks=tuple(raw.get("required_checks") or ()),
            shadow_required=bool(raw.get("shadow_required", True)),
            canary_required=bool(raw.get("canary_required", True)),
            promotion_allowed=bool(raw.get("promotion_allowed", False)),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            status=raw.get("status", PolicyEvolutionPlanStatus.PROPOSED),
            reasons=tuple(raw.get("reasons") or ()),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
            proposed_changes=dict(raw.get("proposed_changes") or {}),
        )


@dataclass(frozen=True)
class PolicyEvolutionWorkflow:
    """Single auditable workflow tying together policy evolution artifacts."""

    workflow_id: str
    trigger_id: str
    plan_id: str
    source_policy_id: str
    source_policy_version: str
    candidate_policy_id: str
    candidate_policy_version: str
    training_job_id: str | None = None
    candidate_artifact_id: str | None = None
    shadow_proposal_id: str | None = None
    shadow_approval_id: str | None = None
    shadow_schedule_id: str | None = None
    shadow_result_id: str | None = None
    canary_proposal_id: str | None = None
    canary_approval_id: str | None = None
    canary_schedule_id: str | None = None
    canary_result_id: str | None = None
    final_promotion_proposal_id: str | None = None
    final_approval_id: str | None = None
    weight_tuning_proposal_ids: tuple[str, ...] = ()
    structure_proposal_ids: tuple[str, ...] = ()
    current_stage: PolicyEvolutionStage | str = PolicyEvolutionStage.TRIGGERED
    status: PolicyEvolutionWorkflowStatus | str = PolicyEvolutionWorkflowStatus.ACTIVE
    rollback_policy_id: str | None = None
    rollback_policy_version: str | None = None
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyEvolutionWorkflow:
        return cls(
            workflow_id=str(raw.get("workflow_id", "")),
            trigger_id=str(raw.get("trigger_id", "")),
            plan_id=str(raw.get("plan_id", "")),
            source_policy_id=str(raw.get("source_policy_id", "")),
            source_policy_version=str(raw.get("source_policy_version", "")),
            candidate_policy_id=str(raw.get("candidate_policy_id", "")),
            candidate_policy_version=str(raw.get("candidate_policy_version", "")),
            training_job_id=raw.get("training_job_id"),
            candidate_artifact_id=raw.get("candidate_artifact_id"),
            shadow_proposal_id=raw.get("shadow_proposal_id"),
            shadow_approval_id=raw.get("shadow_approval_id"),
            shadow_schedule_id=raw.get("shadow_schedule_id"),
            shadow_result_id=raw.get("shadow_result_id"),
            canary_proposal_id=raw.get("canary_proposal_id"),
            canary_approval_id=raw.get("canary_approval_id"),
            canary_schedule_id=raw.get("canary_schedule_id"),
            canary_result_id=raw.get("canary_result_id"),
            final_promotion_proposal_id=raw.get("final_promotion_proposal_id"),
            final_approval_id=raw.get("final_approval_id"),
            weight_tuning_proposal_ids=tuple(raw.get("weight_tuning_proposal_ids") or ()),
            structure_proposal_ids=tuple(raw.get("structure_proposal_ids") or ()),
            current_stage=raw.get("current_stage", PolicyEvolutionStage.TRIGGERED),
            status=raw.get("status", PolicyEvolutionWorkflowStatus.ACTIVE),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
        )


@dataclass(frozen=True)
class PolicyEvolutionAuditLogEntry:
    """Append-only audit record for workflow transitions and attachments."""

    entry_id: str
    workflow_id: str
    actor_type: PolicyEvolutionAuditActorType | str
    action: str
    from_stage: PolicyEvolutionStage | str
    to_stage: PolicyEvolutionStage | str
    reason: str
    guard_allowed: bool
    guard_violations: tuple[dict[str, Any], ...] = ()
    guard_warnings: tuple[dict[str, Any], ...] = ()
    timestamp: str = field(default_factory=lambda: _now_iso())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class CandidatePolicyTrainingJob:
    """Offline training job derived from a policy-evolution plan."""

    job_id: str
    plan_id: str
    source_policy_id: str
    source_policy_version: str
    candidate_policy_id: str
    candidate_policy_version: str
    dataset_version: str | None
    feature_schema_version: str
    reward_version: str
    training_mode: CandidatePolicyTrainingMode | str
    training_config: dict[str, Any] = field(default_factory=dict)
    status: CandidatePolicyTrainingJobStatus | str = CandidatePolicyTrainingJobStatus.CREATED
    failure_reason: str | None = None
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class CandidatePolicyArtifact:
    """Offline candidate policy artifact and evaluation summary."""

    policy_id: str
    policy_version: str
    parent_policy_id: str
    parent_policy_version: str
    artifact_type: str
    training_mode: CandidatePolicyTrainingMode | str
    dataset_version: str | None
    feature_schema_version: str
    reward_version: str
    training_summary: dict[str, Any]
    offline_evaluation_summary: dict[str, Any]
    safety_summary: dict[str, Any]
    eligible_for_shadow_proposal: bool
    eligible_for_canary_proposal: bool = False
    shadow_promotion_eligible: bool = False
    shadow_promotion_reason: str = ""
    registry_entry_preview: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class ShadowPromotionProposal:
    """Proposal to run an offline-evaluated candidate in shadow mode."""

    proposal_id: str
    plan_id: str
    training_job_id: str
    candidate_policy_id: str
    candidate_policy_version: str
    source_policy_id: str
    source_policy_version: str
    dataset_version: str | None
    feature_schema_version: str
    reward_version: str
    offline_evaluation_summary: dict[str, Any]
    dataset_audit_summary: dict[str, Any]
    reward_sanity_summary: dict[str, Any]
    safety_summary: dict[str, Any]
    counterfactual_uncertainty_summary: dict[str, Any]
    rollback_policy_id: str | None
    rollback_policy_version: str | None
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    required_approvals: tuple[str, ...]
    status: ShadowPromotionProposalStatus | str = ShadowPromotionProposalStatus.PROPOSED
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShadowPromotionProposal:
        return cls(
            proposal_id=str(raw.get("proposal_id", "")),
            plan_id=str(raw.get("plan_id", "")),
            training_job_id=str(raw.get("training_job_id", "")),
            candidate_policy_id=str(raw.get("candidate_policy_id", "")),
            candidate_policy_version=str(raw.get("candidate_policy_version", "")),
            source_policy_id=str(raw.get("source_policy_id", "")),
            source_policy_version=str(raw.get("source_policy_version", "")),
            dataset_version=raw.get("dataset_version"),
            feature_schema_version=str(raw.get("feature_schema_version", "")),
            reward_version=str(raw.get("reward_version", "")),
            offline_evaluation_summary=dict(raw.get("offline_evaluation_summary") or {}),
            dataset_audit_summary=dict(raw.get("dataset_audit_summary") or {}),
            reward_sanity_summary=dict(raw.get("reward_sanity_summary") or {}),
            safety_summary=dict(raw.get("safety_summary") or {}),
            counterfactual_uncertainty_summary=dict(raw.get("counterfactual_uncertainty_summary") or {}),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            eligible=bool(raw.get("eligible", False)),
            eligibility_reasons=tuple(raw.get("eligibility_reasons") or ()),
            required_approvals=tuple(raw.get("required_approvals") or ()),
            status=raw.get("status", ShadowPromotionProposalStatus.PROPOSED),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
        )


@dataclass(frozen=True)
class ShadowApprovalRecord:
    """Explicit human/config/test approval to schedule a shadow run."""

    approval_id: str
    proposal_id: str
    policy_id: str
    policy_version: str
    approved_by: str
    approval_mode: ShadowApprovalMode | str
    approval_reason: str
    approved_at: str = field(default_factory=lambda: _now_iso())
    expires_at: str | None = None
    max_shadow_rounds: int = 0
    allowed_campaign_ids: tuple[str, ...] = ()
    allowed_objective_levels: tuple[str, ...] = ()
    revoked: bool = False
    revoked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShadowApprovalRecord:
        return cls(
            approval_id=str(raw.get("approval_id", "")),
            proposal_id=str(raw.get("proposal_id", "")),
            policy_id=str(raw.get("policy_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            approved_by=str(raw.get("approved_by", "")),
            approval_mode=raw.get("approval_mode", ShadowApprovalMode.MANUAL),
            approval_reason=str(raw.get("approval_reason", "")),
            approved_at=str(raw.get("approved_at") or _now_iso()),
            expires_at=raw.get("expires_at"),
            max_shadow_rounds=int(raw.get("max_shadow_rounds") or 0),
            allowed_campaign_ids=tuple(raw.get("allowed_campaign_ids") or ()),
            allowed_objective_levels=tuple(raw.get("allowed_objective_levels") or ()),
            revoked=bool(raw.get("revoked", False)),
            revoked_reason=raw.get("revoked_reason"),
        )


@dataclass(frozen=True)
class ShadowRunSchedule:
    """Approved shadow-only run schedule."""

    schedule_id: str
    approval_id: str
    policy_id: str
    policy_version: str
    campaign_allowlist: tuple[str, ...] = ()
    objective_allowlist: tuple[str, ...] = ()
    max_rounds: int = 0
    status: ShadowRunScheduleStatus | str = ShadowRunScheduleStatus.SCHEDULED
    created_at: str = field(default_factory=lambda: _now_iso())
    started_at: str | None = None
    completed_at: str | None = None
    cancellation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class ShadowRunResult:
    """Aggregate result from a shadow-only policy run."""

    run_id: str
    schedule_id: str
    policy_id: str
    policy_version: str
    campaign_ids: tuple[str, ...] = ()
    round_count: int = 0
    intent_agreement_rate: float = 0.0
    mode_agreement_rate: float = 0.0
    backend_agreement_rate: float = 0.0
    would_change_top1_rate: float = 0.0
    invalid_suggestion_rate: float = 0.0
    safety_warning_count: int = 0
    confidence_calibration_summary: dict[str, Any] = field(default_factory=dict)
    counterfactual_breakdown: dict[str, int] = field(default_factory=dict)
    recommendation: ShadowRunRecommendation | str = ShadowRunRecommendation.CONTINUE_SHADOW
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class CanaryPromotionProposal:
    """Proposal-only request to move a shadow policy toward canary review."""

    proposal_id: str
    plan_id: str
    shadow_run_id: str | None
    shadow_approval_id: str | None
    policy_id: str
    policy_version: str
    source_policy_id: str
    source_policy_version: str
    shadow_result_summary: dict[str, Any]
    confidence_calibration_summary: dict[str, Any]
    counterfactual_breakdown: dict[str, int]
    safety_summary: dict[str, Any]
    failure_summary: dict[str, Any]
    recommended_canary_scope: dict[str, Any]
    allowed_campaign_ids: tuple[str, ...] = ()
    allowed_objective_levels: tuple[str, ...] = ()
    max_canary_rounds: int = 0
    max_learned_policy_weight: float = 0.0
    max_top1_change_rate: float = 0.0
    rollback_policy_id: str | None = None
    rollback_policy_version: str | None = None
    eligible: bool = False
    eligibility_reasons: tuple[str, ...] = ()
    required_approvals: tuple[str, ...] = ("human_canary_approval",)
    status: CanaryPromotionProposalStatus | str = CanaryPromotionProposalStatus.PROPOSED
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CanaryPromotionProposal:
        return cls(
            proposal_id=str(raw.get("proposal_id", "")),
            plan_id=str(raw.get("plan_id", "")),
            shadow_run_id=raw.get("shadow_run_id"),
            shadow_approval_id=raw.get("shadow_approval_id"),
            policy_id=str(raw.get("policy_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            source_policy_id=str(raw.get("source_policy_id", "")),
            source_policy_version=str(raw.get("source_policy_version", "")),
            shadow_result_summary=dict(raw.get("shadow_result_summary") or {}),
            confidence_calibration_summary=dict(raw.get("confidence_calibration_summary") or {}),
            counterfactual_breakdown=dict(raw.get("counterfactual_breakdown") or {}),
            safety_summary=dict(raw.get("safety_summary") or {}),
            failure_summary=dict(raw.get("failure_summary") or {}),
            recommended_canary_scope=dict(raw.get("recommended_canary_scope") or {}),
            allowed_campaign_ids=tuple(raw.get("allowed_campaign_ids") or ()),
            allowed_objective_levels=tuple(raw.get("allowed_objective_levels") or ()),
            max_canary_rounds=int(raw.get("max_canary_rounds") or 0),
            max_learned_policy_weight=float(raw.get("max_learned_policy_weight") or 0.0),
            max_top1_change_rate=float(raw.get("max_top1_change_rate") or 0.0),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            eligible=bool(raw.get("eligible", False)),
            eligibility_reasons=tuple(raw.get("eligibility_reasons") or ()),
            required_approvals=tuple(raw.get("required_approvals") or ()),
            status=raw.get("status", CanaryPromotionProposalStatus.PROPOSED),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
        )


@dataclass(frozen=True)
class CanaryApprovalRecord:
    """Explicit approval for a bounded learned-policy SAFE_SOFT canary run."""

    approval_id: str
    proposal_id: str
    policy_id: str
    policy_version: str
    approved_by: str
    approval_mode: CanaryApprovalMode | str
    approval_reason: str
    approved_at: str = field(default_factory=lambda: _now_iso())
    expires_at: str | None = None
    allowed_campaign_ids: tuple[str, ...] = ()
    allowed_objective_levels: tuple[str, ...] = ()
    max_canary_rounds: int = 0
    max_learned_policy_weight: float = 0.0
    max_top1_change_rate: float = 0.0
    auto_disable_enabled: bool = True
    revoked: bool = False
    revoked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CanaryApprovalRecord:
        return cls(
            approval_id=str(raw.get("approval_id", "")),
            proposal_id=str(raw.get("proposal_id", "")),
            policy_id=str(raw.get("policy_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            approved_by=str(raw.get("approved_by", "")),
            approval_mode=raw.get("approval_mode", CanaryApprovalMode.MANUAL),
            approval_reason=str(raw.get("approval_reason", "")),
            approved_at=str(raw.get("approved_at") or _now_iso()),
            expires_at=raw.get("expires_at"),
            allowed_campaign_ids=tuple(raw.get("allowed_campaign_ids") or ()),
            allowed_objective_levels=tuple(raw.get("allowed_objective_levels") or ()),
            max_canary_rounds=int(raw.get("max_canary_rounds") or 0),
            max_learned_policy_weight=float(raw.get("max_learned_policy_weight") or 0.0),
            max_top1_change_rate=float(raw.get("max_top1_change_rate") or 0.0),
            auto_disable_enabled=bool(raw.get("auto_disable_enabled", True)),
            revoked=bool(raw.get("revoked", False)),
            revoked_reason=raw.get("revoked_reason"),
        )


@dataclass(frozen=True)
class CanaryRunSchedule:
    """Approved bounded SAFE_SOFT canary run schedule."""

    schedule_id: str
    approval_id: str
    policy_id: str
    policy_version: str
    campaign_allowlist: tuple[str, ...] = ()
    objective_allowlist: tuple[str, ...] = ()
    max_rounds: int = 0
    max_learned_policy_weight: float = 0.0
    max_top1_change_rate: float = 0.0
    status: CanaryRunScheduleStatus | str = CanaryRunScheduleStatus.SCHEDULED
    created_at: str = field(default_factory=lambda: _now_iso())
    started_at: str | None = None
    completed_at: str | None = None
    cancellation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class CanaryRunResult:
    """Aggregate result from a bounded SAFE_SOFT canary run."""

    run_id: str
    schedule_id: str
    policy_id: str
    policy_version: str
    campaign_ids: tuple[str, ...] = ()
    round_count: int = 0
    applied_round_count: int = 0
    top1_changed_count: int = 0
    top1_change_rate: float = 0.0
    reward_vs_baseline: float = 0.0
    reward_vs_safe_influence: float = 0.0
    backend_failure_rate: float = 0.0
    constraint_failure_rate: float = 0.0
    safety_warning_count: int = 0
    auto_disable_triggered: bool = False
    auto_disable_reason: str | None = None
    recommendation: CanaryRunRecommendation | str = CanaryRunRecommendation.CONTINUE_CANARY
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class FinalPromotionProposal:
    """Proposal-only request for final learned-policy promotion review."""

    proposal_id: str
    plan_id: str
    canary_run_id: str | None
    canary_approval_id: str | None
    policy_id: str
    policy_version: str
    source_policy_id: str
    source_policy_version: str
    canary_result_summary: dict[str, Any]
    reward_comparison_summary: dict[str, Any]
    failure_comparison_summary: dict[str, Any]
    safety_summary: dict[str, Any]
    top1_change_summary: dict[str, Any]
    confidence_calibration_summary: dict[str, Any]
    counterfactual_breakdown: dict[str, Any]
    recommended_promotion_scope: dict[str, Any]
    allowed_campaign_ids: tuple[str, ...] = ()
    allowed_objective_levels: tuple[str, ...] = ()
    max_live_weight: float = 0.0
    max_top1_change_rate: float = 0.0
    rollback_policy_id: str | None = None
    rollback_policy_version: str | None = None
    eligible: bool = False
    eligibility_reasons: tuple[str, ...] = ()
    required_approvals: tuple[str, ...] = ("human_promotion_approval",)
    status: FinalPromotionProposalStatus | str = FinalPromotionProposalStatus.PROPOSED
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FinalPromotionProposal:
        return cls(
            proposal_id=str(raw.get("proposal_id", "")),
            plan_id=str(raw.get("plan_id", "")),
            canary_run_id=raw.get("canary_run_id"),
            canary_approval_id=raw.get("canary_approval_id"),
            policy_id=str(raw.get("policy_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            source_policy_id=str(raw.get("source_policy_id", "")),
            source_policy_version=str(raw.get("source_policy_version", "")),
            canary_result_summary=dict(raw.get("canary_result_summary") or {}),
            reward_comparison_summary=dict(raw.get("reward_comparison_summary") or {}),
            failure_comparison_summary=dict(raw.get("failure_comparison_summary") or {}),
            safety_summary=dict(raw.get("safety_summary") or {}),
            top1_change_summary=dict(raw.get("top1_change_summary") or {}),
            confidence_calibration_summary=dict(raw.get("confidence_calibration_summary") or {}),
            counterfactual_breakdown=dict(raw.get("counterfactual_breakdown") or {}),
            recommended_promotion_scope=dict(raw.get("recommended_promotion_scope") or {}),
            allowed_campaign_ids=tuple(raw.get("allowed_campaign_ids") or ()),
            allowed_objective_levels=tuple(raw.get("allowed_objective_levels") or ()),
            max_live_weight=float(raw.get("max_live_weight") or 0.0),
            max_top1_change_rate=float(raw.get("max_top1_change_rate") or 0.0),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            eligible=bool(raw.get("eligible", False)),
            eligibility_reasons=tuple(raw.get("eligibility_reasons") or ()),
            required_approvals=tuple(raw.get("required_approvals") or ()),
            status=raw.get("status", FinalPromotionProposalStatus.PROPOSED),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
        )


@dataclass(frozen=True)
class FinalApprovalRecord:
    """Explicit approval that marks a policy safe-soft eligible."""

    approval_id: str
    proposal_id: str
    policy_id: str
    policy_version: str
    approved_by: str
    approval_mode: FinalApprovalMode | str
    approval_reason: str
    approved_at: str = field(default_factory=lambda: _now_iso())
    expires_at: str | None = None
    allowed_campaign_ids: tuple[str, ...] = ()
    allowed_objective_levels: tuple[str, ...] = ()
    max_live_weight: float = 0.0
    max_top1_change_rate: float = 0.0
    rollback_policy_id: str | None = None
    rollback_policy_version: str | None = None
    revoked: bool = False
    revoked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FinalApprovalRecord:
        return cls(
            approval_id=str(raw.get("approval_id", "")),
            proposal_id=str(raw.get("proposal_id", "")),
            policy_id=str(raw.get("policy_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            approved_by=str(raw.get("approved_by", "")),
            approval_mode=raw.get("approval_mode", FinalApprovalMode.MANUAL),
            approval_reason=str(raw.get("approval_reason", "")),
            approved_at=str(raw.get("approved_at") or _now_iso()),
            expires_at=raw.get("expires_at"),
            allowed_campaign_ids=tuple(raw.get("allowed_campaign_ids") or ()),
            allowed_objective_levels=tuple(raw.get("allowed_objective_levels") or ()),
            max_live_weight=float(raw.get("max_live_weight") or 0.0),
            max_top1_change_rate=float(raw.get("max_top1_change_rate") or 0.0),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            revoked=bool(raw.get("revoked", False)),
            revoked_reason=raw.get("revoked_reason"),
        )


@dataclass(frozen=True)
class WeightTuningEvidence:
    """Metric evidence used to justify a proposal-only weight adjustment."""

    evidence_id: str
    source_type: WeightTuningEvidenceSource | str
    metric_name: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    confidence: float = 0.0
    counterfactual_label: str = "observed_outcome"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WeightTuningEvidence:
        return cls(
            evidence_id=str(raw.get("evidence_id", "")),
            source_type=raw.get("source_type", WeightTuningEvidenceSource.OFFLINE_EVAL),
            metric_name=str(raw.get("metric_name", "")),
            baseline_value=_optional_float(raw.get("baseline_value")),
            candidate_value=_optional_float(raw.get("candidate_value")),
            delta=_optional_float(raw.get("delta")),
            confidence=float(raw.get("confidence") or 0.0),
            counterfactual_label=str(raw.get("counterfactual_label") or "observed_outcome"),
            notes=str(raw.get("notes") or ""),
        )


@dataclass(frozen=True)
class PolicyWeightTuningProposal:
    """Proposal-only review record for influence weight changes."""

    proposal_id: str
    policy_id: str
    policy_version: str
    tuning_target: PolicyWeightTuningTarget | str
    current_weight: float
    proposed_weight: float
    max_allowed_weight: float
    delta: float
    evidence: tuple[WeightTuningEvidence, ...] = ()
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    risk_level: PolicyWeightTuningRiskLevel | str = PolicyWeightTuningRiskLevel.MEDIUM
    expected_effect: str = ""
    rollback_policy_id: str | None = None
    rollback_policy_version: str | None = None
    requires_human_approval: bool = True
    eligible: bool = False
    eligibility_reasons: tuple[str, ...] = ()
    status: PolicyWeightTuningProposalStatus | str = PolicyWeightTuningProposalStatus.PROPOSED
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyWeightTuningProposal:
        return cls(
            proposal_id=str(raw.get("proposal_id", "")),
            policy_id=str(raw.get("policy_id", "")),
            policy_version=str(raw.get("policy_version", "")),
            tuning_target=raw.get("tuning_target", PolicyWeightTuningTarget.LEARNED_POLICY_MAX_WEIGHT),
            current_weight=float(raw.get("current_weight") or 0.0),
            proposed_weight=float(raw.get("proposed_weight") or 0.0),
            max_allowed_weight=float(raw.get("max_allowed_weight") or 0.0),
            delta=float(raw.get("delta") or 0.0),
            evidence=tuple(
                item if isinstance(item, WeightTuningEvidence) else WeightTuningEvidence.from_dict(dict(item))
                for item in (raw.get("evidence") or ())
            ),
            evidence_summary=dict(raw.get("evidence_summary") or {}),
            risk_level=raw.get("risk_level", PolicyWeightTuningRiskLevel.MEDIUM),
            expected_effect=str(raw.get("expected_effect") or ""),
            rollback_policy_id=raw.get("rollback_policy_id"),
            rollback_policy_version=raw.get("rollback_policy_version"),
            requires_human_approval=bool(raw.get("requires_human_approval", True)),
            eligible=bool(raw.get("eligible", False)),
            eligibility_reasons=tuple(raw.get("eligibility_reasons") or ()),
            status=raw.get("status", PolicyWeightTuningProposalStatus.PROPOSED),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
        )


@dataclass(frozen=True)
class PolicyStructureEvidence:
    """Evidence used to justify structural policy/controller proposals."""

    evidence_id: str
    source_type: PolicyStructureEvidenceSource | str
    metric_name: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    confidence: float = 0.0
    counterfactual_label: str = "observed_outcome"
    supporting_trace_ids: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyStructureEvidence:
        return cls(
            evidence_id=str(raw.get("evidence_id", "")),
            source_type=raw.get("source_type", PolicyStructureEvidenceSource.TRACE_ANALYSIS),
            metric_name=str(raw.get("metric_name", "")),
            baseline_value=_optional_float(raw.get("baseline_value")),
            candidate_value=_optional_float(raw.get("candidate_value")),
            delta=_optional_float(raw.get("delta")),
            confidence=float(raw.get("confidence") or 0.0),
            counterfactual_label=str(raw.get("counterfactual_label") or "observed_outcome"),
            supporting_trace_ids=tuple(raw.get("supporting_trace_ids") or ()),
            notes=str(raw.get("notes") or ""),
        )


@dataclass(frozen=True)
class PolicyStructureProposal:
    """Proposal-only review record for policy/controller structure changes."""

    proposal_id: str
    proposal_type: PolicyStructureProposalType | str
    title: str
    description: str
    current_behavior: str
    proposed_behavior: str
    evidence: tuple[PolicyStructureEvidence, ...] = ()
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    affected_components: tuple[str, ...] = ()
    risk_level: PolicyStructureRiskLevel | str = PolicyStructureRiskLevel.MEDIUM
    requires_human_approval: bool = True
    eligible: bool = False
    eligibility_reasons: tuple[str, ...] = ()
    status: PolicyStructureProposalStatus | str = PolicyStructureProposalStatus.PROPOSED
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyStructureProposal:
        return cls(
            proposal_id=str(raw.get("proposal_id", "")),
            proposal_type=raw.get("proposal_type", PolicyStructureProposalType.NEW_POLICY_RULE),
            title=str(raw.get("title", "")),
            description=str(raw.get("description", "")),
            current_behavior=str(raw.get("current_behavior", "")),
            proposed_behavior=str(raw.get("proposed_behavior", "")),
            evidence=tuple(
                item if isinstance(item, PolicyStructureEvidence) else PolicyStructureEvidence.from_dict(dict(item))
                for item in (raw.get("evidence") or ())
            ),
            evidence_summary=dict(raw.get("evidence_summary") or {}),
            affected_components=tuple(raw.get("affected_components") or ()),
            risk_level=raw.get("risk_level", PolicyStructureRiskLevel.MEDIUM),
            requires_human_approval=bool(raw.get("requires_human_approval", True)),
            eligible=bool(raw.get("eligible", False)),
            eligibility_reasons=tuple(raw.get("eligibility_reasons") or ()),
            status=raw.get("status", PolicyStructureProposalStatus.PROPOSED),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
        )


@dataclass(frozen=True)
class PolicyVersionRegistryEntry:
    """Registered policy version metadata and lineage."""

    policy_id: str
    policy_version: str
    parent_policy_id: str | None = None
    parent_policy_version: str | None = None
    trained_on_dataset_version: str | None = None
    feature_schema_version: str = ""
    reward_version: str = ""
    training_config_summary: dict[str, Any] = field(default_factory=dict)
    offline_evaluation_summary: dict[str, Any] = field(default_factory=dict)
    shadow_summary: dict[str, Any] = field(default_factory=dict)
    canary_summary: dict[str, Any] = field(default_factory=dict)
    approved_for_shadow: bool = False
    approved_for_safe_soft: bool = False
    approved_for_live_canary: bool = False
    rollback_target: tuple[str, str] | None = None
    shadow_proposed: bool = False
    shadow_proposal_id: str | None = None
    shadow_proposal_status: str | None = None
    shadow_eligibility_summary: dict[str, Any] = field(default_factory=dict)
    shadow_approval_metadata: dict[str, Any] = field(default_factory=dict)
    shadow_run_schedule_metadata: dict[str, Any] = field(default_factory=dict)
    latest_shadow_run_result_summary: dict[str, Any] = field(default_factory=dict)
    canary_proposed: bool = False
    canary_proposal_id: str | None = None
    canary_proposal_status: str | None = None
    canary_eligibility_summary: dict[str, Any] = field(default_factory=dict)
    recommended_canary_scope: dict[str, Any] = field(default_factory=dict)
    canary_approval_metadata: dict[str, Any] = field(default_factory=dict)
    canary_run_schedule_metadata: dict[str, Any] = field(default_factory=dict)
    latest_canary_run_result_summary: dict[str, Any] = field(default_factory=dict)
    promotion_proposed: bool = False
    promotion_proposal_id: str | None = None
    promotion_proposal_status: str | None = None
    promotion_eligibility_summary: dict[str, Any] = field(default_factory=dict)
    recommended_promotion_scope: dict[str, Any] = field(default_factory=dict)
    recommended_live_weight: float = 0.0
    final_approval_metadata: dict[str, Any] = field(default_factory=dict)
    weight_tuning_proposed: bool = False
    weight_tuning_proposal_id: str | None = None
    weight_tuning_status: str | None = None
    current_weights: dict[str, float] = field(default_factory=dict)
    recommended_weights: dict[str, float] = field(default_factory=dict)
    tuning_eligibility_summary: dict[str, Any] = field(default_factory=dict)
    structure_proposal_id: str | None = None
    structure_proposal_status: str | None = None
    structure_proposal_type: str | None = None
    structure_proposal_summary: dict[str, Any] = field(default_factory=dict)
    structure_affected_components: tuple[str, ...] = ()
    latest_workflow_id: str | None = None
    workflow_status_summary: dict[str, Any] = field(default_factory=dict)
    latest_workflow_report_summary: dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: _now_iso())


@dataclass(frozen=True)
class PolicyVersionRegistry:
    """In-memory policy-version registry for offline lifecycle review."""

    entries: tuple[PolicyVersionRegistryEntry, ...] = ()

    def register(self, entry: PolicyVersionRegistryEntry) -> PolicyVersionRegistry:
        return replace(self, entries=tuple((*self.entries, entry)))

    def get(self, policy_id: str, policy_version: str) -> PolicyVersionRegistryEntry | None:
        for entry in reversed(self.entries):
            if entry.policy_id == policy_id and entry.policy_version == policy_version:
                return entry
        return None

    def get_latest_approved_shadow_policy(self) -> PolicyVersionRegistryEntry | None:
        return _latest(
            entry for entry in self.entries
            if entry.approved_for_shadow
        )

    def get_latest_canary_eligible_policy(self) -> PolicyVersionRegistryEntry | None:
        return _latest(
            entry for entry in self.entries
            if entry.approved_for_shadow
            and entry.approved_for_safe_soft
            and entry.approved_for_live_canary
        )

    def get_policy_lineage(
        self,
        policy_id: str,
        policy_version: str,
    ) -> tuple[PolicyVersionRegistryEntry, ...]:
        lineage: list[PolicyVersionRegistryEntry] = []
        seen: set[tuple[str, str]] = set()
        current = self.get(policy_id, policy_version)
        while current is not None:
            key = (current.policy_id, current.policy_version)
            if key in seen:
                break
            seen.add(key)
            lineage.append(current)
            if not current.parent_policy_id or not current.parent_policy_version:
                break
            current = self.get(current.parent_policy_id, current.parent_policy_version)
        return tuple(lineage)

    def get_rollback_target(
        self,
        policy_id: str,
        policy_version: str,
    ) -> PolicyVersionRegistryEntry | None:
        entry = self.get(policy_id, policy_version)
        if entry is None or entry.rollback_target is None:
            return None
        target_id, target_version = entry.rollback_target
        return self.get(target_id, target_version)

    def register_shadow_proposal(
        self,
        policy_id: str,
        policy_version: str,
        proposal: ShadowPromotionProposal,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            shadow_proposed=True,
            shadow_proposal_id=proposal.proposal_id,
            shadow_proposal_status=str(getattr(proposal.status, "value", proposal.status)),
            shadow_eligibility_summary={
                "eligible": proposal.eligible,
                "eligibility_reasons": proposal.eligibility_reasons,
                "required_approvals": proposal.required_approvals,
            },
            approved_for_shadow=False,
        )
        entries = tuple(
            updated if (
                item.policy_id == policy_id
                and item.policy_version == policy_version
            ) else item
            for item in self.entries
        )
        return replace(self, entries=entries)

    def mark_shadow_approved(
        self,
        policy_id: str,
        policy_version: str,
        approval: ShadowApprovalRecord,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            approved_for_shadow=True,
            approved_for_safe_soft=False,
            approved_for_live_canary=False,
            shadow_approval_metadata=approval.to_dict(),
        )
        return self._replace_entry(updated)

    def register_shadow_schedule(
        self,
        policy_id: str,
        policy_version: str,
        schedule: ShadowRunSchedule,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            shadow_run_schedule_metadata=schedule.to_dict(),
            approved_for_safe_soft=False,
            approved_for_live_canary=False,
        )
        return self._replace_entry(updated)

    def register_shadow_result(
        self,
        policy_id: str,
        policy_version: str,
        result: ShadowRunResult,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            latest_shadow_run_result_summary=result.to_dict(),
            approved_for_safe_soft=False,
            approved_for_live_canary=False,
        )
        return self._replace_entry(updated)

    def register_canary_proposal(
        self,
        policy_id: str,
        policy_version: str,
        proposal: CanaryPromotionProposal,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            canary_proposed=True,
            canary_proposal_id=proposal.proposal_id,
            canary_proposal_status=str(getattr(proposal.status, "value", proposal.status)),
            canary_eligibility_summary={
                "eligible": proposal.eligible,
                "eligibility_reasons": proposal.eligibility_reasons,
                "required_approvals": proposal.required_approvals,
            },
            recommended_canary_scope=dict(proposal.recommended_canary_scope),
            approved_for_safe_soft=False,
            approved_for_live_canary=False,
        )
        return self._replace_entry(updated)

    def mark_canary_approved(
        self,
        policy_id: str,
        policy_version: str,
        approval: CanaryApprovalRecord,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            approved_for_live_canary=True,
            approved_for_safe_soft=False,
            canary_approval_metadata=approval.to_dict(),
        )
        return self._replace_entry(updated)

    def register_canary_schedule(
        self,
        policy_id: str,
        policy_version: str,
        schedule: CanaryRunSchedule,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            canary_run_schedule_metadata=schedule.to_dict(),
            approved_for_safe_soft=False,
        )
        return self._replace_entry(updated)

    def register_canary_result(
        self,
        policy_id: str,
        policy_version: str,
        result: CanaryRunResult,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            latest_canary_run_result_summary=result.to_dict(),
            approved_for_safe_soft=False,
        )
        return self._replace_entry(updated)

    def register_final_promotion_proposal(
        self,
        policy_id: str,
        policy_version: str,
        proposal: FinalPromotionProposal,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            promotion_proposed=True,
            promotion_proposal_id=proposal.proposal_id,
            promotion_proposal_status=str(getattr(proposal.status, "value", proposal.status)),
            promotion_eligibility_summary={
                "eligible": proposal.eligible,
                "eligibility_reasons": proposal.eligibility_reasons,
                "required_approvals": proposal.required_approvals,
            },
            recommended_promotion_scope=dict(proposal.recommended_promotion_scope),
            recommended_live_weight=proposal.max_live_weight,
            approved_for_safe_soft=False,
        )
        return self._replace_entry(updated)

    def mark_final_approved(
        self,
        policy_id: str,
        policy_version: str,
        approval: FinalApprovalRecord,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        approved_scope = {
            "campaign_ids": approval.allowed_campaign_ids,
            "objective_levels": approval.allowed_objective_levels,
        }
        updated = replace(
            entry,
            approved_for_safe_soft=True,
            final_approval_metadata={
                "final_approval_id": approval.approval_id,
                "proposal_id": approval.proposal_id,
                "approved_by": approval.approved_by,
                "approval_mode": str(getattr(approval.approval_mode, "value", approval.approval_mode)),
                "approval_reason": approval.approval_reason,
                "approved_at": approval.approved_at,
                "approved_scope": approved_scope,
                "approved_weight_cap": approval.max_live_weight,
                "approved_top1_change_cap": approval.max_top1_change_rate,
                "rollback_target": (
                    approval.rollback_policy_id,
                    approval.rollback_policy_version,
                ),
                "approval_expiration": approval.expires_at,
                "revoked": False,
                "revoked_reason": None,
            },
        )
        return self._replace_entry(updated)

    def revoke_final_approval(
        self,
        policy_id: str,
        policy_version: str,
        reason: str,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        metadata = dict(entry.final_approval_metadata or {})
        metadata.update({
            "revoked": True,
            "revoked_reason": reason,
            "revoked_at": _now_iso(),
        })
        updated = replace(
            entry,
            approved_for_safe_soft=False,
            final_approval_metadata=metadata,
        )
        return self._replace_entry(updated)

    def register_weight_tuning_proposal(
        self,
        policy_id: str,
        policy_version: str,
        proposal: PolicyWeightTuningProposal,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        target = str(getattr(proposal.tuning_target, "value", proposal.tuning_target))
        current = dict(entry.current_weights or {})
        recommended = dict(entry.recommended_weights or {})
        current[target] = proposal.current_weight
        recommended[target] = proposal.proposed_weight
        updated = replace(
            entry,
            weight_tuning_proposed=True,
            weight_tuning_proposal_id=proposal.proposal_id,
            weight_tuning_status=str(getattr(proposal.status, "value", proposal.status)),
            current_weights=current,
            recommended_weights=recommended,
            tuning_eligibility_summary={
                "eligible": proposal.eligible,
                "eligibility_reasons": proposal.eligibility_reasons,
                "risk_level": str(getattr(proposal.risk_level, "value", proposal.risk_level)),
                "requires_human_approval": proposal.requires_human_approval,
            },
        )
        return self._replace_entry(updated)

    def register_policy_structure_proposal(
        self,
        policy_id: str,
        policy_version: str,
        proposal: PolicyStructureProposal,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            structure_proposal_id=proposal.proposal_id,
            structure_proposal_status=str(getattr(proposal.status, "value", proposal.status)),
            structure_proposal_type=str(getattr(proposal.proposal_type, "value", proposal.proposal_type)),
            structure_proposal_summary={
                "title": proposal.title,
                "eligible": proposal.eligible,
                "eligibility_reasons": proposal.eligibility_reasons,
                "risk_level": str(getattr(proposal.risk_level, "value", proposal.risk_level)),
                "requires_human_approval": proposal.requires_human_approval,
            },
            structure_affected_components=proposal.affected_components,
        )
        return self._replace_entry(updated)

    def register_workflow_metadata(
        self,
        policy_id: str,
        policy_version: str,
        workflow: PolicyEvolutionWorkflow,
        report: Any | None = None,
    ) -> PolicyVersionRegistry:
        entry = self.get(policy_id, policy_version)
        if entry is None:
            return self
        updated = replace(
            entry,
            latest_workflow_id=workflow.workflow_id,
            workflow_status_summary={
                "workflow_id": workflow.workflow_id,
                "current_stage": str(getattr(workflow.current_stage, "value", workflow.current_stage)),
                "status": str(getattr(workflow.status, "value", workflow.status)),
                "updated_at": workflow.updated_at,
            },
            latest_workflow_report_summary=_plain_dict(report) if report is not None else entry.latest_workflow_report_summary,
        )
        return self._replace_entry(updated)

    def _replace_entry(self, updated: PolicyVersionRegistryEntry) -> PolicyVersionRegistry:
        entries = tuple(
            updated if (
                item.policy_id == updated.policy_id
                and item.policy_version == updated.policy_version
            ) else item
            for item in self.entries
        )
        return replace(self, entries=entries)


@dataclass(frozen=True)
class EvolutionGuardResult:
    """Guardrail result for a policy-evolution plan."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = False


class EvolutionGuard:
    """Reject unsafe policy-evolution plans before any promotion review."""

    def __init__(
        self,
        *,
        max_allowed_score_delta_cap: float = 0.01,
        current_reward_version: str = "strategy_reward_v1",
    ) -> None:
        self.max_allowed_score_delta_cap = abs(float(max_allowed_score_delta_cap))
        self.current_reward_version = current_reward_version

    def evaluate(self, plan: PolicyEvolutionPlan) -> EvolutionGuardResult:
        changes = dict(plan.proposed_changes or {})
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if changes.get("change_safety_constraints"):
            violations.append(_violation("change_safety_constraints", "Evolution cannot modify safety constraints"))
        if changes.get("lower_approval_required"):
            violations.append(_violation("lower_approval_required", "Evolution cannot lower approval requirements"))
        if changes.get("unknown_counterfactual_as_ground_truth"):
            violations.append(_violation(
                "unknown_counterfactual_as_ground_truth",
                "Unknown counterfactual outcomes cannot be treated as ground truth reward",
            ))
        if changes.get("penalize_scientific_negative_backend"):
            violations.append(_violation(
                "penalize_scientific_negative_backend",
                "Scientific negative outcomes are evidence, not optimizer backend failures",
            ))
        if changes.get("bypass_promotion_gates"):
            violations.append(_violation("bypass_promotion_gates", "Promotion gates cannot be bypassed"))
        if changes.get("auto_apply_space_revision"):
            violations.append(_violation("auto_apply_space_revision", "Space revisions remain approval-only"))
        if changes.get("learned_policy_hard_veto"):
            violations.append(_violation("learned_policy_hard_veto", "Learned policies cannot hard-veto live choices"))
        if changes.get("learned_policy_add_backend"):
            violations.append(_violation("learned_policy_add_backend", "Learned policies cannot add backends"))
        if changes.get("learned_policy_override_action") or changes.get("learned_policy_override_objective"):
            violations.append(_violation("learned_policy_override_action_objective", "Learned policies cannot override action or objective"))
        if changes.get("enable_live_influence_directly") or changes.get("enable_learned_online_influence"):
            violations.append(_violation("enable_live_influence_directly", "Evolution plans cannot enable live influence"))
        if plan.promotion_allowed:
            violations.append(_violation("auto_promotion", "Evolution plans cannot auto-promote policies"))

        requested_reward = str(changes.get("reward_version") or plan.reward_version or "")
        explicit_bump = bool(changes.get("explicit_reward_version_bump"))
        if requested_reward and requested_reward != self.current_reward_version and not explicit_bump:
            violations.append(_violation(
                "reward_version_without_explicit_bump",
                "Reward version changes require an explicit version bump",
                {"requested_reward_version": requested_reward},
            ))

        requested_cap = changes.get("max_score_delta_cap")
        if requested_cap is not None:
            cap = abs(float(requested_cap or 0.0))
            if cap > self.max_allowed_score_delta_cap:
                violations.append(_violation(
                    "score_delta_cap_too_high",
                    "Requested influence cap exceeds configured maximum",
                    {"requested_cap": cap, "max_allowed_cap": self.max_allowed_score_delta_cap},
                ))
            elif cap > 0:
                warnings.append(_warning(
                    "score_delta_cap_requested",
                    "Any nonzero influence cap still requires promotion review and human approval",
                    {"requested_cap": cap},
                ))

        required_human_approval = bool(
            violations
            or warnings
            or plan.shadow_required
            or plan.canary_required
            or changes.get("explicit_reward_version_bump")
        )
        return EvolutionGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=required_human_approval,
        )


@dataclass(frozen=True)
class TrainingGuardResult:
    """Guardrail result for offline candidate-policy training."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()


class TrainingGuard:
    """Validate offline datasets and evaluations before emitting artifacts."""

    def evaluate(
        self,
        plan: PolicyEvolutionPlan,
        dataset: Any,
        audit: Any,
        reward_sanity: Any,
        *,
        training_config: dict[str, Any] | None = None,
        offline_evaluation_summary: dict[str, Any] | None = None,
    ) -> TrainingGuardResult:
        config = dict(training_config or {})
        evaluation = dict(offline_evaluation_summary or {})
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        record_count = int(getattr(audit, "record_count", 0) or 0)
        missing = dict(getattr(audit, "missing_feature_rates", {}) or {})
        if record_count <= 0:
            violations.append(_violation("dataset_audit_failed", "Dataset audit has no records"))
        for key in ("state_features", "context_features", "available_actions", "candidate_backends"):
            if float(missing.get(key, 0.0) or 0.0) > 0:
                violations.append(_violation(
                    "dataset_audit_failed",
                    f"Dataset audit reports missing {key}",
                    {"missing_rate": missing.get(key)},
                ))
        if float(getattr(audit, "candidate_score_coverage", 0.0) or 0.0) < 1.0:
            violations.append(_violation("dataset_audit_failed", "Candidate backend scores are incomplete"))
        if float(getattr(audit, "candidate_rank_coverage", 0.0) or 0.0) < 1.0:
            violations.append(_violation("dataset_audit_failed", "Candidate backend ranks are incomplete"))

        if not bool(getattr(reward_sanity, "passed", False)):
            violations.append(_violation(
                "reward_sanity_failed",
                "Reward sanity checks failed",
                {"failures": tuple(getattr(reward_sanity, "failures", ()) or ())},
            ))

        if getattr(dataset, "feature_schema_version", None) != plan.feature_schema_version:
            violations.append(_violation(
                "feature_schema_version_mismatch",
                "Dataset feature schema does not match plan",
                {
                    "dataset": getattr(dataset, "feature_schema_version", None),
                    "plan": plan.feature_schema_version,
                },
            ))
        if getattr(dataset, "reward_version", None) != plan.reward_version:
            violations.append(_violation(
                "reward_version_mismatch",
                "Dataset reward version does not match plan",
                {
                    "dataset": getattr(dataset, "reward_version", None),
                    "plan": plan.reward_version,
                },
            ))

        if config.get("use_unknown_counterfactual_as_ground_truth") and _has_unknown_counterfactual(dataset):
            violations.append(_violation(
                "unknown_counterfactual_as_ground_truth",
                "Unknown counterfactual outcomes cannot be used as ground-truth reward",
            ))

        safety = dict(evaluation.get("learned_policy_safety") or evaluation.get("safety_summary") or {})
        if safety and not bool(safety.get("passed", True)):
            violations.append(_violation(
                "offline_safety_violations",
                "Offline evaluator reported learned policy safety violations",
                {"failure_count": safety.get("failure_count", 0)},
            ))

        for warning in getattr(audit, "offline_readiness_warnings", ()) or ():
            warnings.append(_warning("dataset_audit_warning", str(warning)))
        for warning in getattr(reward_sanity, "warnings", ()) or ():
            warnings.append(_warning("reward_sanity_warning", str(warning)))

        return TrainingGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class ShadowPromotionGuardResult:
    """Guardrail result for shadow-promotion proposals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True


class ShadowPromotionGuard:
    """Validate a candidate before proposing shadow deployment."""

    def evaluate(self, proposal: ShadowPromotionProposal) -> ShadowPromotionGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if not proposal.offline_evaluation_summary:
            violations.append(_violation("missing_offline_evaluation", "Candidate artifact has no offline evaluation"))
        audit = proposal.dataset_audit_summary
        if audit and audit.get("passed") is False:
            violations.append(_violation("dataset_audit_failed", "Dataset audit failed"))
        if not audit:
            violations.append(_violation("missing_dataset_audit", "Dataset audit summary is required"))
        reward = proposal.reward_sanity_summary
        if reward and reward.get("passed") is False:
            violations.append(_violation("reward_sanity_failed", "Reward sanity failed"))
        if not reward:
            violations.append(_violation("missing_reward_sanity", "Reward sanity summary is required"))
        safety = proposal.safety_summary
        if safety and not bool(safety.get("passed", False)):
            violations.append(_violation("safety_violations_present", "Offline safety violations are present"))
        if not safety:
            violations.append(_violation("missing_safety_summary", "Safety summary is required"))
        if _unknown_counterfactual_primary_evidence(proposal):
            violations.append(_violation(
                "unknown_counterfactual_primary_evidence",
                "Unknown counterfactual cannot be primary improvement evidence",
            ))
        if (
            proposal.offline_evaluation_summary.get("feature_schema_version")
            and proposal.offline_evaluation_summary.get("feature_schema_version") != proposal.feature_schema_version
        ):
            violations.append(_violation("feature_schema_version_mismatch", "Feature schema version mismatch"))
        if (
            proposal.offline_evaluation_summary.get("reward_version")
            and proposal.offline_evaluation_summary.get("reward_version") != proposal.reward_version
        ):
            violations.append(_violation("reward_version_mismatch", "Reward version mismatch"))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        if not proposal.candidate_policy_id or not proposal.candidate_policy_version:
            violations.append(_violation("incomplete_candidate_artifact", "Candidate policy identity is incomplete"))
        if not proposal.dataset_version or not proposal.feature_schema_version or not proposal.reward_version:
            violations.append(_violation("incomplete_candidate_artifact", "Candidate artifact version metadata is incomplete"))
        if not proposal.eligible:
            warnings.append(_warning("proposal_not_marked_eligible", "Proposal is not marked eligible by artifact metadata"))

        return ShadowPromotionGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
        )


@dataclass(frozen=True)
class ShadowApprovalGuardResult:
    """Guardrail result for explicit shadow approvals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True


class ShadowApprovalGuard:
    """Validate explicit approval before any shadow run is scheduled."""

    BLOCKED_STATUSES = {
        ShadowPromotionProposalStatus.BLOCKED.value,
        ShadowPromotionProposalStatus.REJECTED.value,
        ShadowPromotionProposalStatus.EXPIRED.value,
    }

    def evaluate(
        self,
        proposal: ShadowPromotionProposal,
        approval: ShadowApprovalRecord,
        registry: PolicyVersionRegistry | None = None,
    ) -> ShadowApprovalGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        status = str(getattr(proposal.status, "value", proposal.status))
        if not proposal.eligible:
            violations.append(_violation("proposal_not_eligible", "Proposal is not eligible for shadow approval"))
        if status in self.BLOCKED_STATUSES:
            violations.append(_violation("proposal_status_blocked", "Proposal status blocks approval", {"status": status}))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        required = set(proposal.required_approvals or ())
        if "human_shadow_approval" in required and not approval.approved_by:
            violations.append(_violation("required_approval_missing", "Human shadow approval is required"))
        if approval.revoked:
            violations.append(_violation("approval_revoked", "Approval has been revoked"))
        if approval.proposal_id != proposal.proposal_id:
            violations.append(_violation("approval_proposal_mismatch", "Approval does not match proposal"))
        if approval.policy_id != proposal.candidate_policy_id or approval.policy_version != proposal.candidate_policy_version:
            violations.append(_violation("approval_policy_mismatch", "Approval does not match proposal policy"))
        if registry is not None:
            entry = registry.get(proposal.candidate_policy_id, proposal.candidate_policy_version)
            rollback = registry.get(proposal.rollback_policy_id or "", proposal.rollback_policy_version or "")
            if entry is None:
                violations.append(_violation("policy_lineage_invalid", "Candidate policy is missing from registry"))
            elif (
                entry.parent_policy_id != proposal.source_policy_id
                or entry.parent_policy_version != proposal.source_policy_version
            ):
                violations.append(_violation("policy_lineage_invalid", "Candidate lineage does not match proposal"))
            if rollback is None:
                violations.append(_violation("rollback_target_missing_from_registry", "Rollback target is missing from registry"))
        if approval.approval_mode != ShadowApprovalMode.MANUAL.value:
            warnings.append(_warning("non_manual_shadow_approval", "Non-manual shadow approvals still require audit visibility"))
        return ShadowApprovalGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
        )


@dataclass(frozen=True)
class CanaryPromotionGuardResult:
    """Guardrail result for canary-promotion proposals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True
    recommended_scope: dict[str, Any] = field(default_factory=dict)
    recommended_weight_cap: float = 0.0


class CanaryPromotionGuard:
    """Validate canary promotion proposals without enabling canary."""

    def __init__(
        self,
        *,
        min_shadow_rounds: int = 10,
        max_safety_warning_rate: float = 0.0,
        max_invalid_suggestion_rate: float = 0.05,
        min_confidence_calibration: float = 0.6,
        max_top1_change_rate: float = 0.25,
        max_unknown_counterfactual_rate: float = 0.5,
        max_weight_cap: float = 0.005,
    ) -> None:
        self.min_shadow_rounds = min_shadow_rounds
        self.max_safety_warning_rate = max_safety_warning_rate
        self.max_invalid_suggestion_rate = max_invalid_suggestion_rate
        self.min_confidence_calibration = min_confidence_calibration
        self.max_top1_change_rate = max_top1_change_rate
        self.max_unknown_counterfactual_rate = max_unknown_counterfactual_rate
        self.max_weight_cap = max_weight_cap

    def evaluate(
        self,
        proposal: CanaryPromotionProposal,
        *,
        registry: PolicyVersionRegistry | None = None,
        shadow_approval: ShadowApprovalRecord | None = None,
    ) -> CanaryPromotionGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        result = proposal.shadow_result_summary
        round_count = int(result.get("round_count") or 0)
        safety_warning_count = int(result.get("safety_warning_count") or 0)
        safety_warning_rate = safety_warning_count / round_count if round_count else 1.0
        invalid_rate = float(result.get("invalid_suggestion_rate") or 0.0)
        top1_rate = float(result.get("would_change_top1_rate") or 0.0)
        confidence = _confidence_calibration_score(proposal.confidence_calibration_summary)
        recommendation = str(result.get("recommendation") or "")

        if not result:
            violations.append(_violation("missing_shadow_result", "Shadow run result is required"))
        if recommendation != ShadowRunRecommendation.PROPOSE_CANARY.value:
            violations.append(_violation("shadow_recommendation_not_propose_canary", "Shadow result must recommend propose_canary"))
        if registry is not None:
            entry = registry.get(proposal.policy_id, proposal.policy_version)
            rollback = registry.get(proposal.rollback_policy_id or "", proposal.rollback_policy_version or "")
            if entry is None:
                violations.append(_violation("policy_lineage_invalid", "Policy is missing from registry"))
            else:
                if not entry.approved_for_shadow:
                    violations.append(_violation("policy_not_approved_for_shadow", "Policy must be approved for shadow first"))
                if (
                    entry.parent_policy_id != proposal.source_policy_id
                    or entry.parent_policy_version != proposal.source_policy_version
                ):
                    violations.append(_violation("policy_lineage_invalid", "Policy lineage does not match proposal"))
            if rollback is None:
                violations.append(_violation("rollback_target_missing_from_registry", "Rollback target is missing from registry"))
        if shadow_approval is not None:
            if shadow_approval.revoked:
                violations.append(_violation("shadow_approval_revoked", "Shadow approval has been revoked"))
            if _is_past_iso(shadow_approval.expires_at):
                violations.append(_violation("shadow_approval_expired", "Shadow approval has expired"))
        if round_count < self.min_shadow_rounds:
            violations.append(_violation("insufficient_shadow_rounds", "Shadow round count is below threshold", {"round_count": round_count}))
        if safety_warning_rate > self.max_safety_warning_rate:
            violations.append(_violation("safety_warning_threshold_breached", "Safety warning rate exceeds threshold", {"rate": safety_warning_rate}))
        if invalid_rate > self.max_invalid_suggestion_rate:
            violations.append(_violation("invalid_suggestion_rate_too_high", "Invalid suggestion rate exceeds threshold", {"rate": invalid_rate}))
        if confidence < self.min_confidence_calibration:
            violations.append(_violation("confidence_calibration_too_low", "Confidence calibration is below threshold", {"score": confidence}))
        if top1_rate > self.max_top1_change_rate:
            violations.append(_violation("top1_change_rate_too_high", "Would-change top1 rate exceeds threshold", {"rate": top1_rate}))
        if _unknown_counterfactual_rate(proposal.counterfactual_breakdown) > self.max_unknown_counterfactual_rate:
            violations.append(_violation("counterfactual_uncertainty_too_high", "Counterfactual uncertainty is too high"))
        if _canary_unknown_counterfactual_ground_truth(proposal):
            violations.append(_violation("unknown_counterfactual_as_ground_truth", "Unknown counterfactual cannot be ground truth"))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        if proposal.max_learned_policy_weight > self.max_weight_cap:
            warnings.append(_warning("weight_cap_reduced", "Requested learned policy weight was above guard cap"))

        scope = dict(proposal.recommended_canary_scope or {})
        if not scope:
            scope = {
                "campaign_ids": proposal.allowed_campaign_ids,
                "objective_levels": proposal.allowed_objective_levels,
                "max_rounds": proposal.max_canary_rounds,
            }
        return CanaryPromotionGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
            recommended_scope=scope,
            recommended_weight_cap=min(proposal.max_learned_policy_weight, self.max_weight_cap),
        )


@dataclass(frozen=True)
class CanaryApprovalGuardResult:
    """Guardrail result for explicit canary approvals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True


class CanaryApprovalGuard:
    """Validate explicit approval before any bounded SAFE_SOFT canary schedule."""

    BLOCKED_STATUSES = {
        CanaryPromotionProposalStatus.BLOCKED.value,
        CanaryPromotionProposalStatus.REJECTED.value,
        CanaryPromotionProposalStatus.EXPIRED.value,
    }

    def evaluate(
        self,
        proposal: CanaryPromotionProposal | None,
        approval: CanaryApprovalRecord,
        registry: PolicyVersionRegistry | None = None,
    ) -> CanaryApprovalGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        if proposal is None:
            return CanaryApprovalGuardResult(
                allowed=False,
                violations=(_violation("missing_canary_proposal", "Canary proposal is required"),),
                warnings=(),
                required_human_approval=True,
            )

        status = str(getattr(proposal.status, "value", proposal.status))
        if not proposal.eligible:
            violations.append(_violation("proposal_not_eligible", "Proposal is not eligible for canary approval"))
        if status in self.BLOCKED_STATUSES:
            violations.append(_violation("proposal_status_blocked", "Proposal status blocks approval", {"status": status}))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        required = set(proposal.required_approvals or ())
        if "human_canary_approval" in required and not approval.approved_by:
            violations.append(_violation("required_approval_missing", "Human canary approval is required"))
        if approval.revoked:
            violations.append(_violation("approval_revoked", "Approval has been revoked"))
        if _is_past_iso(approval.expires_at):
            violations.append(_violation("approval_expired", "Approval has expired"))
        if approval.proposal_id != proposal.proposal_id:
            violations.append(_violation("approval_proposal_mismatch", "Approval does not match proposal"))
        if approval.policy_id != proposal.policy_id or approval.policy_version != proposal.policy_version:
            violations.append(_violation("approval_policy_mismatch", "Approval does not match proposal policy"))
        if not _scope_within(approval.allowed_campaign_ids, proposal.allowed_campaign_ids):
            violations.append(_violation("scope_exceeds_proposal", "Campaign approval scope exceeds proposal scope"))
        if not _scope_within(approval.allowed_objective_levels, proposal.allowed_objective_levels):
            violations.append(_violation("scope_exceeds_proposal", "Objective approval scope exceeds proposal scope"))
        if approval.max_learned_policy_weight > proposal.max_learned_policy_weight:
            violations.append(_violation(
                "weight_above_proposal_cap",
                "Requested learned policy weight exceeds proposal cap",
                {
                    "requested": approval.max_learned_policy_weight,
                    "cap": proposal.max_learned_policy_weight,
                },
            ))
        if approval.max_canary_rounds > proposal.max_canary_rounds:
            violations.append(_violation(
                "rounds_above_proposal_cap",
                "Requested canary rounds exceed proposal cap",
                {
                    "requested": approval.max_canary_rounds,
                    "cap": proposal.max_canary_rounds,
                },
            ))
        if approval.max_top1_change_rate > proposal.max_top1_change_rate:
            violations.append(_violation(
                "top1_change_rate_above_proposal_cap",
                "Requested top1 change rate exceeds proposal cap",
                {
                    "requested": approval.max_top1_change_rate,
                    "cap": proposal.max_top1_change_rate,
                },
            ))
        if registry is not None:
            entry = registry.get(proposal.policy_id, proposal.policy_version)
            rollback = registry.get(proposal.rollback_policy_id or "", proposal.rollback_policy_version or "")
            if entry is None:
                violations.append(_violation("policy_lineage_invalid", "Policy is missing from registry"))
            else:
                if not entry.approved_for_shadow:
                    violations.append(_violation("policy_not_approved_for_shadow", "Policy must be approved for shadow before canary"))
                if (
                    entry.parent_policy_id != proposal.source_policy_id
                    or entry.parent_policy_version != proposal.source_policy_version
                ):
                    violations.append(_violation("policy_lineage_invalid", "Policy lineage does not match proposal"))
            if rollback is None:
                violations.append(_violation("rollback_target_missing_from_registry", "Rollback target is missing from registry"))
        if approval.approval_mode != CanaryApprovalMode.MANUAL.value:
            warnings.append(_warning("non_manual_canary_approval", "Non-manual canary approvals still require audit visibility"))
        if approval.auto_disable_enabled is False:
            warnings.append(_warning("auto_disable_disabled", "Canary approval disabled automatic safety stop"))
        return CanaryApprovalGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
        )


@dataclass(frozen=True)
class FinalPromotionGuardResult:
    """Guardrail result for final promotion proposals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True
    recommended_scope: dict[str, Any] = field(default_factory=dict)
    recommended_live_weight: float = 0.0


class FinalPromotionGuard:
    """Validate final promotion proposals without activating learned policies."""

    def __init__(
        self,
        *,
        min_canary_rounds: int = 5,
        min_applied_rounds: int = 3,
        max_safety_warning_count: int = 0,
        max_backend_failure_rate: float = 0.05,
        max_constraint_failure_rate: float = 0.05,
        min_reward_vs_baseline: float = 0.0,
        min_reward_vs_safe_influence: float = 0.0,
        max_top1_change_rate: float = 0.25,
        min_confidence_calibration: float = 0.6,
        max_live_weight: float = 0.005,
    ) -> None:
        self.min_canary_rounds = min_canary_rounds
        self.min_applied_rounds = min_applied_rounds
        self.max_safety_warning_count = max_safety_warning_count
        self.max_backend_failure_rate = max_backend_failure_rate
        self.max_constraint_failure_rate = max_constraint_failure_rate
        self.min_reward_vs_baseline = min_reward_vs_baseline
        self.min_reward_vs_safe_influence = min_reward_vs_safe_influence
        self.max_top1_change_rate = max_top1_change_rate
        self.min_confidence_calibration = min_confidence_calibration
        self.max_live_weight = max_live_weight

    def evaluate(
        self,
        proposal: FinalPromotionProposal,
        *,
        registry: PolicyVersionRegistry | None = None,
        canary_approval: CanaryApprovalRecord | None = None,
    ) -> FinalPromotionGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        result = proposal.canary_result_summary
        recommendation = str(result.get("recommendation") or "")
        round_count = int(result.get("round_count") or 0)
        applied_count = int(result.get("applied_round_count") or 0)
        safety_warning_count = int(result.get("safety_warning_count") or 0)
        backend_failure_rate = float(result.get("backend_failure_rate") or 0.0)
        constraint_failure_rate = float(result.get("constraint_failure_rate") or 0.0)
        reward_vs_baseline = float(result.get("reward_vs_baseline") or 0.0)
        reward_vs_safe = float(result.get("reward_vs_safe_influence") or 0.0)
        top1_change_rate = float(result.get("top1_change_rate") or 0.0)
        confidence = _confidence_calibration_score(proposal.confidence_calibration_summary)

        if not result:
            violations.append(_violation("missing_canary_result", "Canary run result is required"))
        if recommendation != CanaryRunRecommendation.PROPOSE_PROMOTION.value:
            violations.append(_violation("canary_recommendation_not_propose_promotion", "Canary result must recommend propose_promotion"))
        if registry is not None:
            entry = registry.get(proposal.policy_id, proposal.policy_version)
            rollback = registry.get(proposal.rollback_policy_id or "", proposal.rollback_policy_version or "")
            if entry is None:
                violations.append(_violation("policy_lineage_invalid", "Policy is missing from registry"))
            else:
                if not entry.approved_for_live_canary:
                    violations.append(_violation("policy_not_approved_for_live_canary", "Policy must be explicitly approved for live canary first"))
                if (
                    entry.parent_policy_id != proposal.source_policy_id
                    or entry.parent_policy_version != proposal.source_policy_version
                ):
                    violations.append(_violation("policy_lineage_invalid", "Policy lineage does not match proposal"))
            if rollback is None:
                violations.append(_violation("rollback_target_missing_from_registry", "Rollback target is missing from registry"))
        if canary_approval is not None:
            if canary_approval.revoked:
                violations.append(_violation("canary_approval_revoked", "Canary approval has been revoked"))
            if _is_past_iso(canary_approval.expires_at):
                violations.append(_violation("canary_approval_expired", "Canary approval has expired"))
        if round_count < self.min_canary_rounds:
            violations.append(_violation("insufficient_canary_rounds", "Canary round count is below threshold", {"round_count": round_count}))
        if applied_count < self.min_applied_rounds:
            violations.append(_violation("insufficient_applied_rounds", "Applied canary round count is below threshold", {"applied_round_count": applied_count}))
        if safety_warning_count > self.max_safety_warning_count:
            violations.append(_violation("safety_warning_threshold_breached", "Safety warning count exceeds threshold", {"count": safety_warning_count}))
        if bool(result.get("auto_disable_triggered", False)):
            violations.append(_violation("auto_disable_triggered", "Canary auto-disable was triggered"))
        if backend_failure_rate > self.max_backend_failure_rate:
            violations.append(_violation("backend_failure_rate_increased", "Backend failure rate exceeds threshold", {"rate": backend_failure_rate}))
        if constraint_failure_rate > self.max_constraint_failure_rate:
            violations.append(_violation("constraint_failure_rate_increased", "Constraint failure rate exceeds threshold", {"rate": constraint_failure_rate}))
        if reward_vs_baseline < self.min_reward_vs_baseline:
            violations.append(_violation("reward_vs_baseline_too_low", "Reward vs baseline is below threshold", {"reward_delta": reward_vs_baseline}))
        if reward_vs_safe < self.min_reward_vs_safe_influence:
            violations.append(_violation("reward_vs_safe_influence_too_low", "Reward vs safe influence is below threshold", {"reward_delta": reward_vs_safe}))
        if top1_change_rate > self.max_top1_change_rate:
            violations.append(_violation("top1_change_rate_too_high", "Top1 change rate exceeds threshold", {"rate": top1_change_rate}))
        if confidence < self.min_confidence_calibration:
            violations.append(_violation("confidence_calibration_too_low", "Confidence calibration is below threshold", {"score": confidence}))
        if _final_unknown_counterfactual_primary_evidence(proposal):
            violations.append(_violation("unknown_counterfactual_primary_evidence", "Unknown counterfactual cannot be primary promotion evidence"))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        if proposal.max_live_weight > self.max_live_weight:
            warnings.append(_warning("live_weight_reduced", "Requested live weight was above guard cap"))

        scope = dict(proposal.recommended_promotion_scope or {})
        if not scope:
            scope = {
                "campaign_ids": proposal.allowed_campaign_ids,
                "objective_levels": proposal.allowed_objective_levels,
            }
        return FinalPromotionGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
            recommended_scope=scope,
            recommended_live_weight=min(proposal.max_live_weight, self.max_live_weight),
        )


@dataclass(frozen=True)
class FinalApprovalGuardResult:
    """Guardrail result for explicit final approvals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True


class FinalApprovalGuard:
    """Validate final approval before marking a policy safe-soft eligible."""

    BLOCKED_STATUSES = {
        FinalPromotionProposalStatus.BLOCKED.value,
        FinalPromotionProposalStatus.REJECTED.value,
        FinalPromotionProposalStatus.EXPIRED.value,
    }

    def evaluate(
        self,
        proposal: FinalPromotionProposal | None,
        approval: FinalApprovalRecord,
        registry: PolicyVersionRegistry | None = None,
    ) -> FinalApprovalGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        if proposal is None:
            return FinalApprovalGuardResult(
                allowed=False,
                violations=(_violation("missing_final_promotion_proposal", "Final promotion proposal is required"),),
                warnings=(),
                required_human_approval=True,
            )

        status = str(getattr(proposal.status, "value", proposal.status))
        if not proposal.eligible:
            violations.append(_violation("proposal_not_eligible", "Proposal is not eligible for final approval"))
        if status in self.BLOCKED_STATUSES:
            violations.append(_violation("proposal_status_blocked", "Proposal status blocks final approval", {"status": status}))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        required = set(proposal.required_approvals or ())
        if "human_promotion_approval" in required and not approval.approved_by:
            violations.append(_violation("required_approval_missing", "Human final approval is required"))
        if approval.revoked:
            violations.append(_violation("approval_revoked", "Approval has been revoked"))
        if _is_past_iso(approval.expires_at):
            violations.append(_violation("approval_expired", "Approval has expired"))
        if approval.proposal_id != proposal.proposal_id:
            violations.append(_violation("approval_proposal_mismatch", "Approval does not match proposal"))
        if approval.policy_id != proposal.policy_id or approval.policy_version != proposal.policy_version:
            violations.append(_violation("approval_policy_mismatch", "Approval does not match proposal policy"))
        if not _scope_within(approval.allowed_campaign_ids, proposal.allowed_campaign_ids):
            violations.append(_violation("scope_exceeds_proposal", "Campaign approval scope exceeds proposal scope"))
        if not _scope_within(approval.allowed_objective_levels, proposal.allowed_objective_levels):
            violations.append(_violation("scope_exceeds_proposal", "Objective approval scope exceeds proposal scope"))
        if approval.max_live_weight > proposal.max_live_weight:
            violations.append(_violation(
                "weight_above_proposal_cap",
                "Requested live weight exceeds proposal cap",
                {
                    "requested": approval.max_live_weight,
                    "cap": proposal.max_live_weight,
                },
            ))
        if approval.max_top1_change_rate > proposal.max_top1_change_rate:
            violations.append(_violation(
                "top1_change_rate_above_proposal_cap",
                "Requested top1 change rate exceeds proposal cap",
                {
                    "requested": approval.max_top1_change_rate,
                    "cap": proposal.max_top1_change_rate,
                },
            ))
        if not approval.rollback_policy_id or not approval.rollback_policy_version:
            violations.append(_violation("approval_missing_rollback_target", "Approval rollback target is required"))
        if (
            approval.rollback_policy_id != proposal.rollback_policy_id
            or approval.rollback_policy_version != proposal.rollback_policy_version
        ):
            violations.append(_violation("approval_rollback_mismatch", "Approval rollback target does not match proposal"))
        if registry is not None:
            entry = registry.get(proposal.policy_id, proposal.policy_version)
            rollback = registry.get(proposal.rollback_policy_id or "", proposal.rollback_policy_version or "")
            if entry is None:
                violations.append(_violation("policy_lineage_invalid", "Policy is missing from registry"))
            else:
                if not entry.approved_for_live_canary:
                    violations.append(_violation("policy_not_approved_for_live_canary", "Policy must be approved for live canary first"))
                if (
                    entry.parent_policy_id != proposal.source_policy_id
                    or entry.parent_policy_version != proposal.source_policy_version
                ):
                    violations.append(_violation("policy_lineage_invalid", "Policy lineage does not match proposal"))
            if rollback is None:
                violations.append(_violation("rollback_target_missing_from_registry", "Rollback target is missing from registry"))
        if approval.approval_mode != FinalApprovalMode.MANUAL.value:
            warnings.append(_warning("non_manual_final_approval", "Non-manual final approvals still require audit visibility"))
        return FinalApprovalGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
        )


@dataclass(frozen=True)
class PolicyWeightTuningGuardResult:
    """Guardrail result for policy weight tuning proposals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True
    recommended_safe_weight: float = 0.0


class PolicyWeightTuningGuard:
    """Validate proposal-only policy weight adjustments."""

    def evaluate(
        self,
        proposal: PolicyWeightTuningProposal,
        *,
        registry: PolicyVersionRegistry | None = None,
    ) -> PolicyWeightTuningGuardResult:
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        target = str(getattr(proposal.tuning_target, "value", proposal.tuning_target))
        increasing = proposal.proposed_weight > proposal.current_weight

        if proposal.proposed_weight > proposal.max_allowed_weight:
            violations.append(_violation(
                "weight_above_max_allowed",
                "Proposed weight exceeds max allowed weight",
                {
                    "proposed_weight": proposal.proposed_weight,
                    "max_allowed_weight": proposal.max_allowed_weight,
                },
            ))
        if proposal.proposed_weight < 0:
            violations.append(_violation("negative_weight", "Proposed weight cannot be negative"))
        if not proposal.rollback_policy_id or not proposal.rollback_policy_version:
            violations.append(_violation("missing_rollback_target", "Rollback target is required"))
        if proposal.evidence_summary.get("alter_hard_safety_gates") or proposal.evidence_summary.get("lower_approval_requirements"):
            violations.append(_violation("safety_or_approval_gate_change", "Weight tuning cannot alter safety gates or approval requirements"))
        if proposal.evidence_summary.get("auto_apply_space_revision"):
            violations.append(_violation("auto_apply_space_revision", "Weight tuning cannot auto-apply space revisions"))
        if _weight_unknown_counterfactual_primary_evidence(proposal):
            violations.append(_violation("unknown_counterfactual_primary_evidence", "Unknown counterfactual cannot be primary tuning evidence"))
        if _evidence_metric_increased(proposal.evidence, "safety_warning"):
            violations.append(_violation("safety_warnings_increased", "Safety warnings increased in tuning evidence"))
        if _evidence_metric_increased(proposal.evidence, "backend_failure"):
            violations.append(_violation("backend_failure_rate_increased", "Backend failure rate increased in tuning evidence"))
        if _evidence_metric_increased(proposal.evidence, "constraint_failure"):
            violations.append(_violation("constraint_failure_rate_increased", "Constraint failure rate increased in tuning evidence"))
        if (
            target == PolicyWeightTuningTarget.LEARNED_POLICY_MAX_WEIGHT.value
            and increasing
        ):
            entry = registry.get(proposal.policy_id, proposal.policy_version) if registry is not None else None
            if entry is None or not entry.approved_for_safe_soft:
                violations.append(_violation("learned_weight_requires_safe_soft_approval", "Learned policy weight increase requires approved_for_safe_soft"))
        if (
            target == PolicyWeightTuningTarget.BANDIT_MAX_WEIGHT.value
            and increasing
            and not _has_calibration_evidence(proposal.evidence)
        ):
            violations.append(_violation("bandit_increase_requires_calibration", "Bandit weight increase requires calibration evidence"))
        if proposal.delta > 0:
            warnings.append(_warning("weight_increase_requires_approval", "Weight increases remain approval-only and must not auto-apply"))

        recommended = min(max(proposal.proposed_weight, 0.0), max(proposal.max_allowed_weight, 0.0))
        return PolicyWeightTuningGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
            recommended_safe_weight=recommended,
        )


@dataclass(frozen=True)
class PolicyStructureProposalGuardResult:
    """Guardrail result for structural policy proposals."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = True
    recommended_reviewers: tuple[str, ...] = ()


class PolicyStructureProposalGuard:
    """Validate policy structure proposals without modifying live behavior."""

    DEFAULT_BACKEND_REGISTRY = (
        "built_in",
        "nexus_gp_bo",
        "nexus_turbo",
        "nexus_tpe",
        "bomcp",
        "random",
        "sobol",
        "lhs",
    )

    def __init__(self, *, backend_registry: tuple[str, ...] | None = None) -> None:
        self.backend_registry = tuple(backend_registry or self.DEFAULT_BACKEND_REGISTRY)

    def evaluate(self, proposal: PolicyStructureProposal) -> PolicyStructureProposalGuardResult:
        summary = dict(proposal.evidence_summary or {})
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if summary.get("lower_safety_gates"):
            violations.append(_violation("lower_safety_gates", "Structure proposals cannot lower safety gates"))
        if summary.get("lower_approval_requirements"):
            violations.append(_violation("lower_approval_requirements", "Structure proposals cannot lower approval requirements"))
        if summary.get("auto_apply_space_revision"):
            violations.append(_violation("auto_apply_space_revision", "Space revisions remain approval-only"))
        if summary.get("unknown_counterfactual_as_ground_truth") or _structure_unknown_counterfactual_ground_truth(proposal):
            violations.append(_violation("unknown_counterfactual_as_ground_truth", "Unknown counterfactual cannot be ground truth"))
        if summary.get("penalize_scientific_negative_backend"):
            violations.append(_violation("penalize_scientific_negative_backend", "Scientific negative outcomes are evidence, not backend failures"))
        if summary.get("enable_live_hard_veto"):
            violations.append(_violation("enable_live_hard_veto", "Structure proposals cannot introduce live hard vetoes"))
        for backend in tuple(summary.get("added_backends") or ()):
            if backend not in self.backend_registry:
                violations.append(_violation(
                    "backend_outside_registry",
                    "Structure proposal references backend outside registry",
                    {"backend": backend},
                ))
        proposal_type = str(getattr(proposal.proposal_type, "value", proposal.proposal_type))
        reward_change = proposal_type == PolicyStructureProposalType.REWARD_FEATURE_CHANGE.value
        if (reward_change or summary.get("changes_reward_semantics")) and not summary.get("reward_version_bump"):
            violations.append(_violation("reward_semantics_without_version_bump", "Reward semantic changes require reward_version bump metadata"))
        if summary.get("bypass_shadow_canary_promotion_lifecycle"):
            violations.append(_violation("bypass_lifecycle", "Structure proposals cannot bypass shadow/canary/promotion lifecycle"))

        if not proposal.evidence:
            warnings.append(_warning("missing_evidence", "Structure proposal has no supporting evidence"))
        reviewers = _structure_recommended_reviewers(proposal)
        return PolicyStructureProposalGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=True,
            recommended_reviewers=reviewers,
        )


@dataclass(frozen=True)
class PolicyEvolutionWorkflowGuardResult:
    """Guardrail result for workflow stage transitions."""

    allowed: bool
    violations: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    required_human_approval: bool = False


class PolicyEvolutionWorkflowGuard:
    """Validate workflow transitions without touching live runtime behavior."""

    def evaluate(
        self,
        workflow: PolicyEvolutionWorkflow,
        target_stage: PolicyEvolutionStage | str,
        *,
        actor: PolicyEvolutionAuditActorType | str = PolicyEvolutionAuditActorType.SYSTEM,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyEvolutionWorkflowGuardResult:
        target = str(getattr(target_stage, "value", target_stage))
        actor_value = str(getattr(actor, "value", actor))
        data = dict(metadata or {})
        violations: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        if target == PolicyEvolutionStage.SHADOW_PROPOSED.value and not workflow.candidate_artifact_id:
            violations.append(_violation("shadow_before_offline_evaluated", "Shadow proposal requires offline-evaluated candidate artifact"))
        if target == PolicyEvolutionStage.SHADOW_APPROVED.value and not data.get("shadow_approval_id", workflow.shadow_approval_id):
            violations.append(_violation("missing_shadow_approval", "Shadow approval requires explicit ShadowApprovalRecord"))
        if target == PolicyEvolutionStage.SHADOW_RUNNING.value and not data.get("shadow_schedule_id", workflow.shadow_schedule_id):
            violations.append(_violation("missing_shadow_schedule", "Shadow running requires ShadowRunSchedule"))
        if target == PolicyEvolutionStage.CANARY_PROPOSED.value and not workflow.shadow_result_id:
            violations.append(_violation("canary_before_shadow_completed", "Canary proposal requires completed shadow result"))
        if target == PolicyEvolutionStage.CANARY_APPROVED.value and not data.get("canary_approval_id", workflow.canary_approval_id):
            violations.append(_violation("missing_canary_approval", "Canary approval requires explicit CanaryApprovalRecord"))
        if target == PolicyEvolutionStage.CANARY_RUNNING.value and not data.get("canary_schedule_id", workflow.canary_schedule_id):
            violations.append(_violation("missing_canary_schedule", "Canary running requires CanaryRunSchedule"))
        if target == PolicyEvolutionStage.PROMOTION_PROPOSED.value and not workflow.canary_result_id:
            violations.append(_violation("promotion_before_canary_completed", "Final promotion proposal requires completed canary result"))
        if target == PolicyEvolutionStage.FINAL_APPROVED.value and not data.get("final_approval_id", workflow.final_approval_id):
            violations.append(_violation("missing_final_approval", "Final approval requires explicit FinalApprovalRecord"))
        if target == PolicyEvolutionStage.COMPLETED.value:
            missing = tuple(
                stage for stage, value in (
                    ("shadow_approval", workflow.shadow_approval_id),
                    ("canary_approval", workflow.canary_approval_id),
                    ("final_approval", workflow.final_approval_id),
                )
                if not value
            )
            if missing:
                violations.append(_violation("missing_required_approval_stages", "Completed workflow requires approval stages", {"missing": missing}))
        if target == PolicyEvolutionStage.WEIGHT_TUNING_PROPOSED.value and data.get("apply_weight_tuning"):
            violations.append(_violation("auto_apply_weight_tuning", "Weight tuning proposals must not auto-apply"))
        if target == PolicyEvolutionStage.STRUCTURE_REVIEW_PROPOSED.value and data.get("apply_structure_proposal"):
            violations.append(_violation("auto_apply_structure_proposal", "Structure proposals must not auto-apply"))
        if data.get("auto_apply_space_revision"):
            violations.append(_violation("auto_apply_space_revision", "Space revisions remain approval-only"))
        if data.get("enable_live_influence") or data.get("enable_learned_online_influence"):
            violations.append(_violation("auto_enable_live_influence", "Workflow transitions cannot enable live influence automatically"))
        if data.get("modify_safety_gates") or data.get("lower_approval_requirements"):
            violations.append(_violation("safety_or_approval_gate_change", "Workflow cannot modify safety gates or approval requirements"))

        approval_targets = {
            PolicyEvolutionStage.SHADOW_APPROVED.value,
            PolicyEvolutionStage.CANARY_APPROVED.value,
            PolicyEvolutionStage.FINAL_APPROVED.value,
        }
        required_human = target in approval_targets and actor_value not in {
            PolicyEvolutionAuditActorType.HUMAN.value,
            PolicyEvolutionAuditActorType.CONFIG.value,
            PolicyEvolutionAuditActorType.TEST.value,
        }
        if required_human:
            warnings.append(_warning("approval_actor_review", "Approval transitions should be performed by human/config/test actors"))
        return PolicyEvolutionWorkflowGuardResult(
            allowed=not violations,
            violations=tuple(violations),
            warnings=tuple(warnings),
            required_human_approval=required_human,
        )


@dataclass(frozen=True)
class PolicyEvolutionWorkflowReport:
    """Auditable workflow summary."""

    workflow_id: str
    current_stage: PolicyEvolutionStage | str
    status: PolicyEvolutionWorkflowStatus | str
    completed_stages: tuple[str, ...]
    blocked_stages: tuple[str, ...]
    approval_history: tuple[dict[str, Any], ...]
    training_summary: dict[str, Any]
    offline_evaluation_summary: dict[str, Any]
    shadow_summary: dict[str, Any]
    canary_summary: dict[str, Any]
    final_approval_summary: dict[str, Any]
    weight_tuning_summary: dict[str, Any]
    structure_proposal_summary: dict[str, Any]
    rollback_target: tuple[str | None, str | None]
    guard_violations: tuple[dict[str, Any], ...]
    audit_log_count: int
    recommendation: PolicyEvolutionWorkflowRecommendation | str

    def to_dict(self) -> dict[str, Any]:
        return _plain_dict(self)


@dataclass(frozen=True)
class PolicyEvolutionReport:
    """Review report for one policy-evolution plan."""

    plan_id: str
    trigger_summary: dict[str, Any]
    dataset_readiness: dict[str, Any]
    audit_status: dict[str, Any]
    reward_sanity_status: dict[str, Any]
    offline_benchmark_status: dict[str, Any]
    shadow_eligibility: dict[str, Any]
    canary_eligibility: dict[str, Any]
    guard_violations: tuple[dict[str, Any], ...]
    guard_warnings: tuple[dict[str, Any], ...]
    recommendation: PolicyEvolutionRecommendation | str
    report_version: str = "policy_evolution_report_v1"


class PolicyEvolutionManager:
    """Create, guard, and report policy-evolution plans without execution."""

    DEFAULT_REQUIRED_CHECKS: tuple[str, ...] = (
        "dataset_audit",
        "reward_sanity",
        "offline_benchmark",
        "shadow_analysis",
        "canary_analysis",
        "evolution_guard",
    )

    def __init__(
        self,
        *,
        guard: EvolutionGuard | None = None,
        shadow_promotion_guard: ShadowPromotionGuard | None = None,
        shadow_approval_guard: ShadowApprovalGuard | None = None,
        canary_promotion_guard: CanaryPromotionGuard | None = None,
        canary_approval_guard: CanaryApprovalGuard | None = None,
        final_promotion_guard: FinalPromotionGuard | None = None,
        final_approval_guard: FinalApprovalGuard | None = None,
        weight_tuning_guard: PolicyWeightTuningGuard | None = None,
        structure_proposal_guard: PolicyStructureProposalGuard | None = None,
    ) -> None:
        self.guard = guard or EvolutionGuard()
        self.shadow_promotion_guard = shadow_promotion_guard or ShadowPromotionGuard()
        self.shadow_approval_guard = shadow_approval_guard or ShadowApprovalGuard()
        self.canary_promotion_guard = canary_promotion_guard or CanaryPromotionGuard()
        self.canary_approval_guard = canary_approval_guard or CanaryApprovalGuard()
        self.final_promotion_guard = final_promotion_guard or FinalPromotionGuard()
        self.final_approval_guard = final_approval_guard or FinalApprovalGuard()
        self.weight_tuning_guard = weight_tuning_guard or PolicyWeightTuningGuard()
        self.structure_proposal_guard = structure_proposal_guard or PolicyStructureProposalGuard()

    def create_evolution_plan(
        self,
        trigger: PolicyEvolutionTrigger,
        registry: PolicyVersionRegistry,
        dataset_summary: dict[str, Any] | None = None,
        audit_summary: dict[str, Any] | None = None,
        *,
        source_policy_id: str | None = None,
        source_policy_version: str | None = None,
        candidate_policy_id: str | None = None,
        candidate_policy_version: str | None = None,
        proposed_changes: dict[str, Any] | None = None,
    ) -> PolicyEvolutionPlan:
        latest = (
            registry.get_latest_canary_eligible_policy()
            or registry.get_latest_approved_shadow_policy()
            or _latest(registry.entries)
        )
        source_id = source_policy_id or (latest.policy_id if latest else "current_policy")
        source_version = source_policy_version or (latest.policy_version if latest else "v0")
        dataset_version = (
            trigger.dataset_version
            or (dataset_summary or {}).get("dataset_version")
            or (latest.trained_on_dataset_version if latest else None)
        )
        feature_schema_version = str(
            (dataset_summary or {}).get("feature_schema_version")
            or (latest.feature_schema_version if latest else "policy_feature_schema_v1")
        )
        reward_version = str(
            (dataset_summary or {}).get("reward_version")
            or (latest.reward_version if latest else "strategy_reward_v1")
        )
        plan = PolicyEvolutionPlan(
            plan_id=f"evo-{_compact_timestamp()}-{source_id}-{source_version}",
            source_policy_id=source_id,
            source_policy_version=source_version,
            candidate_policy_id=candidate_policy_id or f"{source_id}-candidate",
            candidate_policy_version=candidate_policy_version or _next_candidate_version(source_version),
            trigger=trigger,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema_version,
            reward_version=reward_version,
            required_checks=self.DEFAULT_REQUIRED_CHECKS,
            rollback_policy_id=source_id,
            rollback_policy_version=source_version,
            promotion_allowed=False,
            reasons=tuple(_plan_reasons(trigger, dataset_summary, audit_summary)),
            proposed_changes=dict(proposed_changes or {}),
        )
        return plan

    def evaluate_plan_guard(self, plan: PolicyEvolutionPlan) -> EvolutionGuardResult:
        return self.guard.evaluate(plan)

    def create_training_job(
        self,
        plan: PolicyEvolutionPlan,
        *,
        training_mode: CandidatePolicyTrainingMode | str = CandidatePolicyTrainingMode.IMITATION,
        training_config: dict[str, Any] | None = None,
    ) -> CandidatePolicyTrainingJob:
        return CandidatePolicyTrainingJob(
            job_id=f"train-{_compact_timestamp()}-{plan.candidate_policy_id}-{plan.candidate_policy_version}",
            plan_id=plan.plan_id,
            source_policy_id=plan.source_policy_id,
            source_policy_version=plan.source_policy_version,
            candidate_policy_id=plan.candidate_policy_id,
            candidate_policy_version=plan.candidate_policy_version,
            dataset_version=plan.dataset_version,
            feature_schema_version=plan.feature_schema_version,
            reward_version=plan.reward_version,
            training_mode=str(getattr(training_mode, "value", training_mode)),
            training_config=dict(training_config or {}),
        )

    def update_training_job_status(
        self,
        job: CandidatePolicyTrainingJob,
        new_status: CandidatePolicyTrainingJobStatus | str,
        *,
        failure_reason: str | None = None,
    ) -> CandidatePolicyTrainingJob:
        status = str(getattr(new_status, "value", new_status))
        return replace(
            job,
            status=status,
            failure_reason=failure_reason,
            updated_at=_now_iso(),
            completed_at=_now_iso() if status in {
                CandidatePolicyTrainingJobStatus.OFFLINE_EVALUATED.value,
                CandidatePolicyTrainingJobStatus.FAILED.value,
            } else job.completed_at,
        )

    def attach_training_result(
        self,
        plan: PolicyEvolutionPlan,
        artifact: CandidatePolicyArtifact,
    ) -> PolicyEvolutionPlan:
        if not artifact.offline_evaluation_summary:
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.REJECTED,
                "candidate artifact has no offline evaluation summary",
            )
        return self.update_plan_status(
            plan,
            PolicyEvolutionPlanStatus.OFFLINE_EVALUATED,
            f"candidate artifact ready:{artifact.policy_id}:{artifact.policy_version}",
        )

    def create_shadow_promotion_proposal(
        self,
        plan: PolicyEvolutionPlan,
        training_job: CandidatePolicyTrainingJob,
        artifact: CandidatePolicyArtifact | None,
    ) -> ShadowPromotionProposal:
        artifact_dict = artifact.to_dict() if artifact is not None else {}
        offline = dict(artifact_dict.get("offline_evaluation_summary") or {})
        safety = dict(artifact_dict.get("safety_summary") or {})
        proposal = ShadowPromotionProposal(
            proposal_id=f"shadow-{_compact_timestamp()}-{plan.candidate_policy_id}-{plan.candidate_policy_version}",
            plan_id=plan.plan_id,
            training_job_id=training_job.job_id,
            candidate_policy_id=artifact_dict.get("policy_id", plan.candidate_policy_id),
            candidate_policy_version=artifact_dict.get("policy_version", plan.candidate_policy_version),
            source_policy_id=artifact_dict.get("parent_policy_id", plan.source_policy_id),
            source_policy_version=artifact_dict.get("parent_policy_version", plan.source_policy_version),
            dataset_version=artifact_dict.get("dataset_version", plan.dataset_version),
            feature_schema_version=artifact_dict.get("feature_schema_version", plan.feature_schema_version),
            reward_version=artifact_dict.get("reward_version", plan.reward_version),
            offline_evaluation_summary=offline,
            dataset_audit_summary=dict(offline.get("dataset_audit") or artifact_dict.get("dataset_audit_summary") or {}),
            reward_sanity_summary=dict(offline.get("reward_sanity") or artifact_dict.get("reward_sanity_summary") or {}),
            safety_summary=safety,
            counterfactual_uncertainty_summary=dict(
                offline.get("counterfactual_uncertainty_summary")
                or offline.get("counterfactual_uncertainty_breakdown")
                or artifact_dict.get("counterfactual_uncertainty_summary")
                or {}
            ),
            rollback_policy_id=plan.rollback_policy_id,
            rollback_policy_version=plan.rollback_policy_version,
            eligible=bool(artifact_dict.get("shadow_promotion_eligible", False)),
            eligibility_reasons=(
                (artifact_dict.get("shadow_promotion_reason"),)
                if artifact_dict.get("shadow_promotion_reason") else ()
            ),
            required_approvals=("human_shadow_approval",),
            status=ShadowPromotionProposalStatus.PROPOSED,
        )
        guard = self.evaluate_shadow_promotion_guard(proposal)
        return replace(
            proposal,
            status=(
                ShadowPromotionProposalStatus.ELIGIBLE.value
                if guard.allowed else ShadowPromotionProposalStatus.BLOCKED.value
            ),
            eligible=guard.allowed and proposal.eligible,
            eligibility_reasons=tuple((
                *proposal.eligibility_reasons,
                *tuple(v["check"] for v in guard.violations),
            )),
            updated_at=_now_iso(),
        )

    def evaluate_shadow_promotion_guard(
        self,
        proposal: ShadowPromotionProposal,
    ) -> ShadowPromotionGuardResult:
        return self.shadow_promotion_guard.evaluate(proposal)

    def evaluate_shadow_approval_guard(
        self,
        proposal: ShadowPromotionProposal,
        approval_record: ShadowApprovalRecord,
        registry: PolicyVersionRegistry | None = None,
    ) -> ShadowApprovalGuardResult:
        return self.shadow_approval_guard.evaluate(proposal, approval_record, registry)

    def approve_shadow_proposal(
        self,
        proposal: ShadowPromotionProposal,
        approval_record: ShadowApprovalRecord,
        *,
        registry: PolicyVersionRegistry | None = None,
    ) -> tuple[ShadowPromotionProposal, PolicyVersionRegistry | None, ShadowApprovalGuardResult]:
        guard = self.evaluate_shadow_approval_guard(proposal, approval_record, registry)
        if not guard.allowed:
            return proposal, registry, guard
        approved = replace(
            proposal,
            status=ShadowPromotionProposalStatus.APPROVED.value,
            updated_at=_now_iso(),
        )
        if registry is not None:
            registry = registry.mark_shadow_approved(
                approved.candidate_policy_id,
                approved.candidate_policy_version,
                approval_record,
            )
        return approved, registry, guard

    def schedule_shadow_run(
        self,
        approval_record: ShadowApprovalRecord,
    ) -> ShadowRunSchedule:
        return ShadowRunSchedule(
            schedule_id=f"shadow-run-{_compact_timestamp()}-{approval_record.policy_id}-{approval_record.policy_version}",
            approval_id=approval_record.approval_id,
            policy_id=approval_record.policy_id,
            policy_version=approval_record.policy_version,
            campaign_allowlist=approval_record.allowed_campaign_ids,
            objective_allowlist=approval_record.allowed_objective_levels,
            max_rounds=approval_record.max_shadow_rounds,
        )

    def update_shadow_run_status(
        self,
        schedule: ShadowRunSchedule,
        status: ShadowRunScheduleStatus | str,
        reason: str | None = None,
    ) -> ShadowRunSchedule:
        value = str(getattr(status, "value", status))
        return replace(
            schedule,
            status=value,
            started_at=_now_iso() if value == ShadowRunScheduleStatus.RUNNING.value else schedule.started_at,
            completed_at=_now_iso() if value in {
                ShadowRunScheduleStatus.COMPLETED.value,
                ShadowRunScheduleStatus.CANCELLED.value,
                ShadowRunScheduleStatus.EXPIRED.value,
            } else schedule.completed_at,
            cancellation_reason=reason if value == ShadowRunScheduleStatus.CANCELLED.value else schedule.cancellation_reason,
        )

    def attach_shadow_run_result(
        self,
        plan: PolicyEvolutionPlan,
        result: ShadowRunResult,
    ) -> PolicyEvolutionPlan:
        if _shadow_result_passes_canary_thresholds(result):
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.SHADOW_ELIGIBLE,
                f"shadow result supports canary proposal:{result.run_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"shadow result does not support canary:{result.run_id}",
        )

    def create_canary_promotion_proposal(
        self,
        plan: PolicyEvolutionPlan,
        shadow_run_result: ShadowRunResult | None,
        *,
        shadow_approval: ShadowApprovalRecord | None = None,
        registry: PolicyVersionRegistry | None = None,
    ) -> CanaryPromotionProposal:
        result_dict = shadow_run_result.to_dict() if shadow_run_result is not None else {}
        entry = (
            registry.get(result_dict.get("policy_id", ""), result_dict.get("policy_version", ""))
            if registry is not None and result_dict else None
        )
        approval = shadow_approval or _approval_from_registry(entry)
        proposal = CanaryPromotionProposal(
            proposal_id=f"canary-{_compact_timestamp()}-{plan.candidate_policy_id}-{plan.candidate_policy_version}",
            plan_id=plan.plan_id,
            shadow_run_id=result_dict.get("run_id"),
            shadow_approval_id=approval.approval_id if approval else None,
            policy_id=result_dict.get("policy_id", plan.candidate_policy_id),
            policy_version=result_dict.get("policy_version", plan.candidate_policy_version),
            source_policy_id=plan.source_policy_id,
            source_policy_version=plan.source_policy_version,
            shadow_result_summary=result_dict,
            confidence_calibration_summary=dict(result_dict.get("confidence_calibration_summary") or {}),
            counterfactual_breakdown=dict(result_dict.get("counterfactual_breakdown") or {}),
            safety_summary={
                "safety_warning_count": result_dict.get("safety_warning_count", 0),
                "safety_warning_rate": (
                    float(result_dict.get("safety_warning_count", 0) or 0)
                    / float(result_dict.get("round_count", 0) or 1)
                ),
            },
            failure_summary=dict(result_dict.get("failure_summary") or {}),
            recommended_canary_scope={
                "campaign_ids": approval.allowed_campaign_ids if approval else (),
                "objective_levels": approval.allowed_objective_levels if approval else (),
                "max_rounds": min(5, approval.max_shadow_rounds if approval else 0),
            },
            allowed_campaign_ids=approval.allowed_campaign_ids if approval else (),
            allowed_objective_levels=approval.allowed_objective_levels if approval else (),
            max_canary_rounds=min(5, approval.max_shadow_rounds if approval else 0),
            max_learned_policy_weight=0.005,
            max_top1_change_rate=0.25,
            rollback_policy_id=plan.rollback_policy_id,
            rollback_policy_version=plan.rollback_policy_version,
            eligible=bool(shadow_run_result is not None and _shadow_result_passes_canary_thresholds(shadow_run_result)),
            eligibility_reasons=("shadow result passed canary proposal thresholds",)
            if shadow_run_result is not None and _shadow_result_passes_canary_thresholds(shadow_run_result) else (),
            required_approvals=("human_canary_approval",),
        )
        guard = self.evaluate_canary_promotion_guard(
            proposal,
            registry=registry,
            shadow_approval=approval,
        )
        return replace(
            proposal,
            status=(
                CanaryPromotionProposalStatus.ELIGIBLE.value
                if guard.allowed else CanaryPromotionProposalStatus.BLOCKED.value
            ),
            eligible=guard.allowed and proposal.eligible,
            eligibility_reasons=tuple((
                *proposal.eligibility_reasons,
                *tuple(v["check"] for v in guard.violations),
            )),
            recommended_canary_scope=guard.recommended_scope or proposal.recommended_canary_scope,
            max_learned_policy_weight=guard.recommended_weight_cap,
            updated_at=_now_iso(),
        )

    def evaluate_canary_promotion_guard(
        self,
        proposal: CanaryPromotionProposal,
        *,
        registry: PolicyVersionRegistry | None = None,
        shadow_approval: ShadowApprovalRecord | None = None,
    ) -> CanaryPromotionGuardResult:
        return self.canary_promotion_guard.evaluate(
            proposal,
            registry=registry,
            shadow_approval=shadow_approval,
        )

    def attach_canary_proposal(
        self,
        plan: PolicyEvolutionPlan,
        proposal: CanaryPromotionProposal,
    ) -> PolicyEvolutionPlan:
        guard = self.evaluate_canary_promotion_guard(proposal)
        if guard.allowed and proposal.eligible:
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.CANARY_ELIGIBLE,
                f"canary proposal eligible:{proposal.proposal_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"canary proposal blocked:{proposal.proposal_id}",
        )

    def evaluate_canary_approval_guard(
        self,
        proposal: CanaryPromotionProposal | None,
        approval_record: CanaryApprovalRecord,
        registry: PolicyVersionRegistry | None = None,
    ) -> CanaryApprovalGuardResult:
        return self.canary_approval_guard.evaluate(proposal, approval_record, registry)

    def approve_canary_proposal(
        self,
        proposal: CanaryPromotionProposal | None,
        approval_record: CanaryApprovalRecord,
        *,
        registry: PolicyVersionRegistry | None = None,
    ) -> tuple[CanaryPromotionProposal | None, PolicyVersionRegistry | None, CanaryApprovalGuardResult]:
        guard = self.evaluate_canary_approval_guard(proposal, approval_record, registry)
        if not guard.allowed or proposal is None:
            return proposal, registry, guard
        approved = replace(
            proposal,
            status=CanaryPromotionProposalStatus.APPROVED.value,
            updated_at=_now_iso(),
        )
        if registry is not None:
            registry = registry.mark_canary_approved(
                approved.policy_id,
                approved.policy_version,
                approval_record,
            )
        return approved, registry, guard

    def schedule_canary_run(
        self,
        approval_record: CanaryApprovalRecord,
    ) -> CanaryRunSchedule:
        return CanaryRunSchedule(
            schedule_id=f"canary-run-{_compact_timestamp()}-{approval_record.policy_id}-{approval_record.policy_version}",
            approval_id=approval_record.approval_id,
            policy_id=approval_record.policy_id,
            policy_version=approval_record.policy_version,
            campaign_allowlist=approval_record.allowed_campaign_ids,
            objective_allowlist=approval_record.allowed_objective_levels,
            max_rounds=approval_record.max_canary_rounds,
            max_learned_policy_weight=approval_record.max_learned_policy_weight,
            max_top1_change_rate=approval_record.max_top1_change_rate,
        )

    def update_canary_run_status(
        self,
        schedule: CanaryRunSchedule,
        status: CanaryRunScheduleStatus | str,
        reason: str | None = None,
    ) -> CanaryRunSchedule:
        value = str(getattr(status, "value", status))
        return replace(
            schedule,
            status=value,
            started_at=_now_iso() if value == CanaryRunScheduleStatus.RUNNING.value else schedule.started_at,
            completed_at=_now_iso() if value in {
                CanaryRunScheduleStatus.COMPLETED.value,
                CanaryRunScheduleStatus.CANCELLED.value,
                CanaryRunScheduleStatus.EXPIRED.value,
                CanaryRunScheduleStatus.AUTO_DISABLED.value,
            } else schedule.completed_at,
            cancellation_reason=reason if value in {
                CanaryRunScheduleStatus.CANCELLED.value,
                CanaryRunScheduleStatus.AUTO_DISABLED.value,
            } else schedule.cancellation_reason,
        )

    def attach_canary_run_result(
        self,
        plan: PolicyEvolutionPlan,
        result: CanaryRunResult,
    ) -> PolicyEvolutionPlan:
        if _canary_result_passes_promotion_thresholds(result):
            return self.update_plan_status(
                plan,
                plan.status,
                f"canary result supports promotion proposal:{result.run_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"canary result does not support promotion:{result.run_id}",
        )

    def create_final_promotion_proposal(
        self,
        plan: PolicyEvolutionPlan,
        canary_run_result: CanaryRunResult | None,
        *,
        canary_approval: CanaryApprovalRecord | None = None,
        registry: PolicyVersionRegistry | None = None,
    ) -> FinalPromotionProposal:
        result_dict = canary_run_result.to_dict() if canary_run_result is not None else {}
        entry = (
            registry.get(result_dict.get("policy_id", ""), result_dict.get("policy_version", ""))
            if registry is not None and result_dict else None
        )
        approval = canary_approval or _canary_approval_from_registry(entry)
        proposal = FinalPromotionProposal(
            proposal_id=f"promotion-{_compact_timestamp()}-{plan.candidate_policy_id}-{plan.candidate_policy_version}",
            plan_id=plan.plan_id,
            canary_run_id=result_dict.get("run_id"),
            canary_approval_id=approval.approval_id if approval else None,
            policy_id=result_dict.get("policy_id", plan.candidate_policy_id),
            policy_version=result_dict.get("policy_version", plan.candidate_policy_version),
            source_policy_id=plan.source_policy_id,
            source_policy_version=plan.source_policy_version,
            canary_result_summary=result_dict,
            reward_comparison_summary={
                "reward_vs_baseline": result_dict.get("reward_vs_baseline", 0.0),
                "reward_vs_safe_influence": result_dict.get("reward_vs_safe_influence", 0.0),
            },
            failure_comparison_summary={
                "backend_failure_rate": result_dict.get("backend_failure_rate", 0.0),
                "constraint_failure_rate": result_dict.get("constraint_failure_rate", 0.0),
            },
            safety_summary={
                "safety_warning_count": result_dict.get("safety_warning_count", 0),
                "auto_disable_triggered": result_dict.get("auto_disable_triggered", False),
                "auto_disable_reason": result_dict.get("auto_disable_reason"),
            },
            top1_change_summary={
                "top1_changed_count": result_dict.get("top1_changed_count", 0),
                "top1_change_rate": result_dict.get("top1_change_rate", 0.0),
            },
            confidence_calibration_summary=dict(result_dict.get("confidence_calibration_summary") or {"calibration_score": 1.0}),
            counterfactual_breakdown=dict(result_dict.get("counterfactual_breakdown") or {}),
            recommended_promotion_scope={
                "campaign_ids": approval.allowed_campaign_ids if approval else (),
                "objective_levels": approval.allowed_objective_levels if approval else (),
            },
            allowed_campaign_ids=approval.allowed_campaign_ids if approval else (),
            allowed_objective_levels=approval.allowed_objective_levels if approval else (),
            max_live_weight=min(0.005, approval.max_learned_policy_weight if approval else 0.0),
            max_top1_change_rate=approval.max_top1_change_rate if approval else 0.25,
            rollback_policy_id=plan.rollback_policy_id,
            rollback_policy_version=plan.rollback_policy_version,
            eligible=bool(canary_run_result is not None and _canary_result_passes_promotion_thresholds(canary_run_result)),
            eligibility_reasons=("canary result passed final promotion proposal thresholds",)
            if canary_run_result is not None and _canary_result_passes_promotion_thresholds(canary_run_result) else (),
            required_approvals=("human_promotion_approval",),
        )
        guard = self.evaluate_final_promotion_guard(
            proposal,
            registry=registry,
            canary_approval=approval,
        )
        return replace(
            proposal,
            status=(
                FinalPromotionProposalStatus.ELIGIBLE.value
                if guard.allowed else FinalPromotionProposalStatus.BLOCKED.value
            ),
            eligible=guard.allowed and proposal.eligible,
            eligibility_reasons=tuple((
                *proposal.eligibility_reasons,
                *tuple(v["check"] for v in guard.violations),
            )),
            recommended_promotion_scope=guard.recommended_scope or proposal.recommended_promotion_scope,
            max_live_weight=guard.recommended_live_weight,
            updated_at=_now_iso(),
        )

    def evaluate_final_promotion_guard(
        self,
        proposal: FinalPromotionProposal,
        *,
        registry: PolicyVersionRegistry | None = None,
        canary_approval: CanaryApprovalRecord | None = None,
    ) -> FinalPromotionGuardResult:
        return self.final_promotion_guard.evaluate(
            proposal,
            registry=registry,
            canary_approval=canary_approval,
        )

    def attach_final_promotion_proposal(
        self,
        plan: PolicyEvolutionPlan,
        proposal: FinalPromotionProposal,
    ) -> PolicyEvolutionPlan:
        guard = self.evaluate_final_promotion_guard(proposal)
        if guard.allowed and proposal.eligible:
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.PROMOTION_ELIGIBLE,
                f"final promotion proposal eligible:{proposal.proposal_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"final promotion proposal blocked:{proposal.proposal_id}",
        )

    def evaluate_final_approval_guard(
        self,
        proposal: FinalPromotionProposal | None,
        approval_record: FinalApprovalRecord,
        registry: PolicyVersionRegistry | None = None,
    ) -> FinalApprovalGuardResult:
        return self.final_approval_guard.evaluate(proposal, approval_record, registry)

    def approve_final_promotion(
        self,
        proposal: FinalPromotionProposal | None,
        approval_record: FinalApprovalRecord,
        *,
        registry: PolicyVersionRegistry | None = None,
    ) -> tuple[FinalPromotionProposal | None, PolicyVersionRegistry | None, FinalApprovalGuardResult]:
        guard = self.evaluate_final_approval_guard(proposal, approval_record, registry)
        if not guard.allowed or proposal is None:
            return proposal, registry, guard
        approved = replace(
            proposal,
            status=FinalPromotionProposalStatus.APPROVED.value,
            updated_at=_now_iso(),
        )
        if registry is not None:
            registry = registry.mark_final_approved(
                approved.policy_id,
                approved.policy_version,
                approval_record,
            )
        return approved, registry, guard

    def revoke_final_approval(
        self,
        registry: PolicyVersionRegistry,
        policy_id: str,
        policy_version: str,
        reason: str,
    ) -> PolicyVersionRegistry:
        return registry.revoke_final_approval(policy_id, policy_version, reason)

    def get_active_safe_soft_policy(
        self,
        registry: PolicyVersionRegistry,
        *,
        campaign_id: str | None = None,
        objective_level: str | None = None,
    ) -> dict[str, Any] | None:
        for entry in sorted(registry.entries, key=lambda item: item.registered_at, reverse=True):
            if not entry.approved_for_safe_soft:
                continue
            metadata = dict(entry.final_approval_metadata or {})
            if not metadata or metadata.get("revoked"):
                continue
            if _is_past_iso(metadata.get("approval_expiration")):
                continue
            scope = dict(metadata.get("approved_scope") or {})
            if campaign_id and not _scope_contains(campaign_id, tuple(scope.get("campaign_ids") or ())):
                continue
            if objective_level and not _scope_contains(objective_level, tuple(scope.get("objective_levels") or ())):
                continue
            return {
                "policy_id": entry.policy_id,
                "policy_version": entry.policy_version,
                "approved_for_safe_soft": True,
                "final_approval_metadata": metadata,
                "live_selector_activation": False,
            }
        return None

    def create_weight_tuning_proposal(
        self,
        policy: PolicyVersionRegistryEntry,
        target: PolicyWeightTuningTarget | str,
        current_weight: float,
        proposed_weight: float,
        evidence: tuple[WeightTuningEvidence, ...] | list[WeightTuningEvidence],
        *,
        max_allowed_weight: float | None = None,
        expected_effect: str = "",
        evidence_summary: dict[str, Any] | None = None,
        registry: PolicyVersionRegistry | None = None,
    ) -> PolicyWeightTuningProposal:
        target_value = str(getattr(target, "value", target))
        evidence_tuple = tuple(evidence or ())
        max_weight = float(max_allowed_weight if max_allowed_weight is not None else _default_weight_cap(target_value))
        delta = float(proposed_weight) - float(current_weight)
        proposal = PolicyWeightTuningProposal(
            proposal_id=f"weight-tuning-{_compact_timestamp()}-{policy.policy_id}-{policy.policy_version}-{target_value}",
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            tuning_target=target_value,
            current_weight=float(current_weight),
            proposed_weight=float(proposed_weight),
            max_allowed_weight=max_weight,
            delta=delta,
            evidence=evidence_tuple,
            evidence_summary=dict(evidence_summary or _weight_evidence_summary(evidence_tuple)),
            risk_level=_weight_risk_level(delta, proposed_weight, max_weight),
            expected_effect=expected_effect,
            rollback_policy_id=(policy.rollback_target or (None, None))[0],
            rollback_policy_version=(policy.rollback_target or (None, None))[1],
            requires_human_approval=True,
            eligible=bool(evidence_tuple),
            eligibility_reasons=("evidence available; proposal requires explicit approval",) if evidence_tuple else (),
        )
        guard = self.evaluate_weight_tuning_guard(proposal, registry=registry)
        return replace(
            proposal,
            status=(
                PolicyWeightTuningProposalStatus.ELIGIBLE.value
                if guard.allowed else PolicyWeightTuningProposalStatus.BLOCKED.value
            ),
            eligible=guard.allowed and proposal.eligible,
            eligibility_reasons=tuple((
                *proposal.eligibility_reasons,
                *tuple(v["check"] for v in guard.violations),
            )),
            proposed_weight=guard.recommended_safe_weight,
            updated_at=_now_iso(),
        )

    def evaluate_weight_tuning_guard(
        self,
        proposal: PolicyWeightTuningProposal,
        *,
        registry: PolicyVersionRegistry | None = None,
    ) -> PolicyWeightTuningGuardResult:
        return self.weight_tuning_guard.evaluate(proposal, registry=registry)

    def attach_weight_tuning_proposal(
        self,
        plan: PolicyEvolutionPlan,
        proposal: PolicyWeightTuningProposal,
    ) -> PolicyEvolutionPlan:
        status = str(getattr(proposal.status, "value", proposal.status))
        if status == PolicyWeightTuningProposalStatus.ELIGIBLE.value and proposal.eligible:
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.WEIGHT_TUNING_ELIGIBLE,
                f"weight tuning proposal eligible:{proposal.proposal_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"weight tuning proposal blocked:{proposal.proposal_id}",
        )

    def create_policy_structure_proposal(
        self,
        proposal_type: PolicyStructureProposalType | str,
        title: str,
        description: str,
        current_behavior: str,
        proposed_behavior: str,
        evidence: tuple[PolicyStructureEvidence, ...] | list[PolicyStructureEvidence],
        *,
        affected_components: tuple[str, ...] = (),
        evidence_summary: dict[str, Any] | None = None,
        risk_level: PolicyStructureRiskLevel | str | None = None,
    ) -> PolicyStructureProposal:
        evidence_tuple = tuple(evidence or ())
        proposal_type_value = str(getattr(proposal_type, "value", proposal_type))
        proposal = PolicyStructureProposal(
            proposal_id=f"structure-{_compact_timestamp()}-{proposal_type_value}",
            proposal_type=proposal_type_value,
            title=title,
            description=description,
            current_behavior=current_behavior,
            proposed_behavior=proposed_behavior,
            evidence=evidence_tuple,
            evidence_summary=dict(evidence_summary or _structure_evidence_summary(evidence_tuple)),
            affected_components=tuple(affected_components),
            risk_level=str(getattr(risk_level, "value", risk_level)) if risk_level else _structure_risk_level(affected_components),
            requires_human_approval=True,
            eligible=bool(evidence_tuple),
            eligibility_reasons=("evidence available; proposal requires explicit review",) if evidence_tuple else (),
        )
        guard = self.evaluate_policy_structure_guard(proposal)
        return replace(
            proposal,
            status=(
                PolicyStructureProposalStatus.ELIGIBLE.value
                if guard.allowed else PolicyStructureProposalStatus.BLOCKED.value
            ),
            eligible=guard.allowed and proposal.eligible,
            eligibility_reasons=tuple((
                *proposal.eligibility_reasons,
                *tuple(v["check"] for v in guard.violations),
            )),
            updated_at=_now_iso(),
        )

    def evaluate_policy_structure_guard(
        self,
        proposal: PolicyStructureProposal,
    ) -> PolicyStructureProposalGuardResult:
        return self.structure_proposal_guard.evaluate(proposal)

    def attach_policy_structure_proposal(
        self,
        plan: PolicyEvolutionPlan,
        proposal: PolicyStructureProposal,
    ) -> PolicyEvolutionPlan:
        status = str(getattr(proposal.status, "value", proposal.status))
        if status == PolicyStructureProposalStatus.ELIGIBLE.value and proposal.eligible:
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.STRUCTURE_REVIEW_ELIGIBLE,
                f"policy structure proposal eligible:{proposal.proposal_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"policy structure proposal blocked:{proposal.proposal_id}",
        )

    def attach_shadow_proposal(
        self,
        plan: PolicyEvolutionPlan,
        proposal: ShadowPromotionProposal,
    ) -> PolicyEvolutionPlan:
        guard = self.evaluate_shadow_promotion_guard(proposal)
        if guard.allowed and proposal.eligible:
            return self.update_plan_status(
                plan,
                PolicyEvolutionPlanStatus.SHADOW_ELIGIBLE,
                f"shadow proposal eligible:{proposal.proposal_id}",
            )
        return self.update_plan_status(
            plan,
            plan.status,
            f"shadow proposal blocked:{proposal.proposal_id}",
        )

    def update_plan_status(
        self,
        plan: PolicyEvolutionPlan,
        new_status: PolicyEvolutionPlanStatus | str,
        reason: str,
    ) -> PolicyEvolutionPlan:
        return replace(
            plan,
            status=str(getattr(new_status, "value", new_status)),
            reasons=tuple((*plan.reasons, reason)),
            updated_at=_now_iso(),
        )

    def recommend_next_step(self, plan: PolicyEvolutionPlan) -> PolicyEvolutionRecommendation:
        guard = self.evaluate_plan_guard(plan)
        if not guard.allowed or str(plan.status) == PolicyEvolutionPlanStatus.REJECTED.value:
            return PolicyEvolutionRecommendation.REJECT
        if str(plan.status) == PolicyEvolutionPlanStatus.ROLLED_BACK.value:
            return PolicyEvolutionRecommendation.ROLLBACK
        if str(plan.status) == PolicyEvolutionPlanStatus.PROPOSED.value:
            return (
                PolicyEvolutionRecommendation.TRAIN_CANDIDATE
                if plan.dataset_version
                else PolicyEvolutionRecommendation.PREPARE_DATASET
            )
        if str(plan.status) == PolicyEvolutionPlanStatus.DATASET_READY.value:
            return PolicyEvolutionRecommendation.RUN_OFFLINE_EVAL
        if str(plan.status) == PolicyEvolutionPlanStatus.OFFLINE_EVALUATED.value:
            return PolicyEvolutionRecommendation.APPROVE_SHADOW
        if str(plan.status) == PolicyEvolutionPlanStatus.SHADOW_ELIGIBLE.value:
            if any("shadow result supports canary proposal" in reason for reason in plan.reasons):
                return PolicyEvolutionRecommendation.APPROVE_CANARY
            return PolicyEvolutionRecommendation.KEEP_CURRENT
        if str(plan.status) == PolicyEvolutionPlanStatus.CANARY_ELIGIBLE.value:
            if any("canary result supports promotion proposal" in reason for reason in plan.reasons):
                return PolicyEvolutionRecommendation.PROPOSE_PROMOTION
            return PolicyEvolutionRecommendation.APPROVE_CANARY
        if str(plan.status) == PolicyEvolutionPlanStatus.PROMOTION_ELIGIBLE.value:
            return PolicyEvolutionRecommendation.APPROVE_PROMOTION
        if str(plan.status) == PolicyEvolutionPlanStatus.WEIGHT_TUNING_ELIGIBLE.value:
            return PolicyEvolutionRecommendation.APPROVE_WEIGHT_TUNING
        if str(plan.status) == PolicyEvolutionPlanStatus.STRUCTURE_REVIEW_ELIGIBLE.value:
            return PolicyEvolutionRecommendation.REVIEW_STRUCTURE_PROPOSAL
        if str(plan.status) == PolicyEvolutionPlanStatus.PROMOTED.value:
            return PolicyEvolutionRecommendation.KEEP_CURRENT
        return PolicyEvolutionRecommendation.KEEP_CURRENT

    def build_report(
        self,
        plan: PolicyEvolutionPlan,
        *,
        dataset_summary: dict[str, Any] | None = None,
        audit_summary: dict[str, Any] | None = None,
        reward_sanity_summary: dict[str, Any] | None = None,
        offline_benchmark_summary: dict[str, Any] | None = None,
        shadow_summary: dict[str, Any] | None = None,
        canary_summary: dict[str, Any] | None = None,
    ) -> PolicyEvolutionReport:
        guard = self.evaluate_plan_guard(plan)
        recommendation = self.recommend_next_step(plan)
        return PolicyEvolutionReport(
            plan_id=plan.plan_id,
            trigger_summary=plan.trigger.to_dict(),
            dataset_readiness=dict(dataset_summary or {}),
            audit_status=dict(audit_summary or {}),
            reward_sanity_status=dict(reward_sanity_summary or {}),
            offline_benchmark_status=dict(offline_benchmark_summary or {}),
            shadow_eligibility=dict(shadow_summary or {}),
            canary_eligibility=dict(canary_summary or {}),
            guard_violations=guard.violations,
            guard_warnings=guard.warnings,
            recommendation=recommendation,
        )


class PolicyEvolutionWorkflowManager:
    """Orchestrate policy evolution artifacts as an auditable, proposal-only workflow."""

    def __init__(self, *, guard: PolicyEvolutionWorkflowGuard | None = None) -> None:
        self.guard = guard or PolicyEvolutionWorkflowGuard()
        self.audit_log: tuple[PolicyEvolutionAuditLogEntry, ...] = ()

    def create_workflow(self, trigger: PolicyEvolutionTrigger, plan: PolicyEvolutionPlan) -> PolicyEvolutionWorkflow:
        workflow = PolicyEvolutionWorkflow(
            workflow_id=f"workflow-{_compact_timestamp()}-{plan.candidate_policy_id}-{plan.candidate_policy_version}",
            trigger_id=f"{str(trigger.trigger_type)}:{trigger.created_at}",
            plan_id=plan.plan_id,
            source_policy_id=plan.source_policy_id,
            source_policy_version=plan.source_policy_version,
            candidate_policy_id=plan.candidate_policy_id,
            candidate_policy_version=plan.candidate_policy_version,
            current_stage=PolicyEvolutionStage.TRIGGERED.value,
            status=PolicyEvolutionWorkflowStatus.ACTIVE.value,
            rollback_policy_id=plan.rollback_policy_id,
            rollback_policy_version=plan.rollback_policy_version,
        )
        return self.transition_stage(
            workflow,
            PolicyEvolutionStage.PLANNED,
            PolicyEvolutionAuditActorType.SYSTEM,
            "workflow planned from trigger and evolution plan",
        )

    def attach_training_job(self, workflow: PolicyEvolutionWorkflow, job: CandidatePolicyTrainingJob) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, training_job_id=job.job_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.TRAINING_REQUESTED, PolicyEvolutionAuditActorType.SYSTEM, "training job attached")

    def attach_candidate_artifact(self, workflow: PolicyEvolutionWorkflow, artifact: CandidatePolicyArtifact) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, candidate_artifact_id=f"{artifact.policy_id}:{artifact.policy_version}")
        return self.transition_stage(workflow, PolicyEvolutionStage.OFFLINE_EVALUATED, PolicyEvolutionAuditActorType.SYSTEM, "candidate artifact attached")

    def attach_shadow_proposal(self, workflow: PolicyEvolutionWorkflow, proposal: ShadowPromotionProposal) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, shadow_proposal_id=proposal.proposal_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.SHADOW_PROPOSED, PolicyEvolutionAuditActorType.SYSTEM, "shadow proposal attached")

    def attach_shadow_approval(self, workflow: PolicyEvolutionWorkflow, approval: ShadowApprovalRecord) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, shadow_approval_id=approval.approval_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.SHADOW_APPROVED, _actor_from_approval(approval.approval_mode), "shadow approval attached", {"shadow_approval_id": approval.approval_id})

    def attach_shadow_schedule(self, workflow: PolicyEvolutionWorkflow, schedule: ShadowRunSchedule) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, shadow_schedule_id=schedule.schedule_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.SHADOW_RUNNING, PolicyEvolutionAuditActorType.SYSTEM, "shadow schedule attached", {"shadow_schedule_id": schedule.schedule_id})

    def attach_shadow_result(self, workflow: PolicyEvolutionWorkflow, result: ShadowRunResult) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, shadow_result_id=result.run_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.SHADOW_COMPLETED, PolicyEvolutionAuditActorType.SYSTEM, "shadow result attached")

    def attach_canary_proposal(self, workflow: PolicyEvolutionWorkflow, proposal: CanaryPromotionProposal) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, canary_proposal_id=proposal.proposal_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.CANARY_PROPOSED, PolicyEvolutionAuditActorType.SYSTEM, "canary proposal attached")

    def attach_canary_approval(self, workflow: PolicyEvolutionWorkflow, approval: CanaryApprovalRecord) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, canary_approval_id=approval.approval_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.CANARY_APPROVED, _actor_from_approval(approval.approval_mode), "canary approval attached", {"canary_approval_id": approval.approval_id})

    def attach_canary_schedule(self, workflow: PolicyEvolutionWorkflow, schedule: CanaryRunSchedule) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, canary_schedule_id=schedule.schedule_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.CANARY_RUNNING, PolicyEvolutionAuditActorType.SYSTEM, "canary schedule attached", {"canary_schedule_id": schedule.schedule_id})

    def attach_canary_result(self, workflow: PolicyEvolutionWorkflow, result: CanaryRunResult) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, canary_result_id=result.run_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.CANARY_COMPLETED, PolicyEvolutionAuditActorType.SYSTEM, "canary result attached")

    def attach_final_promotion_proposal(self, workflow: PolicyEvolutionWorkflow, proposal: FinalPromotionProposal) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, final_promotion_proposal_id=proposal.proposal_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.PROMOTION_PROPOSED, PolicyEvolutionAuditActorType.SYSTEM, "final promotion proposal attached")

    def attach_final_approval(self, workflow: PolicyEvolutionWorkflow, approval: FinalApprovalRecord) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, final_approval_id=approval.approval_id)
        return self.transition_stage(workflow, PolicyEvolutionStage.FINAL_APPROVED, _actor_from_approval(approval.approval_mode), "final approval attached", {"final_approval_id": approval.approval_id})

    def attach_weight_tuning_proposal(self, workflow: PolicyEvolutionWorkflow, proposal: PolicyWeightTuningProposal) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, weight_tuning_proposal_ids=tuple((*workflow.weight_tuning_proposal_ids, proposal.proposal_id)))
        return self.transition_stage(workflow, PolicyEvolutionStage.WEIGHT_TUNING_PROPOSED, PolicyEvolutionAuditActorType.SYSTEM, "weight tuning proposal attached")

    def attach_structure_proposal(self, workflow: PolicyEvolutionWorkflow, proposal: PolicyStructureProposal) -> PolicyEvolutionWorkflow:
        workflow = replace(workflow, structure_proposal_ids=tuple((*workflow.structure_proposal_ids, proposal.proposal_id)))
        return self.transition_stage(workflow, PolicyEvolutionStage.STRUCTURE_REVIEW_PROPOSED, PolicyEvolutionAuditActorType.SYSTEM, "structure proposal attached")

    def transition_stage(
        self,
        workflow: PolicyEvolutionWorkflow,
        target_stage: PolicyEvolutionStage | str,
        actor: PolicyEvolutionAuditActorType | str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyEvolutionWorkflow:
        from_stage = str(getattr(workflow.current_stage, "value", workflow.current_stage))
        target = str(getattr(target_stage, "value", target_stage))
        guard = self.guard.evaluate(workflow, target, actor=actor, metadata=metadata)
        status = _workflow_status_for_transition(target, guard)
        updated = workflow
        if guard.allowed:
            updated = replace(
                workflow,
                current_stage=target,
                status=status,
                updated_at=_now_iso(),
            )
        else:
            updated = replace(workflow, status=PolicyEvolutionWorkflowStatus.BLOCKED.value, updated_at=_now_iso())
        self.audit_log = tuple((*self.audit_log, PolicyEvolutionAuditLogEntry(
            entry_id=f"audit-{_compact_timestamp()}-{len(self.audit_log) + 1}",
            workflow_id=workflow.workflow_id,
            actor_type=str(getattr(actor, "value", actor)),
            action="transition_stage",
            from_stage=from_stage,
            to_stage=target,
            reason=reason,
            guard_allowed=guard.allowed,
            guard_violations=guard.violations,
            guard_warnings=guard.warnings,
            metadata=dict(metadata or {}),
        )))
        return updated

    def build_report(self, workflow: PolicyEvolutionWorkflow) -> PolicyEvolutionWorkflowReport:
        log = tuple(entry for entry in self.audit_log if entry.workflow_id == workflow.workflow_id)
        violations = tuple(v for entry in log for v in entry.guard_violations)
        completed = tuple(dict.fromkeys(entry.to_stage for entry in log if entry.guard_allowed))
        blocked = tuple(dict.fromkeys(entry.to_stage for entry in log if not entry.guard_allowed))
        approvals = tuple(
            entry.to_dict() for entry in log
            if entry.to_stage in {
                PolicyEvolutionStage.SHADOW_APPROVED.value,
                PolicyEvolutionStage.CANARY_APPROVED.value,
                PolicyEvolutionStage.FINAL_APPROVED.value,
            }
        )
        return PolicyEvolutionWorkflowReport(
            workflow_id=workflow.workflow_id,
            current_stage=workflow.current_stage,
            status=workflow.status,
            completed_stages=completed,
            blocked_stages=blocked,
            approval_history=approvals,
            training_summary={"training_job_id": workflow.training_job_id},
            offline_evaluation_summary={"candidate_artifact_id": workflow.candidate_artifact_id},
            shadow_summary={"proposal_id": workflow.shadow_proposal_id, "approval_id": workflow.shadow_approval_id, "result_id": workflow.shadow_result_id},
            canary_summary={"proposal_id": workflow.canary_proposal_id, "approval_id": workflow.canary_approval_id, "result_id": workflow.canary_result_id},
            final_approval_summary={"promotion_proposal_id": workflow.final_promotion_proposal_id, "final_approval_id": workflow.final_approval_id},
            weight_tuning_summary={"proposal_ids": workflow.weight_tuning_proposal_ids},
            structure_proposal_summary={"proposal_ids": workflow.structure_proposal_ids},
            rollback_target=(workflow.rollback_policy_id, workflow.rollback_policy_version),
            guard_violations=violations,
            audit_log_count=len(log),
            recommendation=_workflow_recommendation(workflow),
        )


class PolicyAutoTrainer:
    """Build offline candidate artifacts from policy evolution plans."""

    def __init__(
        self,
        *,
        training_guard: TrainingGuard | None = None,
        manager: PolicyEvolutionManager | None = None,
    ) -> None:
        self.training_guard = training_guard or TrainingGuard()
        self.manager = manager or PolicyEvolutionManager()

    def train_candidate(
        self,
        plan: PolicyEvolutionPlan,
        *,
        records: tuple[Any, ...] | list[Any] | None = None,
        dataset: Any | None = None,
        training_mode: CandidatePolicyTrainingMode | str = CandidatePolicyTrainingMode.IMITATION,
        training_config: dict[str, Any] | None = None,
        registry: PolicyVersionRegistry | None = None,
    ) -> tuple[CandidatePolicyTrainingJob, CandidatePolicyArtifact | None, PolicyVersionRegistry | None]:
        from app.services.learned_policy import (
            ImitationPolicy,
            LearnedBackendReranker,
            LearnedMetaPolicy,
            OfflineMetaPolicyTrainer,
            OfflinePolicyEvaluator,
            PolicyDatasetBuilder,
            PolicyDatasetAuditor,
            RewardSanityChecker,
        )

        config = dict(training_config or {})
        mode = str(getattr(training_mode, "value", training_mode))
        job = self.manager.create_training_job(
            plan,
            training_mode=mode,
            training_config=config,
        )
        if dataset is None:
            dataset = PolicyDatasetBuilder().build(tuple(records or ()))
        job = self.manager.update_training_job_status(job, CandidatePolicyTrainingJobStatus.DATASET_BUILT)

        audit = PolicyDatasetAuditor().audit(dataset)
        reward_sanity = RewardSanityChecker().check(dataset)
        pre_guard = self.training_guard.evaluate(
            plan,
            dataset,
            audit,
            reward_sanity,
            training_config=config,
        )
        if not pre_guard.allowed:
            return (
                self.manager.update_training_job_status(
                    job,
                    CandidatePolicyTrainingJobStatus.FAILED,
                    failure_reason=_guard_failure_reason(pre_guard),
                ),
                None,
                registry,
            )
        job = self.manager.update_training_job_status(job, CandidatePolicyTrainingJobStatus.AUDIT_PASSED)
        job = self.manager.update_training_job_status(job, CandidatePolicyTrainingJobStatus.REWARD_SANITY_PASSED)

        if mode == CandidatePolicyTrainingMode.IMITATION.value:
            policy = ImitationPolicy().fit(dataset)
            training_summary = {
                "training_mode": mode,
                "evaluation": policy.evaluate(dataset),
                "online_enabled": False,
            }
            artifact_type = "imitation_policy"
        elif mode == CandidatePolicyTrainingMode.BACKEND_RERANKER.value:
            reranker = LearnedBackendReranker(
                max_delta=float(config.get("max_delta", 0.01))
            ).fit(dataset)
            training_summary = {
                "training_mode": mode,
                "backend_reward_means": dict(reranker.backend_rewards),
                "online_enabled": False,
            }
            artifact_type = "learned_backend_reranker"
        elif mode == CandidatePolicyTrainingMode.META_POLICY.value:
            policy = LearnedMetaPolicy(
                max_delta=float(config.get("max_delta", 0.01))
            ).fit_imitation(dataset)
            meta_summary = OfflineMetaPolicyTrainer().train_imitation(dataset)
            training_summary = {
                "training_mode": mode,
                "evaluation": meta_summary["evaluation"],
                "imitation_pretrained": policy.imitation_pretrained,
                "online_enabled": False,
            }
            artifact_type = "learned_meta_policy"
        else:
            failed = self.manager.update_training_job_status(
                job,
                CandidatePolicyTrainingJobStatus.FAILED,
                failure_reason=f"unsupported training mode:{mode}",
            )
            return failed, None, registry

        job = self.manager.update_training_job_status(job, CandidatePolicyTrainingJobStatus.TRAINED)
        offline_evaluation = OfflinePolicyEvaluator(
            learned_delta_cap=float(config.get("learned_delta_cap", 0.01))
        ).evaluate_dataset(dataset)
        post_guard = self.training_guard.evaluate(
            plan,
            dataset,
            audit,
            reward_sanity,
            training_config=config,
            offline_evaluation_summary=offline_evaluation,
        )
        if not post_guard.allowed:
            return (
                self.manager.update_training_job_status(
                    job,
                    CandidatePolicyTrainingJobStatus.FAILED,
                    failure_reason=_guard_failure_reason(post_guard),
                ),
                None,
                registry,
            )

        safety_summary = dict(offline_evaluation.get("learned_policy_safety") or {})
        offline_summary = {
            **_compact_offline_summary(offline_evaluation),
            "dataset_audit": _audit_shadow_summary(audit),
            "reward_sanity": _reward_shadow_summary(reward_sanity),
            "feature_schema_version": dataset.feature_schema_version,
            "reward_version": dataset.reward_version,
            "counterfactual_uncertainty_summary": _counterfactual_summary(dataset),
        }
        registry_preview = PolicyVersionRegistryEntry(
            policy_id=plan.candidate_policy_id,
            policy_version=plan.candidate_policy_version,
            parent_policy_id=plan.source_policy_id,
            parent_policy_version=plan.source_policy_version,
            trained_on_dataset_version=dataset.dataset_version,
            feature_schema_version=dataset.feature_schema_version,
            reward_version=dataset.reward_version,
            training_config_summary=config,
            offline_evaluation_summary=offline_summary,
            approved_for_shadow=False,
            approved_for_safe_soft=False,
            approved_for_live_canary=False,
            rollback_target=(plan.source_policy_id, plan.source_policy_version),
        )
        artifact = CandidatePolicyArtifact(
            policy_id=plan.candidate_policy_id,
            policy_version=plan.candidate_policy_version,
            parent_policy_id=plan.source_policy_id,
            parent_policy_version=plan.source_policy_version,
            artifact_type=artifact_type,
            training_mode=mode,
            dataset_version=dataset.dataset_version,
            feature_schema_version=dataset.feature_schema_version,
            reward_version=dataset.reward_version,
            training_summary=training_summary,
            offline_evaluation_summary=offline_summary,
            safety_summary=safety_summary,
            eligible_for_shadow_proposal=bool(
                offline_evaluation.get("learned_policy_safety", {}).get("passed", False)
            ),
            eligible_for_canary_proposal=False,
            shadow_promotion_eligible=bool(
                offline_evaluation.get("learned_policy_safety", {}).get("passed", False)
            ),
            shadow_promotion_reason="offline evaluation passed; human shadow approval still required",
            registry_entry_preview=_plain_dict(registry_preview),
        )
        if registry is not None:
            registry = registry.register(registry_preview)
        return (
            self.manager.update_training_job_status(
                job,
                CandidatePolicyTrainingJobStatus.OFFLINE_EVALUATED,
            ),
            artifact,
            registry,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _latest(entries: Any) -> PolicyVersionRegistryEntry | None:
    ordered = list(entries)
    if not ordered:
        return None
    return sorted(ordered, key=lambda entry: (entry.registered_at, entry.policy_version))[-1]


def _next_candidate_version(source_version: str) -> str:
    return f"{source_version}.candidate"


def _plan_reasons(
    trigger: PolicyEvolutionTrigger,
    dataset_summary: dict[str, Any] | None,
    audit_summary: dict[str, Any] | None,
) -> list[str]:
    reasons = [f"trigger:{str(trigger.trigger_type)}"]
    if dataset_summary and dataset_summary.get("dataset_version"):
        reasons.append(f"dataset:{dataset_summary['dataset_version']}")
    if audit_summary and audit_summary.get("audit_version"):
        reasons.append(f"audit:{audit_summary['audit_version']}")
    return reasons


def _has_unknown_counterfactual(dataset: Any) -> bool:
    for row in getattr(dataset, "records", ()) or ():
        outcome = row.get("outcome") or {}
        if outcome.get("counterfactual_label") == "unknown_counterfactual":
            return True
    return False


def _guard_failure_reason(result: TrainingGuardResult) -> str:
    return "; ".join(
        str(violation.get("check", "training_guard_violation"))
        for violation in result.violations
    ) or "training_guard_violation"


def _compact_offline_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "imitation_policy_summary": dict(report.get("imitation_policy_summary") or {}),
        "learned_reranker_summary": dict(report.get("learned_reranker_summary") or {}),
        "learned_policy_safety": dict(report.get("learned_policy_safety") or {}),
        "n_learned_policy_traces": len(report.get("learned_policy_traces") or ()),
    }


def _audit_shadow_summary(audit: Any) -> dict[str, Any]:
    missing = dict(getattr(audit, "missing_feature_rates", {}) or {})
    return {
        "passed": (
            int(getattr(audit, "record_count", 0) or 0) > 0
            and all(float(missing.get(key, 0.0) or 0.0) == 0.0 for key in (
                "state_features",
                "context_features",
                "available_actions",
                "candidate_backends",
            ))
            and float(getattr(audit, "candidate_score_coverage", 0.0) or 0.0) == 1.0
            and float(getattr(audit, "candidate_rank_coverage", 0.0) or 0.0) == 1.0
        ),
        "record_count": int(getattr(audit, "record_count", 0) or 0),
        "missing_feature_rates": missing,
        "candidate_score_coverage": float(getattr(audit, "candidate_score_coverage", 0.0) or 0.0),
        "candidate_rank_coverage": float(getattr(audit, "candidate_rank_coverage", 0.0) or 0.0),
    }


def _reward_shadow_summary(reward_sanity: Any) -> dict[str, Any]:
    return {
        "passed": bool(getattr(reward_sanity, "passed", False)),
        "failures": tuple(getattr(reward_sanity, "failures", ()) or ()),
        "warnings": tuple(getattr(reward_sanity, "warnings", ()) or ()),
        "reward_version_distribution": dict(getattr(reward_sanity, "reward_version_distribution", {}) or {}),
    }


def _counterfactual_summary(dataset: Any) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in getattr(dataset, "records", ()) or ():
        label = str((row.get("outcome") or {}).get("counterfactual_label") or "unknown_counterfactual")
        counts[label] = counts.get(label, 0) + 1
    return {
        "label_distribution": counts,
        "primary_improvement_evidence": "observed_or_replay_reward",
    }


def _unknown_counterfactual_primary_evidence(proposal: ShadowPromotionProposal) -> bool:
    summary = proposal.counterfactual_uncertainty_summary or {}
    primary = str(summary.get("primary_improvement_evidence") or "")
    return primary == "unknown_counterfactual"


def _shadow_result_passes_canary_thresholds(result: ShadowRunResult) -> bool:
    return (
        str(getattr(result.recommendation, "value", result.recommendation))
        == ShadowRunRecommendation.PROPOSE_CANARY.value
        and result.round_count > 0
        and result.safety_warning_count == 0
        and result.invalid_suggestion_rate <= 0.05
        and result.backend_agreement_rate >= 0.7
    )


def _approval_from_registry(entry: PolicyVersionRegistryEntry | None) -> ShadowApprovalRecord | None:
    if entry is None or not entry.shadow_approval_metadata:
        return None
    return ShadowApprovalRecord.from_dict(entry.shadow_approval_metadata)


def _canary_approval_from_registry(entry: PolicyVersionRegistryEntry | None) -> CanaryApprovalRecord | None:
    if entry is None or not entry.canary_approval_metadata:
        return None
    return CanaryApprovalRecord.from_dict(entry.canary_approval_metadata)


def _is_past_iso(value: str | None) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed < datetime.now(timezone.utc)


def _confidence_calibration_score(summary: dict[str, Any]) -> float:
    if "calibration_score" in summary:
        return float(summary.get("calibration_score") or 0.0)
    if "score" in summary:
        return float(summary.get("score") or 0.0)
    if not summary:
        return 1.0
    return float(summary.get("mean_confidence_alignment", 1.0) or 0.0)


def _unknown_counterfactual_rate(breakdown: dict[str, int]) -> float:
    total = sum(int(value or 0) for value in breakdown.values())
    if total <= 0:
        return 0.0
    return int(breakdown.get("unknown_counterfactual", 0) or 0) / total


def _canary_unknown_counterfactual_ground_truth(proposal: CanaryPromotionProposal) -> bool:
    return bool(
        proposal.failure_summary.get("unknown_counterfactual_as_ground_truth")
        or proposal.shadow_result_summary.get("unknown_counterfactual_as_ground_truth")
    )


def _scope_within(requested: tuple[str, ...], allowed: tuple[str, ...]) -> bool:
    if not requested:
        return True
    if not allowed:
        return False
    return set(requested).issubset(set(allowed))


def _scope_contains(value: str, allowed: tuple[str, ...]) -> bool:
    return not allowed or value in set(allowed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _default_weight_cap(target: str) -> float:
    if target == PolicyWeightTuningTarget.LEARNED_POLICY_MAX_WEIGHT.value:
        return 0.005
    if target == PolicyWeightTuningTarget.BANDIT_MAX_WEIGHT.value:
        return 0.01
    if target == PolicyWeightTuningTarget.TOTAL_INFLUENCE_CAP.value:
        return 0.05
    return 0.03


def _weight_evidence_summary(evidence: tuple[WeightTuningEvidence, ...]) -> dict[str, Any]:
    return {
        "evidence_count": len(evidence),
        "source_types": tuple(str(getattr(item.source_type, "value", item.source_type)) for item in evidence),
        "metric_names": tuple(item.metric_name for item in evidence),
        "primary_improvement_evidence": "observed_or_replay_reward",
    }


def _weight_risk_level(delta: float, proposed_weight: float, max_allowed_weight: float) -> str:
    if delta <= 0:
        return PolicyWeightTuningRiskLevel.LOW.value
    if max_allowed_weight > 0 and proposed_weight >= max_allowed_weight:
        return PolicyWeightTuningRiskLevel.HIGH.value
    return PolicyWeightTuningRiskLevel.MEDIUM.value


def _has_calibration_evidence(evidence: tuple[WeightTuningEvidence, ...]) -> bool:
    return any("calibration" in item.metric_name for item in evidence)


def _evidence_metric_increased(evidence: tuple[WeightTuningEvidence, ...], metric_fragment: str) -> bool:
    for item in evidence:
        if metric_fragment not in item.metric_name:
            continue
        delta = item.delta
        if delta is None and item.baseline_value is not None and item.candidate_value is not None:
            delta = item.candidate_value - item.baseline_value
        if delta is not None and delta > 0:
            return True
    return False


def _weight_unknown_counterfactual_primary_evidence(proposal: PolicyWeightTuningProposal) -> bool:
    primary = str(proposal.evidence_summary.get("primary_improvement_evidence") or "")
    if primary == "unknown_counterfactual":
        return True
    return any(item.counterfactual_label == "unknown_counterfactual" and item.confidence >= 0.8 for item in proposal.evidence)


def _structure_evidence_summary(evidence: tuple[PolicyStructureEvidence, ...]) -> dict[str, Any]:
    return {
        "evidence_count": len(evidence),
        "source_types": tuple(str(getattr(item.source_type, "value", item.source_type)) for item in evidence),
        "metric_names": tuple(item.metric_name for item in evidence),
        "primary_improvement_evidence": "observed_or_replay_reward",
    }


def _structure_risk_level(affected_components: tuple[str, ...]) -> str:
    high_risk = {"reward", "safety", "transition_guard", "backend_prior"}
    if any(component in high_risk for component in affected_components):
        return PolicyStructureRiskLevel.HIGH.value
    if affected_components:
        return PolicyStructureRiskLevel.MEDIUM.value
    return PolicyStructureRiskLevel.LOW.value


def _structure_unknown_counterfactual_ground_truth(proposal: PolicyStructureProposal) -> bool:
    if str(proposal.evidence_summary.get("primary_improvement_evidence") or "") == "unknown_counterfactual":
        return True
    return any(item.counterfactual_label == "unknown_counterfactual" and item.confidence >= 0.8 for item in proposal.evidence)


def _structure_recommended_reviewers(proposal: PolicyStructureProposal) -> tuple[str, ...]:
    reviewers = {"policy_owner"}
    components = set(proposal.affected_components)
    proposal_type = str(getattr(proposal.proposal_type, "value", proposal.proposal_type))
    if "reward" in components or proposal_type == PolicyStructureProposalType.REWARD_FEATURE_CHANGE.value:
        reviewers.add("reward_owner")
    if "safety" in components or "transition_guard" in components:
        reviewers.add("safety_reviewer")
    if "backend_prior" in components:
        reviewers.add("optimization_owner")
    return tuple(sorted(reviewers))


def _actor_from_approval(mode: Any) -> str:
    value = str(getattr(mode, "value", mode))
    if value == "manual":
        return PolicyEvolutionAuditActorType.HUMAN.value
    if value == "config":
        return PolicyEvolutionAuditActorType.CONFIG.value
    if value == "test":
        return PolicyEvolutionAuditActorType.TEST.value
    return PolicyEvolutionAuditActorType.SYSTEM.value


def _workflow_status_for_transition(target: str, guard: PolicyEvolutionWorkflowGuardResult) -> str:
    if not guard.allowed:
        return PolicyEvolutionWorkflowStatus.BLOCKED.value
    if target == PolicyEvolutionStage.COMPLETED.value:
        return PolicyEvolutionWorkflowStatus.COMPLETED.value
    if target == PolicyEvolutionStage.REJECTED.value:
        return PolicyEvolutionWorkflowStatus.REJECTED.value
    if target == PolicyEvolutionStage.ROLLED_BACK.value:
        return PolicyEvolutionWorkflowStatus.ROLLED_BACK.value
    if target == PolicyEvolutionStage.CANCELLED.value:
        return PolicyEvolutionWorkflowStatus.CANCELLED.value
    if target in {
        PolicyEvolutionStage.SHADOW_PROPOSED.value,
        PolicyEvolutionStage.CANARY_PROPOSED.value,
        PolicyEvolutionStage.PROMOTION_PROPOSED.value,
        PolicyEvolutionStage.WEIGHT_TUNING_PROPOSED.value,
        PolicyEvolutionStage.STRUCTURE_REVIEW_PROPOSED.value,
    }:
        return PolicyEvolutionWorkflowStatus.WAITING_FOR_APPROVAL.value
    return PolicyEvolutionWorkflowStatus.ACTIVE.value


def _workflow_recommendation(workflow: PolicyEvolutionWorkflow) -> str:
    stage = str(getattr(workflow.current_stage, "value", workflow.current_stage))
    status = str(getattr(workflow.status, "value", workflow.status))
    if status == PolicyEvolutionWorkflowStatus.BLOCKED.value:
        return PolicyEvolutionWorkflowRecommendation.REJECT.value
    if stage in {PolicyEvolutionStage.PLANNED.value, PolicyEvolutionStage.TRAINING_REQUESTED.value}:
        return PolicyEvolutionWorkflowRecommendation.RUN_TRAINING.value
    if stage == PolicyEvolutionStage.SHADOW_PROPOSED.value:
        return PolicyEvolutionWorkflowRecommendation.WAIT_FOR_APPROVAL.value
    if stage == PolicyEvolutionStage.SHADOW_APPROVED.value:
        return PolicyEvolutionWorkflowRecommendation.RUN_SHADOW.value
    if stage == PolicyEvolutionStage.CANARY_PROPOSED.value:
        return PolicyEvolutionWorkflowRecommendation.WAIT_FOR_APPROVAL.value
    if stage == PolicyEvolutionStage.CANARY_APPROVED.value:
        return PolicyEvolutionWorkflowRecommendation.RUN_CANARY.value
    if stage == PolicyEvolutionStage.PROMOTION_PROPOSED.value:
        return PolicyEvolutionWorkflowRecommendation.APPROVE_PROMOTION.value
    if stage == PolicyEvolutionStage.WEIGHT_TUNING_PROPOSED.value:
        return PolicyEvolutionWorkflowRecommendation.REVIEW_WEIGHT_TUNING.value
    if stage == PolicyEvolutionStage.STRUCTURE_REVIEW_PROPOSED.value:
        return PolicyEvolutionWorkflowRecommendation.REVIEW_STRUCTURE_PROPOSAL.value
    if stage == PolicyEvolutionStage.FINAL_APPROVED.value:
        return PolicyEvolutionWorkflowRecommendation.COMPLETE.value
    if stage == PolicyEvolutionStage.ROLLED_BACK.value:
        return PolicyEvolutionWorkflowRecommendation.ROLLBACK.value
    if stage == PolicyEvolutionStage.REJECTED.value:
        return PolicyEvolutionWorkflowRecommendation.REJECT.value
    if stage == PolicyEvolutionStage.COMPLETED.value:
        return PolicyEvolutionWorkflowRecommendation.COMPLETE.value
    return PolicyEvolutionWorkflowRecommendation.CONTINUE_WORKFLOW.value


def _canary_result_passes_promotion_thresholds(result: CanaryRunResult) -> bool:
    return (
        str(getattr(result.recommendation, "value", result.recommendation))
        == CanaryRunRecommendation.PROPOSE_PROMOTION.value
        and result.round_count > 0
        and result.applied_round_count > 0
        and result.safety_warning_count == 0
        and not result.auto_disable_triggered
        and result.backend_failure_rate <= 0.05
        and result.constraint_failure_rate <= 0.05
        and result.reward_vs_baseline >= 0.0
        and result.reward_vs_safe_influence >= 0.0
    )


def _final_unknown_counterfactual_primary_evidence(proposal: FinalPromotionProposal) -> bool:
    primary = str(
        proposal.counterfactual_breakdown.get("primary_improvement_evidence")
        or proposal.canary_result_summary.get("primary_improvement_evidence")
        or ""
    )
    return primary == "unknown_counterfactual"


def _plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        raw = asdict(value)
    else:
        raw = dict(value)
    return _plain_value(raw)


def _plain_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _violation(check: str, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"check": check, "reason": reason, "metadata": dict(metadata or {})}


def _warning(check: str, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"check": check, "reason": reason, "metadata": dict(metadata or {})}
