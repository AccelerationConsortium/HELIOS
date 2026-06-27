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
    ) -> None:
        self.guard = guard or EvolutionGuard()
        self.shadow_promotion_guard = shadow_promotion_guard or ShadowPromotionGuard()
        self.shadow_approval_guard = shadow_approval_guard or ShadowApprovalGuard()
        self.canary_promotion_guard = canary_promotion_guard or CanaryPromotionGuard()

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
            return PolicyEvolutionRecommendation.APPROVE_CANARY
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
