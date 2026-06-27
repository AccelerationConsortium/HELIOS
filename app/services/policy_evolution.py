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

    def __init__(self, *, guard: EvolutionGuard | None = None) -> None:
        self.guard = guard or EvolutionGuard()

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
            return PolicyEvolutionRecommendation.PREPARE_DATASET
        if str(plan.status) == PolicyEvolutionPlanStatus.DATASET_READY.value:
            return PolicyEvolutionRecommendation.RUN_OFFLINE_EVAL
        if str(plan.status) == PolicyEvolutionPlanStatus.OFFLINE_EVALUATED.value:
            return PolicyEvolutionRecommendation.APPROVE_SHADOW
        if str(plan.status) == PolicyEvolutionPlanStatus.SHADOW_ELIGIBLE.value:
            return PolicyEvolutionRecommendation.APPROVE_CANARY
        if str(plan.status) == PolicyEvolutionPlanStatus.CANARY_ELIGIBLE.value:
            return PolicyEvolutionRecommendation.PROMOTE
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
