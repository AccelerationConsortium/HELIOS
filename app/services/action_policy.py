"""Trace-only campaign action policy and transition audits."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.services.strategy_models import (
    ActionPolicyDecision,
    ActionTransitionRecord,
    CampaignIntent,
    CampaignSnapshot,
    FailureType,
    ObjectiveLevel,
    OptimizationMode,
    StrategyEvidence,
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _add_prior(priors: dict[str, float], key: str, amount: float) -> None:
    priors[key] = round(priors.get(key, 0.0) + amount, 4)


class ActionPolicyMatrix:
    """Explicit, auditable priors for campaign intent/mode/backend pools.

    The matrix is deliberately trace-only. It emits StrategyEvidence for every
    boost, penalty, and veto, but does not change the selected action/backend.
    """

    OBJECTIVE_RULES: dict[str, tuple[tuple[str, str, float, str], ...]] = {
        ObjectiveLevel.FEASIBILITY.value: (
            ("intent", CampaignIntent.STABILIZE.value, 0.35, "feasibility work favors stabilization"),
            ("intent", CampaignIntent.RECOVER.value, 0.25, "feasibility work favors recovery from setup issues"),
            ("intent", CampaignIntent.DIAGNOSE.value, 0.2, "feasibility work favors diagnosis before optimization"),
            ("mode", OptimizationMode.BASELINE_CALIBRATION.value, 0.25, "feasibility work needs calibration priors"),
        ),
        ObjectiveLevel.DATA_QUALITY.value: (
            ("intent", CampaignIntent.STABILIZE.value, 0.35, "data-quality work favors replication"),
            ("mode", OptimizationMode.REPLICATE.value, 0.3, "data-quality work favors replicate mode"),
            ("mode", OptimizationMode.BASELINE_CALIBRATION.value, 0.25, "data-quality work favors calibration"),
        ),
        ObjectiveLevel.BASELINE.value: (
            ("intent", CampaignIntent.DISCOVER.value, 0.4, "baseline work favors broad exploration"),
            ("mode", OptimizationMode.EXPLORE.value, 0.35, "baseline work favors explore mode"),
        ),
        ObjectiveLevel.PERFORMANCE.value: (
            ("intent", CampaignIntent.OPTIMIZE.value, 0.4, "performance work favors optimization"),
            ("mode", OptimizationMode.EXPLOIT.value, 0.3, "performance work favors exploit mode"),
            ("mode", OptimizationMode.REFINE.value, 0.15, "performance work can benefit from local refinement"),
        ),
        ObjectiveLevel.MECHANISM.value: (
            ("intent", CampaignIntent.VALIDATE.value, 0.4, "mechanism work favors validation"),
            ("intent", CampaignIntent.HYPOTHESIS_TEST.value, 0.25, "mechanism work favors hypothesis testing"),
            ("mode", OptimizationMode.MECHANISM_VALIDATION.value, 0.35, "mechanism work favors mechanism validation"),
            ("mode", OptimizationMode.HYPOTHESIS_TEST.value, 0.25, "mechanism work favors explicit hypothesis-test mode"),
        ),
        ObjectiveLevel.GENERALIZATION.value: (
            ("intent", CampaignIntent.TRANSFER.value, 0.3, "generalization work favors transfer"),
            ("intent", CampaignIntent.PIVOT.value, 0.3, "generalization work favors route pivot proposals"),
            ("mode", OptimizationMode.WARM_START.value, 0.25, "generalization work favors warm-start evidence"),
            ("mode", OptimizationMode.ROUTE_SWITCH.value, 0.25, "generalization work favors route-switch evidence"),
        ),
    }

    FAILURE_RULES: dict[str, tuple[tuple[str, str, float, str, str], ...]] = {
        FailureType.HARDWARE.value: (
            ("intent", CampaignIntent.RECOVER.value, 0.45, "boost", "hardware failure favors recovery"),
            ("intent", CampaignIntent.DIAGNOSE.value, 0.25, "boost", "hardware failure favors diagnosis"),
            ("intent", CampaignIntent.OPTIMIZE.value, 1.0, "veto", "hardware failure should not push optimization directly"),
        ),
        FailureType.MEASUREMENT.value: (
            ("intent", CampaignIntent.STABILIZE.value, 0.35, "boost", "measurement failure favors stabilization"),
            ("intent", CampaignIntent.DIAGNOSE.value, 0.3, "boost", "measurement failure favors diagnosis"),
            ("mode", OptimizationMode.BASELINE_CALIBRATION.value, 0.3, "boost", "measurement failure favors calibration"),
        ),
        FailureType.CONSTRAINT.value: (
            ("intent", CampaignIntent.DIAGNOSE.value, 0.35, "boost", "constraint failure favors diagnosis"),
            ("intent", CampaignIntent.PIVOT.value, 0.25, "boost", "constraint failure can justify space revision"),
            ("intent", CampaignIntent.REVISE_SPACE.value, 0.3, "boost", "constraint failure favors explicit space revision review"),
            ("mode", OptimizationMode.CONSTRAINED_SEARCH.value, 0.35, "boost", "constraint failure favors constrained search"),
            ("mode", OptimizationMode.FAILURE_AVOIDANCE.value, 0.3, "boost", "constraint failure favors failure-region avoidance"),
            ("mode", OptimizationMode.SPACE_REVISION.value, 0.3, "boost", "constraint failure favors space-revision mode"),
        ),
        FailureType.BACKEND.value: (
            ("intent", CampaignIntent.RECOVER.value, 0.35, "boost", "backend failure favors recovery"),
            ("intent", CampaignIntent.DIAGNOSE.value, 0.25, "boost", "backend failure favors failure localization"),
            ("mode", OptimizationMode.FAILURE_LOCALIZATION.value, 0.3, "boost", "backend failure favors localization"),
        ),
        FailureType.SCIENTIFIC_NEGATIVE.value: (
            ("intent", CampaignIntent.VALIDATE.value, 0.35, "boost", "scientific negative result is evidence for validation"),
            ("intent", CampaignIntent.HYPOTHESIS_GENERATE.value, 0.25, "boost", "scientific negative result can generate revised hypotheses"),
            ("intent", CampaignIntent.HYPOTHESIS_TEST.value, 0.25, "boost", "scientific negative result should be tested against hypotheses"),
            ("mode", OptimizationMode.MECHANISM_VALIDATION.value, 0.3, "boost", "scientific negative result favors hypothesis update"),
            ("mode", OptimizationMode.HYPOTHESIS_GENERATION.value, 0.25, "boost", "scientific negative result favors hypothesis generation"),
            ("mode", OptimizationMode.HYPOTHESIS_TEST.value, 0.25, "boost", "scientific negative result favors hypothesis testing"),
        ),
    }

    INTENT_BACKEND_PRIORS: dict[str, tuple[tuple[str, float, str], ...]] = {
        CampaignIntent.DISCOVER.value: (
            ("lhs", 0.25, "discover intent favors LHS broad exploration"),
            ("nexus_lhs", 0.25, "discover intent favors Nexus LHS broad exploration"),
            ("nexus_sobol", 0.25, "discover intent favors Sobol broad exploration"),
            ("random_sampling", 0.2, "discover intent favors random broad exploration"),
        ),
        CampaignIntent.OPTIMIZE.value: (
            ("bomcp", 0.3, "optimize intent favors BO MCP backends"),
            ("nexus_gp_bo", 0.3, "optimize intent favors Nexus GP-BO style backends"),
            ("nexus_turbo", 0.25, "optimize intent favors TuRBO style backends"),
            ("nexus_tpe", 0.2, "optimize intent favors TPE style backends"),
            ("optuna_tpe", 0.2, "optimize intent favors TPE style backends"),
        ),
        CampaignIntent.VALIDATE.value: (
            ("built_in", 0.2, "validate intent favors local probe and replication paths"),
            ("mechanism_validation", 0.25, "validate intent favors mechanism-validation modes"),
            ("replicate", 0.25, "validate intent favors replicate modes"),
        ),
        CampaignIntent.STABILIZE.value: (
            ("built_in", 0.2, "stabilize intent favors measurement-check paths"),
            ("calibration", 0.25, "stabilize intent favors calibration modes"),
            ("replicate", 0.25, "stabilize intent favors replication modes"),
        ),
        CampaignIntent.DIAGNOSE.value: (
            ("control_probe", 0.25, "diagnose intent favors control probes"),
            ("failure_localization", 0.25, "diagnose intent favors failure localization"),
            ("built_in", 0.15, "diagnose intent favors conservative local checks"),
        ),
        CampaignIntent.PIVOT.value: (
            ("route_switch", 0.3, "pivot intent favors route-switch proposals"),
            ("space_revision", 0.3, "pivot intent favors space-revision proposals"),
            ("revise_space", 0.3, "pivot intent favors explicit space-revision review"),
        ),
        CampaignIntent.REVISE_SPACE.value: (
            ("space_revision", 0.35, "revise-space intent favors space-revision proposals"),
            ("revise_space", 0.35, "revise-space intent favors explicit parameter-space review"),
        ),
        CampaignIntent.TRANSFER.value: (
            ("warm_start", 0.3, "transfer intent favors prior-campaign warm starts"),
            ("prior_campaign", 0.25, "transfer intent favors prior-campaign paths"),
            ("nexus_gp_bo", 0.15, "transfer intent can reuse Nexus optimization priors"),
        ),
        CampaignIntent.HYPOTHESIS_GENERATE.value: (
            ("hypothesis_generate", 0.35, "hypothesis-generate intent favors revised mechanism hypotheses"),
            ("mechanism_validation", 0.15, "hypothesis-generate intent can feed mechanism validation"),
        ),
        CampaignIntent.HYPOTHESIS_TEST.value: (
            ("hypothesis_test", 0.35, "hypothesis-test intent favors discriminative validation"),
            ("mechanism_validation", 0.25, "hypothesis-test intent favors mechanism-validation modes"),
        ),
    }

    def evaluate(
        self,
        snapshot: CampaignSnapshot,
        *,
        selected_intent: CampaignIntent | str,
        selected_mode: OptimizationMode | str,
        selected_backend: str,
    ) -> ActionPolicyDecision:
        intent_priors: dict[str, float] = defaultdict(float)
        mode_priors: dict[str, float] = defaultdict(float)
        backend_priors: dict[str, float] = defaultdict(float)
        vetoes: list[str] = []
        evidence: list[StrategyEvidence] = []

        context = snapshot.campaign_context
        level = _value(context.current_objective_level if context else ObjectiveLevel.PERFORMANCE)
        for bucket, target, strength, reason in self.OBJECTIVE_RULES.get(level, ()):
            self._record_rule(
                bucket, target, strength, "boost", reason,
                intent_priors, mode_priors, backend_priors, vetoes, evidence,
                source="action_policy:objective",
                metadata={"objective_level": level},
            )

        for failure in snapshot.failure_events:
            ftype = _value(failure.failure_type)
            for bucket, target, strength, effect, reason in self.FAILURE_RULES.get(ftype, ()):
                self._record_rule(
                    bucket, target, strength, effect, reason,
                    intent_priors, mode_priors, backend_priors, vetoes, evidence,
                    source="action_policy:failure",
                    metadata={"failure_type": ftype, "backend_name": failure.backend_name},
                )
            if ftype == FailureType.SCIENTIFIC_NEGATIVE.value:
                evidence.append(StrategyEvidence(
                    source="hypothesis_policy",
                    target="hypothesis_update",
                    effect="boost",
                    strength=0.35,
                    reason="Scientific negative result should update active hypotheses",
                    metadata={"failure_reason": failure.reason},
                ))

        selected_intent_value = _value(selected_intent)
        for backend, strength, reason in self.INTENT_BACKEND_PRIORS.get(selected_intent_value, ()):
            _add_prior(backend_priors, backend, strength)
            evidence.append(StrategyEvidence(
                source="intent_backend_prior",
                target=backend,
                effect="boost",
                strength=strength,
                reason=reason,
                metadata={"selected_intent": selected_intent_value},
            ))

        if context and level == ObjectiveLevel.MECHANISM.value and context.domain_hypotheses:
            self._record_rule(
                "intent", CampaignIntent.VALIDATE.value, 0.35, "boost",
                "Active mechanism hypothesis favors validation",
                intent_priors, mode_priors, backend_priors, vetoes, evidence,
                source="hypothesis_policy",
                metadata={"n_hypotheses": len(context.domain_hypotheses)},
            )
            self._record_rule(
                "mode", OptimizationMode.MECHANISM_VALIDATION.value, 0.35, "boost",
                "Active mechanism hypothesis favors mechanism validation",
                intent_priors, mode_priors, backend_priors, vetoes, evidence,
                source="hypothesis_policy",
                metadata={"n_hypotheses": len(context.domain_hypotheses)},
            )
            self._record_rule(
                "intent", CampaignIntent.HYPOTHESIS_TEST.value, 0.3, "boost",
                "Active mechanism hypothesis favors explicit hypothesis testing",
                intent_priors, mode_priors, backend_priors, vetoes, evidence,
                source="hypothesis_policy",
                metadata={"n_hypotheses": len(context.domain_hypotheses)},
            )
            self._record_rule(
                "mode", OptimizationMode.HYPOTHESIS_TEST.value, 0.3, "boost",
                "Active mechanism hypothesis favors hypothesis-test mode",
                intent_priors, mode_priors, backend_priors, vetoes, evidence,
                source="hypothesis_policy",
                metadata={"n_hypotheses": len(context.domain_hypotheses)},
            )
            if len(context.domain_hypotheses) > 1:
                evidence.append(StrategyEvidence(
                    source="hypothesis_policy",
                    target="discriminative_validation",
                    effect="boost",
                    strength=0.3,
                    reason="Competing hypotheses propose discriminative validation actions",
                    metadata={"hypotheses": list(context.domain_hypotheses)},
                ))

        if context and context.prior_campaigns:
            self._record_rule(
                "intent", CampaignIntent.TRANSFER.value, 0.2, "boost",
                "Prior campaigns provide transfer evidence",
                intent_priors, mode_priors, backend_priors, vetoes, evidence,
                source="action_policy:context",
                metadata={"n_prior_campaigns": len(context.prior_campaigns)},
            )

        evidence.append(StrategyEvidence(
            source="action_policy:selected",
            target=f"{selected_intent_value}:{_value(selected_mode)}:{selected_backend}",
            effect="context",
            strength=0.0,
            reason="Action policy recorded priors without changing execution behavior",
        ))

        return ActionPolicyDecision(
            intent_priors=dict(intent_priors),
            mode_priors=dict(mode_priors),
            backend_priors=dict(backend_priors),
            vetoes=tuple(vetoes),
            evidence=tuple(evidence),
        )

    @staticmethod
    def _record_rule(
        bucket: str,
        target: str,
        strength: float,
        effect: str,
        reason: str,
        intent_priors: dict[str, float],
        mode_priors: dict[str, float],
        backend_priors: dict[str, float],
        vetoes: list[str],
        evidence: list[StrategyEvidence],
        *,
        source: str,
        metadata: dict[str, Any],
    ) -> None:
        if effect == "veto":
            vetoes.append(target)
        elif bucket == "intent":
            _add_prior(intent_priors, target, strength)
        elif bucket == "mode":
            _add_prior(mode_priors, target, strength)
        elif bucket == "backend":
            _add_prior(backend_priors, target, strength)
        evidence.append(StrategyEvidence(
            source=source,
            target=target,
            effect=effect,
            strength=strength,
            reason=reason,
            metadata={**metadata, "bucket": bucket},
        ))


class ActionTransitionGuard:
    """Trace-only guard for campaign-intent transitions."""

    ALLOWED_TRANSITIONS: dict[str, set[str]] = {
        CampaignIntent.DISCOVER.value: {
            CampaignIntent.DISCOVER.value,
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.REVISE_SPACE.value,
        },
        CampaignIntent.OPTIMIZE.value: {
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.VALIDATE.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.PIVOT.value,
            CampaignIntent.HYPOTHESIS_TEST.value,
        },
        CampaignIntent.VALIDATE.value: {
            CampaignIntent.VALIDATE.value,
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.PIVOT.value,
            CampaignIntent.HYPOTHESIS_TEST.value,
            CampaignIntent.HYPOTHESIS_GENERATE.value,
        },
        CampaignIntent.STABILIZE.value: {
            CampaignIntent.STABILIZE.value,
            CampaignIntent.DISCOVER.value,
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.VALIDATE.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.HYPOTHESIS_TEST.value,
        },
        CampaignIntent.RECOVER.value: {
            CampaignIntent.RECOVER.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.DISCOVER.value,
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.REVISE_SPACE.value,
        },
        CampaignIntent.DIAGNOSE.value: {
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.RECOVER.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.PIVOT.value,
            CampaignIntent.REVISE_SPACE.value,
            CampaignIntent.DISCOVER.value,
            CampaignIntent.VALIDATE.value,
        },
        CampaignIntent.PIVOT.value: {
            CampaignIntent.PIVOT.value,
            CampaignIntent.DISCOVER.value,
            CampaignIntent.TRANSFER.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.REVISE_SPACE.value,
        },
        CampaignIntent.TRANSFER.value: {
            CampaignIntent.TRANSFER.value,
            CampaignIntent.DISCOVER.value,
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.VALIDATE.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.HYPOTHESIS_TEST.value,
        },
        CampaignIntent.REVISE_SPACE.value: {
            CampaignIntent.REVISE_SPACE.value,
            CampaignIntent.PIVOT.value,
            CampaignIntent.DIAGNOSE.value,
            CampaignIntent.DISCOVER.value,
            CampaignIntent.OPTIMIZE.value,
        },
        CampaignIntent.HYPOTHESIS_GENERATE.value: {
            CampaignIntent.HYPOTHESIS_GENERATE.value,
            CampaignIntent.HYPOTHESIS_TEST.value,
            CampaignIntent.VALIDATE.value,
            CampaignIntent.DISCOVER.value,
        },
        CampaignIntent.HYPOTHESIS_TEST.value: {
            CampaignIntent.HYPOTHESIS_TEST.value,
            CampaignIntent.VALIDATE.value,
            CampaignIntent.OPTIMIZE.value,
            CampaignIntent.STABILIZE.value,
            CampaignIntent.HYPOTHESIS_GENERATE.value,
        },
    }

    def evaluate(
        self,
        previous_intent: CampaignIntent | str | None,
        selected_intent: CampaignIntent | str,
        evidence: tuple[StrategyEvidence, ...],
    ) -> ActionTransitionRecord:
        to_intent = _value(selected_intent)
        from_intent = _value(previous_intent) if previous_intent else None
        if not from_intent:
            transition_evidence = (
                StrategyEvidence(
                    source="transition_guard",
                    target=to_intent,
                    effect="context",
                    strength=0.0,
                    reason="No previous intent; transition audit initialized",
                ),
            )
            return ActionTransitionRecord(
                from_intent=None,
                to_intent=to_intent,
                allowed=True,
                unstable=False,
                reason="initial intent",
                evidence=transition_evidence,
            )

        allowed = to_intent in self.ALLOWED_TRANSITIONS.get(from_intent, set())
        support = self._supporting_evidence(to_intent, evidence)
        unstable = (not allowed) or (to_intent in {CampaignIntent.PIVOT.value, CampaignIntent.TRANSFER.value} and not support)
        effect = "penalize" if unstable else "boost"
        reason = (
            "transition has supporting policy evidence"
            if support
            else "transition lacks explicit supporting policy evidence"
        )
        if not allowed:
            reason = "transition is outside the allowed audit matrix"

        transition_evidence = (
            StrategyEvidence(
                source="transition_guard",
                target=f"{from_intent}->{to_intent}",
                effect=effect,
                strength=0.7 if unstable else 0.2,
                reason=reason,
                metadata={
                    "allowed": allowed,
                    "unstable": unstable,
                    "supporting_reasons": [item.reason for item in support[:3]],
                },
            ),
        )
        return ActionTransitionRecord(
            from_intent=from_intent,
            to_intent=to_intent,
            allowed=allowed,
            unstable=unstable,
            reason=reason,
            evidence=transition_evidence,
        )

    @staticmethod
    def _supporting_evidence(
        selected_intent: str,
        evidence: tuple[StrategyEvidence, ...],
    ) -> tuple[StrategyEvidence, ...]:
        return tuple(
            item for item in evidence
            if item.target == selected_intent
            and item.effect in {"boost", "context", "veto"}
        )
