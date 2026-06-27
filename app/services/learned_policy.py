"""Offline-only learned meta-policy scaffolding.

This module never participates in live backend selection. It consumes
PolicyTrainingRecord/PolicyReplayRecord rows and StrategyTrace dictionaries
for offline imitation, reranking, and safety evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import StrEnum
from typing import Any

from app.services.policy_evaluation import PolicyEvaluationRunner
from app.services.strategy_models import (
    LearnedPolicyDeploymentMode,
    LearnedPolicyInfluenceRecord,
    LearnedPolicyPromotionGateResult,
    LearnedPolicyRegistryEntry,
    LearnedPolicyShadowRecord,
    OnlineInfluenceOutcome,
    PolicyInfluenceConfig,
    PolicyReplayRecord,
    PolicyTrainingRecord,
    RankingInfluenceRecord,
    StrategyTrace,
    policy_replay_record_from_trace,
    policy_training_record_from_trace,
)
from app.services.strategy_selector import (
    PhaseConfig,
    select_strategy,
    strategy_trace_to_dict,
)

DATASET_VERSION = "policy_dataset_v1"
FEATURE_SCHEMA_VERSION = "policy_feature_schema_v1"
LEARNED_POLICY_TRACE_VERSION = "learned_policy_trace_v1"
POLICY_BENCHMARK_REPORT_VERSION = "policy_benchmark_report_v1"


class CounterfactualOutcomeLabel(StrEnum):
    """Explicit outcome provenance for offline policy comparisons."""

    OBSERVED_OUTCOME = "observed_outcome"
    REPLAY_OUTCOME = "replay_outcome"
    SYNTHETIC_OUTCOME = "synthetic_outcome"
    UNKNOWN_COUNTERFACTUAL = "unknown_counterfactual"


@dataclass(frozen=True)
class PolicyDataset:
    """Versioned offline policy dataset."""

    records: tuple[dict[str, Any], ...]
    dataset_version: str = DATASET_VERSION
    record_version: str = "policy_training_record_v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    reward_version: str = "strategy_reward_v1"
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LearnedPolicyTrace:
    """Offline-only learned policy suggestion trace."""

    suggested_intent: str | None = None
    suggested_mode: str | None = None
    suggested_backend: str | None = None
    confidence: float = 0.0
    score_deltas: tuple[dict[str, Any], ...] = ()
    top1_changed: bool = False
    reasons: tuple[str, ...] = ()
    features_used: tuple[str, ...] = ()
    trace_version: str = LEARNED_POLICY_TRACE_VERSION


@dataclass(frozen=True)
class PolicyDatasetAudit:
    """Dataset quality and coverage summary for offline policy learning."""

    record_count: int
    missing_feature_rates: dict[str, float]
    objective_level_distribution: dict[str, int]
    intent_distribution: dict[str, int]
    mode_distribution: dict[str, int]
    backend_distribution: dict[str, int]
    reward_coverage: float
    outcome_coverage: float
    failure_type_distribution: dict[str, int]
    safety_flag_distribution: dict[str, int]
    candidate_backend_coverage: dict[str, int]
    candidate_score_coverage: float = 0.0
    candidate_rank_coverage: float = 0.0
    offline_readiness_warnings: tuple[str, ...] = ()
    class_imbalance_warnings: tuple[str, ...] = ()
    audit_version: str = "policy_dataset_audit_v1"


@dataclass(frozen=True)
class RewardSanityReport:
    """Reward attribution sanity checks for offline policy data."""

    passed: bool
    reward_version_distribution: dict[str, int]
    objective_level_reward_distribution: dict[str, dict[str, float]]
    failures: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    report_version: str = "reward_sanity_report_v1"


@dataclass(frozen=True)
class PolicyOfflineCompletenessReport:
    """End-to-end offline readiness report for learned policy experiments."""

    passed: bool
    dataset_version: str
    feature_schema_version: str
    reward_version: str
    record_count: int
    required_sections: dict[str, bool]
    failure_count: int
    warning_count: int
    failures: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    report_version: str = "policy_offline_completeness_v1"


class PolicyDatasetBuilder:
    """Build versioned datasets from canonical policy records."""

    def build(
        self,
        records: list[PolicyTrainingRecord | PolicyReplayRecord | dict[str, Any]]
        | tuple[PolicyTrainingRecord | PolicyReplayRecord | dict[str, Any], ...],
    ) -> PolicyDataset:
        rows = tuple(self._record_row(record) for record in records)
        reward_version = self._reward_version(rows)
        record_version = rows[0].get("record_version", "policy_training_record_v1") if rows else "policy_training_record_v1"
        return PolicyDataset(
            records=rows,
            record_version=str(record_version),
            reward_version=reward_version,
            metadata={"n_records": len(rows)},
        )

    def from_traces(self, traces: list[Any] | tuple[Any, ...]) -> PolicyDataset:
        records = [
            policy_training_record_from_trace(_trace_obj(trace))
            for trace in traces
            if trace is not None and not isinstance(trace, dict)
        ]
        records.extend(
            _training_record_from_trace_dict(trace)
            for trace in traces
            if isinstance(trace, dict)
        )
        return self.build(records)

    @staticmethod
    def _record_row(
        record: PolicyTrainingRecord | PolicyReplayRecord | dict[str, Any],
    ) -> dict[str, Any]:
        if is_dataclass(record):
            row = asdict(record)
        else:
            row = dict(record)
        selected_intent = str(row.get("selected_intent") or "")
        selected_mode = str(row.get("selected_mode") or "")
        return {
            "campaign_id": row.get("campaign_id", ""),
            "loop_id": row.get("loop_id", ""),
            "state_features": dict(row.get("state_features") or {}),
            "context_features": dict(row.get("context_features") or {}),
            "available_actions": _normalize_available_actions(
                row.get("available_actions") or (),
                selected_intent=selected_intent,
                selected_mode=selected_mode,
            ),
            "selected_intent": selected_intent,
            "selected_mode": selected_mode,
            "selected_backend": row.get("selected_backend", ""),
            "candidate_backends": _normalize_candidate_backends(
                row.get("candidate_backends") or ()
            ),
            "applied_influences": list(row.get("applied_influences") or []),
            "reward": row.get("reward"),
            "outcome": _normalize_outcome(row.get("outcome")),
            "safety_flags": list(row.get("safety_flags") or []),
            "record_version": row.get("record_version", "policy_training_record_v1"),
        }

    @staticmethod
    def _reward_version(rows: tuple[dict[str, Any], ...]) -> str:
        for row in rows:
            reward = row.get("reward") or {}
            version = reward.get("reward_version")
            if version:
                return str(version)
        return "strategy_reward_v1"


class PolicyDatasetAuditor:
    """Audit offline policy datasets before learned-policy experiments."""

    FEATURE_KEYS: tuple[str, ...] = (
        "state_features",
        "context_features",
        "available_actions",
        "candidate_backends",
        "reward",
        "outcome",
    )

    def audit(self, dataset: PolicyDataset) -> PolicyDatasetAudit:
        rows = dataset.records
        n = len(rows)
        missing = {
            key: self._missing_rate(rows, key)
            for key in self.FEATURE_KEYS
        }
        objective_levels = Counter(
            str((row.get("context_features") or {}).get("current_objective_level") or "unknown")
            for row in rows
        )
        intents = Counter(str(row.get("selected_intent") or "unknown") for row in rows)
        modes = Counter(str(row.get("selected_mode") or "unknown") for row in rows)
        backends = Counter(str(row.get("selected_backend") or "unknown") for row in rows)
        failures = Counter(
            failure_type
            for row in rows
            for failure_type in _failure_types_from_row(row)
        )
        safety = Counter(
            str(flag)
            for row in rows
            for flag in row.get("safety_flags") or []
        )
        candidate_coverage = Counter(
            str(candidate.get("name") or "unknown")
            for row in rows
            for candidate in row.get("candidate_backends") or []
        )
        candidate_count = sum(len(row.get("candidate_backends") or []) for row in rows)
        candidate_score_count = sum(
            1
            for row in rows
            for candidate in row.get("candidate_backends") or []
            if candidate.get("total") is not None
        )
        candidate_rank_count = sum(
            1
            for row in rows
            for candidate in row.get("candidate_backends") or []
            if candidate.get("rank") is not None
        )
        readiness_warnings = self._offline_readiness_warnings(
            rows,
            missing,
            candidate_count=candidate_count,
            candidate_score_count=candidate_score_count,
            candidate_rank_count=candidate_rank_count,
        )
        return PolicyDatasetAudit(
            record_count=n,
            missing_feature_rates=missing,
            objective_level_distribution=dict(objective_levels),
            intent_distribution=dict(intents),
            mode_distribution=dict(modes),
            backend_distribution=dict(backends),
            reward_coverage=_coverage(rows, "reward"),
            outcome_coverage=_coverage(rows, "outcome"),
            failure_type_distribution=dict(failures),
            safety_flag_distribution=dict(safety),
            candidate_backend_coverage=dict(candidate_coverage),
            candidate_score_coverage=round(candidate_score_count / candidate_count, 4) if candidate_count else 0.0,
            candidate_rank_coverage=round(candidate_rank_count / candidate_count, 4) if candidate_count else 0.0,
            offline_readiness_warnings=tuple(readiness_warnings),
            class_imbalance_warnings=tuple(
                self._imbalance_warnings({
                    "intent": intents,
                    "mode": modes,
                    "backend": backends,
                }, n)
            ),
        )

    @staticmethod
    def _missing_rate(rows: tuple[dict[str, Any], ...], key: str) -> float:
        if not rows:
            return 0.0
        missing = sum(1 for row in rows if not row.get(key))
        return round(missing / len(rows), 4)

    @staticmethod
    def _imbalance_warnings(counters: dict[str, Counter[str]], n: int) -> list[str]:
        warnings: list[str] = []
        if n == 0:
            return warnings
        for label, counter in counters.items():
            if len(counter) <= 1 and n > 1:
                warnings.append(f"{label} has a single observed class")
                continue
            top, count = counter.most_common(1)[0] if counter else ("", 0)
            if count / n >= 0.8:
                warnings.append(f"{label} class '{top}' dominates {count}/{n} records")
        return warnings

    @staticmethod
    def _offline_readiness_warnings(
        rows: tuple[dict[str, Any], ...],
        missing: dict[str, float],
        *,
        candidate_count: int,
        candidate_score_count: int,
        candidate_rank_count: int,
    ) -> list[str]:
        warnings: list[str] = []
        if not rows:
            warnings.append("dataset has no records")
            return warnings
        for key in ("state_features", "context_features", "available_actions", "candidate_backends"):
            if missing.get(key, 0.0) > 0:
                warnings.append(f"{key} missing in {missing[key]:.0%} of records")
        if candidate_count == 0:
            warnings.append("candidate backend ranking is unavailable")
        elif candidate_score_count < candidate_count:
            warnings.append("candidate backend scores are incomplete")
        if candidate_count and candidate_rank_count < candidate_count:
            warnings.append("candidate backend ranks are incomplete")
        if _coverage(rows, "reward") < 1.0:
            warnings.append("reward coverage is incomplete")
        if _coverage(rows, "outcome") < 1.0:
            warnings.append("outcome coverage is incomplete")
        return warnings


class PolicyOfflineCompletenessChecker:
    """Validate that offline learned-policy evaluation can run end to end."""

    def check(
        self,
        dataset: PolicyDataset,
        *,
        audit: PolicyDatasetAudit | None = None,
        reward_sanity: RewardSanityReport | None = None,
        learned_safety: dict[str, Any] | None = None,
        benchmark_report: dict[str, Any] | None = None,
    ) -> PolicyOfflineCompletenessReport:
        audit = audit or PolicyDatasetAuditor().audit(dataset)
        reward_sanity = reward_sanity or RewardSanityChecker().check(dataset)
        learned_safety = learned_safety or {"passed": True, "failures": ()}
        benchmark_report = benchmark_report or {}
        failures: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        required_sections = {
            "dataset_records": len(dataset.records) > 0,
            "state_context_features": (
                audit.missing_feature_rates.get("state_features", 1.0) == 0.0
                and audit.missing_feature_rates.get("context_features", 1.0) == 0.0
            ),
            "available_actions": audit.missing_feature_rates.get("available_actions", 1.0) == 0.0,
            "candidate_backend_rankings": (
                audit.missing_feature_rates.get("candidate_backends", 1.0) == 0.0
                and audit.candidate_score_coverage == 1.0
                and audit.candidate_rank_coverage == 1.0
            ),
            "reward_sanity": reward_sanity.passed,
            "learned_safety": bool(learned_safety.get("passed", False)),
            "benchmark_variants": _benchmark_has_required_variants(benchmark_report),
            "counterfactual_labels": all(
                (row.get("outcome") or {}).get("counterfactual_label") in {item.value for item in CounterfactualOutcomeLabel}
                for row in dataset.records
            ),
        }
        for section, passed in required_sections.items():
            if not passed:
                failures.append({"check": section, "reason": "offline requirement not satisfied"})
        warnings.extend({"check": "dataset_audit", "reason": warning} for warning in audit.offline_readiness_warnings)
        warnings.extend(reward_sanity.warnings)
        return PolicyOfflineCompletenessReport(
            passed=not failures,
            dataset_version=dataset.dataset_version,
            feature_schema_version=dataset.feature_schema_version,
            reward_version=dataset.reward_version,
            record_count=len(dataset.records),
            required_sections=required_sections,
            failure_count=len(failures),
            warning_count=len(warnings),
            failures=tuple(failures),
            warnings=tuple(warnings),
        )


class RewardSanityChecker:
    """Check reward coverage and failure attribution before offline learning."""

    REQUIRED_REWARD_FIELDS: tuple[str, ...] = (
        "objective_improvement",
        "information_gain",
        "constraint_satisfaction",
        "data_quality_gain",
        "novelty",
        "failure_penalty",
        "cost_penalty",
        "time_penalty",
        "composite_reward",
        "reward_version",
    )

    def check(self, dataset: PolicyDataset) -> RewardSanityReport:
        failures: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        versions: Counter[str] = Counter()
        objective_rewards: dict[str, list[float]] = defaultdict(list)
        for idx, row in enumerate(dataset.records):
            reward = row.get("reward") or {}
            if not reward:
                failures.append({"index": idx, "check": "missing_reward"})
                continue
            missing = [field for field in self.REQUIRED_REWARD_FIELDS if field not in reward]
            if missing:
                failures.append({
                    "index": idx,
                    "check": "missing_reward_fields",
                    "fields": missing,
                })
            version = str(reward.get("reward_version") or "missing")
            versions[version] += 1
            if version != dataset.reward_version:
                failures.append({
                    "index": idx,
                    "check": "reward_version_inconsistent",
                    "reward_version": version,
                    "dataset_reward_version": dataset.reward_version,
                })
            level = str((row.get("context_features") or {}).get("current_objective_level") or "unknown")
            objective_rewards[level].append(float(reward.get("composite_reward") or 0.0))
            failure_types = set(_failure_types_from_row(row))
            failure_penalty = float(reward.get("failure_penalty") or 0.0)
            if "scientific_negative" in failure_types and failure_penalty > 0:
                failures.append({
                    "index": idx,
                    "check": "scientific_negative_penalized_as_failure",
                    "failure_penalty": failure_penalty,
                })
            backend = str(row.get("selected_backend") or "")
            if failure_types & {"hardware", "measurement"} and backend and failure_penalty > 0:
                failures.append({
                    "index": idx,
                    "check": "execution_failure_contaminates_backend_reward",
                    "failure_types": sorted(failure_types),
                    "backend": backend,
                    "failure_penalty": failure_penalty,
                })
            if row.get("outcome") and row["outcome"].get("counterfactual_label") == CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value:
                warnings.append({
                    "index": idx,
                    "check": "unknown_counterfactual_reward_not_observed",
                })
        return RewardSanityReport(
            passed=not failures,
            reward_version_distribution=dict(versions),
            objective_level_reward_distribution={
                level: _reward_stats(values)
                for level, values in objective_rewards.items()
            },
            failures=tuple(failures),
            warnings=tuple(warnings),
        )


class ImitationPolicy:
    """Simple offline majority/count baseline for selector imitation."""

    def __init__(self) -> None:
        self.intent_counts: Counter[str] = Counter()
        self.mode_counts: Counter[str] = Counter()
        self.backend_counts: Counter[str] = Counter()
        self.backend_by_context: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, dataset: PolicyDataset) -> ImitationPolicy:
        for row in dataset.records:
            intent = str(row.get("selected_intent") or "")
            mode = str(row.get("selected_mode") or "")
            backend = str(row.get("selected_backend") or "")
            if intent:
                self.intent_counts[intent] += 1
            if mode:
                self.mode_counts[mode] += 1
            if backend:
                self.backend_counts[backend] += 1
                self.backend_by_context[_context_bucket(row)][backend] += 1
        return self

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        bucket = _context_bucket(row)
        backend_counter = self.backend_by_context.get(bucket) or self.backend_counts
        backend, backend_conf = _most_common_with_confidence(backend_counter)
        intent, intent_conf = _most_common_with_confidence(self.intent_counts)
        mode, mode_conf = _most_common_with_confidence(self.mode_counts)
        return {
            "intent": intent,
            "mode": mode,
            "backend": backend,
            "confidence": round((intent_conf + mode_conf + backend_conf) / 3, 4),
            "reason": f"majority imitation for context bucket {bucket}",
        }

    def evaluate(self, dataset: PolicyDataset) -> dict[str, Any]:
        n = len(dataset.records)
        if n == 0:
            return {
                "n_records": 0,
                "intent_accuracy": 0.0,
                "mode_accuracy": 0.0,
                "backend_top1_accuracy": 0.0,
            }
        intent_hits = 0
        mode_hits = 0
        backend_hits = 0
        for row in dataset.records:
            pred = self.predict(row)
            intent_hits += int(pred["intent"] == row.get("selected_intent"))
            mode_hits += int(pred["mode"] == row.get("selected_mode"))
            backend_hits += int(pred["backend"] == row.get("selected_backend"))
        return {
            "n_records": n,
            "intent_accuracy": round(intent_hits / n, 4),
            "mode_accuracy": round(mode_hits / n, 4),
            "backend_top1_accuracy": round(backend_hits / n, 4),
        }


class LearnedBackendReranker:
    """Offline-only learned backend reranker scaffold."""

    def __init__(self, *, max_delta: float = 0.02) -> None:
        self.max_delta = max_delta
        self.backend_rewards: dict[str, float] = {}

    def fit(self, dataset: PolicyDataset) -> LearnedBackendReranker:
        totals: dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for row in dataset.records:
            backend = str(row.get("selected_backend") or "")
            if not backend:
                continue
            reward = float((row.get("reward") or {}).get("composite_reward") or 0.0)
            totals[backend] += reward
            counts[backend] += 1
        self.backend_rewards = {
            backend: totals[backend] / counts[backend]
            for backend in counts
        }
        return self

    def score_deltas(
        self,
        row: dict[str, Any],
        *,
        cap: float | None = None,
    ) -> tuple[dict[str, Any], ...]:
        limit = self.max_delta if cap is None else min(abs(cap), self.max_delta)
        if not row.get("candidate_backends"):
            return ()
        best_mean = max(self.backend_rewards.values(), default=0.0)
        records: list[dict[str, Any]] = []
        for candidate in row.get("candidate_backends") or []:
            backend = str(candidate.get("name") or "")
            raw = self.backend_rewards.get(backend, 0.0) - best_mean
            bounded = max(-limit, min(limit, raw * limit))
            records.append({
                "source": "learned_backend_reranker",
                "target": backend,
                "raw_signal": round(raw, 4),
                "score_delta": round(bounded, 4),
                "cap": limit,
                "capped": abs(bounded - raw * limit) > 1e-12,
                "reason": "offline learned reward mean delta; not applied to live ranking",
            })
        return tuple(records)

    def trace_for(self, row: dict[str, Any], *, cap: float | None = None) -> LearnedPolicyTrace:
        deltas = self.score_deltas(row, cap=cap)
        base_top = _top_backend(row.get("candidate_backends") or ())
        reranked_top = _reranked_top(row.get("candidate_backends") or (), deltas)
        return LearnedPolicyTrace(
            suggested_backend=reranked_top,
            confidence=_delta_confidence(deltas),
            score_deltas=deltas,
            top1_changed=bool(base_top and reranked_top and base_top != reranked_top),
            reasons=("offline learned reranker suggestion only",),
            features_used=("candidate_backends", "context_features", "reward"),
        )


@dataclass(frozen=True)
class SimulationStep:
    """One offline simulation transition."""

    observation: dict[str, Any]
    reward: float | None
    done: bool
    info: dict[str, Any]


class PolicySimulationEnvironment:
    """Replay/simulation-only environment over canonical policy records."""

    def __init__(self, records: PolicyDataset | tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
        if isinstance(records, PolicyDataset):
            self.dataset = records
        else:
            self.dataset = PolicyDatasetBuilder().build(tuple(records))
        self._idx = 0

    def reset(self) -> dict[str, Any]:
        self._idx = 0
        return self._observation()

    def step(self, action: dict[str, Any] | None = None) -> SimulationStep:
        row = self._current_row()
        proposed = action or {}
        safety = self._validate_action(row, proposed)
        reward_info = RewardModel().compute(row)
        info = {
            "safety": safety,
            "available_actions": row.get("available_actions") or [],
            "available_backends": _candidate_names(row),
            "counterfactual_label": reward_info["counterfactual_label"],
            "ground_truth": reward_info["ground_truth"],
            "reward_version": reward_info["reward_version"],
            "space_revision_auto_applied": False,
        }
        if not safety["valid"]:
            info["safety_violation"] = True
        self._idx += 1
        done = self._idx >= len(self.dataset.records)
        return SimulationStep(
            observation=self._observation() if not done else {},
            reward=reward_info["reward"],
            done=done,
            info=info,
        )

    def action_space(self) -> dict[str, tuple[str, ...]]:
        intents: set[str] = set()
        modes: set[str] = set()
        backends: set[str] = set()
        for row in self.dataset.records:
            intents.update(str(action.get("intent")) for action in row.get("available_actions") or [] if action.get("intent"))
            modes.update(str(action.get("mode")) for action in row.get("available_actions") or [] if action.get("mode"))
            backends.update(_candidate_names(row))
        intents.update(str(row.get("selected_intent")) for row in self.dataset.records if row.get("selected_intent"))
        modes.update(str(row.get("selected_mode")) for row in self.dataset.records if row.get("selected_mode"))
        return {
            "campaign_intents": tuple(sorted(intents)),
            "optimization_modes": tuple(sorted(modes)),
            "backends": tuple(sorted(backends)),
            "backend_score_delta": ("bounded_delta",),
        }

    def safety_mask(self) -> dict[str, tuple[str, ...]]:
        row = self._current_row()
        return {
            "allowed_backends": tuple(sorted(_candidate_names(row))),
            "allowed_actions": tuple(str(action.get("name")) for action in row.get("available_actions") or [] if action.get("name")),
            "allowed_intents": tuple(sorted(_allowed_intents(row))),
            "allowed_modes": tuple(sorted(_allowed_modes(row))),
            "blocked_operations": (
                "add_backend",
                "hard_veto",
                "auto_apply_space_revision",
                "live_selector_influence",
            ),
        }

    def _observation(self) -> dict[str, Any]:
        row = self._current_row()
        return {
            "state_features": dict(row.get("state_features") or {}),
            "context_features": dict(row.get("context_features") or {}),
            "available_actions": list(row.get("available_actions") or []),
            "candidate_backends": list(row.get("candidate_backends") or []),
            "safety_mask": self.safety_mask() if self.dataset.records else {},
            "counterfactual_label": (row.get("outcome") or {}).get(
                "counterfactual_label",
                CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value,
            ),
        }

    def _current_row(self) -> dict[str, Any]:
        if not self.dataset.records:
            return {}
        return self.dataset.records[min(self._idx, len(self.dataset.records) - 1)]

    @staticmethod
    def _validate_action(row: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        violations: list[str] = []
        backend = action.get("backend") or action.get("suggested_backend")
        if backend and str(backend) not in _candidate_names(row):
            violations.append("backend_not_available")
        intent = action.get("intent") or action.get("suggested_intent")
        if intent and str(intent) not in _allowed_intents(row):
            violations.append("intent_not_available")
        mode = action.get("mode") or action.get("suggested_mode")
        if mode and str(mode) not in _allowed_modes(row):
            violations.append("mode_not_available")
        if action.get("hard_veto"):
            violations.append("hard_veto_not_allowed")
        if action.get("auto_apply_space_revision"):
            violations.append("space_revision_auto_apply_not_allowed")
        for delta in action.get("score_deltas") or []:
            target = str(delta.get("target") or "")
            if target not in _candidate_names(row):
                violations.append("score_delta_target_not_available")
        return {
            "valid": not violations,
            "violations": tuple(violations),
        }


class OutcomeModel:
    """Outcome label handling for offline reward use."""

    SUPPORTED_REWARD_LABELS = {
        CounterfactualOutcomeLabel.OBSERVED_OUTCOME.value,
        CounterfactualOutcomeLabel.REPLAY_OUTCOME.value,
        CounterfactualOutcomeLabel.SYNTHETIC_OUTCOME.value,
    }

    def classify(self, row: dict[str, Any]) -> dict[str, Any]:
        outcome = _normalize_outcome(row.get("outcome")) or {}
        label = outcome.get("counterfactual_label", CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value)
        return {
            "counterfactual_label": label,
            "reward_supported": label in self.SUPPORTED_REWARD_LABELS,
            "ground_truth": label in {
                CounterfactualOutcomeLabel.OBSERVED_OUTCOME.value,
                CounterfactualOutcomeLabel.REPLAY_OUTCOME.value,
            },
            "reason": (
                "reward can be used for offline simulation"
                if label in self.SUPPORTED_REWARD_LABELS
                else "unknown counterfactual outcome is not ground truth"
            ),
        }


class RewardModel:
    """Reward access that refuses unknown counterfactual ground truth."""

    def compute(self, row: dict[str, Any]) -> dict[str, Any]:
        classification = OutcomeModel().classify(row)
        reward = row.get("reward") or {}
        version = str(reward.get("reward_version") or "strategy_reward_v1")
        if not classification["reward_supported"]:
            return {
                **classification,
                "reward": None,
                "reward_version": version,
            }
        return {
            **classification,
            "reward": float(reward.get("composite_reward") or 0.0),
            "reward_version": version,
        }


class LearnedMetaPolicy:
    """Offline/simulation-only learned meta-policy scaffold."""

    def __init__(self, *, max_delta: float = 0.01, imitation_pretrained: bool = True) -> None:
        self.max_delta = max_delta
        self.imitation_pretrained = imitation_pretrained
        self.imitation = ImitationPolicy()
        self.reranker = LearnedBackendReranker(max_delta=max_delta)
        self._fitted = False

    def fit_imitation(self, dataset: PolicyDataset) -> LearnedMetaPolicy:
        self.imitation.fit(dataset)
        self.reranker.fit(dataset)
        self._fitted = True
        return self

    def predict(self, observation: dict[str, Any]) -> LearnedPolicyTrace:
        row = _row_from_observation(observation)
        pred = self.imitation.predict(row) if self._fitted else {"intent": "", "mode": "", "backend": "", "confidence": 0.0}
        candidate_names = _candidate_names(row)
        backend = str(pred.get("backend") or "")
        if backend not in candidate_names:
            backend = _top_backend(row.get("candidate_backends") or ())
        deltas = tuple(
            delta for delta in self.reranker.score_deltas(row, cap=self.max_delta)
            if str(delta.get("target") or "") in candidate_names
        )
        return LearnedPolicyTrace(
            suggested_intent=str(pred.get("intent") or "") or None,
            suggested_mode=str(pred.get("mode") or "") or None,
            suggested_backend=backend or None,
            confidence=float(pred.get("confidence") or 0.0),
            score_deltas=deltas,
            top1_changed=bool(backend and backend != _top_backend(row.get("candidate_backends") or ())),
            reasons=("offline simulation meta-policy suggestion only",),
            features_used=("state_features", "context_features", "available_actions", "candidate_backends"),
        )


class OfflineMetaPolicyTrainer:
    """Train learned meta-policies only on replay/simulation records."""

    def train_imitation(self, dataset: PolicyDataset) -> dict[str, Any]:
        policy = LearnedMetaPolicy().fit_imitation(dataset)
        evaluation = policy.imitation.evaluate(dataset)
        return {
            "policy": policy,
            "training_mode": "supervised_imitation",
            "n_records": len(dataset.records),
            "evaluation": evaluation,
            "online_enabled": False,
        }

    def train_policy_gradient_style(
        self,
        environment: PolicySimulationEnvironment,
        *,
        episodes: int = 1,
    ) -> dict[str, Any]:
        """Simulation-only placeholder for future policy-gradient experiments."""
        rewards: list[float] = []
        for _ in range(max(episodes, 0)):
            obs = environment.reset()
            done = False
            while not done:
                step = environment.step({"backend": _top_backend(obs.get("candidate_backends") or [])})
                if step.reward is not None:
                    rewards.append(step.reward)
                obs = step.observation
                done = step.done
        return {
            "training_mode": "simulation_policy_gradient_placeholder",
            "episodes": episodes,
            "mean_supported_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "online_enabled": False,
        }


class OfflineMetaPolicyEvaluator:
    """Evaluate trained meta-policy outputs in replay/simulation only."""

    def __init__(self, *, learned_delta_cap: float = 0.01) -> None:
        self.learned_delta_cap = learned_delta_cap

    def evaluate_snapshots(self, snapshots: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        offline_report = OfflinePolicyEvaluator(
            learned_delta_cap=self.learned_delta_cap
        ).evaluate_snapshots(snapshots)
        dataset = PolicyDatasetBuilder().build(
            replay_records_from_traces(offline_report["traces_by_variant"]["baseline"])
        )
        trained = OfflineMetaPolicyTrainer().train_imitation(dataset)
        policy: LearnedMetaPolicy = trained["policy"]
        env = PolicySimulationEnvironment(dataset)
        traces: list[LearnedPolicyTrace] = []
        reward_results: list[dict[str, Any]] = []
        obs = env.reset()
        done = len(dataset.records) == 0
        while not done:
            trace = policy.predict(obs)
            traces.append(trace)
            step = env.step({
                "backend": trace.suggested_backend,
                "score_deltas": trace.score_deltas,
            })
            reward_results.append({
                "reward": step.reward,
                "counterfactual_label": step.info["counterfactual_label"],
                "ground_truth": step.info.get("ground_truth", False),
                "safety": step.info["safety"],
            })
            obs = step.observation
            done = step.done
        safety = OfflinePolicyEvaluator(
            learned_delta_cap=self.learned_delta_cap
        )._safety_check(tuple(traces), dataset)
        return {
            **offline_report,
            "trained_meta_policy": {
                "training_mode": trained["training_mode"],
                "imitation_evaluation": trained["evaluation"],
                "online_enabled": False,
            },
            "trained_meta_policy_summary": self._summary(dataset, traces, reward_results, safety),
            "trained_meta_policy_traces": tuple(asdict(trace) for trace in traces),
            "counterfactual_uncertainty_breakdown": dict(Counter(
                result["counterfactual_label"] for result in reward_results
            )),
        }

    def _summary(
        self,
        dataset: PolicyDataset,
        traces: list[LearnedPolicyTrace],
        reward_results: list[dict[str, Any]],
        safety: dict[str, Any],
    ) -> dict[str, Any]:
        n = len(traces)
        supported_rewards = [
            float(result["reward"])
            for result in reward_results
            if result["reward"] is not None
        ]
        return {
            "n_records": n,
            "reward_delta": 0.0,
            "mean_supported_reward": (
                round(sum(supported_rewards) / len(supported_rewards), 4)
                if supported_rewards else None
            ),
            "safety_violations": safety.get("failure_count", 0),
            "unstable_transitions": _dataset_safety_metrics(dataset).get("unstable_transition_count", 0),
            "top1_change_rate": (
                sum(1 for trace in traces if trace.top1_changed) / n if n else 0.0
            ),
            "calibration": {"method": "offline_meta_policy_imitation"},
            "counterfactual_uncertainty_breakdown": dict(Counter(
                result["counterfactual_label"] for result in reward_results
            )),
        }


class LearnedPolicyPromotionGate:
    """Conservative eligibility gate for learned-policy tiny influence."""

    def __init__(
        self,
        *,
        min_shadow_rounds: int = 10,
        max_safety_warning_rate: float = 0.05,
        max_invalid_suggestion_rate: float = 0.05,
        min_confidence_calibration: float = 0.6,
        min_top_k_agreement: float = 0.7,
        require_offline_benchmark_pass: bool = True,
        require_reward_sanity_pass: bool = True,
    ) -> None:
        self.min_shadow_rounds = min_shadow_rounds
        self.max_safety_warning_rate = max_safety_warning_rate
        self.max_invalid_suggestion_rate = max_invalid_suggestion_rate
        self.min_confidence_calibration = min_confidence_calibration
        self.min_top_k_agreement = min_top_k_agreement
        self.require_offline_benchmark_pass = require_offline_benchmark_pass
        self.require_reward_sanity_pass = require_reward_sanity_pass

    def evaluate(
        self,
        *,
        registry_entry: LearnedPolicyRegistryEntry | None,
        shadow_summary: dict[str, Any] | None = None,
    ) -> LearnedPolicyPromotionGateResult:
        summary = dict(shadow_summary or {})
        evaluation = dict(registry_entry.evaluation_summary if registry_entry else {})
        shadow_rounds = int(
            summary["n_records"]
            if "n_records" in summary
            else evaluation.get("shadow_rounds", 0)
        )
        safety_warning_rate = _bounded_rate(
            summary.get("safety_warning_rate"),
            numerator=summary.get("safety_warning_count"),
            denominator=shadow_rounds,
        )
        invalid_suggestion_rate = _bounded_rate(
            summary.get("invalid_suggestion_rate"),
            numerator=sum((summary.get("invalid_suggestion_distribution") or {}).values()),
            denominator=shadow_rounds,
        )
        confidence_calibration = float(
            evaluation.get(
                "confidence_calibration",
                summary.get("confidence_calibration", 0.0),
            )
            or 0.0
        )
        top_k_agreement = float(
            evaluation.get(
                "top_k_agreement",
                summary.get("top_k_agreement", summary.get("backend_agreement_rate", 0.0)),
            )
            or 0.0
        )
        offline_passed = bool(evaluation.get("offline_benchmark_pass", False))
        reward_passed = bool(evaluation.get("reward_sanity_pass", False))

        reasons: list[str] = []
        if registry_entry is None:
            reasons.append("missing_policy_registry_entry")
        elif not registry_entry.approved_for_shadow:
            reasons.append("policy_not_approved_for_shadow")
        if registry_entry is not None and not registry_entry.approved_for_safe_soft:
            reasons.append("policy_not_approved_for_safe_soft")
        if shadow_rounds < self.min_shadow_rounds:
            reasons.append("insufficient_shadow_rounds")
        if safety_warning_rate > self.max_safety_warning_rate:
            reasons.append("safety_warning_rate_too_high")
        if invalid_suggestion_rate > self.max_invalid_suggestion_rate:
            reasons.append("invalid_suggestion_rate_too_high")
        if confidence_calibration < self.min_confidence_calibration:
            reasons.append("confidence_calibration_too_low")
        if top_k_agreement < self.min_top_k_agreement:
            reasons.append("top_k_agreement_too_low")
        if self.require_offline_benchmark_pass and not offline_passed:
            reasons.append("offline_benchmark_not_passed")
        if self.require_reward_sanity_pass and not reward_passed:
            reasons.append("reward_sanity_not_passed")

        return LearnedPolicyPromotionGateResult(
            eligible=not reasons,
            reasons=tuple(reasons),
            shadow_rounds=shadow_rounds,
            safety_warning_rate=round(safety_warning_rate, 4),
            invalid_suggestion_rate=round(invalid_suggestion_rate, 4),
            confidence_calibration=round(confidence_calibration, 4),
            top_k_agreement=round(top_k_agreement, 4),
            offline_benchmark_passed=offline_passed,
            reward_sanity_passed=reward_passed,
        )


class LearnedPolicyShadowRunner:
    """Run learned policies in OFF, SHADOW, or gated SAFE_SOFT mode."""

    def __init__(
        self,
        *,
        registry_entry: LearnedPolicyRegistryEntry | None = None,
        policy: LearnedMetaPolicy | None = None,
        mode: LearnedPolicyDeploymentMode | str = LearnedPolicyDeploymentMode.OFF,
        promotion_gate: LearnedPolicyPromotionGate | None = None,
        shadow_summary: dict[str, Any] | None = None,
        max_safe_soft_delta: float = 0.005,
    ) -> None:
        self.registry_entry = registry_entry
        self.policy = policy
        self.mode = str(getattr(mode, "value", mode))
        self.promotion_gate = promotion_gate or LearnedPolicyPromotionGate()
        self.shadow_summary = dict(shadow_summary or {})
        self.max_safe_soft_delta = abs(float(max_safe_soft_delta))

    def run(self, trace: StrategyTrace) -> StrategyTrace:
        record = self.shadow_record(trace)
        if record is None:
            return trace
        influence = self.influence_record(trace, record)
        if influence is None:
            return replace(trace, learned_policy_shadow=record)
        online_outcome = _attach_learned_influence_to_online_outcome(
            trace.online_influence_outcome,
            influence,
        )
        return replace(
            trace,
            learned_policy_shadow=record,
            learned_policy_influence=influence,
            online_influence_outcome=online_outcome,
        )

    def shadow_record(self, trace: StrategyTrace) -> LearnedPolicyShadowRecord | None:
        if self.mode == LearnedPolicyDeploymentMode.OFF.value:
            return None
        if self.mode == LearnedPolicyDeploymentMode.SAFE_SOFT.value:
            return self._suggestion_record(
                trace,
                deployment_mode=LearnedPolicyDeploymentMode.SAFE_SOFT,
                reason="safe-soft learned policy suggestion; influence requires promotion gate",
            )
        if self.mode != LearnedPolicyDeploymentMode.SHADOW.value:
            return self._blocked_record(
                trace,
                ("unsupported_learned_policy_deployment_mode",),
                "Unsupported learned policy deployment mode",
            )
        return self._suggestion_record(
            trace,
            deployment_mode=LearnedPolicyDeploymentMode.SHADOW,
            reason="shadow-only learned policy record; no live ranking influence",
        )

    def influence_record(
        self,
        trace: StrategyTrace,
        shadow_record: LearnedPolicyShadowRecord | None = None,
    ) -> LearnedPolicyInfluenceRecord | None:
        if self.mode != LearnedPolicyDeploymentMode.SAFE_SOFT.value:
            return None
        record = shadow_record or self.shadow_record(trace)
        gate = self.promotion_gate.evaluate(
            registry_entry=self.registry_entry,
            shadow_summary=self.shadow_summary,
        )
        if record is None:
            return self._blocked_influence_record(
                trace,
                gate,
                ("missing_learned_policy_suggestion",),
                "No learned policy suggestion available",
            )
        warnings = list(record.safety_warnings)
        if not gate.eligible:
            return self._blocked_influence_record(
                trace,
                gate,
                tuple((*warnings, *gate.reasons)),
                "Promotion gate blocked learned safe-soft influence",
                record=record,
            )
        if not record.safety_mask_valid:
            return self._blocked_influence_record(
                trace,
                gate,
                tuple((*warnings, "safety_mask_invalid")),
                "Safety mask blocked learned safe-soft influence",
                record=record,
            )
        raw_target, raw_delta = _largest_learned_delta(record.score_deltas)
        if not raw_target or raw_target not in _trace_candidate_names(trace):
            return self._blocked_influence_record(
                trace,
                gate,
                tuple((*warnings, "unavailable_backend_suggestion")),
                "Learned policy did not produce an available backend delta",
                record=record,
            )
        capped = abs(raw_delta) > self.max_safe_soft_delta
        applied_delta = max(-self.max_safe_soft_delta, min(self.max_safe_soft_delta, raw_delta))
        base_top = _top_backend(trace.candidate_backends)
        influenced_top = _reranked_top(
            trace.candidate_backends,
            ({"target": raw_target, "score_delta": applied_delta},),
        )
        return LearnedPolicyInfluenceRecord(
            policy_id=self.registry_entry.policy_id if self.registry_entry else "",
            policy_version=self.registry_entry.policy_version if self.registry_entry else "",
            eligibility=gate,
            suggested_backend=record.suggested_backend,
            target_backend=raw_target,
            raw_delta=round(float(raw_delta), 6),
            applied_delta=round(float(applied_delta), 6),
            capped=capped,
            confidence=record.confidence,
            would_change_top1=record.would_change_top1,
            changed_top1=bool(base_top and influenced_top and base_top != influenced_top),
            safety_mask_valid=True,
            safety_warnings=tuple(warnings),
            reason="learned safe-soft tiny bounded score delta; not passed into live rank_backends",
        )

    def ranking_influence_record(
        self,
        influence: LearnedPolicyInfluenceRecord | None,
        *,
        source: str = "learned_policy_safe_soft",
    ) -> RankingInfluenceRecord | None:
        if influence is None or not influence.eligibility.eligible or not influence.target_backend:
            return None
        return RankingInfluenceRecord(
            source=source,
            target=influence.target_backend,
            raw_signal=influence.raw_delta,
            applied_weight=abs(influence.applied_delta),
            score_delta=influence.applied_delta,
            capped=influence.capped,
            reason=influence.reason,
        )

    def _suggestion_record(
        self,
        trace: StrategyTrace,
        *,
        deployment_mode: LearnedPolicyDeploymentMode,
        reason: str,
    ) -> LearnedPolicyShadowRecord:
        if self.registry_entry is None or self.policy is None:
            return self._blocked_record(
                trace,
                ("missing_policy_or_registry_entry",),
                "No approved learned policy available for shadow run",
            )
        if not self.registry_entry.approved_for_shadow:
            return self._blocked_record(
                trace,
                ("policy_not_approved_for_shadow",),
                "Policy registry entry is not approved for shadow",
            )
        observation = _observation_from_trace(trace)
        suggestion = self.policy.predict(observation)
        candidate_names = {
            str(candidate.get("name"))
            for candidate in trace.candidate_backends
            if candidate.get("name")
        }
        invalid: list[str] = []
        if suggestion.suggested_backend and suggestion.suggested_backend not in candidate_names:
            invalid.append("suggested_backend_unavailable")
            invalid.append("backend_addition_attempt")
        for reason_name in (
            ("hard_veto", "hard_veto_attempt"),
            ("auto_apply_space_revision", "space_revision_auto_apply_attempt"),
            ("objective_override", "objective_override_attempt"),
            ("action_override", "action_override_attempt"),
        ):
            if bool(getattr(suggestion, reason_name[0], False)) or bool(
                getattr(self.policy, reason_name[0], False)
            ):
                invalid.append(reason_name[1])
        safe_deltas: list[dict[str, Any]] = []
        for delta in suggestion.score_deltas:
            target = str(delta.get("target") or "")
            if target not in candidate_names:
                invalid.append("score_delta_target_unavailable")
                continue
            safe_deltas.append(dict(delta))
        base_top = _top_backend(tuple(dict(row) for row in trace.candidate_backends))
        would_top = _reranked_top(tuple(dict(row) for row in trace.candidate_backends), tuple(safe_deltas))
        suggested_backend = (
            suggestion.suggested_backend
            if suggestion.suggested_backend in candidate_names
            else None
        )
        return LearnedPolicyShadowRecord(
            policy_id=self.registry_entry.policy_id,
            policy_version=self.registry_entry.policy_version,
            deployment_mode=deployment_mode,
            suggested_intent=suggestion.suggested_intent,
            suggested_mode=suggestion.suggested_mode,
            suggested_backend=suggested_backend,
            score_deltas=tuple(safe_deltas),
            confidence=suggestion.confidence,
            safety_mask_valid=not invalid,
            invalid_suggestion_reasons=tuple(invalid),
            safety_warnings=tuple(invalid),
            actual_intent=str(getattr(trace.selected_intent, "value", trace.selected_intent)),
            actual_mode=str(getattr(trace.selected_mode, "value", trace.selected_mode)),
            actual_backend=trace.selected_backend,
            intent_agrees=suggestion.suggested_intent == str(getattr(trace.selected_intent, "value", trace.selected_intent)),
            mode_agrees=suggestion.suggested_mode == str(getattr(trace.selected_mode, "value", trace.selected_mode)),
            backend_agrees=suggested_backend == trace.selected_backend,
            would_change_top1=bool(base_top and would_top and base_top != would_top),
            counterfactual_label=_trace_counterfactual_label(trace),
            reason=reason,
        )

    def _blocked_influence_record(
        self,
        trace: StrategyTrace,
        eligibility: LearnedPolicyPromotionGateResult,
        warnings: tuple[str, ...],
        reason: str,
        *,
        record: LearnedPolicyShadowRecord | None = None,
    ) -> LearnedPolicyInfluenceRecord:
        return LearnedPolicyInfluenceRecord(
            policy_id=self.registry_entry.policy_id if self.registry_entry else "",
            policy_version=self.registry_entry.policy_version if self.registry_entry else "",
            eligibility=eligibility,
            suggested_backend=record.suggested_backend if record else None,
            target_backend=None,
            raw_delta=0.0,
            applied_delta=0.0,
            capped=False,
            confidence=record.confidence if record else 0.0,
            would_change_top1=record.would_change_top1 if record else False,
            changed_top1=False,
            safety_mask_valid=False,
            safety_warnings=tuple(dict.fromkeys(warnings)),
            reason=reason,
        )

    def _blocked_record(
        self,
        trace: StrategyTrace,
        reasons: tuple[str, ...],
        reason: str,
    ) -> LearnedPolicyShadowRecord:
        entry = self.registry_entry
        return LearnedPolicyShadowRecord(
            policy_id=entry.policy_id if entry else "",
            policy_version=entry.policy_version if entry else "",
            deployment_mode=self.mode,
            confidence=0.0,
            safety_mask_valid=False,
            invalid_suggestion_reasons=reasons,
            safety_warnings=reasons,
            actual_intent=str(getattr(trace.selected_intent, "value", trace.selected_intent)),
            actual_mode=str(getattr(trace.selected_mode, "value", trace.selected_mode)),
            actual_backend=trace.selected_backend,
            counterfactual_label=_trace_counterfactual_label(trace),
            reason=reason,
        )


class LearnedPolicyShadowAnalyzer:
    """Aggregate learned policy shadow records across traces."""

    def summarize(self, traces: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        records = [
            _shadow_record_dict(trace)
            for trace in traces
            if _shadow_record_dict(trace)
        ]
        n = len(records)
        if n == 0:
            return {
                "n_records": 0,
                "intent_agreement_rate": None,
                "mode_agreement_rate": None,
                "backend_agreement_rate": None,
                "top1_would_change_rate": None,
                "confidence_buckets_vs_outcome": {},
                "safety_warning_count": 0,
                "invalid_suggestion_distribution": {},
                "counterfactual_uncertainty_breakdown": {},
            }
        return {
            "n_records": n,
            "intent_agreement_rate": _rate(records, "intent_agrees"),
            "mode_agreement_rate": _rate(records, "mode_agrees"),
            "backend_agreement_rate": _rate(records, "backend_agrees"),
            "top1_would_change_rate": _rate(records, "would_change_top1"),
            "confidence_buckets_vs_outcome": _shadow_confidence_buckets(records),
            "safety_warning_count": sum(len(record.get("safety_warnings") or []) for record in records),
            "invalid_suggestion_distribution": dict(Counter(
                reason
                for record in records
                for reason in record.get("invalid_suggestion_reasons") or []
            )),
            "counterfactual_uncertainty_breakdown": dict(Counter(
                record.get("counterfactual_label", CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value)
                for record in records
            )),
        }


class OfflinePolicyEvaluator:
    """Compare safe offline variants and learned policy scaffolds."""

    def __init__(
        self,
        *,
        learned_delta_cap: float = 0.02,
        safe_variants: dict[str, PolicyInfluenceConfig] | None = None,
    ) -> None:
        self.learned_delta_cap = learned_delta_cap
        self.safe_variants = safe_variants

    def evaluate_snapshots(self, snapshots: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        runner = PolicyEvaluationRunner(self.safe_variants)
        base_report = runner.evaluate_snapshots(snapshots)
        baseline_traces = base_report["traces_by_variant"]["baseline"]
        training_records = [
            _training_record_from_trace_dict(trace)
            for trace in baseline_traces
        ]
        dataset = PolicyDatasetBuilder().build(training_records)
        imitation = ImitationPolicy().fit(dataset)
        reranker = LearnedBackendReranker(max_delta=self.learned_delta_cap).fit(dataset)
        learned = self.evaluate_dataset(dataset, reranker=reranker, imitation=imitation)
        audit = PolicyDatasetAuditor().audit(dataset)
        reward_sanity = RewardSanityChecker().check(dataset)
        benchmark = self.benchmark_report(base_report, dataset, learned)
        offline_completeness = PolicyOfflineCompletenessChecker().check(
            dataset,
            audit=audit,
            reward_sanity=reward_sanity,
            learned_safety=learned["learned_policy_safety"],
            benchmark_report=benchmark,
        )
        return {
            **base_report,
            "dataset_summary": {
                "dataset_version": dataset.dataset_version,
                "record_version": dataset.record_version,
                "feature_schema_version": dataset.feature_schema_version,
                "reward_version": dataset.reward_version,
                "n_records": len(dataset.records),
            },
            "dataset_audit": asdict(audit),
            "reward_sanity": asdict(reward_sanity),
            "offline_completeness": asdict(offline_completeness),
            "feature_ablation": self.evaluate_feature_ablation(dataset),
            "learned_policy_benchmark_report": benchmark,
            "imitation_policy_summary": learned["imitation_policy_summary"],
            "learned_reranker_summary": learned["learned_reranker_summary"],
            "learned_policy_traces": learned["learned_policy_traces"],
            "learned_policy_safety": learned["learned_policy_safety"],
        }

    def evaluate_dataset(
        self,
        dataset: PolicyDataset,
        *,
        reranker: LearnedBackendReranker | None = None,
        imitation: ImitationPolicy | None = None,
    ) -> dict[str, Any]:
        fitted_imitation = imitation or ImitationPolicy().fit(dataset)
        fitted_reranker = reranker or LearnedBackendReranker(
            max_delta=self.learned_delta_cap
        ).fit(dataset)
        learned_traces = tuple(
            fitted_reranker.trace_for(row, cap=self.learned_delta_cap)
            for row in dataset.records
        )
        safety = self._safety_check(learned_traces, dataset)
        return {
            "imitation_policy_summary": fitted_imitation.evaluate(dataset),
            "learned_reranker_summary": self._reranker_summary(dataset, learned_traces),
            "learned_policy_traces": tuple(asdict(trace) for trace in learned_traces),
            "learned_policy_safety": safety,
        }

    def evaluate_feature_ablation(self, dataset: PolicyDataset) -> dict[str, Any]:
        """Evaluate imitation stability under feature ablations."""
        variants = {
            "full_features": dataset,
            "without_objective_hierarchy": _ablated_dataset(dataset, ("objective_hierarchy", "current_objective_level")),
            "without_failure_taxonomy": _ablated_dataset(dataset, ("failure_type_distribution", "failure_events")),
            "without_backend_memory": _ablated_dataset(dataset, ("backend_memory",), drop_influences=("backend_memory",)),
            "without_nexus_recommendation": _ablated_dataset(dataset, ("nexus_recommendation",)),
            "without_route_budget_data_quality_prior_campaign": _ablated_dataset(
                dataset,
                ("route_context", "budget_context", "data_quality_context", "prior_campaign_context"),
            ),
        }
        baseline = ImitationPolicy().fit(dataset).evaluate(dataset)
        return {
            name: {
                **ImitationPolicy().fit(variant).evaluate(variant),
                "backend_top3_accuracy": _backend_topk_accuracy(variant, k=3),
                "reward_correlation": _reward_match_correlation(variant),
                "safety_metrics": _dataset_safety_metrics(variant),
                "delta_backend_top1_vs_full": round(
                    ImitationPolicy().fit(variant).evaluate(variant)["backend_top1_accuracy"]
                    - baseline["backend_top1_accuracy"],
                    4,
                ),
            }
            for name, variant in variants.items()
        }

    def benchmark_report(
        self,
        base_report: dict[str, Any],
        dataset: PolicyDataset,
        learned: dict[str, Any],
    ) -> dict[str, Any]:
        """Stable offline benchmark report across rule/safe/learned policies."""
        variants = base_report.get("policy_variants") or {}
        safety = base_report.get("safety_check_summary") or {}
        ranking = base_report.get("ranking_change_summary") or {}
        imitation = learned.get("imitation_policy_summary") or {}
        reranker = learned.get("learned_reranker_summary") or {}
        learned_safety = learned.get("learned_policy_safety") or {}
        return {
            "report_version": POLICY_BENCHMARK_REPORT_VERSION,
            "counterfactual_policy": "unknown counterfactual rewards are reported as ranking-only unless observed/replayed",
            "baseline_rule_policy": _benchmark_variant_summary(
                variants.get("baseline", {}),
                safety.get("baseline", {}),
                {},
            ),
            "safe_influence_variants": {
                name: _benchmark_variant_summary(
                    summary,
                    safety.get(name, {}),
                    ranking.get(name, {}),
                )
                for name, summary in variants.items()
                if name not in {"baseline", "combined_safe_influence"}
            },
            "bandit_soft_influence_variant": _benchmark_variant_summary(
                variants.get("combined_safe_influence", {}),
                safety.get("combined_safe_influence", {}),
                ranking.get("combined_safe_influence", {}),
            ),
            "imitation_policy": {
                "intent_accuracy": imitation.get("intent_accuracy", 0.0),
                "mode_accuracy": imitation.get("mode_accuracy", 0.0),
                "backend_top1_accuracy": imitation.get("backend_top1_accuracy", 0.0),
                "reward_delta": 0.0,
                "safety_violations": 0,
            },
            "learned_backend_reranker": {
                **reranker,
                "safety_violations": learned_safety.get("failure_count", 0),
                "cap_violations": _learned_cap_violations(learned.get("learned_policy_traces") or (), self.learned_delta_cap),
                "counterfactual_label": CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value,
            },
            "rule_plus_learned_correction": {
                "reward_delta": 0.0,
                "top1_change_rate": reranker.get("top1_change_rate", 0.0),
                "safety_violations": learned_safety.get("failure_count", 0),
                "unstable_transitions": _dataset_safety_metrics(dataset).get("unstable_transition_count", 0),
                "cap_violations": _learned_cap_violations(learned.get("learned_policy_traces") or (), self.learned_delta_cap),
                "counterfactual_label": CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value,
            },
        }

    def _reranker_summary(
        self,
        dataset: PolicyDataset,
        traces: tuple[LearnedPolicyTrace, ...],
    ) -> dict[str, Any]:
        n = len(traces)
        rewards = [
            float((row.get("reward") or {}).get("composite_reward") or 0.0)
            for row in dataset.records
        ]
        return {
            "n_records": n,
            "reward_delta_vs_baseline": 0.0,
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "top1_change_rate": (
                sum(1 for trace in traces if trace.top1_changed) / n if n else 0.0
            ),
            "max_abs_delta": max(
                (
                    abs(float(delta.get("score_delta") or 0.0))
                    for trace in traces
                    for delta in trace.score_deltas
                ),
                default=0.0,
            ),
            "calibration": {"method": "offline_imitation_baseline"},
            "unstable_transition_count": sum(
                1 for row in dataset.records if "unstable_transition" in (row.get("safety_flags") or [])
            ),
            "failure_attribution_changes": 0,
        }

    def _safety_check(
        self,
        traces: tuple[LearnedPolicyTrace, ...],
        dataset: PolicyDataset,
    ) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        for idx, trace in enumerate(traces):
            row = dataset.records[idx] if idx < len(dataset.records) else {}
            candidate_names = {
                str(candidate.get("name"))
                for candidate in row.get("candidate_backends") or []
                if candidate.get("name")
            }
            for delta in trace.score_deltas:
                value = abs(float(delta.get("score_delta") or 0.0))
                if value - self.learned_delta_cap > 1e-9:
                    failures.append({
                        "index": idx,
                        "check": "learned_delta_cap",
                        "target": delta.get("target"),
                        "score_delta": delta.get("score_delta"),
                        "cap": self.learned_delta_cap,
                    })
                if str(delta.get("target") or "") not in candidate_names:
                    failures.append({
                        "index": idx,
                        "check": "learned_backend_must_exist",
                        "target": delta.get("target"),
                    })
            if trace.suggested_backend and trace.suggested_backend not in candidate_names:
                failures.append({
                    "index": idx,
                    "check": "learned_suggestion_must_exist",
                    "target": trace.suggested_backend,
                })
            if "space_revision_auto_applied" in (row.get("safety_flags") or []):
                failures.append({
                    "index": idx,
                    "check": "space_revision_approval_only",
                })
        return {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        }


def _training_record_from_trace_dict(trace: dict[str, Any]) -> PolicyTrainingRecord:
    reward = trace.get("strategy_reward") or (trace.get("outcome") or {}).get("reward")
    outcome = _normalize_outcome(trace.get("outcome") or trace.get("online_influence_outcome"))
    flags: list[str] = []
    if (trace.get("transition_guard") or {}).get("unstable"):
        flags.append("unstable_transition")
    if (trace.get("space_revision") or {}).get("auto_applied"):
        flags.append("space_revision_auto_applied")
    online = trace.get("online_influence_outcome") or {}
    flags.extend(online.get("safety_warnings") or [])
    return PolicyTrainingRecord(
        campaign_id=str(trace.get("campaign_id") or ""),
        loop_id=f"round-{trace.get('round_number')}",
        state_features=dict(trace.get("state_summary") or {}),
        context_features=dict(trace.get("context_summary") or {}),
        available_actions=tuple(dict(row) for row in trace.get("available_actions") or []),
        selected_intent=str(trace.get("selected_intent") or ""),
        selected_mode=str(trace.get("selected_mode") or ""),
        selected_backend=str(trace.get("selected_backend") or ""),
        candidate_backends=tuple(dict(row) for row in trace.get("candidate_backends") or []),
        applied_influences=tuple(dict(row) for row in trace.get("ranking_influences") or []),
        reward=dict(reward) if reward else None,
        outcome=dict(outcome) if outcome else None,
        safety_flags=tuple(flags),
    )


def replay_records_from_traces(traces: list[Any] | tuple[Any, ...]) -> tuple[PolicyReplayRecord, ...]:
    """Convert traces to canonical replay records for offline evaluation."""
    records: list[PolicyReplayRecord] = []
    for trace in traces:
        if isinstance(trace, dict):
            record = _training_record_from_trace_dict(trace)
            records.append(PolicyReplayRecord(
                campaign_id=record.campaign_id,
                loop_id=record.loop_id,
                state_features=record.state_features,
                context_features=record.context_features,
                available_actions=record.available_actions,
                selected_intent=record.selected_intent,
                selected_mode=record.selected_mode,
                selected_backend=record.selected_backend,
                candidate_backends=record.candidate_backends,
                applied_influences=record.applied_influences,
                reward=record.reward,
                outcome=record.outcome,
                safety_flags=record.safety_flags,
            ))
        else:
            records.append(policy_replay_record_from_trace(_trace_obj(trace)))
    return tuple(records)


def _trace_obj(trace: Any) -> Any:
    return trace


def _normalize_available_actions(
    actions: Any,
    *,
    selected_intent: str = "",
    selected_mode: str = "",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for action in actions or []:
        row = dict(action) if isinstance(action, dict) else {"name": str(action)}
        name = str(row.get("name") or "")
        if not row.get("intent"):
            row["intent"] = _intent_from_action_name(name) or selected_intent
        if not row.get("mode"):
            row["mode"] = _mode_from_action_name(name) or selected_mode
        normalized.append(row)
    return normalized


def _normalize_candidate_backends(candidates: Any) -> list[dict[str, Any]]:
    rows = [
        dict(candidate) if isinstance(candidate, dict) else {"name": str(candidate)}
        for candidate in candidates or []
    ]
    rows.sort(key=lambda candidate: float(candidate.get("total", 0.0) or 0.0), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row.setdefault("total", 0.0)
        row.setdefault("rank", idx)
        row.setdefault("score_components", {
            key: row.get(key)
            for key in (
                "phase_score",
                "fingerprint_boost",
                "failure_penalty",
                "influence_delta",
            )
            if key in row
        })
    return rows


def _intent_from_action_name(name: str) -> str:
    mapping = {
        "explore": "discover",
        "exploit": "optimize",
        "refine": "optimize",
        "stabilize": "stabilize",
        "diagnose": "diagnose",
        "validate": "validate",
        "recover": "recover",
        "pivot": "pivot",
        "transfer": "transfer",
        "revise_space": "revise_space",
        "hypothesis_generate": "hypothesis_generate",
        "hypothesis_test": "hypothesis_test",
    }
    return mapping.get(name, "")


def _mode_from_action_name(name: str) -> str:
    mapping = {
        "explore": "explore",
        "exploit": "exploit",
        "refine": "refine",
        "stabilize": "replicate",
        "diagnose": "failure_localization",
        "validate": "mechanism_validation",
        "recover": "failure_avoidance",
        "pivot": "route_switch",
        "transfer": "warm_start",
        "revise_space": "revise_space",
        "hypothesis_generate": "hypothesis_generate",
        "hypothesis_test": "hypothesis_test",
    }
    return mapping.get(name, "")


def _candidate_names(row: dict[str, Any]) -> set[str]:
    return {
        str(candidate.get("name"))
        for candidate in row.get("candidate_backends") or []
        if candidate.get("name")
    }


def _allowed_intents(row: dict[str, Any]) -> set[str]:
    intents = {
        str(action.get("intent"))
        for action in row.get("available_actions") or []
        if action.get("intent")
    }
    if row.get("selected_intent"):
        intents.add(str(row["selected_intent"]))
    return intents


def _allowed_modes(row: dict[str, Any]) -> set[str]:
    modes = {
        str(action.get("mode"))
        for action in row.get("available_actions") or []
        if action.get("mode")
    }
    if row.get("selected_mode"):
        modes.add(str(row["selected_mode"]))
    return modes


def _row_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_features": dict(observation.get("state_features") or {}),
        "context_features": dict(observation.get("context_features") or {}),
        "available_actions": list(observation.get("available_actions") or []),
        "candidate_backends": list(observation.get("candidate_backends") or []),
        "outcome": {
            "counterfactual_label": observation.get(
                "counterfactual_label",
                CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value,
            )
        },
    }


def _observation_from_trace(trace: StrategyTrace) -> dict[str, Any]:
    return {
        "state_features": dict(trace.state_summary),
        "context_features": dict(trace.context_summary),
        "available_actions": list(trace.available_actions),
        "candidate_backends": [dict(row) for row in trace.candidate_backends],
        "counterfactual_label": _trace_counterfactual_label(trace),
    }


def _trace_counterfactual_label(trace: StrategyTrace) -> str:
    outcome = trace.outcome
    if outcome is not None and outcome.observed:
        return CounterfactualOutcomeLabel.OBSERVED_OUTCOME.value
    if outcome is not None and outcome.outcome:
        return CounterfactualOutcomeLabel.REPLAY_OUTCOME.value
    return CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value


def _shadow_record_dict(trace: Any) -> dict[str, Any] | None:
    if is_dataclass(trace):
        record = getattr(trace, "learned_policy_shadow", None)
        return asdict(record) if record is not None else None
    if isinstance(trace, dict):
        record = trace.get("learned_policy_shadow")
        return dict(record) if record else None
    return None


def _bounded_rate(
    value: Any,
    *,
    numerator: Any = None,
    denominator: Any = None,
) -> float:
    if value is not None:
        return max(0.0, min(1.0, float(value or 0.0)))
    if denominator:
        return max(0.0, min(1.0, float(numerator or 0.0) / float(denominator)))
    return 0.0


def _trace_candidate_names(trace: StrategyTrace) -> set[str]:
    return {
        str(candidate.get("name"))
        for candidate in trace.candidate_backends
        if candidate.get("name")
    }


def _largest_learned_delta(deltas: tuple[dict[str, Any], ...]) -> tuple[str, float]:
    valid = [
        (str(delta.get("target") or ""), float(delta.get("score_delta") or 0.0))
        for delta in deltas
        if delta.get("target")
    ]
    if not valid:
        return "", 0.0
    return max(valid, key=lambda item: abs(item[1]))


def _attach_learned_influence_to_online_outcome(
    outcome: OnlineInfluenceOutcome | None,
    influence: LearnedPolicyInfluenceRecord,
) -> OnlineInfluenceOutcome | None:
    if outcome is None:
        return None
    warnings = tuple(dict.fromkeys((*outcome.safety_warnings, *influence.safety_warnings)))
    return replace(
        outcome,
        learned_policy_influences=tuple((*outcome.learned_policy_influences, influence)),
        safety_warnings=warnings,
        auto_disabled=outcome.auto_disabled or _learned_influence_auto_disable(influence),
        reason="; ".join(warnings) if warnings else outcome.reason,
    )


def _learned_influence_auto_disable(influence: LearnedPolicyInfluenceRecord) -> bool:
    warnings = set(influence.safety_warnings)
    if "cap_violation" in warnings:
        return True
    return bool(warnings & {
        "ranking_changed_without_influence_record",
        "unavailable_backend_suggestion",
        "suggested_backend_unavailable",
        "score_delta_target_unavailable",
        "hard_veto_attempt",
        "space_revision_auto_apply_attempt",
        "learned_high_confidence_underperformance",
        "learned_backend_or_constraint_failure_increase",
    })


def _rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return round(sum(1 for record in records if record.get(key)) / len(records), 4)


def _shadow_confidence_buckets(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = {
        "low": Counter(),
        "medium": Counter(),
        "high": Counter(),
    }
    for record in records:
        confidence = float(record.get("confidence") or 0.0)
        bucket = "low" if confidence < 0.34 else "medium" if confidence < 0.67 else "high"
        label = str(record.get("counterfactual_label") or CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value)
        buckets[bucket][label] += 1
    return {
        bucket: dict(counter)
        for bucket, counter in buckets.items()
        if counter
    }


def _normalize_outcome(outcome: Any) -> dict[str, Any] | None:
    if outcome is None:
        return {
            "counterfactual_label": CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value,
            "outcome": None,
        }
    if is_dataclass(outcome):
        raw = asdict(outcome)
    elif isinstance(outcome, dict):
        raw = dict(outcome)
    else:
        raw = {"outcome": outcome}
    if "counterfactual_label" not in raw:
        if raw.get("observed") is True:
            label = CounterfactualOutcomeLabel.OBSERVED_OUTCOME.value
        elif raw.get("outcome") is not None:
            label = CounterfactualOutcomeLabel.REPLAY_OUTCOME.value
        else:
            label = CounterfactualOutcomeLabel.UNKNOWN_COUNTERFACTUAL.value
        raw["counterfactual_label"] = label
    return raw


def _coverage(rows: tuple[dict[str, Any], ...], key: str) -> float:
    if not rows:
        return 0.0
    present = sum(1 for row in rows if row.get(key))
    return round(present / len(rows), 4)


def _failure_types_from_row(row: dict[str, Any]) -> tuple[str, ...]:
    outcome = row.get("outcome") or {}
    failures = outcome.get("failure_events") or []
    types: list[str] = []
    for failure in failures:
        if isinstance(failure, dict):
            ftype = failure.get("failure_type")
        else:
            ftype = getattr(failure, "failure_type", None)
        if ftype:
            types.append(str(getattr(ftype, "value", ftype)))
    context_failures = (row.get("context_features") or {}).get("failure_type_distribution") or {}
    types.extend(str(key) for key, count in context_failures.items() for _ in range(int(count or 0)))
    return tuple(types)


def _reward_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _ablated_dataset(
    dataset: PolicyDataset,
    context_keys: tuple[str, ...],
    *,
    drop_influences: tuple[str, ...] = (),
) -> PolicyDataset:
    rows: list[dict[str, Any]] = []
    for row in dataset.records:
        copy = dict(row)
        context = dict(copy.get("context_features") or {})
        for key in context_keys:
            context.pop(key, None)
        copy["context_features"] = context
        if drop_influences:
            copy["applied_influences"] = [
                influence for influence in copy.get("applied_influences") or []
                if not any(str(influence.get("source", "")).startswith(prefix) for prefix in drop_influences)
            ]
        rows.append(copy)
    return PolicyDataset(
        records=tuple(rows),
        dataset_version=dataset.dataset_version,
        record_version=dataset.record_version,
        feature_schema_version=f"{dataset.feature_schema_version}:ablated",
        reward_version=dataset.reward_version,
        metadata={**(dataset.metadata or {}), "ablation_keys": list(context_keys)},
    )


def _backend_topk_accuracy(dataset: PolicyDataset, *, k: int) -> float:
    if not dataset.records:
        return 0.0
    hits = 0
    for row in dataset.records:
        selected = row.get("selected_backend")
        ranked = sorted(
            row.get("candidate_backends") or [],
            key=lambda candidate: float(candidate.get("total", 0.0)),
            reverse=True,
        )
        topk = {candidate.get("name") for candidate in ranked[:k]}
        hits += int(selected in topk)
    return round(hits / len(dataset.records), 4)


def _reward_match_correlation(dataset: PolicyDataset) -> float | None:
    rewards = [
        float((row.get("reward") or {}).get("composite_reward") or 0.0)
        for row in dataset.records
    ]
    if len(rewards) < 2 or len(set(rewards)) <= 1:
        return None
    selected_scores = []
    for row in dataset.records:
        selected = row.get("selected_backend")
        score = 0.0
        for candidate in row.get("candidate_backends") or []:
            if candidate.get("name") == selected:
                score = float(candidate.get("total", 0.0))
                break
        selected_scores.append(score)
    if len(set(selected_scores)) <= 1:
        return None
    return round(_pearson(rewards, selected_scores), 4)


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _dataset_safety_metrics(dataset: PolicyDataset) -> dict[str, Any]:
    flags = Counter(
        str(flag)
        for row in dataset.records
        for flag in row.get("safety_flags") or []
    )
    return {
        "safety_flag_count": sum(flags.values()),
        "safety_flag_distribution": dict(flags),
        "unstable_transition_count": flags.get("unstable_transition", 0),
    }


def _benchmark_variant_summary(
    summary: dict[str, Any],
    safety: dict[str, Any],
    ranking: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reward_delta": summary.get("reward_delta_vs_baseline", 0.0),
        "mean_reward": summary.get("mean_reward", 0.0),
        "top1_change_rate": ranking.get("top1_changed_rate", summary.get("backend_changed_rate", 0.0)),
        "safety_violations": safety.get("failure_count", 0),
        "unstable_transitions": summary.get("unstable_transition_count", 0),
        "cap_violations": ranking.get("cap_violation_count", 0),
    }


def _benchmark_has_required_variants(report: dict[str, Any]) -> bool:
    if not report:
        return False
    required = {
        "baseline_rule_policy",
        "safe_influence_variants",
        "bandit_soft_influence_variant",
        "imitation_policy",
        "learned_backend_reranker",
        "rule_plus_learned_correction",
    }
    return required.issubset(set(report))


def _learned_cap_violations(traces: tuple[Any, ...], cap: float) -> int:
    count = 0
    for trace in traces:
        if is_dataclass(trace):
            deltas = trace.score_deltas
        else:
            deltas = trace.get("score_deltas") or []
        for delta in deltas:
            if abs(float(delta.get("score_delta") or 0.0)) - cap > 1e-9:
                count += 1
    return count


def _context_bucket(row: dict[str, Any]) -> str:
    context = row.get("context_features") or {}
    state = row.get("state_features") or {}
    level = context.get("current_objective_level", "performance")
    n_obs = int(state.get("n_observations") or 0)
    obs_bucket = "tiny" if n_obs < 5 else "small" if n_obs < 20 else "large"
    return f"{level}:{obs_bucket}"


def _most_common_with_confidence(counter: Counter[str]) -> tuple[str, float]:
    if not counter:
        return "", 0.0
    key, count = counter.most_common(1)[0]
    total = sum(counter.values())
    return key, count / total if total else 0.0


def _top_backend(candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    if not candidates:
        return ""
    ranked = sorted(candidates, key=lambda row: float(row.get("total", 0.0)), reverse=True)
    return str(ranked[0].get("name") or "")


def _reranked_top(
    candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    deltas: tuple[dict[str, Any], ...],
) -> str:
    delta_by_backend = {
        str(delta.get("target")): float(delta.get("score_delta") or 0.0)
        for delta in deltas
    }
    if not candidates:
        return ""
    ranked = sorted(
        candidates,
        key=lambda row: (
            -(float(row.get("total", 0.0)) + delta_by_backend.get(str(row.get("name")), 0.0)),
            str(row.get("name") or ""),
        ),
    )
    return str(ranked[0].get("name") or "")


def _delta_confidence(deltas: tuple[dict[str, Any], ...]) -> float:
    if not deltas:
        return 0.0
    return round(
        min(1.0, max(abs(float(delta.get("score_delta") or 0.0)) for delta in deltas) * 10),
        4,
    )


def baseline_traces_from_snapshots(snapshots: list[Any] | tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """Build default-selector traces without policy influence."""
    return tuple(
        strategy_trace_to_dict(
            select_strategy(snapshot, config=PhaseConfig()).strategy_trace
        )
        for snapshot in snapshots
    )
