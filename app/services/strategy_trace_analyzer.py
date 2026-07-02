"""Aggregate StrategyTrace records for policy evaluation."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any


class StrategyTraceAnalyzer:
    """Summarise trace-only controller behavior across rounds/campaigns."""

    def __init__(self, traces: list[Any] | tuple[Any, ...] = ()) -> None:
        self.traces = [self._as_dict(trace) for trace in traces if trace is not None]

    def add(self, trace: Any) -> None:
        if trace is not None:
            self.traces.append(self._as_dict(trace))

    def summarize(self, *, top_n: int = 5) -> dict[str, Any]:
        n = len(self.traces)
        bandit_records = [
            trace.get("shadow_bandit_record") or trace.get("bandit_decision")
            for trace in self.traces
            if trace.get("shadow_bandit_record") or trace.get("bandit_decision")
        ]
        agreements = [
            bool(record.get("agrees_with_actual", False))
            for record in bandit_records
        ]
        return {
            "n_traces": n,
            "intent_distribution": self._counter("selected_intent"),
            "mode_distribution": self._counter("selected_mode"),
            "backend_distribution": self._counter("selected_backend"),
            "objective_level_distribution": self._objective_levels(),
            "failure_type_distribution": self._failure_types(),
            "bandit_agreement_rate": (
                sum(agreements) / len(agreements) if agreements else None
            ),
            "top_veto_penalty_boost_reasons": self._top_evidence_reasons(top_n),
            "space_revision_counts": self._space_revision_counts(),
            "transition_counts": self._transition_counts(),
            "unstable_transition_warnings": self._unstable_transition_warnings(),
            "most_common_transition_evidence": self._transition_evidence(top_n),
            "intent_duration_per_campaign": self._duration_per_campaign("selected_intent"),
            "mode_duration_per_campaign": self._duration_per_campaign("selected_mode"),
            "bandit_calibration_summary": self._bandit_calibration_summary(),
        }

    @staticmethod
    def _as_dict(trace: Any) -> dict[str, Any]:
        if is_dataclass(trace):
            return asdict(trace)
        return dict(trace)

    def _counter(self, key: str) -> dict[str, int]:
        return dict(Counter(str(trace.get(key, "")) for trace in self.traces if trace.get(key)))

    def _objective_levels(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for trace in self.traces:
            context = trace.get("context_summary") or {}
            level = context.get("current_objective_level")
            if level:
                counter[str(level)] += 1
        return dict(counter)

    def _failure_types(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for trace in self.traces:
            for evidence in trace.get("evidence") or ():
                source = evidence.get("source", "")
                if source.startswith("failure:"):
                    counter[source.split(":", 1)[1]] += 1
        return dict(counter)

    def _top_evidence_reasons(self, top_n: int) -> list[dict[str, Any]]:
        counter: Counter[tuple[str, str, str]] = Counter()
        for trace in self.traces:
            for evidence in trace.get("evidence") or ():
                effect = evidence.get("effect")
                if effect in {"veto", "penalize", "boost"}:
                    counter[
                        (
                            str(effect),
                            str(evidence.get("source", "")),
                            str(evidence.get("reason", "")),
                        )
                    ] += 1
        return [
            {
                "effect": effect,
                "source": source,
                "reason": reason,
                "count": count,
            }
            for (effect, source, reason), count in counter.most_common(top_n)
        ]

    def _space_revision_counts(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for trace in self.traces:
            revision = trace.get("space_revision")
            if revision:
                counter[str(revision.get("revision_type", "unknown"))] += 1
        return dict(counter)

    def _transition_counts(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for trace in self.traces:
            guard = trace.get("transition_guard") or {}
            from_intent = guard.get("from_intent")
            to_intent = guard.get("to_intent") or trace.get("selected_intent")
            if not to_intent:
                continue
            label = f"{from_intent or 'initial'}->{to_intent}"
            counter[label] += 1
        return dict(counter)

    def _unstable_transition_warnings(self) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for trace in self.traces:
            guard = trace.get("transition_guard") or {}
            if not guard.get("unstable"):
                continue
            warnings.append({
                "campaign_id": trace.get("campaign_id") or "default",
                "round_number": trace.get("round_number"),
                "transition": (
                    f"{guard.get('from_intent') or 'initial'}->{guard.get('to_intent')}"
                ),
                "reason": guard.get("reason", ""),
            })
        return warnings

    def _transition_evidence(self, top_n: int) -> list[dict[str, Any]]:
        counter: Counter[tuple[str, str, str]] = Counter()
        for trace in self.traces:
            guard = trace.get("transition_guard") or {}
            for evidence in guard.get("evidence") or ():
                counter[
                    (
                        str(evidence.get("effect", "")),
                        str(evidence.get("target", "")),
                        str(evidence.get("reason", "")),
                    )
                ] += 1
        return [
            {
                "effect": effect,
                "transition": transition,
                "reason": reason,
                "count": count,
            }
            for (effect, transition, reason), count in counter.most_common(top_n)
        ]

    def _duration_per_campaign(self, key: str) -> dict[str, dict[str, int]]:
        grouped: dict[str, Counter[str]] = {}
        for trace in self.traces:
            value = trace.get(key)
            if not value:
                continue
            campaign_id = str(trace.get("campaign_id") or "default")
            grouped.setdefault(campaign_id, Counter())[str(value)] += 1
        return {campaign_id: dict(counter) for campaign_id, counter in grouped.items()}

    def _bandit_calibration_summary(self) -> dict[str, Any]:
        records = [
            trace.get("shadow_bandit_record") or {}
            for trace in self.traces
            if trace.get("shadow_bandit_record")
        ]
        if not records:
            return {
                "n_records": 0,
                "confidence_buckets": {},
                "agreement_reward_summary": {},
                "confidence_vs_outcome": [],
            }
        buckets: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
        reward_by_agreement: dict[str, list[float]] = {"agree": [], "disagree": []}
        confidence_vs_outcome: list[dict[str, Any]] = []
        for record in records:
            confidence = float(record.get("bandit_confidence") or 0.0)
            reward = record.get("actual_reward")
            reward_value = float(reward) if reward is not None else 0.0
            bucket = "low" if confidence < 0.34 else "medium" if confidence < 0.67 else "high"
            buckets[bucket].append(reward_value)
            agreement_key = "agree" if record.get("agrees_with_actual") else "disagree"
            reward_by_agreement[agreement_key].append(reward_value)
            confidence_vs_outcome.append({
                "confidence": confidence,
                "reward": reward if reward is not None else None,
                "agrees_with_actual": bool(record.get("agrees_with_actual")),
                "outcome": record.get("outcome"),
            })
        return {
            "n_records": len(records),
            "confidence_buckets": {
                key: self._reward_stats(values)
                for key, values in buckets.items()
                if values
            },
            "agreement_reward_summary": {
                key: self._reward_stats(values)
                for key, values in reward_by_agreement.items()
                if values
            },
            "confidence_vs_outcome": confidence_vs_outcome,
        }

    @staticmethod
    def _reward_stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"n": 0, "mean_reward": 0.0}
        return {
            "n": len(values),
            "mean_reward": round(sum(values) / len(values), 4),
        }

    @staticmethod
    def summarize_policy_evaluation(report: dict[str, Any]) -> dict[str, Any]:
        """Compact analyzer view of offline policy-variant evaluation."""
        safety = report.get("safety_check_summary") or {}
        ranking = report.get("ranking_change_summary") or {}
        rewards = report.get("reward_comparison_summary") or {}
        return {
            "baseline_summary": report.get("baseline_summary", {}),
            "safe_influence_summary": report.get("safe_influence_summary", {}),
            "ranking_change_summary": {
                name: {
                    "top1_changed_count": summary.get("top1_changed_count", 0),
                    "topk_changed_count": summary.get("topk_changed_count", 0),
                    "unexplained_change_count": summary.get("unexplained_change_count", 0),
                    "cap_violation_count": summary.get("cap_violation_count", 0),
                }
                for name, summary in ranking.items()
            },
            "reward_comparison_summary": rewards,
            "safety_check_summary": {
                name: {
                    "passed": summary.get("passed", False),
                    "failure_count": summary.get("failure_count", 0),
                    "warning_count": summary.get("warning_count", 0),
                }
                for name, summary in safety.items()
            },
        }
