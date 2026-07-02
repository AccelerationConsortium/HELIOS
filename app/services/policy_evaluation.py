"""Offline replay evaluation for safe policy influence variants."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any

from app.services.strategy_models import (
    CampaignSnapshot,
    PolicyInfluenceConfig,
)
from app.services.strategy_selector import PhaseConfig, select_strategy, strategy_trace_to_dict
from app.services.strategy_trace_analyzer import StrategyTraceAnalyzer


class PolicyEvaluationRunner:
    """Compare bounded policy-influence variants on replay snapshots/traces."""

    DEFAULT_VARIANTS: dict[str, PolicyInfluenceConfig] = {
        "baseline": PolicyInfluenceConfig(),
        "action_policy_rerank": PolicyInfluenceConfig(enable_action_policy_rerank=True),
        "backend_memory_rerank": PolicyInfluenceConfig(enable_backend_memory_rerank=True),
        "transition_guard_penalty": PolicyInfluenceConfig(enable_transition_guard_penalty=True),
        "combined_safe_influence": PolicyInfluenceConfig(
            enable_action_policy_rerank=True,
            enable_backend_memory_rerank=True,
            enable_transition_guard_penalty=True,
            enable_bandit_rerank=True,
        ),
    }

    def __init__(
        self,
        variants: dict[str, PolicyInfluenceConfig] | None = None,
    ) -> None:
        self.variants = variants or dict(self.DEFAULT_VARIANTS)

    def evaluate_snapshots(
        self,
        snapshots: list[CampaignSnapshot] | tuple[CampaignSnapshot, ...],
    ) -> dict[str, Any]:
        """Run replay snapshots under every configured policy variant."""
        traces_by_variant: dict[str, list[dict[str, Any]]] = {}
        for name, influence in self.variants.items():
            traces_by_variant[name] = [
                self._trace_for(snapshot, influence)
                for snapshot in snapshots
            ]

        baseline = traces_by_variant.get("baseline", [])
        variants = {
            name: self._variant_summary(name, traces, baseline)
            for name, traces in traces_by_variant.items()
        }
        ranking_change_summary = {
            name: self._counterfactual_ranking_replay(baseline, traces)
            for name, traces in traces_by_variant.items()
            if name != "baseline"
        }
        safety = {
            name: self._safety_check(traces, self.variants[name])
            for name, traces in traces_by_variant.items()
        }
        report = {
            "baseline_summary": variants.get("baseline", {}),
            "safe_influence_summary": variants.get("combined_safe_influence", {}),
            "policy_variants": variants,
            "ranking_change_summary": ranking_change_summary,
            "reward_comparison_summary": self._reward_comparison(variants),
            "safety_check_summary": safety,
            "bandit_calibration_evaluation": {
                name: StrategyTraceAnalyzer(traces).summarize()["bandit_calibration_summary"]
                for name, traces in traces_by_variant.items()
            },
            "traces_by_variant": traces_by_variant,
        }
        report["analyzer_policy_variant_summary"] = (
            StrategyTraceAnalyzer.summarize_policy_evaluation(report)
        )
        return report

    def evaluate_traces(
        self,
        traces: list[Any] | tuple[Any, ...],
    ) -> dict[str, Any]:
        """Evaluate persisted StrategyTrace records when snapshots are unavailable."""
        trace_dicts = [self._as_dict(trace) for trace in traces if trace is not None]
        analyzer_summary = StrategyTraceAnalyzer(trace_dicts).summarize()
        safety = self._safety_check(trace_dicts, PolicyInfluenceConfig())
        return {
            "baseline_summary": analyzer_summary,
            "safe_influence_summary": {},
            "policy_variants": {
                "persisted_traces": {
                    "n_traces": len(trace_dicts),
                    "mean_reward": self._mean_reward(trace_dicts),
                    "unstable_transition_count": len(
                        analyzer_summary.get("unstable_transition_warnings", [])
                    ),
                }
            },
            "ranking_change_summary": {},
            "reward_comparison_summary": {},
            "safety_check_summary": {"persisted_traces": safety},
            "bandit_calibration_evaluation": analyzer_summary.get(
                "bandit_calibration_summary", {}
            ),
        }

    @staticmethod
    def _trace_for(
        snapshot: CampaignSnapshot,
        influence: PolicyInfluenceConfig,
    ) -> dict[str, Any]:
        decision = select_strategy(
            snapshot,
            config=PhaseConfig(policy_influence=influence),
        )
        return strategy_trace_to_dict(decision.strategy_trace)

    def _variant_summary(
        self,
        name: str,
        traces: list[dict[str, Any]],
        baseline: list[dict[str, Any]],
    ) -> dict[str, Any]:
        baseline_backends = [trace.get("selected_backend") for trace in baseline]
        changed = sum(
            1
            for idx, trace in enumerate(traces)
            if idx < len(baseline_backends)
            and trace.get("selected_backend") != baseline_backends[idx]
        )
        mean_reward = self._mean_reward(traces)
        baseline_reward = self._mean_reward(baseline)
        analyzer = StrategyTraceAnalyzer(traces).summarize()
        return {
            "variant": name,
            "n_traces": len(traces),
            "backend_changed_count": changed,
            "backend_changed_rate": changed / len(traces) if traces else 0.0,
            "mean_reward": mean_reward,
            "reward_delta_vs_baseline": round(mean_reward - baseline_reward, 4),
            "unstable_transition_count": len(
                analyzer.get("unstable_transition_warnings", [])
            ),
            "safety_warning_count": 0,
            "analyzer_summary": analyzer,
        }

    @staticmethod
    def _counterfactual_ranking_replay(
        baseline: list[dict[str, Any]],
        variant: list[dict[str, Any]],
        *,
        top_k: int = 3,
    ) -> dict[str, Any]:
        top1_changes = 0
        topk_changes = 0
        unexplained_changes = 0
        cap_violations = 0
        records: list[dict[str, Any]] = []
        for idx, (base_trace, variant_trace) in enumerate(zip(baseline, variant, strict=False)):
            base_rank = _ranked_backends(base_trace)
            variant_rank = _ranked_backends(variant_trace)
            if not base_rank or not variant_rank:
                continue
            top1_changed = base_rank[0] != variant_rank[0]
            topk_changed = set(base_rank[:top_k]) != set(variant_rank[:top_k])
            influences = variant_trace.get("ranking_influences") or []
            nonzero_influences = [
                record for record in influences
                if abs(float(record.get("score_delta") or 0.0)) > 0
            ]
            explained = bool(nonzero_influences)
            if top1_changed:
                top1_changes += 1
            if topk_changed:
                topk_changes += 1
            if (top1_changed or topk_changed) and not explained:
                unexplained_changes += 1
            for record in influences:
                applied_weight = abs(float(record.get("applied_weight") or 0.0))
                score_delta = abs(float(record.get("score_delta") or 0.0))
                if score_delta - applied_weight > 1e-9:
                    cap_violations += 1
            records.append({
                "round_number": variant_trace.get("round_number"),
                "index": idx,
                "baseline_top1": base_rank[0],
                "variant_top1": variant_rank[0],
                "top1_changed": top1_changed,
                "topk_changed": topk_changed,
                "explained_by_influence": explained,
                "influence_records": influences,
            })
        return {
            "n_replayed": len(records),
            "top1_changed_count": top1_changes,
            "top1_changed_rate": top1_changes / len(records) if records else 0.0,
            "topk_changed_count": topk_changes,
            "unexplained_change_count": unexplained_changes,
            "cap_violation_count": cap_violations,
            "records": records,
        }

    @staticmethod
    def _safety_check(
        traces: list[dict[str, Any]],
        config: PolicyInfluenceConfig,
    ) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        total_cap = config.max_total_score_delta
        source_caps = {
            "action_policy": config.max_action_policy_weight,
            "backend_memory": config.max_backend_memory_weight,
            "backend_memory_failure_event": config.max_backend_memory_weight,
            "transition_guard": config.max_transition_guard_weight,
            "bandit_shadow": 0.0,
            "bandit_soft": config.max_bandit_weight,
            "action_policy_veto_shadow": 0.0,
        }
        for trace in traces:
            round_number = trace.get("round_number")
            influences = trace.get("ranking_influences") or []
            by_target: Counter[str] = Counter()
            for record in influences:
                source = str(record.get("source", ""))
                target = str(record.get("target", ""))
                delta = float(record.get("score_delta") or 0.0)
                by_target[target] += delta
                cap = source_caps.get(source, abs(float(record.get("applied_weight") or 0.0)))
                if abs(delta) - cap > 1e-9:
                    failures.append({
                        "round_number": round_number,
                        "check": "score_delta_within_source_cap",
                        "source": source,
                        "target": target,
                        "score_delta": delta,
                        "cap": cap,
                    })
                if source == "bandit_shadow" and abs(delta) > 1e-12:
                    failures.append({
                        "round_number": round_number,
                        "check": "bandit_shadow_only",
                        "target": target,
                    })
                if source == "bandit_soft":
                    bandit = trace.get("bandit_influence") or {}
                    eligibility = bandit.get("eligibility") or {}
                    if not eligibility.get("eligible", False) and abs(delta) > 1e-12:
                        failures.append({
                            "round_number": round_number,
                            "check": "bandit_ineligible_nonzero_delta",
                            "target": target,
                        })
                if source == "action_policy_veto_shadow" and abs(delta) > 1e-12:
                    failures.append({
                        "round_number": round_number,
                        "check": "action_policy_no_hard_veto_by_default",
                        "target": target,
                    })
            for target, total in by_target.items():
                if abs(float(total)) - total_cap > 1e-9:
                    failures.append({
                        "round_number": round_number,
                        "check": "score_delta_within_total_cap",
                        "target": target,
                        "total_delta": float(total),
                        "cap": total_cap,
                    })
            for evidence in trace.get("evidence") or []:
                if evidence.get("source") != "ranking_influence:backend_memory":
                    continue
                failure_type = (evidence.get("metadata") or {}).get("failure_type")
                if failure_type in {"hardware", "measurement", "scientific_negative"}:
                    failures.append({
                        "round_number": round_number,
                        "check": "backend_memory_attribution",
                        "failure_type": failure_type,
                        "target": evidence.get("target"),
                    })
            if (trace.get("space_revision") or {}).get("auto_applied"):
                failures.append({
                    "round_number": round_number,
                    "check": "space_revision_approval_only",
                })
            if influences and not any(trace.get("candidate_backends") or []):
                warnings.append({
                    "round_number": round_number,
                    "check": "influence_without_candidate_table",
                })
        return {
            "passed": not failures,
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "failures": failures,
            "warnings": warnings,
        }

    @staticmethod
    def _reward_comparison(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
        baseline = variants.get("baseline", {}).get("mean_reward", 0.0)
        return {
            name: {
                "mean_reward": summary.get("mean_reward", 0.0),
                "delta_vs_baseline": round(summary.get("mean_reward", 0.0) - baseline, 4),
            }
            for name, summary in variants.items()
        }

    @staticmethod
    def _mean_reward(traces: list[dict[str, Any]]) -> float:
        rewards = [
            ((trace.get("strategy_reward") or {}).get("composite_reward"))
            for trace in traces
        ]
        values = [float(value) for value in rewards if value is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    @staticmethod
    def _as_dict(trace: Any) -> dict[str, Any]:
        if is_dataclass(trace):
            return asdict(trace)
        return dict(trace)


def _ranked_backends(trace: dict[str, Any]) -> list[str]:
    rows = list(trace.get("candidate_backends") or [])
    rows.sort(key=lambda row: float(row.get("total", 0.0)), reverse=True)
    return [str(row.get("name")) for row in rows if row.get("name")]
