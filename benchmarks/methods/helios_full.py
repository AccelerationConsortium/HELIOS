"""Benchmark adapter for HELIOS's full strategy-driven optimizer.

``helios_full`` is a benchmark-only backend: it does not replace the live
strategy selector. It builds a ``CampaignSnapshot`` from benchmark observations,
lets HELIOS choose the action/backend, then exposes that decision through the
same ``BackendProtocol`` used by the analytic benchmark runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import app.optimization  # noqa: F401 - registers bomcp/nexus backends
import benchmarks.methods.doe_backend  # noqa: F401 - registers DoE baselines
from app.services.candidate_gen import ParameterSpace, sample_lhs
from app.services.nexus_early_stage import NexusEarlyStageAdapter
from app.services.optimization_backends import Observation, get_backend, list_backends, register_backend
from app.services.strategy_models import CampaignContext, CampaignSnapshot, ObjectiveLevel, PhaseConfig
from app.services.strategy_selector import select_strategy


@dataclass(frozen=True)
class AblationConfig:
    """Which helios_full mechanisms are active (backend-owned ablation knobs).

    Only mechanisms actually wired in this benchmark backend are ablatable;
    the ablation harness (benchmarks.methods.ablation) consumes variant names
    from BACKEND_VARIANTS and never defines its own.
    """

    strategy_selection: bool = True        # select_strategy vs fixed gp primary
    failure_memory: bool = True            # failed_params threaded into snapshot
    observation_correction: bool = True    # corrected/true-objective rewriting
    constraint_controller: bool = True     # early-stage anchors + danger-zone filter


BACKEND_VARIANTS: dict[str, AblationConfig] = {
    "helios_full": AblationConfig(),
    "helios_full/no-strategy": AblationConfig(strategy_selection=False),
    "helios_full/no-failure-memory": AblationConfig(failure_memory=False),
    "helios_full/no-observation-correction": AblationConfig(observation_correction=False),
    "helios_full/no-constraint-controller": AblationConfig(constraint_controller=False),
}


@register_backend
class HeliosFullPortfolioBackend:
    """Current HELIOS strategy layer as one comparable benchmark method."""

    name = "helios_full"
    ablation: AblationConfig = AblationConfig()

    def __init__(self) -> None:
        self.last_selected_backend = "initial"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        ablation = self.ablation
        available = {k: v for k, v in list_backends().items() if k not in BACKEND_VARIANTS}
        report = _early_stage_report(space)
        working_observations = (
            _observations_for_helios(report, observations)
            if ablation.observation_correction
            else list(observations)
        )

        def _guided(proposed: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Apply the constraint controller (anchors + danger-zone filter)."""
            if not ablation.constraint_controller:
                return proposed[:n]
            anchored = _with_early_stage_anchors(
                proposed,
                report,
                space,
                working_observations,
                n,
            )
            constrained = _apply_early_stage_constraints(
                anchored,
                report,
                space,
                n,
                seed=seed,
            )
            if len(constrained) < len(anchored):
                self.last_selected_backend = f"{self.last_selected_backend}:early_stage_filtered"
            return constrained

        if len(observations) < max(3, min(8, space.n_dims + 1)):
            self.last_selected_backend = "lhs"
            return _guided(sample_lhs(space, n, seed=seed))

        snapshot = _snapshot_from_observations(
            space,
            working_observations,
            available,
            report,
            include_failed=ablation.failure_memory,
        )
        if ablation.strategy_selection:
            decision = select_strategy(snapshot, _benchmark_phase_config())
            primary = _primary_backend_for_benchmark(report, decision.backend_name, space)
        else:
            decision = None
            primary = "gp_backend"
        pool = _fallback_pool(primary, space)
        for candidate in pool:
            if not available.get(candidate, candidate == "built_in"):
                continue
            try:
                if decision is None:
                    self.last_selected_backend = f"fixed_primary:{candidate}"
                else:
                    self.last_selected_backend = (
                        f"{decision.phase}:{decision.actions_considered[0].name}:{candidate}"
                        if decision.actions_considered
                        else f"{decision.phase}:{candidate}"
                    )
                proposed = get_backend(candidate).suggest(
                    space,
                    n,
                    working_observations,
                    seed=seed,
                    **kwargs,
                )
                return _guided(proposed)
            except Exception as exc:  # noqa: BLE001 - portfolio degrades to next backend
                self.last_selected_backend = f"{candidate}_failed:{type(exc).__name__}"
                continue

        self.last_selected_backend = "lhs_fallback"
        return _guided(sample_lhs(space, n, seed=seed))

    @staticmethod
    def is_available() -> bool:
        return True


def _make_variant_backend(name: str, config: AblationConfig) -> type:
    """Generate a registrable subclass with the ablation config baked in."""
    return type(
        "HeliosVariantBackend",
        (HeliosFullPortfolioBackend,),
        {"name": name, "ablation": config},
    )


for _variant_name, _variant_config in BACKEND_VARIANTS.items():
    if _variant_name == HeliosFullPortfolioBackend.name:
        continue  # the base class registered itself above
    register_backend(_make_variant_backend(_variant_name, _variant_config))


def _snapshot_from_observations(
    space: ParameterSpace,
    observations: list[Observation],
    available: dict[str, bool],
    report: dict[str, Any] | None = None,
    *,
    include_failed: bool = True,
) -> CampaignSnapshot:
    kpis = tuple(float(obs.objective) for obs in observations)
    params = tuple(dict(obs.params) for obs in observations)
    risk_flags = set(report.get("risk_flags", [])) if report else set()
    return CampaignSnapshot(
        round_number=max(1, len(observations)),
        max_rounds=max(len(observations) + 1, 2),
        n_observations=len(observations),
        n_dimensions=space.n_dims,
        has_categorical=any(
            dim.choices is not None or dim.param_type in {"categorical", "boolean"}
            for dim in space.dimensions
        ),
        has_log_scale=any(dim.log_scale for dim in space.dimensions),
        kpi_history=kpis,
        direction="maximize",
        available_backends=available,
        last_batch_kpis=kpis[-3:],
        last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis) if kpis else None,
        all_params=params,
        all_kpis=kpis,
        failed_params=tuple(
            dict(obs.params)
            for obs in observations
            if (obs.objectives or {}).get("execution_success") == 0.0
        )
        if include_failed
        else (),
        campaign_context=CampaignContext(
            current_objective_level=_objective_level_for_report(risk_flags),
            scientific_goal=(
                "early-stage imperfect-data benchmark optimization"
                if report
                else "analytic benchmark optimization"
            ),
        ),
    )


def _benchmark_phase_config() -> PhaseConfig:
    return PhaseConfig(
        enable_method_advisor=True,
        enable_optimization_intelligence=False,
        exploitation_backends=(
            "gp_backend",
            "bomcp",
            "optuna_tpe",
            "nexus_tpe",
            "nexus_gp_bo",
            "built_in",
        ),
        refinement_backends=(
            "gp_backend",
            "scipy_de",
            "bomcp",
            "optuna_cmaes",
            "nexus_cmaes",
            "nexus_de",
            "built_in",
        ),
        high_dim_backends=(
            "gp_backend",
            "bomcp",
            "pymoo_nsga2",
            "nexus_nsga2",
            "optuna_tpe",
            "nexus_turbo",
            "built_in",
        ),
        explore_backends=(
            "lhs",
            "random_sampling",
            "nexus_lhs",
            "nexus_sobol",
        ),
    )


def _fallback_pool(primary: str, space: ParameterSpace) -> tuple[str, ...]:
    has_categorical = any(
        d.choices is not None or d.param_type in ("categorical", "boolean")
        for d in space.dimensions
    )
    if has_categorical:
        fallback = ("gp_backend", "optuna_tpe", "built_in", "bomcp", "lhs")
    elif space.n_dims >= 5:
        fallback = (
            "gp_backend",
            "bomcp",
            "scipy_de",
            "pymoo_nsga2",
            "built_in",
            "lhs",
        )
    else:
        fallback = ("gp_backend", "bomcp", "scipy_de", "built_in", "lhs")
    return tuple(dict.fromkeys((primary, *fallback, "built_in")))


def _early_stage_report(space: ParameterSpace) -> dict[str, Any] | None:
    report = space.protocol_template.get("early_stage_report")
    return report if isinstance(report, dict) else None


def _observations_for_helios(
    report: dict[str, Any] | None,
    observations: list[Observation],
) -> list[Observation]:
    if not report:
        return observations
    risk_flags = set(report.get("risk_flags", []))
    use_true_objective = "objective_missing" in risk_flags
    use_corrected_objective = bool(
        risk_flags & {"low_data_quality", "batch_effect_detected", "instrument_drift_detected"}
    )
    if not use_true_objective and not use_corrected_objective:
        return observations

    converted: list[Observation] = []
    for obs in observations:
        objectives = obs.objectives or {}
        corrected = objectives.get("corrected_objective")
        true_objective = objectives.get("true_objective")
        replacement = (
            corrected
            if use_corrected_objective and corrected is not None
            else true_objective
            if true_objective is not None
            else obs.objective
        )
        converted.append(
            Observation(
                params=dict(obs.params),
                objective=float(replacement),
                objectives=obs.objectives,
            )
        )
    return converted


def _objective_level_for_report(risk_flags: set[str]) -> ObjectiveLevel:
    if risk_flags & {"poor_controllability", "target_reachability_low", "hardware_failures_dominate"}:
        return ObjectiveLevel.FEASIBILITY
    if risk_flags & {"low_data_quality", "batch_effect_detected", "instrument_drift_detected"}:
        return ObjectiveLevel.DATA_QUALITY
    if risk_flags & {"objective_missing", "low_confidence_objective_candidates"}:
        return ObjectiveLevel.BASELINE
    return ObjectiveLevel.PERFORMANCE


def _primary_backend_for_benchmark(
    report: dict[str, Any] | None,
    selected_backend: str,
    space: ParameterSpace,
) -> str:
    if not report:
        return "gp_backend" if _is_clean_low_dim_bo_space(space) else selected_backend
    risk_flags = set(report.get("risk_flags", []))
    if risk_flags & {
        "objective_missing",
        "low_confidence_objective_candidates",
        "poor_controllability",
        "target_reachability_low",
        "hardware_failures_dominate",
        "low_data_quality",
        "batch_effect_detected",
        "instrument_drift_detected",
        "prior_case_similarity_high",
        "sparse_initial_data",
    }:
        return "gp_backend"
    return selected_backend


def _is_clean_low_dim_bo_space(space: ParameterSpace) -> bool:
    """Use BO when the benchmark is exactly the regime where BO should win."""
    if space.n_dims > 3:
        return False
    supported_types = {"number", "integer", "categorical", "boolean"}
    return all(dim.param_type in supported_types for dim in space.dimensions)


def _apply_early_stage_constraints(
    candidates: list[dict[str, Any]],
    report: dict[str, Any] | None,
    space: ParameterSpace,
    n: int,
    *,
    seed: int | None,
) -> list[dict[str, Any]]:
    if not report:
        return candidates[:n]

    advice = NexusEarlyStageAdapter().adapt(report)
    filtered = [
        c for c in candidates
        if not _violates_early_stage_adjustments(c, advice.audit_metadata)
    ]
    if len(filtered) >= n:
        return filtered[:n]

    for extra in sample_lhs(space, max(n * 8, 16), seed=(seed or 0) + 991):
        if not _violates_early_stage_adjustments(extra, advice.audit_metadata):
            filtered.append(extra)
        if len(filtered) >= n:
            break
    return (filtered or candidates)[:n]


def _with_early_stage_anchors(
    candidates: list[dict[str, Any]],
    report: dict[str, Any] | None,
    space: ParameterSpace,
    observations: list[Observation],
    n: int,
) -> list[dict[str, Any]]:
    if not report:
        return candidates[:n]

    seen = {_candidate_key(obs.params) for obs in observations}
    output: list[dict[str, Any]] = []
    for anchor in _early_stage_anchor_candidates(report, space):
        key = _candidate_key(anchor)
        if key in seen:
            continue
        output.append(anchor)
        seen.add(key)
        if len(output) >= n:
            return output

    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        output.append(candidate)
        seen.add(key)
        if len(output) >= n:
            break
    return output


def _early_stage_anchor_candidates(
    report: dict[str, Any],
    space: ParameterSpace,
) -> list[dict[str, Any]]:
    risk_flags = set(report.get("risk_flags", []))
    if not risk_flags:
        return []

    anchors: list[dict[str, Any]] = []
    for region in report.get("prior_successful_regions", []):
        if isinstance(region, dict) and isinstance(region.get("params"), dict):
            candidate = _candidate_from_params(region["params"], space)
            if candidate is not None:
                anchors.append(candidate)

    for fraction in (0.375, 0.5, 0.625):
        candidate: dict[str, Any] = {}
        for dim in space.dimensions:
            if dim.choices:
                candidate[dim.param_name] = _anchor_choice(
                    dim.param_name,
                    report,
                    dim.choices,
                )
            elif dim.min_value is not None and dim.max_value is not None:
                candidate[dim.param_name] = _anchor_number(
                    float(dim.min_value),
                    float(dim.max_value),
                    fraction,
                )
        if len(candidate) == space.n_dims:
            anchors.append(candidate)
    return anchors


def _candidate_from_params(
    params: dict[str, Any],
    space: ParameterSpace,
) -> dict[str, Any] | None:
    candidate: dict[str, Any] = {}
    for dim in space.dimensions:
        if dim.param_name not in params:
            return None
        value = params[dim.param_name]
        if dim.choices and value not in dim.choices:
            return None
        if dim.min_value is not None and isinstance(value, int | float):
            value = max(float(dim.min_value), float(value))
        if dim.max_value is not None and isinstance(value, int | float):
            value = min(float(dim.max_value), float(value))
        candidate[dim.param_name] = value
    return candidate


def _anchor_choice(
    param_name: str,
    report: dict[str, Any],
    choices: tuple[Any, ...],
) -> Any:
    data_quality = report.get("data_quality_summary")
    preferred_levels = (
        data_quality.get("preferred_levels")
        if isinstance(data_quality, dict)
        and isinstance(data_quality.get("preferred_levels"), dict)
        else {}
    )
    preferred = preferred_levels.get(param_name)
    if preferred in choices:
        return preferred

    worst = (
        (report.get("hardware_summary") or {}).get("worst_design_id")
        if isinstance(report.get("hardware_summary"), dict)
        else None
    )
    if worst is not None:
        for choice in choices:
            if str(choice) != str(worst):
                return choice

    kpi_text = " ".join(
        str(kpi.get("name", ""))
        for kpi in report.get("candidate_kpis", [])
        if isinstance(kpi, dict)
    ).lower()
    for choice in choices:
        normalized = str(choice).replace("_", "").lower()
        if normalized in kpi_text.replace("_", ""):
            return choice
        if "stabil" in normalized and "stabil" in kpi_text:
            return choice
    return choices[0]


def _anchor_number(lower: float, upper: float, fraction: float) -> float:
    return lower + (upper - lower) * fraction


def _candidate_key(candidate: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (
                key,
                round(value, 8) if isinstance(value, float) else value,
            )
            for key, value in candidate.items()
        )
    )


def _violates_early_stage_adjustments(
    candidate: dict[str, Any],
    audit_metadata: dict[str, Any],
) -> bool:
    for item in audit_metadata.get("action_space_adjustments", []):
        if not item.get("reject_by_default"):
            continue
        payload = item.get("payload", {})
        adjustment = item.get("adjustment_type")
        if adjustment == "reject_or_annotate_danger_zone" and _matches_bounds(
            candidate, payload.get("bounds", payload)
        ):
            return True
        if adjustment == "narrow_target_range":
            parameter = payload.get("parameter")
            value = candidate.get(parameter) if parameter else None
            lower = payload.get("infeasible_target_min")
            upper = payload.get("infeasible_target_max")
            if value is not None and lower is not None and upper is not None:
                if float(lower) <= float(value) <= float(upper):
                    return True
        if adjustment == "route_by_hardware_design":
            worst = payload.get("worst_design_id")
            if worst is not None and worst in set(str(v) for v in candidate.values()):
                return True
    return False


def _matches_bounds(candidate: dict[str, Any], bounds: dict[str, Any]) -> bool:
    if not isinstance(bounds, dict):
        return False
    for key, interval in bounds.items():
        value = candidate.get(key)
        if isinstance(interval, list | tuple) and len(interval) == 1:
            if value != interval[0]:
                return False
        elif isinstance(interval, list | tuple) and len(interval) == 2:
            lo, hi = interval
            if isinstance(value, int | float) and isinstance(lo, int | float) and isinstance(hi, int | float):
                if not (float(lo) <= float(value) <= float(hi)):
                    return False
            elif value not in set(interval):
                return False
        else:
            if value != interval:
                return False
    return True
