"""Analytic optimization test problems with KNOWN ground-truth optima.

The defining guarantee of every problem here is that evaluating ``objective``
at the stated ``optimum_x`` returns the stated ``optimum`` value (within
tolerance).  Without that ruler, regret -- and therefore the whole
method-comparison benchmark -- has no meaning.

Sign convention
---------------
``objective`` returns the *raw* test-function value in textbook **minimization**
form (lower = better), and ``optimum`` is that minimum.  The study runner flips
the sign before handing values to the maximizing HELIOS backends; see the
package docstring.  ``ProblemTags`` operationalizes the project's
"factors influencing method selection".
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.services.candidate_gen import ParameterSpace, SearchDimension

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProblemTags:
    """Structural factors that influence which method works best."""

    n_dims: int
    multimodal: bool
    noise_std: float
    separable: bool
    has_categorical: bool
    surface_class: str  # e.g. "convex" | "valley" | "multimodal" | "mixed"


@dataclass(frozen=True)
class OptProblem:
    """An analytic problem: space + objective + known optimum + tags."""

    id: str
    name: str
    space: ParameterSpace
    objective: Callable[[dict], float]  # raw minimization value (lower = better)
    optimum: float                      # f(optimum_x), the global minimum
    optimum_x: dict | None              # argmin (None if not in closed form)
    tags: ProblemTags
    evaluator: Callable[[dict], ProblemEvaluation] | None = None

    def evaluate(self, params: dict) -> ProblemEvaluation:
        """Return the benchmark evaluation seen by scoring and optimizers."""
        if self.evaluator is not None:
            return self.evaluator(params)
        value = float(self.objective(params))
        return ProblemEvaluation(raw_value=value, observed_value=value)


@dataclass(frozen=True)
class ProblemEvaluation:
    """One evaluation with true score plus imperfect early-stage observation."""

    raw_value: float
    observed_value: float | None = None
    execution_success: bool = True
    qc_passed: bool = True
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    objective_values: dict[str, float] = field(default_factory=dict)

    @property
    def optimizer_value(self) -> float:
        """Minimization value exposed to ordinary optimizers."""
        return self.raw_value if self.observed_value is None else self.observed_value

    def observation_objectives(self) -> dict[str, float]:
        """JSON-safe numeric metrics attached to an optimizer observation."""
        values = {
            "true_objective": -float(self.raw_value),
            "observed_objective": -float(self.optimizer_value),
            "execution_success": 1.0 if self.execution_success else 0.0,
            "qc_passed": 1.0 if self.qc_passed else 0.0,
        }
        values.update({str(k): float(v) for k, v in self.objective_values.items()})
        return values


# ---------------------------------------------------------------------------
# Space helpers
# ---------------------------------------------------------------------------


def _cont_space(bounds: dict[str, tuple[float, float]]) -> ParameterSpace:
    """Build a continuous ParameterSpace from {name: (min, max)}."""
    dims = tuple(
        SearchDimension(
            param_name=name,
            param_type="number",
            min_value=lo,
            max_value=hi,
        )
        for name, (lo, hi) in bounds.items()
    )
    return ParameterSpace(dimensions=dims, protocol_template={})


# ---------------------------------------------------------------------------
# Analytic objectives (textbook minimization form)
# ---------------------------------------------------------------------------


def _sphere(p: dict) -> float:
    """Sphere: f = sum(x_i^2). Separable, convex, unimodal. Min 0 at origin."""
    return sum(v * v for v in p.values())


def _branin(p: dict) -> float:
    """Branin-Hoo (2D). Min 0.397887 at three symmetric points.

    f(x1,x2) = a(x2 - b x1^2 + c x1 - r)^2 + s(1-t)cos(x1) + s
    """
    x1, x2 = p["x1"], p["x2"]
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


# Hartmann6 constants (standard).
_H6_ALPHA = (1.0, 1.2, 3.0, 3.2)
_H6_A = (
    (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
    (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
    (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
    (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
)
_H6_P = (
    (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
    (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
    (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
    (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
)


def _hartmann6(p: dict) -> float:
    """Hartmann 6D. Global min approx -3.32237 at the standard argmin."""
    x = [p[f"x{i + 1}"] for i in range(6)]
    total = 0.0
    for i in range(4):
        inner = sum(_H6_A[i][j] * (x[j] - _H6_P[i][j]) ** 2 for j in range(6))
        total += _H6_ALPHA[i] * math.exp(-inner)
    return -total


def _ackley(p: dict) -> float:
    """Ackley (n-D). Multimodal, near-separable. Min 0 at origin."""
    xs = list(p.values())
    n = len(xs)
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    sum_sq = sum(x * x for x in xs)
    sum_cos = sum(math.cos(c * x) for x in xs)
    term1 = -a * math.exp(-b * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    return term1 + term2 + a + math.e


def _rosenbrock(p: dict) -> float:
    """Rosenbrock 2D. Curved valley, unimodal. Min 0 at (1, 1)."""
    x, y = p["x1"], p["x2"]
    return 100.0 * (y - x * x) ** 2 + (1.0 - x) ** 2


def _rastrigin(p: dict) -> float:
    """Rastrigin (n-D). Highly multimodal, separable. Min 0 at origin."""
    xs = list(p.values())
    n = len(xs)
    return 10.0 * n + sum(x * x - 10.0 * math.cos(2.0 * math.pi * x) for x in xs)


def _mixed_categorical(p: dict) -> float:
    """Mixed problem: 1 categorical "shape" + 1 continuous x.

    Each shape picks a parabola centred at a different point; only the
    "circle" shape can reach the global minimum 0 at x = 0.5.
    Tests handling of categorical dimensions.
    """
    shape = p["shape"]
    x = p["x"]
    offset = {"circle": 0.0, "square": 1.5, "triangle": 3.0}.get(shape, 5.0)
    return (x - 0.5) ** 2 + offset


def _early_stage_controllability(p: dict) -> float:
    target = float(p["target_temp"])
    flow = float(p["flow_rate"])
    actual = _actual_temperature(target, flow)
    control_error = abs(actual - target)
    failure = control_error > 8.0
    return ((actual - 72.0) / 12.0) ** 2 + ((flow - 4.0) / 3.0) ** 2 + (4.0 if failure else 0.0)


def _early_stage_controllability_eval(p: dict) -> ProblemEvaluation:
    target = float(p["target_temp"])
    flow = float(p["flow_rate"])
    actual = _actual_temperature(target, flow)
    control_error = abs(actual - target)
    failure = control_error > 8.0
    raw = _early_stage_controllability(p)
    return ProblemEvaluation(
        raw_value=raw,
        observed_value=raw,
        execution_success=not failure,
        qc_passed=control_error <= 6.0,
        failure_type="controllability" if failure else None,
        metadata={
            "target_temp": target,
            "actual_temp": actual,
            "control_error": control_error,
        },
        objective_values={
            "actual_temp": actual,
            "control_error": control_error,
        },
    )


def _actual_temperature(target: float, flow: float) -> float:
    undershoot = max(0.0, target - 78.0) * 0.75 + max(0.0, flow - 6.0) * 1.8
    return target - undershoot


def _hardware_zone(p: dict) -> float:
    temp = float(p["temp"])
    pressure = float(p["pressure"])
    design = str(p["reactor_design"])
    failure = design == "thin_wall" and temp >= 78.0 and pressure >= 4.0
    design_offset = 0.0 if design == "thick_wall" else 0.25
    return ((temp - 70.0) / 12.0) ** 2 + ((pressure - 3.0) / 2.0) ** 2 + design_offset + (5.0 if failure else 0.0)


def _hardware_zone_eval(p: dict) -> ProblemEvaluation:
    raw = _hardware_zone(p)
    temp = float(p["temp"])
    pressure = float(p["pressure"])
    design = str(p["reactor_design"])
    failure = design == "thin_wall" and temp >= 78.0 and pressure >= 4.0
    return ProblemEvaluation(
        raw_value=raw,
        observed_value=raw,
        execution_success=not failure,
        qc_passed=not failure,
        failure_type="hardware" if failure else None,
        metadata={
            "reactor_design": design,
            "temp": temp,
            "pressure": pressure,
        },
    )


def _objective_uncertain_true(p: dict) -> float:
    temp = float(p["temp"])
    dwell = float(p["dwell_h"])
    additive = str(p["additive"])
    additive_penalty = 0.0 if additive == "stabilizer" else 0.8
    return ((temp - 62.0) / 10.0) ** 2 + ((dwell - 5.0) / 3.0) ** 2 + additive_penalty


def _objective_uncertain_proxy(p: dict) -> float:
    temp = float(p["temp"])
    dwell = float(p["dwell_h"])
    additive = str(p["additive"])
    additive_bonus = -0.35 if additive == "fast_yield" else 0.0
    return ((temp - 86.0) / 12.0) ** 2 + ((dwell - 1.2) / 2.5) ** 2 + additive_bonus


def _objective_uncertain_eval(p: dict) -> ProblemEvaluation:
    true_value = _objective_uncertain_true(p)
    proxy_value = _objective_uncertain_proxy(p)
    return ProblemEvaluation(
        raw_value=true_value,
        observed_value=proxy_value,
        execution_success=True,
        qc_passed=True,
        metadata={"proxy_objective": "fast_yield", "true_objective": "stability"},
        objective_values={
            "candidate_kpi_stability": -true_value,
            "candidate_kpi_fast_yield": -proxy_value,
        },
    )


def _batch_effect_true(p: dict) -> float:
    temp = float(p["temp"])
    hold = float(p["hold_h"])
    ligand = str(p["ligand"])
    protocol = str(p["screen_protocol"])
    ligand_penalty = 0.0 if ligand == "L2" else 0.7
    protocol_penalty = 0.0 if protocol == "replicate_qc" else 0.25
    return ((temp - 64.0) / 9.0) ** 2 + ((hold - 3.5) / 1.8) ** 2 + ligand_penalty + protocol_penalty


def _batch_effect_observed(p: dict) -> float:
    temp = float(p["temp"])
    hold = float(p["hold_h"])
    ligand = str(p["ligand"])
    protocol = str(p["screen_protocol"])
    fast_bias = -1.1 if protocol == "fast_screen" else 0.0
    ligand_bias = -0.25 if ligand == "L1" else 0.0
    return ((temp - 82.0) / 12.0) ** 2 + ((hold - 1.2) / 1.4) ** 2 + fast_bias + ligand_bias


def _batch_effect_eval(p: dict) -> ProblemEvaluation:
    true_value = _batch_effect_true(p)
    observed_value = _batch_effect_observed(p)
    protocol = str(p["screen_protocol"])
    return ProblemEvaluation(
        raw_value=true_value,
        observed_value=observed_value,
        execution_success=True,
        qc_passed=protocol == "replicate_qc",
        failure_type=None if protocol == "replicate_qc" else "batch_effect",
        metadata={
            "screen_protocol": protocol,
            "observed_bias": observed_value - true_value,
        },
        objective_values={
            "corrected_objective": -true_value,
            "batch_biased_objective": -observed_value,
        },
    )


def _prior_warm_start(p: dict) -> float:
    temp = float(p["temp"])
    residence = float(p["residence_min"])
    catalyst = str(p["catalyst"])
    solvent = str(p["solvent"])
    catalyst_penalty = {"cat_c": 0.0, "cat_b": 0.35, "cat_d": 0.8}.get(catalyst, 1.2)
    solvent_penalty = 0.0 if solvent == "polar" else 0.45
    return ((temp - 68.0) / 8.0) ** 2 + ((residence - 7.0) / 3.0) ** 2 + catalyst_penalty + solvent_penalty


def _early_stage_report(
    *,
    mode: str,
    risk_flags: list[str],
    recommendations: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    report = {
        "contract_version": "early_stage_system_characterization.v1",
        "recommended_campaign_mode": mode,
        "confidence": 0.9,
        "risk_flags": risk_flags,
        "diagnostic_recommendations": recommendations,
    }
    report.update(extra)
    return report


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, OptProblem]:
    problems: list[OptProblem] = []

    # Sphere 2D -- convex, separable, unimodal.
    sphere_space = _cont_space({"x1": (-5.0, 5.0), "x2": (-5.0, 5.0)})
    problems.append(
        OptProblem(
            id="sphere_2d",
            name="Sphere (2D)",
            space=sphere_space,
            objective=_sphere,
            optimum=0.0,
            optimum_x={"x1": 0.0, "x2": 0.0},
            tags=ProblemTags(
                n_dims=2,
                multimodal=False,
                noise_std=0.0,
                separable=True,
                has_categorical=False,
                surface_class="convex",
            ),
        )
    )

    # Branin -- 2D classic, multimodal (3 global minima).
    branin_space = _cont_space({"x1": (-5.0, 10.0), "x2": (0.0, 15.0)})
    problems.append(
        OptProblem(
            id="branin",
            name="Branin-Hoo (2D)",
            space=branin_space,
            objective=_branin,
            optimum=0.397887,
            optimum_x={"x1": math.pi, "x2": 2.275},
            tags=ProblemTags(
                n_dims=2,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=False,
                surface_class="multimodal",
            ),
        )
    )

    # Hartmann6 -- 6D, multimodal.
    h6_space = _cont_space({f"x{i + 1}": (0.0, 1.0) for i in range(6)})
    problems.append(
        OptProblem(
            id="hartmann6",
            name="Hartmann (6D)",
            space=h6_space,
            objective=_hartmann6,
            optimum=-3.32237,
            optimum_x={
                "x1": 0.20169,
                "x2": 0.150011,
                "x3": 0.476874,
                "x4": 0.275332,
                "x5": 0.311652,
                "x6": 0.6573,
            },
            tags=ProblemTags(
                n_dims=6,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=False,
                surface_class="multimodal",
            ),
        )
    )

    # Ackley 5D -- multimodal, near-separable.
    ackley_space = _cont_space({f"x{i + 1}": (-5.0, 5.0) for i in range(5)})
    problems.append(
        OptProblem(
            id="ackley_5d",
            name="Ackley (5D)",
            space=ackley_space,
            objective=_ackley,
            optimum=0.0,
            optimum_x={f"x{i + 1}": 0.0 for i in range(5)},
            tags=ProblemTags(
                n_dims=5,
                multimodal=True,
                noise_std=0.0,
                separable=True,
                has_categorical=False,
                surface_class="multimodal",
            ),
        )
    )

    # Rosenbrock 2D -- curved valley, unimodal.
    rosen_space = _cont_space({"x1": (-2.0, 2.0), "x2": (-1.0, 3.0)})
    problems.append(
        OptProblem(
            id="rosenbrock_2d",
            name="Rosenbrock (2D)",
            space=rosen_space,
            objective=_rosenbrock,
            optimum=0.0,
            optimum_x={"x1": 1.0, "x2": 1.0},
            tags=ProblemTags(
                n_dims=2,
                multimodal=False,
                noise_std=0.0,
                separable=False,
                has_categorical=False,
                surface_class="valley",
            ),
        )
    )

    # Rastrigin 2D -- highly multimodal, separable.
    rast_space = _cont_space({"x1": (-5.12, 5.12), "x2": (-5.12, 5.12)})
    problems.append(
        OptProblem(
            id="rastrigin_2d",
            name="Rastrigin (2D)",
            space=rast_space,
            objective=_rastrigin,
            optimum=0.0,
            optimum_x={"x1": 0.0, "x2": 0.0},
            tags=ProblemTags(
                n_dims=2,
                multimodal=True,
                noise_std=0.0,
                separable=True,
                has_categorical=False,
                surface_class="multimodal",
            ),
        )
    )

    # Mixed categorical -- 1 categorical + 1 continuous.
    mixed_space = ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="shape",
                param_type="categorical",
                choices=("circle", "square", "triangle"),
            ),
            SearchDimension(
                param_name="x",
                param_type="number",
                min_value=0.0,
                max_value=1.0,
            ),
        ),
        protocol_template={},
    )
    problems.append(
        OptProblem(
            id="mixed_categorical",
            name="Mixed Categorical (1 cat + 1 cont)",
            space=mixed_space,
            objective=_mixed_categorical,
            optimum=0.0,
            optimum_x={"shape": "circle", "x": 0.5},
            tags=ProblemTags(
                n_dims=2,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=True,
                surface_class="mixed",
            ),
        )
    )

    controllability_space = ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="target_temp",
                param_type="number",
                min_value=58.0,
                max_value=96.0,
            ),
            SearchDimension(
                param_name="flow_rate",
                param_type="number",
                min_value=1.0,
                max_value=9.0,
            ),
        ),
        protocol_template={
            "early_stage_report": _early_stage_report(
                mode="controllability_mapping",
                risk_flags=["poor_controllability", "target_reachability_low"],
                recommendations=[
                    {"action_type": "run_controllability_mapping", "priority": "high"},
                    {"action_type": "validate_target_reachability", "priority": "high"},
                ],
                target_feasibility_summary=[
                    {
                        "parameter": "target_temp",
                        "infeasible_target_min": 84.0,
                        "infeasible_target_max": 96.0,
                    }
                ],
                insights=["Actual temperature undershoots high target settings."],
            )
        },
    )
    problems.append(
        OptProblem(
            id="early_stage_controllability",
            name="Early-stage target-vs-actual controllability",
            space=controllability_space,
            objective=_early_stage_controllability,
            optimum=0.0,
            optimum_x={"target_temp": 72.0, "flow_rate": 4.0},
            tags=ProblemTags(
                n_dims=2,
                multimodal=False,
                noise_std=0.0,
                separable=False,
                has_categorical=False,
                surface_class="early_stage_controllability",
            ),
            evaluator=_early_stage_controllability_eval,
        )
    )

    hardware_space = ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="reactor_design",
                param_type="categorical",
                choices=("thin_wall", "thick_wall"),
            ),
            SearchDimension(
                param_name="temp",
                param_type="number",
                min_value=55.0,
                max_value=92.0,
            ),
            SearchDimension(
                param_name="pressure",
                param_type="number",
                min_value=1.0,
                max_value=6.0,
            ),
        ),
        protocol_template={
            "early_stage_report": _early_stage_report(
                mode="hardware_feasibility_discovery",
                risk_flags=["hardware_failures_dominate", "hardware_design_changed"],
                recommendations=[
                    {"action_type": "map_hardware_feasibility", "priority": "high"},
                    {"action_type": "shrink_or_annotate_action_space", "priority": "high"},
                ],
                feasibility_summary={
                    "danger_zones": [
                        {
                            "bounds": {
                                "reactor_design": ["thin_wall"],
                                "temp": [78.0, 92.0],
                                "pressure": [4.0, 6.0],
                            },
                            "reason": "thin_wall timeout and leak region",
                        }
                    ]
                },
                hardware_summary={"worst_design_id": "thin_wall"},
                insights=["Thin-wall hardware fails under high temp and pressure."],
            )
        },
    )
    problems.append(
        OptProblem(
            id="early_stage_hardware_zone",
            name="Early-stage hardware infeasible zone",
            space=hardware_space,
            objective=_hardware_zone,
            optimum=0.0,
            optimum_x={"reactor_design": "thick_wall", "temp": 70.0, "pressure": 3.0},
            tags=ProblemTags(
                n_dims=3,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=True,
                surface_class="early_stage_hardware",
            ),
            evaluator=_hardware_zone_eval,
        )
    )

    objective_space = ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="additive",
                param_type="categorical",
                choices=("fast_yield", "stabilizer"),
            ),
            SearchDimension(
                param_name="temp",
                param_type="number",
                min_value=45.0,
                max_value=95.0,
            ),
            SearchDimension(
                param_name="dwell_h",
                param_type="number",
                min_value=0.5,
                max_value=8.0,
            ),
        ),
        protocol_template={
            "early_stage_report": _early_stage_report(
                mode="objective_discovery",
                risk_flags=["objective_missing", "low_confidence_objective_candidates"],
                recommendations=[
                    {"action_type": "run_objective_discovery", "priority": "high"}
                ],
                candidate_kpis=[
                    {
                        "name": "candidate_kpi_stability",
                        "direction": "maximize",
                        "confidence": 0.86,
                    },
                    {
                        "name": "candidate_kpi_fast_yield",
                        "direction": "maximize",
                        "confidence": 0.42,
                    },
                ],
                insights=["Fast yield is a misleading proxy for stability."],
            )
        },
    )
    problems.append(
        OptProblem(
            id="early_stage_objective_uncertainty",
            name="Early-stage objective uncertainty",
            space=objective_space,
            objective=_objective_uncertain_true,
            optimum=0.0,
            optimum_x={"additive": "stabilizer", "temp": 62.0, "dwell_h": 5.0},
            tags=ProblemTags(
                n_dims=3,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=True,
                surface_class="early_stage_objective_uncertainty",
            ),
            evaluator=_objective_uncertain_eval,
        )
    )

    batch_effect_space = ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="screen_protocol",
                param_type="categorical",
                choices=("fast_screen", "replicate_qc"),
            ),
            SearchDimension(
                param_name="ligand",
                param_type="categorical",
                choices=("L1", "L2", "L3"),
            ),
            SearchDimension(
                param_name="temp",
                param_type="number",
                min_value=45.0,
                max_value=90.0,
            ),
            SearchDimension(
                param_name="hold_h",
                param_type="number",
                min_value=0.5,
                max_value=6.0,
            ),
        ),
        protocol_template={
            "early_stage_report": _early_stage_report(
                mode="data_quality_diagnostic",
                risk_flags=["low_data_quality", "batch_effect_detected"],
                recommendations=[
                    {"action_type": "run_data_quality_diagnostic", "priority": "high"},
                    {"action_type": "replicate_suspicious_hits", "priority": "high"},
                ],
                data_quality_summary={
                    "preferred_levels": {
                        "screen_protocol": "replicate_qc",
                        "ligand": "L2",
                    },
                    "biased_protocol": "fast_screen",
                },
                insights=["Fast-screen observations are batch-biased and overstate yield."],
            )
        },
    )
    problems.append(
        OptProblem(
            id="early_stage_batch_effect",
            name="Early-stage batch effect and data-quality drift",
            space=batch_effect_space,
            objective=_batch_effect_true,
            optimum=0.0,
            optimum_x={
                "screen_protocol": "replicate_qc",
                "ligand": "L2",
                "temp": 64.0,
                "hold_h": 3.5,
            },
            tags=ProblemTags(
                n_dims=4,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=True,
                surface_class="early_stage_batch_effect",
            ),
            evaluator=_batch_effect_eval,
        )
    )

    prior_space = ParameterSpace(
        dimensions=(
            SearchDimension(
                param_name="catalyst",
                param_type="categorical",
                choices=("cat_a", "cat_b", "cat_c", "cat_d"),
            ),
            SearchDimension(
                param_name="solvent",
                param_type="categorical",
                choices=("polar", "apolar"),
            ),
            SearchDimension(
                param_name="temp",
                param_type="number",
                min_value=35.0,
                max_value=95.0,
            ),
            SearchDimension(
                param_name="residence_min",
                param_type="number",
                min_value=1.0,
                max_value=20.0,
            ),
        ),
        protocol_template={
            "early_stage_report": _early_stage_report(
                mode="optimization_ready",
                risk_flags=["prior_case_similarity_high", "sparse_initial_data"],
                recommendations=[
                    {"action_type": "seed_from_prior_cases", "priority": "high"},
                    {"action_type": "proceed_with_guarded_optimization", "priority": "medium"},
                ],
                prior_successful_regions=[
                    {
                        "params": {
                            "catalyst": "cat_c",
                            "solvent": "polar",
                            "temp": 68.0,
                            "residence_min": 7.0,
                        },
                        "source": "similar-literature-case",
                        "confidence": 0.88,
                    },
                    {
                        "params": {
                            "catalyst": "cat_b",
                            "solvent": "polar",
                            "temp": 66.0,
                            "residence_min": 8.5,
                        },
                        "source": "neighboring-substrate-case",
                        "confidence": 0.72,
                    },
                ],
                insights=["Similar prior cases suggest a narrow productive region."],
            )
        },
    )
    problems.append(
        OptProblem(
            id="early_stage_prior_warm_start",
            name="Early-stage prior-case warm start",
            space=prior_space,
            objective=_prior_warm_start,
            optimum=0.0,
            optimum_x={
                "catalyst": "cat_c",
                "solvent": "polar",
                "temp": 68.0,
                "residence_min": 7.0,
            },
            tags=ProblemTags(
                n_dims=4,
                multimodal=True,
                noise_std=0.0,
                separable=False,
                has_categorical=True,
                surface_class="early_stage_prior_warm_start",
            ),
        )
    )

    return {p.id: p for p in problems}


_REGISTRY: dict[str, OptProblem] = _build_registry()


def get_problems() -> list[OptProblem]:
    """Return all registered problems (stable order)."""
    return list(_REGISTRY.values())


def get_problem(problem_id: str) -> OptProblem:
    """Return one problem by id. Raises ``KeyError`` if unknown."""
    return _REGISTRY[problem_id]
