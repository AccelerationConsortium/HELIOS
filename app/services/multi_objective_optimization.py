"""Multi-Objective Optimization - Pareto Front Computation.

Implements multi-objective optimization using NSGA-II inspired approach
for lab automation campaigns with multiple competing objectives.

Key Features:
1. **Pareto Dominance** - Non-dominated sorting of solutions
2. **Crowding Distance** - Diversity preservation in Pareto front
3. **Multi-Objective Convergence** - Hypervolume-based convergence detection
4. **Pure Python** - No external dependencies (no scipy/sklearn)

Typical use cases:
- Maximize yield AND minimize cost
- Maximize throughput AND minimize time
- Maximize quality AND maximize stability
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoSolution:
    """A solution in multi-objective space."""

    candidate_id: str  # Identifier for this solution
    objectives: tuple[float, ...]  # Objective values (e.g., kpi1, kpi2, ...)
    parameters: dict[str, Any] | None = None  # Optional: decision variables
    rank: int = 0  # Pareto rank (0 = non-dominated front)
    crowding_distance: float = 0.0  # Crowding distance for diversity


@dataclass
class ParetoFront:
    """Collection of non-dominated solutions forming Pareto front."""

    solutions: list[ParetoSolution]
    objectives_names: list[str]  # Names of objectives (e.g., ["kpi1", "kpi2"])
    maximize: list[bool]  # Whether to maximize each objective
    hypervolume: float | None = None  # Hypervolume indicator (quality metric)

    def __post_init__(self):
        """Sort solutions by crowding distance (descending)."""
        self.solutions.sort(key=lambda s: s.crowding_distance, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_solutions": len(self.solutions),
            "objectives_names": self.objectives_names,
            "maximize": self.maximize,
            "hypervolume": self.hypervolume,
            "solutions": [
                {
                    "candidate_id": sol.candidate_id,
                    "objectives": list(sol.objectives),
                    "rank": sol.rank,
                    "crowding_distance": sol.crowding_distance,
                }
                for sol in self.solutions
            ],
        }


# ---------------------------------------------------------------------------
# Pareto Dominance
# ---------------------------------------------------------------------------


def dominates(
    a: tuple[float, ...],
    b: tuple[float, ...],
    maximize: list[bool],
) -> bool:
    """Check if solution A dominates solution B.

    A dominates B if:
    - A is at least as good as B in all objectives
    - A is strictly better than B in at least one objective

    Args:
        a: Objective values for solution A
        b: Objective values for solution B
        maximize: Whether to maximize each objective

    Returns:
        True if A dominates B
    """
    if len(a) != len(b) or len(a) != len(maximize):
        raise ValueError("Objective vectors must have same length")

    at_least_as_good = True
    strictly_better = False

    for obj_a, obj_b, maximize_obj in zip(a, b, maximize):
        if maximize_obj:
            # Maximization: A dominates if obj_a >= obj_b
            if obj_a < obj_b:
                at_least_as_good = False
                break
            if obj_a > obj_b:
                strictly_better = True
        else:
            # Minimization: A dominates if obj_a <= obj_b
            if obj_a > obj_b:
                at_least_as_good = False
                break
            if obj_a < obj_b:
                strictly_better = True

    return at_least_as_good and strictly_better


def non_dominated_sort(
    solutions: list[ParetoSolution],
    maximize: list[bool],
) -> list[list[ParetoSolution]]:
    """Non-dominated sorting (NSGA-II).

    Partitions solutions into Pareto fronts (ranks).

    Args:
        solutions: List of solutions to sort
        maximize: Whether to maximize each objective

    Returns:
        List of fronts, where front[0] is non-dominated front
    """
    n = len(solutions)

    # Domination count and dominated set for each solution
    domination_count = [0] * n  # How many solutions dominate this one
    dominated_by: list[list[int]] = [[] for _ in range(n)]  # Indices dominated by this solution

    # Compute domination relationships
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(solutions[i].objectives, solutions[j].objectives, maximize):
                # i dominates j
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(solutions[j].objectives, solutions[i].objectives, maximize):
                # j dominates i
                dominated_by[j].append(i)
                domination_count[i] += 1

    # Extract fronts
    fronts: list[list[ParetoSolution]] = []
    current_front_indices = [i for i in range(n) if domination_count[i] == 0]

    while current_front_indices:
        # Current front
        current_front = [solutions[i] for i in current_front_indices]
        fronts.append(current_front)

        # Find next front
        next_front_indices = []
        for i in current_front_indices:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front_indices.append(j)

        current_front_indices = next_front_indices

    return fronts


# ---------------------------------------------------------------------------
# Crowding Distance
# ---------------------------------------------------------------------------


def compute_crowding_distance(
    solutions: list[ParetoSolution],
) -> list[ParetoSolution]:
    """Compute crowding distance for solutions.

    Crowding distance measures how close a solution is to its neighbors.
    Higher distance = more isolated = better for diversity.

    Args:
        solutions: Solutions in the same front

    Returns:
        Solutions with updated crowding_distance attribute
    """
    n = len(solutions)

    if n <= 2:
        # Boundary solutions get infinite distance
        return [
            ParetoSolution(
                candidate_id=sol.candidate_id,
                objectives=sol.objectives,
                parameters=sol.parameters,
                rank=sol.rank,
                crowding_distance=float('inf'),
            )
            for sol in solutions
        ]

    n_objectives = len(solutions[0].objectives)

    # Initialize distances to 0
    distances = [0.0] * n

    # For each objective
    for obj_idx in range(n_objectives):
        # Sort solutions by this objective
        sorted_indices = sorted(
            range(n),
            key=lambda i: solutions[i].objectives[obj_idx]
        )

        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Objective range
        obj_min = solutions[sorted_indices[0]].objectives[obj_idx]
        obj_max = solutions[sorted_indices[-1]].objectives[obj_idx]

        obj_range = obj_max - obj_min
        if obj_range < 1e-10:
            obj_range = 1.0  # Avoid division by zero

        # Compute crowding distance for interior solutions
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            if not math.isinf(distances[idx]):
                # Distance = sum of normalized distances to neighbors
                prev_obj = solutions[sorted_indices[i - 1]].objectives[obj_idx]
                next_obj = solutions[sorted_indices[i + 1]].objectives[obj_idx]
                distances[idx] += (next_obj - prev_obj) / obj_range

    # Create new solutions with updated crowding distance
    return [
        ParetoSolution(
            candidate_id=sol.candidate_id,
            objectives=sol.objectives,
            parameters=sol.parameters,
            rank=sol.rank,
            crowding_distance=distances[i],
        )
        for i, sol in enumerate(solutions)
    ]


# ---------------------------------------------------------------------------
# Pareto Front Extraction
# ---------------------------------------------------------------------------


def compute_pareto_front(
    solutions: list[ParetoSolution],
    objectives_names: list[str],
    maximize: list[bool],
) -> ParetoFront:
    """Compute Pareto front from set of solutions.

    Args:
        solutions: Candidate solutions
        objectives_names: Names of objectives
        maximize: Whether to maximize each objective

    Returns:
        ParetoFront with non-dominated solutions
    """
    if not solutions:
        return ParetoFront(
            solutions=[],
            objectives_names=objectives_names,
            maximize=maximize,
            hypervolume=0.0,
        )

    # Non-dominated sorting
    fronts = non_dominated_sort(solutions, maximize)

    # First front is the Pareto front
    pareto_front = fronts[0] if fronts else []

    # Assign ranks
    pareto_solutions = [
        ParetoSolution(
            candidate_id=sol.candidate_id,
            objectives=sol.objectives,
            parameters=sol.parameters,
            rank=0,
            crowding_distance=0.0,
        )
        for sol in pareto_front
    ]

    # Compute crowding distance
    pareto_solutions = compute_crowding_distance(pareto_solutions)

    # Compute hypervolume (quality metric)
    hypervolume = compute_hypervolume(pareto_solutions, maximize)

    return ParetoFront(
        solutions=pareto_solutions,
        objectives_names=objectives_names,
        maximize=maximize,
        hypervolume=hypervolume,
    )


# ---------------------------------------------------------------------------
# Hypervolume Indicator
# ---------------------------------------------------------------------------


def compute_hypervolume(
    solutions: list[ParetoSolution],
    maximize: list[bool],
    reference_point: tuple[float, ...] | None = None,
) -> float:
    """Compute hypervolume indicator for Pareto front.

    Hypervolume measures the volume of objective space dominated by the front.
    Higher hypervolume = better front quality.

    Uses simplified 2D/3D computation (not full WFG algorithm).

    Args:
        solutions: Solutions in Pareto front
        maximize: Whether to maximize each objective
        reference_point: Reference point for hypervolume (default: auto)

    Returns:
        Hypervolume value
    """
    if not solutions:
        return 0.0

    n_objectives = len(solutions[0].objectives)

    # Auto-compute reference point if not provided
    if reference_point is None:
        reference_point = _compute_reference_point(solutions, maximize)

    # For 2D, use simple area computation
    if n_objectives == 2:
        return _compute_hypervolume_2d(solutions, maximize, reference_point)

    # For 3D, use layered area approach
    elif n_objectives == 3:
        return _compute_hypervolume_3d(solutions, maximize, reference_point)

    # For higher dimensions, use Monte Carlo approximation
    else:
        return _compute_hypervolume_monte_carlo(solutions, maximize, reference_point)


def _compute_reference_point(
    solutions: list[ParetoSolution],
    maximize: list[bool],
) -> tuple[float, ...]:
    """Compute reference point (nadir point with margin)."""
    n_objectives = len(solutions[0].objectives)
    reference = []

    for obj_idx in range(n_objectives):
        obj_values = [sol.objectives[obj_idx] for sol in solutions]

        if maximize[obj_idx]:
            # For maximization: reference = min - margin
            ref = min(obj_values) - 0.1 * abs(max(obj_values) - min(obj_values))
        else:
            # For minimization: reference = max + margin
            ref = max(obj_values) + 0.1 * abs(max(obj_values) - min(obj_values))

        reference.append(ref)

    return tuple(reference)


def _compute_hypervolume_2d(
    solutions: list[ParetoSolution],
    maximize: list[bool],
    reference_point: tuple[float, ...],
) -> float:
    """2D hypervolume computation (exact)."""
    # Sort by first objective
    sorted_sols = sorted(solutions, key=lambda s: s.objectives[0])

    hypervolume = 0.0
    prev_x = reference_point[0]

    for sol in sorted_sols:
        x = sol.objectives[0]
        y = sol.objectives[1]

        # Rectangle area
        if maximize[0]:
            width = x - prev_x
        else:
            width = prev_x - x

        if maximize[1]:
            height = y - reference_point[1]
        else:
            height = reference_point[1] - y

        if width > 0 and height > 0:
            hypervolume += width * height

        prev_x = x

    return hypervolume


def _compute_hypervolume_3d(
    solutions: list[ParetoSolution],
    maximize: list[bool],
    reference_point: tuple[float, ...],
) -> float:
    """3D hypervolume computation (layered approach)."""
    # Simplified: sum of 2D slices
    # Sort by first objective
    sorted_sols = sorted(solutions, key=lambda s: s.objectives[0])

    hypervolume = 0.0
    prev_x = reference_point[0]

    for sol in sorted_sols:
        x = sol.objectives[0]

        # Slice thickness
        if maximize[0]:
            thickness = x - prev_x
        else:
            thickness = prev_x - x

        if thickness > 0:
            # Compute 2D area in y-z plane
            y = sol.objectives[1]
            z = sol.objectives[2]

            if maximize[1]:
                height_y = y - reference_point[1]
            else:
                height_y = reference_point[1] - y

            if maximize[2]:
                height_z = z - reference_point[2]
            else:
                height_z = reference_point[2] - z

            if height_y > 0 and height_z > 0:
                area = height_y * height_z
                hypervolume += thickness * area

        prev_x = x

    return hypervolume


def _compute_hypervolume_monte_carlo(
    solutions: list[ParetoSolution],
    maximize: list[bool],
    reference_point: tuple[float, ...],
    n_samples: int = 10000,
) -> float:
    """Monte Carlo hypervolume approximation for high dimensions."""
    import random

    n_objectives = len(solutions[0].objectives)

    # Compute bounding box
    bounds = []
    for obj_idx in range(n_objectives):
        obj_values = [sol.objectives[obj_idx] for sol in solutions]
        bounds.append((min(obj_values), max(obj_values)))

    # Monte Carlo sampling
    dominated_count = 0

    for _ in range(n_samples):
        # Random point in bounding box
        point = tuple(
            random.uniform(bounds[i][0], bounds[i][1])
            for i in range(n_objectives)
        )

        # Check if dominated by any solution
        for sol in solutions:
            if dominates(sol.objectives, point, maximize):
                dominated_count += 1
                break

    # Hypervolume approximation
    box_volume = 1.0
    for i in range(n_objectives):
        box_volume *= bounds[i][1] - bounds[i][0]

    return (dominated_count / n_samples) * box_volume


# ---------------------------------------------------------------------------
# Multi-Objective Convergence Detection
# ---------------------------------------------------------------------------


@dataclass
class MultiObjectiveConvergenceStatus:
    """Convergence status for multi-objective optimization."""

    hypervolume_history: list[float]
    current_hypervolume: float
    improvement_rate: float  # Recent hypervolume improvement
    converged: bool
    confidence: float


def detect_multi_objective_convergence(
    hypervolume_history: list[float],
    window_size: int = 5,
    improvement_threshold: float = 0.01,
) -> MultiObjectiveConvergenceStatus:
    """Detect convergence in multi-objective optimization.

    Uses hypervolume improvement rate as convergence indicator.

    Args:
        hypervolume_history: History of hypervolume values
        window_size: Window for computing improvement rate
        improvement_threshold: Threshold for convergence (e.g., 1% improvement)

    Returns:
        MultiObjectiveConvergenceStatus
    """
    if len(hypervolume_history) < window_size:
        return MultiObjectiveConvergenceStatus(
            hypervolume_history=hypervolume_history,
            current_hypervolume=hypervolume_history[-1] if hypervolume_history else 0.0,
            improvement_rate=float('inf'),
            converged=False,
            confidence=0.0,
        )

    current_hv = hypervolume_history[-1]
    recent_hvs = hypervolume_history[-window_size:]

    # Compute improvement rate
    first_hv = recent_hvs[0]
    if first_hv < 1e-10:
        improvement_rate = float('inf')
    else:
        improvement_rate = (current_hv - first_hv) / first_hv

    # Converged if improvement rate below threshold
    converged = improvement_rate < improvement_threshold

    # Confidence based on consistency
    improvements = [
        (hypervolume_history[i] - hypervolume_history[i-1]) / max(1e-10, hypervolume_history[i-1])
        for i in range(len(hypervolume_history) - window_size + 1, len(hypervolume_history))
    ]

    # Confidence: 1.0 if all improvements < threshold
    confidence = sum(1 for imp in improvements if imp < improvement_threshold) / len(improvements)

    return MultiObjectiveConvergenceStatus(
        hypervolume_history=hypervolume_history,
        current_hypervolume=current_hv,
        improvement_rate=improvement_rate,
        converged=converged,
        confidence=confidence,
    )
