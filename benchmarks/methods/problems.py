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
from dataclasses import dataclass

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

    return {p.id: p for p in problems}


_REGISTRY: dict[str, OptProblem] = _build_registry()


def get_problems() -> list[OptProblem]:
    """Return all registered problems (stable order)."""
    return list(_REGISTRY.values())


def get_problem(problem_id: str) -> OptProblem:
    """Return one problem by id. Raises ``KeyError`` if unknown."""
    return _REGISTRY[problem_id]
