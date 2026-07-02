"""Classical Design-of-Experiments backends (the project's "method gap" #7).

These are *one-shot* designs: ``suggest`` ignores ``observations`` and returns
a fixed grid of design points.  They conform to ``BackendProtocol`` and are
registered in the shared registry so the study runner treats them like any
other method.

- ``full_factorial``: the full product grid of per-dimension levels
  (``prod(levels)`` points), truncated to ``n``.
- ``fractional_factorial``: a reduced, deterministic subset of that grid
  (every k-th point), useful when the full grid is too large for the budget.

Pure Python -- always available.
"""
from __future__ import annotations

from typing import Any

from app.services.candidate_gen import ParameterSpace, sample_grid
from app.services.optimization_backends import Observation, register_backend


def _grid_levels(space: ParameterSpace, n: int) -> int:
    """Pick a per-dimension level count whose full grid is >= n (capped).

    Categorical/boolean dims enumerate all choices regardless; this only sets
    the resolution of continuous dims.  Bounded to [2, 7] to stay cheap.
    """
    n_cont = sum(
        1
        for d in space.dimensions
        if d.choices is None and d.param_type not in ("boolean",)
    )
    if n_cont <= 0:
        return 2
    levels = 2
    while levels < 7 and levels**n_cont < n:
        levels += 1
    return levels


@register_backend
class FullFactorialBackend:
    """Full-factorial DoE: the complete product grid of levels."""

    name = "full_factorial"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        levels: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        n_per_dim = levels if levels is not None else _grid_levels(space, n)
        grid = sample_grid(space, n_per_dim=n_per_dim)
        return grid[:n]

    @staticmethod
    def is_available() -> bool:
        return True


@register_backend
class FractionalFactorialBackend:
    """Fractional-factorial DoE: an evenly-spaced subset of the full grid.

    Takes the full grid and keeps every ``stride``-th point so the design still
    spans the space at a fraction of the cost.  ``stride`` is chosen so the
    kept count is close to ``n``.
    """

    name = "fractional_factorial"

    def suggest(
        self,
        space: ParameterSpace,
        n: int,
        observations: list[Observation],
        *,
        seed: int | None = None,
        levels: int | None = None,
        fraction: float = 0.5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Build a denser grid than full-factorial so the reduced subset still
        # offers >= n distinct points.
        n_target = max(n, 1)
        n_per_dim = levels if levels is not None else _grid_levels(
            space, int(n_target / max(fraction, 1e-6))
        )
        grid = sample_grid(space, n_per_dim=n_per_dim)
        if not grid:
            return []
        stride = max(1, len(grid) // n_target)
        reduced = grid[::stride]
        return reduced[:n]

    @staticmethod
    def is_available() -> bool:
        return True
