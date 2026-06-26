"""Method-comparison benchmark layer.

Analytic test problems with KNOWN optima, classical-DoE backends, a study
runner, metrics, an aggregating scoreboard, a tag->method recommender and a
markdown/CSV reporter.  The package reuses the live backend registry in
``app.services.optimization_backends`` so every optimization method HELIOS
ships can be benchmarked on the same ruler.

Sign convention
---------------
The analytic ``OptProblem.objective`` returns the *raw* test-function value in
its textbook **minimization** form (e.g. ``sphere(0)==0``, ``branin``
min ``0.397887``).  ``OptProblem.optimum`` stores that same minimization value,
so regret = ``best_observed_f - optimum >= 0``.

HELIOS backends *maximize* (``Observation.objective`` is higher-is-better).
The runner therefore feeds ``-objective(params)`` to the backends and flips the
sign back when reading results -- the negation lives in one place
(:mod:`benchmarks.methods.runner`), not in the problem definitions.
"""
from __future__ import annotations

__version__ = "0.1.0"
