"""HELIOS optimization-intelligence integration layer.

HELIOS can delegate *optimization intelligence* (algorithm portfolio, problem
profiling, candidate generation) to Nexus (``optimization_copilot``) while
retaining authority as the adaptive campaign decision layer: optimization
strategy, validation, safety, recovery, context acquisition, objective and
constraint handling, execution routing, and provenance.

Importing this package is safe even when Nexus is not installed -- the Nexus
backends simply report ``is_available() is False`` and HELIOS falls back to its
built-in optimizer.
"""
from __future__ import annotations

# Importing the bridges registers their backends in the shared
# optimization-backend registry (no-op effects beyond registration).
from app.optimization import bomcp_backend  # noqa: F401
from app.optimization import nexus_backend  # noqa: F401

__all__ = ["bomcp_backend", "nexus_backend"]
