"""HELIOS optimization-intelligence integration layer.

HELIOS delegates *optimization intelligence* (algorithm portfolio, problem
profiling, candidate generation) to Nexus (``optimization_copilot``) while
retaining authority over the scientific campaign loop: validation, safety,
recovery, execution, and provenance.

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
