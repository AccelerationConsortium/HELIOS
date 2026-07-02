"""Compatibility re-export for backend-selection utilities."""
from __future__ import annotations

from app.services.backend_selection import BackendScore, BackendSelection, rank_backends

__all__ = ["BackendScore", "BackendSelection", "rank_backends"]
