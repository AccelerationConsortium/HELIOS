"""Compatibility facade for the services method advisor."""
from __future__ import annotations

from app.services import method_advisor as _impl

DEFAULT_DECISION_TABLE = _impl.DEFAULT_DECISION_TABLE
DECISION_TABLE_PATH = _impl.DECISION_TABLE_PATH
problem_profile = _impl.problem_profile


def load_decision_table(
    path: str | None = None,
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    return _impl.load_decision_table(DECISION_TABLE_PATH if path is None else path)


def recommend_backends(snapshot, diag=None) -> tuple[str, ...]:
    return _impl.recommend_backends(snapshot, diag, table_path=DECISION_TABLE_PATH)


__all__ = [
    "DEFAULT_DECISION_TABLE",
    "DECISION_TABLE_PATH",
    "load_decision_table",
    "problem_profile",
    "recommend_backends",
]
