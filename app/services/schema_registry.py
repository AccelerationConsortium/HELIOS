"""Schema registry: SQLite schema introspection with LRU caching.

Provides structured schema context for the QueryAgent's NL → SQL compilation.
Uses the same caching pattern as primitives_registry.py.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.db import connection


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnInfo:
    """Column metadata from PRAGMA table_info()."""

    name: str
    dtype: str  # TEXT, INTEGER, REAL, BLOB, NUMERIC, ""
    is_pk: bool
    nullable: bool
    default_value: Any = None


# ---------------------------------------------------------------------------
# Core introspection
# ---------------------------------------------------------------------------

def _introspect_schema() -> dict[str, list[ColumnInfo]]:
    """Read all tables and their columns from sqlite_master + PRAGMA table_info."""
    schema: dict[str, list[ColumnInfo]] = {}
    with connection() as conn:
        # Include both tables AND views so the DB Agent can query canonical views
        entities = conn.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for row in entities:
            entity_name = row["name"]
            cols = conn.execute(f"PRAGMA table_info({entity_name})").fetchall()
            schema[entity_name] = [
                ColumnInfo(
                    name=c["name"],
                    dtype=c["type"].upper() if c["type"] else "TEXT",
                    is_pk=bool(c["pk"]),
                    nullable=not bool(c["notnull"]),
                    default_value=c["dflt_value"],
                )
                for c in cols
            ]
    return schema


# ---------------------------------------------------------------------------
# Cached accessors (same pattern as primitives_registry)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_schema() -> dict[str, list[ColumnInfo]]:
    """Return the full DB schema. Cached after first call."""
    return _introspect_schema()


def refresh_schema() -> dict[str, list[ColumnInfo]]:
    """Clear cached schema and reload from DB."""
    get_schema.cache_clear()
    _get_schema_version.cache_clear()
    return get_schema()


@lru_cache(maxsize=1)
def _get_schema_version() -> str:
    """Compute a deterministic hash of the current schema."""
    schema = get_schema()
    # Build a stable string representation
    parts: list[str] = []
    for table in sorted(schema.keys()):
        cols = schema[table]
        col_strs = [f"{c.name}:{c.dtype}:pk={c.is_pk}" for c in cols]
        parts.append(f"{table}({','.join(col_strs)})")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_schema_version() -> str:
    """Return a short hash representing the current schema version."""
    return _get_schema_version()


def get_table_names() -> list[str]:
    """Return sorted list of all table names."""
    return sorted(get_schema().keys())


def get_schema_context_for_prompt() -> str:
    """Format schema as text for LLM context injection.

    Returns a compact but complete listing of tables and columns,
    suitable for inclusion in a system prompt.
    """
    schema = get_schema()
    lines: list[str] = ["## Database Schema (SQLite)", ""]
    for table in sorted(schema.keys()):
        cols = schema[table]
        col_parts: list[str] = []
        for c in cols:
            pk_marker = " PK" if c.is_pk else ""
            null_marker = "" if c.nullable else " NOT NULL"
            col_parts.append(f"  {c.name} {c.dtype}{pk_marker}{null_marker}")
        lines.append(f"### {table}")
        lines.extend(col_parts)
        lines.append("")
    return "\n".join(lines)


def get_schema_as_dict() -> dict[str, list[dict[str, Any]]]:
    """Return schema as plain dicts (for JSON serialisation)."""
    schema = get_schema()
    return {
        table: [
            {
                "name": c.name,
                "dtype": c.dtype,
                "is_pk": c.is_pk,
                "nullable": c.nullable,
            }
            for c in cols
        ]
        for table, cols in schema.items()
    }
