"""Query contract: typed Pydantic models for the DB Retrieval Agent.

The QueryAgent is NOT a chatbot — it's a NL → QueryPlan compiler.
It produces only: SQL + params + expected_schema + constraints.
No free-text answers allowed.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.db import utcnow_iso


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_query_plan_id() -> str:
    return f"qp-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ColumnSpec(BaseModel):
    """Expected column in query result."""

    name: str
    dtype: Literal["TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"] = "TEXT"


class QueryConstraints(BaseModel):
    """Hard limits on query execution."""

    max_rows: int = Field(default=1000, ge=1, le=10000)
    timeout_ms: int = Field(default=5000, ge=100, le=30000)
    allowed_tables: list[str] = Field(default_factory=list)
    denied_tables: list[str] = Field(default_factory=list)
    require_where: bool = True
    max_joins: int = Field(default=3, ge=0, le=10)
    max_subquery_depth: int = Field(default=2, ge=0, le=5)


# ---------------------------------------------------------------------------
# QueryPlan — the ONLY compilation product
# ---------------------------------------------------------------------------

class QueryPlan(BaseModel):
    """Deterministic compilation product.

    This is the ONLY thing the QueryAgent produces.  It is fully
    serialisable, cacheable, and versioned for reproducibility.
    """

    plan_id: str = Field(default_factory=new_query_plan_id)
    sql: str  # parameterised SQL (? placeholders)
    params: list[Any] = Field(default_factory=list)
    expected_columns: list[ColumnSpec] = Field(default_factory=list)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)

    # Reproducibility keys
    prompt_hash: str = ""  # SHA-256 of normalised prompt
    schema_version: str = ""  # hash of current DB schema
    agent_version: str = "1.0"
    snapshot_id: str | None = None  # if created with snapshot_mode
    created_at: str = Field(default_factory=utcnow_iso)


# ---------------------------------------------------------------------------
# Agent IO
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Agent input — NL prompt + optional constraints."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)

    # Optional disambiguation context
    campaign_id: str | None = None
    run_id: str | None = None

    # Snapshot mode: capture result as immutable dataset
    snapshot_mode: bool = False
    snapshot_name: str | None = None


class QueryResult(BaseModel):
    """Agent output — structured data, never prose."""

    plan: QueryPlan
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    execution_ms: float = 0.0
    cache_hit: bool = False
    snapshot_id: str | None = None  # set when snapshot_mode=True
