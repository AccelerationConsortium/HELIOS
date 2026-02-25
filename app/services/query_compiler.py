"""Deterministic DSL → SQL compiler (Layer 1 of double-layer query validation).

Translates an ExperimentQuery into a QueryPlan with fully-parameterised SQL.
No LLM is involved — every compilation is deterministic and fully auditable.

Guarantees:
  - All user-supplied values go through ? placeholders (no string interpolation)
  - ORDER BY / GROUP BY columns are validated against per-entity whitelists
  - LIMIT is capped by QueryConstraints.max_rows
"""
from __future__ import annotations

import hashlib
from typing import Any

from app.contracts.query_contract import ColumnSpec, QueryConstraints, QueryPlan, new_query_plan_id
from app.contracts.query_dsl import (
    ENTITY_SOURCE,
    AggSpec,
    ExperimentQuery,
    _VALID_COLUMNS,
    is_valid_column,
)
from app.core.db import utcnow_iso

_COMPILER_VERSION = "dsl-1.0"

# Default SELECT column list for each entity (no aggregation)
_DEFAULT_SELECT: dict[str, str] = {
    "runs": (
        "run_id, campaign_id, round_number, candidate_index, "
        "status, kpi_value, kpi_name, created_at"
    ),
    "candidates": (
        "candidate_id, campaign_id, round_number, candidate_index, "
        "params_json, predicted_kpi, status, created_at"
    ),
    "kpis": (
        "run_id, campaign_id, round_number, "
        "kpi_name, kpi_value, qc_status, created_at"
    ),
    "artifacts": (
        "artifact_id, run_id, campaign_id, "
        "artifact_type, uri, created_at"
    ),
    "rounds": (
        "round_id, campaign_id, round_number, "
        "n_candidates, best_kpi, mean_kpi, status, started_at, finished_at"
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_dsl(
    query: ExperimentQuery,
    constraints: QueryConstraints | None = None,
) -> QueryPlan:
    """Compile an ExperimentQuery into a parameterised QueryPlan.

    Parameters
    ----------
    query:
        The typed DSL query to compile.
    constraints:
        Optional execution constraints (max_rows, timeout_ms …).
        Defaults to ``QueryConstraints()``.

    Returns
    -------
    QueryPlan
        Ready to pass directly to ``QueryAgent._execute_readonly()``.

    Raises
    ------
    ValueError
        If a column used in ORDER BY or GROUP BY is not whitelisted for
        the given entity, or if an unknown aggregation func is requested.
    """
    if constraints is None:
        constraints = QueryConstraints()

    source = ENTITY_SOURCE[query.entity]
    params: list[Any] = []
    where_clauses: list[str] = []

    # --- WHERE clauses (all values via ? placeholders) ---------------------
    if query.campaign_id is not None:
        where_clauses.append("campaign_id = ?")
        params.append(query.campaign_id)

    if query.run_id is not None:
        where_clauses.append("run_id = ?")
        params.append(query.run_id)

    if query.round_number is not None:
        where_clauses.append("round_number = ?")
        params.append(query.round_number)
    else:
        if query.round_min is not None:
            where_clauses.append("round_number >= ?")
            params.append(query.round_min)
        if query.round_max is not None:
            where_clauses.append("round_number <= ?")
            params.append(query.round_max)

    if query.candidate_index is not None:
        where_clauses.append("candidate_index = ?")
        params.append(query.candidate_index)

    if query.status is not None:
        where_clauses.append("status = ?")
        params.append(query.status)

    if query.kpi_name is not None:
        where_clauses.append("kpi_name = ?")
        params.append(query.kpi_name)

    if query.kpi_min is not None:
        where_clauses.append("kpi_value >= ?")
        params.append(query.kpi_min)

    if query.kpi_max is not None:
        where_clauses.append("kpi_value <= ?")
        params.append(query.kpi_max)

    if query.qc_passed_only:
        # Literal constant — not user input, safe to embed directly
        where_clauses.append("qc_status = 'passed'")

    # --- SELECT + GROUP BY ------------------------------------------------
    if query.agg is not None:
        select_clause, group_by_clause, expected_cols = _build_agg_select(
            query.entity, query.agg
        )
    else:
        select_clause = _DEFAULT_SELECT[query.entity]
        group_by_clause = ""
        expected_cols = _infer_columns(select_clause)

    # --- ORDER BY (whitelist-validated) -----------------------------------
    order_clause = ""
    if query.order_by is not None:
        if not is_valid_column(query.entity, query.order_by):
            valid = sorted(_VALID_COLUMNS.get(query.entity, frozenset()))
            raise ValueError(
                f"Column '{query.order_by}' is not allowed for ORDER BY on "
                f"entity '{query.entity}'. Valid columns: {valid}"
            )
        order_clause = f"ORDER BY {query.order_by} {query.order_dir}"

    # --- Assemble SQL -----------------------------------------------------
    sql_parts = [f"SELECT {select_clause}", f"FROM {source}"]

    if where_clauses:
        sql_parts.append("WHERE " + " AND ".join(where_clauses))

    if group_by_clause:
        sql_parts.append(f"GROUP BY {group_by_clause}")

    if order_clause:
        sql_parts.append(order_clause)

    effective_limit = min(query.limit, constraints.max_rows)
    sql_parts.append(f"LIMIT {effective_limit}")

    sql = "\n".join(sql_parts)

    # --- Reproducibility key (hash of DSL JSON, not NL) ------------------
    dsl_json = query.model_dump_json()
    prompt_hash = hashlib.sha256(dsl_json.encode()).hexdigest()[:16]

    return QueryPlan(
        plan_id=new_query_plan_id(),
        sql=sql,
        params=params,
        expected_columns=expected_cols,
        constraints=constraints,
        prompt_hash=f"dsl:{prompt_hash}",
        schema_version="dsl",
        agent_version=_COMPILER_VERSION,
        created_at=utcnow_iso(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_agg_select(
    entity: str,
    agg: AggSpec,
) -> tuple[str, str, list[ColumnSpec]]:
    """Build SELECT + GROUP BY SQL fragments for an aggregation query.

    Returns
    -------
    (select_clause, group_by_clause, expected_columns)
    """
    # Validate group_by columns
    for col in agg.group_by:
        if not is_valid_column(entity, col):
            valid = sorted(_VALID_COLUMNS.get(entity, frozenset()))
            raise ValueError(
                f"Column '{col}' is not allowed in GROUP BY for entity '{entity}'. "
                f"Valid columns: {valid}"
            )

    # Validate aggregation target column (only when it matters)
    agg_col = agg.on or "kpi_value"
    if agg.func != "count" and agg.on and not is_valid_column(entity, agg.on):
        valid = sorted(_VALID_COLUMNS.get(entity, frozenset()))
        raise ValueError(
            f"Column '{agg.on}' is not allowed as aggregation target for "
            f"entity '{entity}'. Valid columns: {valid}"
        )

    # Build aggregation expression
    if agg.func == "count":
        agg_expr = "COUNT(*) AS count"
        agg_col_spec = ColumnSpec(name="count", dtype="INTEGER")
    elif agg.func == "mean":
        agg_expr = f"AVG({agg_col}) AS mean_{agg_col}"
        agg_col_spec = ColumnSpec(name=f"mean_{agg_col}", dtype="REAL")
    elif agg.func == "max":
        agg_expr = f"MAX({agg_col}) AS max_{agg_col}"
        agg_col_spec = ColumnSpec(name=f"max_{agg_col}", dtype="REAL")
    elif agg.func == "min":
        agg_expr = f"MIN({agg_col}) AS min_{agg_col}"
        agg_col_spec = ColumnSpec(name=f"min_{agg_col}", dtype="REAL")
    elif agg.func == "best":
        sql_func = "MAX" if agg.direction == "maximize" else "MIN"
        agg_expr = f"{sql_func}({agg_col}) AS best_{agg_col}"
        agg_col_spec = ColumnSpec(name=f"best_{agg_col}", dtype="REAL")
    else:
        raise ValueError(f"Unknown aggregation func: {agg.func!r}")

    # Prefix group_by columns into SELECT
    if agg.group_by:
        group_cols_str = ", ".join(agg.group_by)
        select_clause = f"{group_cols_str}, {agg_expr}"
        group_by_clause = group_cols_str
        expected_cols: list[ColumnSpec] = [
            ColumnSpec(name=gc, dtype="TEXT") for gc in agg.group_by
        ] + [agg_col_spec]
    else:
        select_clause = agg_expr
        group_by_clause = ""
        expected_cols = [agg_col_spec]

    return select_clause, group_by_clause, expected_cols


def _infer_columns(select_clause: str) -> list[ColumnSpec]:
    """Infer ColumnSpec list from a bare comma-separated column list string."""
    cols: list[ColumnSpec] = []
    for raw in select_clause.split(","):
        name = raw.strip().split()[-1]   # last token handles "alias AS name" form
        if name.endswith(("_number", "_index", "_count", "n_candidates")):
            dtype: str = "INTEGER"
        elif name.endswith(("_value", "best_kpi", "mean_kpi", "predicted_kpi")):
            dtype = "REAL"
        else:
            dtype = "TEXT"
        cols.append(ColumnSpec(name=name, dtype=dtype))  # type: ignore[arg-type]
    return cols
