"""DB Retrieval Agent: NL → QueryPlan compiler.

This agent is NOT a chatbot that answers questions — it is a strict
NL → SQL QueryPlan compiler.  It produces only:
  SQL + params + expected_columns + constraints

All LLM output goes through SQL AST validation (SqlGuard) before
any query touches the database.  Execution is always read-only.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel

from app.agents.base import BaseAgent
from app.contracts.query_contract import (
    ColumnSpec,
    QueryConstraints,
    QueryPlan,
    QueryRequest,
    QueryResult,
    new_query_plan_id,
)
from app.core.db import connection, utcnow_iso
from app.services.llm_gateway import LLMMessage, LLMProvider, get_llm_provider
from app.services.query_plan_cache import compute_cache_key, get_cached_plan, store_plan
from app.services.schema_registry import (
    get_schema,
    get_schema_context_for_prompt,
    get_schema_version,
    get_table_names,
)
from app.services.sql_guard import validate_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for LLM-based NL → SQL compilation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a SQL compiler for a SQLite database.  You receive a natural-language
query and produce ONLY a JSON object — no explanations, no markdown.

{schema_context}

## Output Format (JSON only)

{{
  "sql": "SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT ...",
  "params": [],
  "expected_columns": [
    {{"name": "col_name", "dtype": "TEXT|INTEGER|REAL|BLOB|NUMERIC"}}
  ]
}}

## Canonical Views (PREFER these over raw tables)

- `v_experiment_runs` — unified run/campaign entity with experiment index dimensions
- `v_experiment_params` — parameters with type/unit metadata from param_schema
- `v_experiment_metrics` — KPI values with QC flags and review verdicts
- `v_experiment_artifacts` — file references (artifact_id, run_id, type, uri, hash)

## Rules

1. Use parameterised queries with ? placeholders for user-provided values.
2. ALWAYS include a LIMIT clause (default 100 unless the user specifies otherwise).
3. ALWAYS include a WHERE clause when querying large tables.
4. Only SELECT statements — never INSERT, UPDATE, DELETE, or DDL.
5. Maximum 3 JOINs per query.
6. JSON columns end with _json — use json_extract() to query inside them.
7. Output ONLY valid JSON.  No markdown fences, no explanations.
8. PREFER canonical views (v_experiment_*) over raw tables when possible.
9. Use qc_flags to filter suspect/failed data points.
"""


# ---------------------------------------------------------------------------
# QueryAgent
# ---------------------------------------------------------------------------


class QueryAgent(BaseAgent[QueryRequest, QueryResult]):
    """NL → QueryPlan compiler agent.

    Pipeline:
      1. Schema introspection (cached)
      2. Cache lookup (prompt_hash + schema_version)
      3. LLM compilation (if cache miss)
      4. SQL validation (SqlGuard)
      5. Read-only execution
      6. Return typed QueryResult
    """

    name = "query_agent"
    description = "NL → SQL QueryPlan compiler with read-only execution"
    layer = "cross-cutting"

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        super().__init__()
        self._llm = llm_provider

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    # -----------------------------------------------------------------------
    # BaseAgent interface
    # -----------------------------------------------------------------------

    def validate_input(self, input_data: QueryRequest) -> list[str]:
        errors: list[str] = []
        if not input_data.prompt or not input_data.prompt.strip():
            errors.append("prompt must be non-empty")
        if input_data.constraints.max_rows < 1:
            errors.append("max_rows must be >= 1")
        if input_data.constraints.timeout_ms < 100:
            errors.append("timeout_ms must be >= 100")
        return errors

    async def process(self, input_data: QueryRequest) -> QueryResult:
        """Main compilation + execution pipeline."""

        # 1. Schema introspection (cached)
        schema = get_schema()
        schema_ver = get_schema_version()
        known_tables = set(get_table_names())

        # 2. Compute cache key
        cache_key = compute_cache_key(
            input_data.prompt, schema_ver, "1.0",
        )

        # 3. Cache lookup
        cached = get_cached_plan(cache_key)
        if cached is not None:
            # Execute cached plan directly
            rows, exec_ms, truncated = self._execute_readonly(
                cached.sql, cached.params, input_data.constraints,
            )
            snap_id = self._maybe_snapshot(
                input_data, rows, cached.plan_id, cache_key,
            )
            return QueryResult(
                plan=cached,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                execution_ms=exec_ms,
                cache_hit=True,
                snapshot_id=snap_id,
            )

        # 4. LLM compilation (NL → SQL)
        plan = await self._compile_nl_to_plan(
            input_data, schema_ver, cache_key,
        )

        # 5. SQL validation
        validation = validate_sql(
            plan.sql, plan.params, input_data.constraints, known_tables,
        )
        if not validation.valid:
            raise ValueError(
                f"Generated SQL failed validation: {validation.violations}"
            )
        # Use potentially-modified SQL (e.g. LIMIT appended)
        plan.sql = validation.sql

        # 6. Store in cache
        store_plan(cache_key, plan)

        # 7. Execute
        rows, exec_ms, truncated = self._execute_readonly(
            plan.sql, plan.params, input_data.constraints,
        )

        # 8. Snapshot (if requested)
        snap_id = self._maybe_snapshot(
            input_data, rows, plan.plan_id, cache_key,
        )
        if snap_id:
            plan.snapshot_id = snap_id

        return QueryResult(
            plan=plan,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            execution_ms=exec_ms,
            cache_hit=False,
            snapshot_id=snap_id,
        )

    # -----------------------------------------------------------------------
    # Internal: NL → QueryPlan compilation
    # -----------------------------------------------------------------------

    async def _compile_nl_to_plan(
        self,
        request: QueryRequest,
        schema_version: str,
        cache_key: str,
    ) -> QueryPlan:
        """Use LLM to compile natural language → QueryPlan."""

        schema_context = get_schema_context_for_prompt()
        system = _SYSTEM_PROMPT_TEMPLATE.format(schema_context=schema_context)

        # Build user message with context
        user_parts: list[str] = [request.prompt]
        if request.campaign_id:
            user_parts.append(f"\nContext: campaign_id = '{request.campaign_id}'")
        if request.run_id:
            user_parts.append(f"\nContext: run_id = '{request.run_id}'")

        messages = [LLMMessage(role="user", content="\n".join(user_parts))]

        response = await self.llm.complete(
            messages=messages,
            system=system,
        )

        # Parse LLM response as JSON
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}"
            ) from e

        # Validate required fields
        if "sql" not in data:
            raise ValueError("LLM response missing 'sql' field")

        expected_columns = [
            ColumnSpec(**col) for col in data.get("expected_columns", [])
        ]

        prompt_hash = cache_key  # cache_key already includes prompt hash

        return QueryPlan(
            plan_id=new_query_plan_id(),
            sql=data["sql"],
            params=data.get("params", []),
            expected_columns=expected_columns,
            constraints=request.constraints,
            prompt_hash=prompt_hash,
            schema_version=schema_version,
            agent_version="1.0",
            created_at=utcnow_iso(),
        )

    # -----------------------------------------------------------------------
    # Internal: read-only execution
    # -----------------------------------------------------------------------

    @staticmethod
    def _execute_readonly(
        sql: str,
        params: list[Any],
        constraints: QueryConstraints,
    ) -> tuple[list[dict[str, Any]], float, bool]:
        """Execute SQL in a read-only transaction.

        Returns (rows, execution_ms, truncated).
        """
        start = time.monotonic()
        with connection() as conn:
            # Set query timeout
            conn.execute(
                f"PRAGMA busy_timeout = {constraints.timeout_ms}"
            )
            try:
                cursor = conn.execute(sql, params)
                rows = [dict(r) for r in cursor.fetchall()]
            except Exception:
                raise
            # Never commit — this is pure read
        elapsed_ms = (time.monotonic() - start) * 1000

        truncated = len(rows) >= constraints.max_rows
        return rows, elapsed_ms, truncated

    # -----------------------------------------------------------------------
    # Internal: snapshot creation
    # -----------------------------------------------------------------------

    @staticmethod
    def _maybe_snapshot(
        request: QueryRequest,
        rows: list[dict[str, Any]],
        plan_id: str,
        cache_key: str,
    ) -> str | None:
        """Create a dataset snapshot if snapshot_mode is enabled."""
        if not request.snapshot_mode:
            return None
        try:
            from app.services.dataset_snapshot import create_snapshot
            snap = create_snapshot(
                rows=rows,
                query_plan_hash=cache_key,
                campaign_id=request.campaign_id,
                query_plan_id=plan_id,
                snapshot_name=request.snapshot_name,
            )
            return snap["snapshot_id"]
        except Exception as e:
            logger.warning("Snapshot creation failed: %s", e)
            return None
