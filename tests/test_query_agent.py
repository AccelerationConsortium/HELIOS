"""Tests for the DB Retrieval Agent (NL → QueryPlan compiler).

Covers:
  - QueryContract Pydantic models
  - SchemaRegistry introspection + caching
  - SqlGuard whitelist validation
  - QueryPlanCache deterministic caching
  - QueryAgent full pipeline with MockProvider
  - Read-only enforcement
  - Reproducibility / determinism
"""
from __future__ import annotations

import json
import os
import tempfile

# Isolate test DB BEFORE any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_query_agent_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "query_agent_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import asyncio  # noqa: E402

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, run_txn  # noqa: E402


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    # Clear caches that depend on DB state
    from app.services.schema_registry import refresh_schema
    refresh_schema()


# ===========================================================================
# Test: QueryContract Pydantic models
# ===========================================================================


class TestQueryContract:
    """Validate Pydantic models for serialisation and defaults."""

    def test_query_request_defaults(self):
        from app.contracts.query_contract import QueryRequest

        req = QueryRequest(prompt="show me all runs")
        assert req.prompt == "show me all runs"
        assert req.constraints.max_rows == 1000
        assert req.constraints.timeout_ms == 5000
        assert req.constraints.require_where is True
        assert req.campaign_id is None

    def test_query_request_min_length(self):
        from app.contracts.query_contract import QueryRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryRequest(prompt="")

    def test_query_plan_serialisation_roundtrip(self):
        from app.contracts.query_contract import (
            ColumnSpec,
            QueryConstraints,
            QueryPlan,
        )

        plan = QueryPlan(
            sql="SELECT id FROM runs WHERE status = ?",
            params=["completed"],
            expected_columns=[ColumnSpec(name="id", dtype="TEXT")],
            constraints=QueryConstraints(max_rows=50),
            prompt_hash="abc123",
            schema_version="def456",
        )
        data = plan.model_dump()
        restored = QueryPlan(**data)
        assert restored.sql == plan.sql
        assert restored.params == plan.params
        assert restored.prompt_hash == plan.prompt_hash
        assert len(restored.expected_columns) == 1

    def test_query_result_structure(self):
        from app.contracts.query_contract import QueryPlan, QueryResult

        plan = QueryPlan(
            sql="SELECT 1",
            prompt_hash="x",
            schema_version="y",
        )
        result = QueryResult(
            plan=plan,
            rows=[{"1": 1}],
            row_count=1,
            truncated=False,
            execution_ms=1.5,
        )
        assert result.cache_hit is False
        assert result.row_count == 1

    def test_new_query_plan_id_format(self):
        from app.contracts.query_contract import new_query_plan_id

        pid = new_query_plan_id()
        assert pid.startswith("qp-")
        assert len(pid) == 15  # "qp-" + 12 hex chars


# ===========================================================================
# Test: SchemaRegistry
# ===========================================================================


class TestSchemaRegistry:
    """Schema introspection, caching, and version hashing."""

    def test_get_schema_returns_tables(self):
        from app.services.schema_registry import get_schema

        schema = get_schema()
        # Should have all 27+ tables
        assert len(schema) >= 20
        assert "runs" in schema
        assert "campaigns" in schema
        assert "run_kpis" in schema
        assert "campaign_state" in schema
        assert "query_plan_cache" in schema

    def test_schema_columns_for_runs(self):
        from app.services.schema_registry import get_schema

        schema = get_schema()
        runs_cols = {c.name for c in schema["runs"]}
        assert "id" in runs_cols
        assert "campaign_id" in runs_cols
        assert "status" in runs_cols
        assert "protocol_json" in runs_cols

    def test_get_schema_version_is_stable(self):
        from app.services.schema_registry import get_schema_version

        v1 = get_schema_version()
        v2 = get_schema_version()
        assert v1 == v2
        assert len(v1) == 16  # 16 hex chars

    def test_refresh_schema_invalidates_cache(self):
        from app.services.schema_registry import (
            get_schema,
            get_schema_version,
            refresh_schema,
        )

        v1 = get_schema_version()
        s1 = get_schema()
        refresh_schema()
        v2 = get_schema_version()
        s2 = get_schema()
        # Same schema, same version after refresh
        assert v1 == v2
        assert set(s1.keys()) == set(s2.keys())

    def test_get_table_names(self):
        from app.services.schema_registry import get_table_names

        names = get_table_names()
        assert isinstance(names, list)
        assert "runs" in names
        assert names == sorted(names)  # sorted

    def test_get_schema_context_for_prompt(self):
        from app.services.schema_registry import get_schema_context_for_prompt

        ctx = get_schema_context_for_prompt()
        assert "## Database Schema (SQLite)" in ctx
        assert "### runs" in ctx
        assert "id TEXT PK NOT NULL" in ctx

    def test_get_schema_as_dict(self):
        from app.services.schema_registry import get_schema_as_dict

        d = get_schema_as_dict()
        assert isinstance(d, dict)
        assert "runs" in d
        assert isinstance(d["runs"], list)
        assert d["runs"][0]["name"] == "id"


# ===========================================================================
# Test: SqlGuard
# ===========================================================================


class TestSqlGuard:
    """SQL whitelist validation."""

    def _constraints(self, **kwargs):
        from app.contracts.query_contract import QueryConstraints
        return QueryConstraints(**kwargs)

    def _tables(self):
        from app.services.schema_registry import get_table_names
        return set(get_table_names())

    def test_valid_select_passes(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT id, status FROM runs WHERE status = ?",
            ["completed"],
            self._constraints(),
            self._tables(),
        )
        assert result.valid
        assert result.violations == []

    def test_insert_blocked(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "INSERT INTO runs (id) VALUES (?)",
            ["test"],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid
        assert any("INSERT" in v for v in result.violations)

    def test_delete_blocked(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "DELETE FROM runs WHERE id = ?",
            ["test"],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid

    def test_drop_blocked(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "DROP TABLE runs",
            [],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid

    def test_update_blocked(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "UPDATE runs SET status = 'failed' WHERE id = ?",
            ["x"],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid

    def test_pragma_blocked(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "PRAGMA table_info(runs)",
            [],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid

    def test_dangerous_function_blocked(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT load_extension('evil.so') FROM runs WHERE id = ?",
            ["x"],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid
        assert any("load_extension" in v for v in result.violations)

    def test_join_limit(self):
        from app.services.sql_guard import validate_sql

        sql = (
            "SELECT r.id FROM runs r "
            "JOIN run_steps s1 ON r.id = s1.run_id "
            "JOIN run_kpis k ON r.id = k.run_id "
            "JOIN artifacts a ON r.id = a.run_id "
            "JOIN provenance_events p ON r.id = p.run_id "
            "WHERE r.status = ?"
        )
        result = validate_sql(
            sql, ["completed"],
            self._constraints(max_joins=3),
            self._tables(),
        )
        assert not result.valid
        assert any("JOIN" in v for v in result.violations)

    def test_join_within_limit_passes(self):
        from app.services.sql_guard import validate_sql

        sql = (
            "SELECT r.id FROM runs r "
            "JOIN run_steps s ON r.id = s.run_id "
            "JOIN run_kpis k ON r.id = k.run_id "
            "WHERE r.status = ?"
        )
        result = validate_sql(
            sql, ["completed"],
            self._constraints(max_joins=3),
            self._tables(),
        )
        assert result.valid

    def test_where_required(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT * FROM runs",
            [],
            self._constraints(require_where=True),
            self._tables(),
        )
        assert not result.valid
        assert any("WHERE" in v for v in result.violations)

    def test_where_not_required(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT COUNT(*) FROM runs",
            [],
            self._constraints(require_where=False),
            self._tables(),
        )
        assert result.valid

    def test_limit_auto_appended(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT id FROM runs WHERE status = ?",
            ["completed"],
            self._constraints(max_rows=50),
            self._tables(),
        )
        assert result.valid
        assert "LIMIT 50" in result.sql
        assert any("Auto-appended" in w for w in result.warnings)

    def test_existing_limit_preserved(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT id FROM runs WHERE status = ? LIMIT 10",
            ["completed"],
            self._constraints(max_rows=1000),
            self._tables(),
        )
        assert result.valid
        assert "LIMIT 10" in result.sql
        assert result.warnings == []

    def test_unknown_table_rejected(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT * FROM nonexistent_table WHERE id = ?",
            ["x"],
            self._constraints(),
            self._tables(),
        )
        assert not result.valid
        assert any("nonexistent_table" in v for v in result.violations)

    def test_denied_table_rejected(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT * FROM runs WHERE id = ?",
            ["x"],
            self._constraints(denied_tables=["runs"]),
            self._tables(),
        )
        assert not result.valid
        assert any("deny list" in v for v in result.violations)

    def test_allowed_tables_whitelist(self):
        from app.services.sql_guard import validate_sql

        result = validate_sql(
            "SELECT * FROM run_kpis WHERE run_id = ?",
            ["x"],
            self._constraints(allowed_tables=["runs"]),
            self._tables(),
        )
        assert not result.valid
        assert any("allow list" in v for v in result.violations)


# ===========================================================================
# Test: QueryPlanCache
# ===========================================================================


class TestQueryPlanCache:
    """Deterministic cache: store, retrieve, invalidate."""

    def test_compute_cache_key_deterministic(self):
        from app.services.query_plan_cache import compute_cache_key

        k1 = compute_cache_key("show runs", "v1", "1.0")
        k2 = compute_cache_key("show runs", "v1", "1.0")
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    def test_cache_key_changes_with_prompt(self):
        from app.services.query_plan_cache import compute_cache_key

        k1 = compute_cache_key("show runs", "v1", "1.0")
        k2 = compute_cache_key("show campaigns", "v1", "1.0")
        assert k1 != k2

    def test_cache_key_changes_with_schema(self):
        from app.services.query_plan_cache import compute_cache_key

        k1 = compute_cache_key("show runs", "v1", "1.0")
        k2 = compute_cache_key("show runs", "v2", "1.0")
        assert k1 != k2

    def test_prompt_normalisation(self):
        from app.services.query_plan_cache import compute_cache_key

        k1 = compute_cache_key("  Show  RUNS  ", "v1", "1.0")
        k2 = compute_cache_key("show runs", "v1", "1.0")
        assert k1 == k2

    def test_store_and_retrieve(self):
        from app.contracts.query_contract import QueryPlan
        from app.services.query_plan_cache import (
            compute_cache_key,
            get_cached_plan,
            store_plan,
        )

        key = compute_cache_key("test prompt", "v1", "1.0")
        plan = QueryPlan(
            sql="SELECT 1",
            prompt_hash=key,
            schema_version="v1",
        )
        store_plan(key, plan)

        cached = get_cached_plan(key)
        assert cached is not None
        assert cached.sql == "SELECT 1"
        assert cached.prompt_hash == key

    def test_cache_miss_returns_none(self):
        from app.services.query_plan_cache import get_cached_plan

        result = get_cached_plan("nonexistent_key_abc123")
        assert result is None

    def test_hit_count_increments(self):
        from app.contracts.query_contract import QueryPlan
        from app.services.query_plan_cache import (
            compute_cache_key,
            get_cached_plan,
            store_plan,
        )

        key = compute_cache_key("hit count test", "v1", "1.0")
        plan = QueryPlan(sql="SELECT 1", prompt_hash=key, schema_version="v1")
        store_plan(key, plan)

        # Access multiple times
        get_cached_plan(key)
        get_cached_plan(key)
        get_cached_plan(key)

        with connection() as conn:
            row = conn.execute(
                "SELECT hit_count FROM query_plan_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        # 1 from store + 3 from get = 4 total (store uses INSERT with hit_count=1)
        assert row["hit_count"] == 4

    def test_invalidate_by_schema(self):
        from app.contracts.query_contract import QueryPlan
        from app.services.query_plan_cache import (
            compute_cache_key,
            get_cached_plan,
            invalidate_by_schema,
            store_plan,
        )

        key = compute_cache_key("invalidate test", "old_v", "1.0")
        plan = QueryPlan(
            sql="SELECT 1",
            prompt_hash=key,
            schema_version="old_v",
        )
        store_plan(key, plan)

        # Invalidate entries with wrong schema version
        deleted = invalidate_by_schema("new_v")
        assert deleted >= 1

        # Cache miss after invalidation
        assert get_cached_plan(key) is None

    def test_clear_cache(self):
        from app.contracts.query_contract import QueryPlan
        from app.services.query_plan_cache import (
            cache_stats,
            clear_cache,
            compute_cache_key,
            store_plan,
        )

        for i in range(5):
            key = compute_cache_key(f"test {i}", "v1", "1.0")
            store_plan(key, QueryPlan(sql=f"SELECT {i}", prompt_hash=key, schema_version="v1"))

        assert cache_stats()["total_entries"] == 5
        cleared = clear_cache()
        assert cleared == 5
        assert cache_stats()["total_entries"] == 0


# ===========================================================================
# Test: QueryAgent (full pipeline with MockProvider)
# ===========================================================================


class TestQueryAgent:
    """Full agent pipeline tests with mock LLM."""

    def _make_mock_response(self, sql, params=None, columns=None):
        """Build a JSON string that the MockProvider will return."""
        data = {"sql": sql, "params": params or []}
        if columns:
            data["expected_columns"] = columns
        return json.dumps(data)

    def test_valid_query_returns_result(self):
        """Full pipeline: mock LLM → validate → execute → return rows."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        # Insert test data
        from app.core.db import run_txn, utcnow_iso
        now = utcnow_iso()
        def _seed(conn):
            conn.execute(
                "INSERT INTO runs (id, campaign_id, trigger_type, trigger_payload_json, "
                "session_key, status, protocol_json, inputs_json, policy_snapshot_json, "
                "created_by, created_at, updated_at) "
                "VALUES (?, NULL, 'manual', '{}', 'sess-1', 'completed', '{}', '{}', "
                "'{\"max_temp_c\": 95}', 'test', ?, ?)",
                ("run-001", now, now),
            )
        run_txn(_seed)

        mock_llm = MockProvider(responses=[
            self._make_mock_response(
                sql="SELECT id, status FROM runs WHERE status = ? LIMIT 10",
                params=["completed"],
                columns=[
                    {"name": "id", "dtype": "TEXT"},
                    {"name": "status", "dtype": "TEXT"},
                ],
            )
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        request = QueryRequest(prompt="show me completed runs")
        result = _run(agent.run(request))

        assert result.success
        assert result.output is not None
        assert result.output.row_count >= 1
        assert result.output.rows[0]["id"] == "run-001"
        assert result.output.cache_hit is False

    def test_cache_hit_skips_llm(self):
        """Second identical query should hit cache — no LLM call."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        mock_llm = MockProvider(responses=[
            self._make_mock_response(
                sql="SELECT COUNT(*) as cnt FROM runs WHERE status = ? LIMIT 100",
                params=["completed"],
            ),
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(prompt="count completed runs")

        # First call — uses LLM
        r1 = _run(agent.run(req))
        assert r1.success
        assert r1.output.cache_hit is False
        assert mock_llm.call_count == 1

        # Second call — same prompt → cache hit, no LLM call
        r2 = _run(agent.run(req))
        assert r2.success
        assert r2.output.cache_hit is True
        assert mock_llm.call_count == 1  # still 1

    def test_invalid_sql_from_llm_not_executed(self):
        """If LLM produces bad SQL (e.g. INSERT), agent rejects it."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        mock_llm = MockProvider(responses=[
            self._make_mock_response(
                sql="INSERT INTO runs (id) VALUES (?)",
                params=["evil"],
            ),
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(prompt="add a run")
        result = _run(agent.run(req))

        assert not result.success
        assert any("validation" in e.lower() or "INSERT" in e for e in result.errors)

    def test_invalid_json_from_llm_handled(self):
        """If LLM returns non-JSON, agent reports error."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        mock_llm = MockProvider(responses=[
            "This is not JSON at all, sorry!"
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(prompt="do something")
        result = _run(agent.run(req))

        assert not result.success
        assert any("JSON" in e for e in result.errors)

    def test_empty_prompt_rejected(self):
        """Empty prompt should fail validation."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryRequest(prompt="")

    def test_markdown_fenced_json_handled(self):
        """LLM sometimes wraps JSON in ```json fences — agent handles it."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        fenced = '```json\n{"sql": "SELECT 1 FROM runs WHERE id = ?", "params": ["x"]}\n```'
        mock_llm = MockProvider(responses=[fenced])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(prompt="test fenced json")
        result = _run(agent.run(req))

        assert result.success
        assert result.output.plan.sql.startswith("SELECT 1")

    def test_context_campaign_id_passed_to_llm(self):
        """campaign_id context should be included in LLM prompt."""
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        mock_llm = MockProvider(responses=[
            self._make_mock_response(
                sql="SELECT status FROM campaign_state WHERE campaign_id = ? LIMIT 100",
                params=["camp-001"],
            )
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(prompt="what is the campaign status", campaign_id="camp-001")
        result = _run(agent.run(req))

        assert result.success
        # Verify LLM received the context
        assert "camp-001" in mock_llm.last_call["messages"][0].content


# ===========================================================================
# Test: Read-only enforcement
# ===========================================================================


class TestReadOnlyEnforcement:
    """Verify that no write operations can sneak through."""

    def test_select_with_semicolon_insert_blocked(self):
        """Attempt SQL injection via semicolon — blocked."""
        from app.services.sql_guard import validate_sql
        from app.contracts.query_contract import QueryConstraints
        from app.services.schema_registry import get_table_names

        result = validate_sql(
            "SELECT 1; INSERT INTO runs (id) VALUES ('evil')",
            [],
            QueryConstraints(),
            set(get_table_names()),
        )
        assert not result.valid

    def test_subquery_with_write_blocked(self):
        """Write operation hidden in subquery — blocked."""
        from app.services.sql_guard import validate_sql
        from app.contracts.query_contract import QueryConstraints
        from app.services.schema_registry import get_table_names

        result = validate_sql(
            "SELECT * FROM (DELETE FROM runs) WHERE 1=1",
            [],
            QueryConstraints(),
            set(get_table_names()),
        )
        assert not result.valid


# ===========================================================================
# Test: Reproducibility
# ===========================================================================


class TestReproducibility:
    """Same prompt + same schema → identical cache key."""

    def test_same_inputs_same_key(self):
        from app.services.query_plan_cache import compute_cache_key
        from app.services.schema_registry import get_schema_version

        sv = get_schema_version()
        k1 = compute_cache_key("show me all completed runs", sv, "1.0")
        k2 = compute_cache_key("show me all completed runs", sv, "1.0")
        assert k1 == k2

    def test_whitespace_normalised(self):
        from app.services.query_plan_cache import compute_cache_key
        from app.services.schema_registry import get_schema_version

        sv = get_schema_version()
        k1 = compute_cache_key("  show   me   all   runs  ", sv, "1.0")
        k2 = compute_cache_key("show me all runs", sv, "1.0")
        assert k1 == k2

    def test_case_normalised(self):
        from app.services.query_plan_cache import compute_cache_key
        from app.services.schema_registry import get_schema_version

        sv = get_schema_version()
        k1 = compute_cache_key("SHOW ME ALL RUNS", sv, "1.0")
        k2 = compute_cache_key("show me all runs", sv, "1.0")
        assert k1 == k2

    def test_different_schema_version_different_key(self):
        from app.services.query_plan_cache import compute_cache_key

        k1 = compute_cache_key("show runs", "schema_v1", "1.0")
        k2 = compute_cache_key("show runs", "schema_v2", "1.0")
        assert k1 != k2

    def test_different_agent_version_different_key(self):
        from app.services.query_plan_cache import compute_cache_key

        k1 = compute_cache_key("show runs", "v1", "1.0")
        k2 = compute_cache_key("show runs", "v1", "2.0")
        assert k1 != k2
