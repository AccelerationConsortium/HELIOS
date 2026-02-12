"""Tests for DB Retrieval Agent infrastructure (Reqs 1-6).

Covers:
  Req 1: Canonical views (v_experiment_*)
  Req 2: Dataset snapshots & watermark
  Req 3: QueryAgent snapshot_mode
  Req 4: QC flags, failure signatures, metric dictionary
  Req 5: Parameter schema
  Req 6: Experiment index dimensions
"""
from __future__ import annotations

import json
import os
import tempfile

# Isolate test DB BEFORE any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_db_agent_infra_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "db_agent_infra_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import asyncio  # noqa: E402

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, json_dumps, run_txn, utcnow_iso  # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _setup_db():
    get_settings.cache_clear()
    init_db()
    from app.services.schema_registry import refresh_schema
    refresh_schema()


def _seed_run(run_id="run-001", campaign_id=None, status="completed"):
    """Insert a minimal run for testing."""
    now = utcnow_iso()
    def _txn(conn):
        conn.execute(
            "INSERT OR IGNORE INTO runs "
            "(id, campaign_id, trigger_type, trigger_payload_json, session_key, "
            "status, protocol_json, inputs_json, policy_snapshot_json, "
            "created_by, created_at, updated_at) "
            "VALUES (?, ?, 'manual', '{}', 'sess-1', ?, '{}', '{}', "
            "'{\"max_temp_c\":95}', 'test', ?, ?)",
            (run_id, campaign_id, status, now, now),
        )
    run_txn(_txn)


def _seed_kpi(run_id, kpi_name, value, unit="pct"):
    """Insert a KPI value."""
    import uuid
    now = utcnow_iso()
    def _txn(conn):
        conn.execute(
            "INSERT INTO run_kpis "
            "(id, run_id, kpi_name, kpi_value, kpi_unit, kpi_schema_version, "
            "details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, '1', '{}', ?)",
            (uuid.uuid4().hex[:12], run_id, kpi_name, value, unit, now),
        )
    run_txn(_txn)


# ===========================================================================
# Req 1: Canonical Views
# ===========================================================================


class TestCanonicalViews:
    """Verify the 4 canonical SQL VIEWs exist and work."""

    def test_views_exist_in_schema(self):
        from app.services.schema_registry import get_table_names
        names = get_table_names()
        assert "v_experiment_runs" in names
        assert "v_experiment_params" in names
        assert "v_experiment_metrics" in names
        assert "v_experiment_artifacts" in names

    def test_v_experiment_runs_columns(self):
        from app.services.schema_registry import get_schema
        schema = get_schema()
        cols = {c.name for c in schema["v_experiment_runs"]}
        assert "run_id" in cols
        assert "campaign_id" in cols
        assert "run_status" in cols
        assert "timestamp" in cols
        assert "lab" in cols
        # Req 6 dimensions
        assert "domain" in cols
        assert "system_id" in cols
        assert "instrument_set" in cols
        assert "protocol_version" in cols

    def test_v_experiment_metrics_columns(self):
        from app.services.schema_registry import get_schema
        schema = get_schema()
        cols = {c.name for c in schema["v_experiment_metrics"]}
        assert "run_id" in cols
        assert "metric_name" in cols
        assert "value" in cols
        assert "unit" in cols
        assert "stage" in cols
        assert "qc_flag" in cols
        assert "review_verdict" in cols

    def test_v_experiment_runs_returns_data(self):
        _seed_run("run-v1")
        with connection() as conn:
            rows = conn.execute(
                "SELECT * FROM v_experiment_runs WHERE run_id = ?",
                ("run-v1",),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-v1"
        assert rows[0]["run_status"] == "completed"

    def test_v_experiment_metrics_returns_data(self):
        _seed_run("run-v2")
        _seed_kpi("run-v2", "volume_accuracy_pct", 95.5, "pct")
        with connection() as conn:
            rows = conn.execute(
                "SELECT * FROM v_experiment_metrics WHERE run_id = ?",
                ("run-v2",),
            ).fetchall()
        assert len(rows) >= 1
        assert rows[0]["metric_name"] == "volume_accuracy_pct"
        assert rows[0]["value"] == 95.5
        assert rows[0]["unit"] == "pct"

    def test_v_experiment_artifacts_columns(self):
        from app.services.schema_registry import get_schema
        schema = get_schema()
        cols = {c.name for c in schema["v_experiment_artifacts"]}
        assert "artifact_id" in cols
        assert "run_id" in cols
        assert "type" in cols
        assert "uri" in cols
        assert "hash" in cols

    def test_sql_guard_allows_views(self):
        """SqlGuard should allow querying canonical views."""
        from app.services.sql_guard import validate_sql
        from app.contracts.query_contract import QueryConstraints
        from app.services.schema_registry import get_table_names

        result = validate_sql(
            "SELECT run_id, run_status FROM v_experiment_runs WHERE run_id = ?",
            ["run-001"],
            QueryConstraints(),
            set(get_table_names()),
        )
        assert result.valid


# ===========================================================================
# Req 2: Dataset Snapshots & Watermark
# ===========================================================================


class TestDatasetSnapshots:
    """Snapshot creation, loading, verification."""

    def test_create_snapshot(self):
        from app.services.dataset_snapshot import create_snapshot

        _seed_run("run-s1")
        rows = [{"run_id": "run-s1", "status": "completed"}]
        snap = create_snapshot(
            rows=rows,
            query_plan_hash="abc123",
            campaign_id=None,
            snapshot_name="test_snap",
        )
        assert snap["snapshot_id"].startswith("snap-")
        assert snap["row_count"] == 1
        assert snap["run_ids"] == ["run-s1"]
        assert snap["result_hash"]  # non-empty
        assert snap["db_watermark"] >= 0

    def test_load_snapshot(self):
        from app.services.dataset_snapshot import create_snapshot, load_snapshot

        _seed_run("run-s2")
        rows = [{"run_id": "run-s2", "val": 42}]
        snap = create_snapshot(rows=rows, query_plan_hash="xyz")
        loaded = load_snapshot(snap["snapshot_id"])
        assert loaded is not None
        assert loaded["snapshot_id"] == snap["snapshot_id"]
        assert loaded["result_hash"] == snap["result_hash"]
        assert loaded["run_ids"] == ["run-s2"]

    def test_load_missing_returns_none(self):
        from app.services.dataset_snapshot import load_snapshot
        assert load_snapshot("snap-nonexistent") is None

    def test_snapshot_run_ids_materialized(self):
        from app.services.dataset_snapshot import (
            create_snapshot,
            get_snapshot_run_ids,
        )

        _seed_run("run-s3a")
        _seed_run("run-s3b")
        rows = [
            {"run_id": "run-s3a", "v": 1},
            {"run_id": "run-s3b", "v": 2},
        ]
        snap = create_snapshot(rows=rows, query_plan_hash="multi")
        run_ids = get_snapshot_run_ids(snap["snapshot_id"])
        assert set(run_ids) == {"run-s3a", "run-s3b"}

    def test_verify_snapshot(self):
        from app.services.dataset_snapshot import create_snapshot, verify_snapshot

        _seed_run("r1")
        rows = [{"run_id": "r1", "v": 100}]
        snap = create_snapshot(rows=rows, query_plan_hash="hash1")

        # Same rows → verified
        assert verify_snapshot(snap["snapshot_id"], rows) is True

        # Different rows → fails
        assert verify_snapshot(snap["snapshot_id"], [{"run_id": "r1", "v": 999}]) is False

    def test_list_snapshots(self):
        from app.services.dataset_snapshot import create_snapshot, list_snapshots

        create_snapshot(rows=[{"a": 1}], query_plan_hash="h1")
        create_snapshot(rows=[{"b": 2}], query_plan_hash="h2")
        snaps = list_snapshots()
        assert len(snaps) >= 2

    def test_get_db_watermark(self):
        from app.services.dataset_snapshot import get_db_watermark

        wm = get_db_watermark()
        assert isinstance(wm, int)
        assert wm >= 0


# ===========================================================================
# Req 3: QueryAgent snapshot_mode
# ===========================================================================


class TestQueryAgentSnapshotMode:
    """snapshot_mode creates a dataset snapshot on query execution."""

    def test_snapshot_mode_creates_snapshot(self):
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.dataset_snapshot import load_snapshot
        from app.services.llm_gateway import MockProvider

        _seed_run("run-snap1")

        mock_llm = MockProvider(responses=[
            json.dumps({
                "sql": "SELECT id as run_id, status FROM runs WHERE status = ? LIMIT 10",
                "params": ["completed"],
            })
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(
            prompt="get completed runs",
            snapshot_mode=True,
            snapshot_name="test_snap_mode",
        )
        result = _run(agent.run(req))

        assert result.success
        assert result.output.snapshot_id is not None

        snap = load_snapshot(result.output.snapshot_id)
        assert snap is not None
        assert snap["snapshot_name"] == "test_snap_mode"
        assert snap["row_count"] >= 1

    def test_no_snapshot_without_flag(self):
        from app.agents.query_agent import QueryAgent
        from app.contracts.query_contract import QueryRequest
        from app.services.llm_gateway import MockProvider

        mock_llm = MockProvider(responses=[
            json.dumps({
                "sql": "SELECT id FROM runs WHERE status = ? LIMIT 10",
                "params": ["completed"],
            })
        ])

        agent = QueryAgent(llm_provider=mock_llm)
        req = QueryRequest(prompt="get runs")  # snapshot_mode=False by default
        result = _run(agent.run(req))

        assert result.success
        assert result.output.snapshot_id is None


# ===========================================================================
# Req 4: QC Flags + Failure Signatures + Metric Dictionary
# ===========================================================================


class TestQCFlags:
    """Structured quality annotations."""

    def test_store_and_get_qc_flag(self):
        from app.services.qc_store import get_qc_flags, store_qc_flag

        _seed_run("run-qc1")
        flag_id = store_qc_flag(
            run_id="run-qc1",
            kpi_name="volume_accuracy_pct",
            flag_value="ok",
            measured_value=95.2,
            threshold=90.0,
        )
        assert flag_id

        flags = get_qc_flags("run-qc1")
        assert len(flags) == 1
        assert flags[0]["kpi_name"] == "volume_accuracy_pct"
        assert flags[0]["flag_value"] == "ok"
        assert flags[0]["measured_value"] == 95.2

    def test_suspect_runs(self):
        from app.services.qc_store import get_suspect_runs, store_qc_flag

        _seed_run("run-qc2")
        _seed_run("run-qc3")
        store_qc_flag(run_id="run-qc2", kpi_name="temp", flag_value="suspect")
        store_qc_flag(run_id="run-qc3", kpi_name="temp", flag_value="ok")

        suspects = get_suspect_runs()
        assert "run-qc2" in suspects
        assert "run-qc3" not in suspects

    def test_qc_flag_appears_in_metrics_view(self):
        """QC flags should be visible through v_experiment_metrics."""
        from app.services.qc_store import store_qc_flag

        _seed_run("run-qcv")
        _seed_kpi("run-qcv", "impedance_ohm", 150.0, "ohm")
        store_qc_flag(
            run_id="run-qcv",
            kpi_name="impedance_ohm",
            flag_value="suspect",
            measured_value=150.0,
        )

        with connection() as conn:
            rows = conn.execute(
                "SELECT * FROM v_experiment_metrics "
                "WHERE run_id = ? AND metric_name = ?",
                ("run-qcv", "impedance_ohm"),
            ).fetchall()
        assert len(rows) >= 1
        assert rows[0]["qc_flag"] == "suspect"


class TestFailureSignatures:
    """Machine-readable failure classification."""

    def test_store_and_get_failure(self):
        from app.services.qc_store import get_failure_signatures, store_failure_signature

        _seed_run("run-fail1")
        sig_id = store_failure_signature(
            run_id="run-fail1",
            failure_type="volume_delivery_failure",
            severity="CRITICAL",
            likely_cause="tip_clog",
            step_key="step_1",
            primitive="aspirate",
            confidence=0.85,
            retryable=True,
            message_code="ERR_VOL_001",
        )
        assert sig_id

        sigs = get_failure_signatures("run-fail1")
        assert len(sigs) == 1
        assert sigs[0]["failure_type"] == "volume_delivery_failure"
        assert sigs[0]["severity"] == "CRITICAL"
        assert sigs[0]["likely_cause"] == "tip_clog"
        assert sigs[0]["retryable"] is True
        assert sigs[0]["message_code"] == "ERR_VOL_001"
        assert sigs[0]["confidence"] == 0.85

    def test_failure_stats_aggregation(self):
        from app.services.qc_store import get_failure_stats, store_failure_signature

        _seed_run("run-f2")
        _seed_run("run-f3")
        store_failure_signature(
            run_id="run-f2", failure_type="tip_shortage",
            severity="HIGH", likely_cause="tip_missing",
        )
        store_failure_signature(
            run_id="run-f3", failure_type="tip_shortage",
            severity="HIGH", likely_cause="tip_missing",
        )
        stats = get_failure_stats()
        tip_stats = [s for s in stats if s["failure_type"] == "tip_shortage"]
        assert len(tip_stats) == 1
        assert tip_stats[0]["count"] == 2


class TestMetricDictionary:
    """Canonical metric registry."""

    def test_register_and_get_metric(self):
        from app.services.qc_store import get_metric_dictionary, register_metric

        register_metric(
            metric_name="test_metric",
            unit="pct",
            definition="A test metric for unit tests",
            scope="run",
        )
        metrics = get_metric_dictionary()
        names = [m["metric_name"] for m in metrics]
        assert "test_metric" in names

    def test_seed_metric_dictionary(self):
        from app.services.qc_store import get_metric_dictionary, seed_metric_dictionary

        count = seed_metric_dictionary()
        assert count >= 12  # 9 step + 3 run KPIs
        metrics = get_metric_dictionary()
        names = [m["metric_name"] for m in metrics]
        assert "volume_accuracy_pct" in names
        assert "run_success_rate" in names

    def test_register_is_idempotent(self):
        from app.services.qc_store import get_metric_dictionary, register_metric

        register_metric(metric_name="idem", unit="x", definition="test")
        register_metric(metric_name="idem", unit="y", definition="updated")
        metrics = get_metric_dictionary()
        idem = [m for m in metrics if m["metric_name"] == "idem"]
        assert len(idem) == 1
        assert idem[0]["unit"] == "y"  # last write wins


# ===========================================================================
# Req 5: Parameter Schema
# ===========================================================================


class TestParamSchema:
    """Campaign-level parameter type/unit storage."""

    def _seed_campaign(self, campaign_id="camp-ps1"):
        now = utcnow_iso()
        def _txn(conn):
            conn.execute(
                "INSERT OR IGNORE INTO campaigns "
                "(id, name, cadence_seconds, protocol_json, inputs_json, "
                "policy_json, next_fire_at, created_at, updated_at) "
                "VALUES (?, 'test', 60, '{}', '{}', '{}', ?, ?, ?)",
                (campaign_id, now, now, now),
            )
        run_txn(_txn)

    def test_store_and_get_param_schema(self):
        from app.services.qc_store import get_param_schema, store_param_schema

        self._seed_campaign("camp-ps1")
        dims = [
            {
                "param_name": "volume_ul",
                "param_type": "number",
                "unit": "uL",
                "min_value": 1.0,
                "max_value": 300.0,
                "log_scale": False,
            },
            {
                "param_name": "temp_c",
                "param_type": "number",
                "unit": "°C",
                "min_value": 20.0,
                "max_value": 95.0,
            },
            {
                "param_name": "reagent",
                "param_type": "categorical",
                "unit": "",
                "choices": ["NaCl", "KCl", "LiCl"],
            },
        ]
        count = store_param_schema("camp-ps1", dims)
        assert count == 3

        schema = get_param_schema("camp-ps1")
        assert len(schema) == 3
        vol = [s for s in schema if s["param_name"] == "volume_ul"][0]
        assert vol["param_type"] == "number"
        assert vol["unit"] == "uL"
        assert vol["min_value"] == 1.0
        assert vol["max_value"] == 300.0
        assert vol["log_scale"] is False

        reagent = [s for s in schema if s["param_name"] == "reagent"][0]
        assert reagent["param_type"] == "categorical"
        assert reagent["choices"] == ["NaCl", "KCl", "LiCl"]


# ===========================================================================
# Req 6: Experiment Index
# ===========================================================================


class TestExperimentIndex:
    """Cross-SDL dimension indexing."""

    def test_index_and_get_experiment(self):
        from app.services.qc_store import get_experiment_index, index_experiment

        _seed_run("run-ei1")
        index_experiment(
            run_id="run-ei1",
            domain="electrochemistry",
            system_id="lab-001",
            instrument_set="ot2|squidstat",
            protocol_version="1.2.0",
            tags=["battery", "cycling"],
        )

        idx = get_experiment_index("run-ei1")
        assert idx is not None
        assert idx["domain"] == "electrochemistry"
        assert idx["system_id"] == "lab-001"
        assert idx["instrument_set"] == "ot2|squidstat"
        assert idx["protocol_version"] == "1.2.0"
        assert idx["tags"] == ["battery", "cycling"]

    def test_search_by_domain(self):
        from app.services.qc_store import index_experiment, search_experiments

        _seed_run("run-ei2")
        _seed_run("run-ei3")
        index_experiment(run_id="run-ei2", domain="liquid_handling", system_id="lab-001")
        index_experiment(run_id="run-ei3", domain="electrochemistry", system_id="lab-002")

        results = search_experiments(domain="electrochemistry")
        run_ids = [r["run_id"] for r in results]
        assert "run-ei3" in run_ids
        assert "run-ei2" not in run_ids

    def test_search_by_multiple_filters(self):
        from app.services.qc_store import index_experiment, search_experiments

        _seed_run("run-ei4")
        _seed_run("run-ei5")
        index_experiment(run_id="run-ei4", domain="echem", system_id="lab-A")
        index_experiment(run_id="run-ei5", domain="echem", system_id="lab-B")

        results = search_experiments(domain="echem", system_id="lab-A")
        assert len(results) == 1
        assert results[0]["run_id"] == "run-ei4"

    def test_index_appears_in_v_experiment_runs(self):
        """Experiment index dimensions should appear in canonical view."""
        from app.services.qc_store import index_experiment

        _seed_run("run-eiv")
        index_experiment(
            run_id="run-eiv",
            domain="SDL1",
            system_id="system-42",
        )

        with connection() as conn:
            rows = conn.execute(
                "SELECT * FROM v_experiment_runs WHERE run_id = ?",
                ("run-eiv",),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["domain"] == "SDL1"
        assert rows[0]["system_id"] == "system-42"

    def test_missing_index_returns_none(self):
        from app.services.qc_store import get_experiment_index
        assert get_experiment_index("nonexistent-run") is None

    def test_upsert_index(self):
        from app.services.qc_store import get_experiment_index, index_experiment

        _seed_run("run-ei6")
        index_experiment(run_id="run-ei6", domain="v1")
        index_experiment(run_id="run-ei6", domain="v2")  # upsert

        idx = get_experiment_index("run-ei6")
        assert idx["domain"] == "v2"
