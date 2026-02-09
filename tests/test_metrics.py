"""Tests for the KPI metrics store and extractor (Phase C2).

Covers:
- Schema table existence (run_kpis)
- Step-level KPI extraction (volume_accuracy, temp_accuracy, impedance, step_duration)
- Run-level KPI extraction (success_rate, duration, recovery_count)
- KPI schema version persistence and filtering
- Source artifact_id FK references
- Details JSON preservation
- Read path (get_run_kpis, get_kpi_summary)
- Missing data graceful handling
- Event listener (run.completed → KPI extraction)
- Failure isolation (extraction failure doesn't block)
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_metrics_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "metrics_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, json_dumps, utcnow_iso  # noqa: E402
from app.services.metrics import (  # noqa: E402
    KPI_SCHEMA_VERSION,
    KpiDefinition,
    KpiValue,
    extract_and_store_kpis,
    extract_charge_passed,
    extract_coulombic_efficiency,
    extract_current_density,
    extract_impedance,
    extract_overpotential,
    extract_run_duration,
    extract_run_success_rate,
    extract_stability_decay,
    extract_step_duration,
    extract_temp_accuracy,
    extract_volume_accuracy,
    get_kpi_summary,
    get_run_kpis,
    start_metrics_listener,
    stop_metrics_listener,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    # Clean tables between tests
    with connection() as conn:
        conn.execute("DELETE FROM evolved_priors")
        conn.execute("DELETE FROM evolution_proposals")
        conn.execute("DELETE FROM protocol_templates")
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM run_reviews")
        conn.execute("DELETE FROM run_kpis")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM provenance_events")
        conn.execute("DELETE FROM run_steps")
        conn.execute("DELETE FROM runs")
        conn.commit()


def _insert_run_with_steps_and_artifacts(
    steps: list[dict],
    run_status: str = "succeeded",
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> str:
    """Insert a run + run_steps + artifact JSON files. Returns run_id."""
    settings = get_settings()
    run_id = str(uuid.uuid4())
    now = utcnow_iso()

    with connection() as conn:
        conn.execute(
            "INSERT INTO runs "
            "(id, campaign_id, trigger_type, trigger_payload_json, session_key, "
            "status, protocol_json, inputs_json, compiled_graph_json, graph_hash, "
            "policy_snapshot_json, created_by, created_at, updated_at, started_at, ended_at) "
            "VALUES (?, NULL, 'manual', '{}', ?, ?, '{}', '{}', '{}', 'h', '{}', 'test', ?, ?, ?, ?)",
            (run_id, run_id, run_status, now, now, started_at, ended_at),
        )
        for step in steps:
            step_id = step.get("id", str(uuid.uuid4()))
            conn.execute(
                "INSERT INTO run_steps "
                "(id, run_id, step_key, primitive, params_json, depends_on_json, "
                "resources_json, status, idempotency_key, started_at, ended_at, error) "
                "VALUES (?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?, ?, ?)",
                (
                    step_id,
                    run_id,
                    step["step_key"],
                    step["primitive"],
                    json_dumps(step.get("params", {})),
                    step.get("status", "succeeded"),
                    f"{run_id}:{step['step_key']}:0",
                    step.get("started_at"),
                    step.get("ended_at"),
                    step.get("error"),
                ),
            )
            # Create artifact if provided
            artifact_data = step.get("artifact_payload")
            if artifact_data is not None:
                artifact_id = str(uuid.uuid4())
                obj_dir = settings.object_store_dir / run_id
                obj_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = obj_dir / f"{artifact_id}.json"
                artifact_path.write_text(
                    json.dumps(artifact_data, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                conn.execute(
                    "INSERT INTO artifacts "
                    "(id, run_id, step_id, kind, uri, checksum, metadata_json, created_at) "
                    "VALUES (?, ?, ?, 'primitive_result', ?, 'test_checksum', ?, ?)",
                    (
                        artifact_id,
                        run_id,
                        step_id,
                        str(artifact_path),
                        json_dumps({"primitive": step["primitive"]}),
                        now,
                    ),
                )
        conn.commit()
    return run_id


# ===========================================================================
# 1. Schema
# ===========================================================================


class TestSchema:
    def test_run_kpis_table_exists(self):
        with connection() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name = 'run_kpis'"
                ).fetchall()
            ]
        assert "run_kpis" in tables


# ===========================================================================
# 2. Step-level KPI extraction
# ===========================================================================


class TestVolumeAccuracy:
    def test_volume_accuracy_extraction(self):
        step_id = str(uuid.uuid4())
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "id": step_id,
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 100.0},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 99.7, "ok": True},
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        vol_kpis = [k for k in kpis if k.kpi_name == "volume_accuracy_pct"]
        assert len(vol_kpis) == 1
        assert abs(vol_kpis[0].kpi_value - 99.7) < 0.01
        assert vol_kpis[0].step_id == step_id
        assert vol_kpis[0].source_artifact_id is not None
        assert vol_kpis[0].details["requested_volume_ul"] == 100.0
        assert vol_kpis[0].details["measured_volume_ul"] == 99.7

    def test_volume_accuracy_missing_param(self):
        """No volume_ul param → KPI skipped."""
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 99.7, "ok": True},
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        vol_kpis = [k for k in kpis if k.kpi_name == "volume_accuracy_pct"]
        assert len(vol_kpis) == 0


class TestTempAccuracy:
    def test_temp_accuracy_extraction(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "heat",
                "params": {"temp_c": 65.0},
                "status": "succeeded",
                "artifact_payload": {"measured_temp_c": 65.3, "ok": True},
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        temp_kpis = [k for k in kpis if k.kpi_name == "temp_accuracy_c"]
        assert len(temp_kpis) == 1
        assert abs(temp_kpis[0].kpi_value - 0.3) < 0.001
        assert temp_kpis[0].details["target_temp_c"] == 65.0
        assert temp_kpis[0].details["measured_temp_c"] == 65.3


class TestImpedance:
    def test_impedance_extraction(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "eis",
                "params": {},
                "status": "succeeded",
                "artifact_payload": {"impedance_ohm": 103.2, "ok": True},
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        imp_kpis = [k for k in kpis if k.kpi_name == "impedance_ohm"]
        assert len(imp_kpis) == 1
        assert abs(imp_kpis[0].kpi_value - 103.2) < 0.001


class TestStepDuration:
    def test_step_duration_extraction(self):
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=5.5)
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "wait",
                "params": {},
                "status": "succeeded",
                "started_at": t0.isoformat(),
                "ended_at": t1.isoformat(),
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        dur_kpis = [k for k in kpis if k.kpi_name == "step_duration_s"]
        assert len(dur_kpis) == 1
        assert abs(dur_kpis[0].kpi_value - 5.5) < 0.01

    def test_step_duration_missing_timestamps(self):
        """No started_at/ended_at → KPI skipped."""
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "wait",
                "params": {},
                "status": "succeeded",
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        dur_kpis = [k for k in kpis if k.kpi_name == "step_duration_s"]
        assert len(dur_kpis) == 0


# ===========================================================================
# 3. Run-level KPI extraction
# ===========================================================================


class TestRunSuccessRate:
    def test_run_success_rate(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {"step_key": "s0", "primitive": "aspirate", "params": {}, "status": "succeeded"},
            {"step_key": "s1", "primitive": "heat", "params": {}, "status": "succeeded"},
            {"step_key": "s2", "primitive": "eis", "params": {}, "status": "failed", "error": "boom"},
        ])
        kpis = extract_and_store_kpis(run_id)
        rate_kpis = [k for k in kpis if k.kpi_name == "run_success_rate"]
        assert len(rate_kpis) == 1
        assert abs(rate_kpis[0].kpi_value - 2 / 3) < 0.001
        assert rate_kpis[0].details["succeeded"] == 2
        assert rate_kpis[0].details["total"] == 3
        assert rate_kpis[0].step_id is None


class TestRunDuration:
    def test_run_duration(self):
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=120.5)
        run_id = _insert_run_with_steps_and_artifacts(
            [{"step_key": "s0", "primitive": "wait", "params": {}, "status": "succeeded"}],
            started_at=t0.isoformat(),
            ended_at=t1.isoformat(),
        )
        kpis = extract_and_store_kpis(run_id)
        dur_kpis = [k for k in kpis if k.kpi_name == "run_duration_s"]
        assert len(dur_kpis) == 1
        assert abs(dur_kpis[0].kpi_value - 120.5) < 0.01

    def test_run_duration_missing_timestamps(self):
        run_id = _insert_run_with_steps_and_artifacts(
            [{"step_key": "s0", "primitive": "wait", "params": {}, "status": "succeeded"}],
        )
        kpis = extract_and_store_kpis(run_id)
        dur_kpis = [k for k in kpis if k.kpi_name == "run_duration_s"]
        assert len(dur_kpis) == 0


class TestRecoveryCount:
    def test_recovery_count(self):
        run_id = _insert_run_with_steps_and_artifacts(
            [{"step_key": "s0", "primitive": "wait", "params": {}, "status": "succeeded"}],
        )
        # Insert provenance events
        now = utcnow_iso()
        with connection() as conn:
            for _ in range(3):
                conn.execute(
                    "INSERT INTO provenance_events "
                    "(id, run_id, actor, action, details_json, created_at) "
                    "VALUES (?, ?, 'recovery-engine', 'recovery.attempted', '{}', ?)",
                    (str(uuid.uuid4()), run_id, now),
                )
            conn.commit()

        kpis = extract_and_store_kpis(run_id)
        rec_kpis = [k for k in kpis if k.kpi_name == "recovery_count"]
        assert len(rec_kpis) == 1
        assert rec_kpis[0].kpi_value == 3.0
        assert rec_kpis[0].details["recovery_attempted_events"] == 3

    def test_recovery_count_zero(self):
        run_id = _insert_run_with_steps_and_artifacts(
            [{"step_key": "s0", "primitive": "wait", "params": {}, "status": "succeeded"}],
        )
        kpis = extract_and_store_kpis(run_id)
        rec_kpis = [k for k in kpis if k.kpi_name == "recovery_count"]
        assert len(rec_kpis) == 1
        assert rec_kpis[0].kpi_value == 0.0


# ===========================================================================
# 4. Schema version + artifact references
# ===========================================================================


class TestSchemaVersion:
    def test_schema_version_stored(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 100},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 99.5, "ok": True},
            },
        ])
        extract_and_store_kpis(run_id)
        with connection() as conn:
            rows = conn.execute(
                "SELECT kpi_schema_version FROM run_kpis WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        assert len(rows) > 0
        for row in rows:
            assert row["kpi_schema_version"] == KPI_SCHEMA_VERSION

    def test_schema_version_filter(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 100},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 99.5, "ok": True},
            },
        ])
        extract_and_store_kpis(run_id)
        # Filter with correct version
        result = get_kpi_summary("volume_accuracy_pct", schema_version="1")
        assert len(result) >= 1
        # Filter with wrong version
        result_v2 = get_kpi_summary("volume_accuracy_pct", schema_version="99")
        assert len(result_v2) == 0


class TestArtifactRef:
    def test_artifact_ref_populated_for_step_kpis(self):
        step_id = str(uuid.uuid4())
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "id": step_id,
                "step_key": "s0",
                "primitive": "eis",
                "params": {},
                "status": "succeeded",
                "artifact_payload": {"impedance_ohm": 99.0, "ok": True},
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        imp_kpis = [k for k in kpis if k.kpi_name == "impedance_ohm"]
        assert len(imp_kpis) == 1
        assert imp_kpis[0].source_artifact_id is not None

    def test_artifact_ref_null_for_run_kpis(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {"step_key": "s0", "primitive": "wait", "params": {}, "status": "succeeded"},
        ])
        kpis = extract_and_store_kpis(run_id)
        rate_kpis = [k for k in kpis if k.kpi_name == "run_success_rate"]
        assert len(rate_kpis) == 1
        assert rate_kpis[0].source_artifact_id is None


class TestDetailsJson:
    def test_details_json_populated(self):
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 200.0},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 198.5, "ok": True},
            },
        ])
        extract_and_store_kpis(run_id)
        with connection() as conn:
            row = conn.execute(
                "SELECT details_json FROM run_kpis WHERE run_id = ? AND kpi_name = 'volume_accuracy_pct'",
                (run_id,),
            ).fetchone()
        assert row is not None
        details = json.loads(row["details_json"])
        assert details["requested_volume_ul"] == 200.0
        assert details["measured_volume_ul"] == 198.5


# ===========================================================================
# 5. Read path
# ===========================================================================


class TestReadPath:
    def test_get_run_kpis(self):
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=10)
        run_id = _insert_run_with_steps_and_artifacts(
            [
                {
                    "step_key": "s0",
                    "primitive": "aspirate",
                    "params": {"volume_ul": 100},
                    "status": "succeeded",
                    "artifact_payload": {"measured_volume_ul": 99.0, "ok": True},
                },
            ],
            started_at=t0.isoformat(),
            ended_at=t1.isoformat(),
        )
        extract_and_store_kpis(run_id)
        result = get_run_kpis(run_id)
        kpi_names = [r["kpi_name"] for r in result]
        # Should have: volume_accuracy_pct, step_duration_s (if timestamps), run_success_rate, run_duration_s, recovery_count
        assert "volume_accuracy_pct" in kpi_names
        assert "run_success_rate" in kpi_names
        assert "run_duration_s" in kpi_names
        assert "recovery_count" in kpi_names
        # details should be parsed from JSON
        for r in result:
            assert isinstance(r["details"], dict)

    def test_get_kpi_summary(self):
        # Create two runs with aspirate
        for _ in range(2):
            run_id = _insert_run_with_steps_and_artifacts([
                {
                    "step_key": "s0",
                    "primitive": "aspirate",
                    "params": {"volume_ul": 100},
                    "status": "succeeded",
                    "artifact_payload": {"measured_volume_ul": 99.5, "ok": True},
                },
            ])
            extract_and_store_kpis(run_id)
        result = get_kpi_summary("volume_accuracy_pct")
        assert len(result) == 2
        for r in result:
            assert r["kpi_name"] == "volume_accuracy_pct"
            assert "run_status" in r


# ===========================================================================
# 6. Graceful handling of missing data
# ===========================================================================


class TestGracefulHandling:
    def test_missing_artifact_graceful(self):
        """No artifact file → step-level KPI skipped, no crash."""
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 100},
                "status": "succeeded",
                # No artifact_payload → no artifact file created
            },
        ])
        kpis = extract_and_store_kpis(run_id)
        vol_kpis = [k for k in kpis if k.kpi_name == "volume_accuracy_pct"]
        assert len(vol_kpis) == 0  # gracefully skipped
        # Run-level KPIs still work
        rate_kpis = [k for k in kpis if k.kpi_name == "run_success_rate"]
        assert len(rate_kpis) == 1

    def test_empty_run(self):
        """Run with no steps → only run-level KPIs attempted."""
        run_id = _insert_run_with_steps_and_artifacts([])
        kpis = extract_and_store_kpis(run_id)
        # run_success_rate returns None when steps is empty
        rate_kpis = [k for k in kpis if k.kpi_name == "run_success_rate"]
        assert len(rate_kpis) == 0  # no steps, no rate
        # recovery_count should still work (0)
        rec_kpis = [k for k in kpis if k.kpi_name == "recovery_count"]
        assert len(rec_kpis) == 1
        assert rec_kpis[0].kpi_value == 0.0

    def test_extraction_failure_isolation(self):
        """Bad run_id → logs warning, no crash."""
        # _on_run_completed should not raise
        loop = asyncio.new_event_loop()
        try:
            from app.services.metrics import _on_run_completed

            loop.run_until_complete(_on_run_completed("nonexistent-run-id"))
        finally:
            loop.close()

    def test_nonexistent_run(self):
        """extract_and_store_kpis with nonexistent run → empty list."""
        kpis = extract_and_store_kpis("no-such-run")
        assert kpis == []


# ===========================================================================
# 7. Event listener E2E
# ===========================================================================


class TestEventListener:
    def test_event_listener_e2e(self):
        """EventBus → run.completed → KPIs extracted automatically."""
        from app.services.event_bus import EventBus, EventMessage

        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 100},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 99.0, "ok": True},
            },
        ])

        async def _run() -> None:
            bus = EventBus()
            await bus.start()
            try:
                sub = await start_metrics_listener(bus)
                try:
                    # Publish a run.completed event
                    bus.publish(
                        EventMessage(
                            id=str(uuid.uuid4()),
                            run_id=run_id,
                            actor="worker",
                            action="run.completed",
                            details={"final_status": "succeeded"},
                            created_at=utcnow_iso(),
                        )
                    )
                    # Give listener time to process
                    await asyncio.sleep(0.3)
                finally:
                    await stop_metrics_listener(sub, bus)
            finally:
                await bus.stop()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        # Verify KPIs were extracted
        result = get_run_kpis(run_id)
        kpi_names = [r["kpi_name"] for r in result]
        assert "volume_accuracy_pct" in kpi_names
        assert "run_success_rate" in kpi_names


# ===========================================================================
# 8. Duplicate protection
# ===========================================================================


class TestDuplicateProtection:
    def test_calling_extract_twice_inserts_twice(self):
        """Calling extract twice creates duplicate rows (no idempotency guard).

        This documents current behavior — if idempotency is needed later,
        add a UNIQUE constraint on (run_id, kpi_name, step_id).
        """
        run_id = _insert_run_with_steps_and_artifacts([
            {
                "step_key": "s0",
                "primitive": "aspirate",
                "params": {"volume_ul": 100},
                "status": "succeeded",
                "artifact_payload": {"measured_volume_ul": 99.5, "ok": True},
            },
        ])
        kpis1 = extract_and_store_kpis(run_id)
        kpis2 = extract_and_store_kpis(run_id)
        # Both extractions produce results
        assert len(kpis1) > 0
        assert len(kpis2) > 0
        # Total rows = 2x
        all_kpis = get_run_kpis(run_id)
        vol_kpis = [k for k in all_kpis if k["kpi_name"] == "volume_accuracy_pct"]
        assert len(vol_kpis) == 2


# ===========================================================================
# 9. Full extraction scenario
# ===========================================================================


class TestFullExtraction:
    def test_full_extraction_scenario(self):
        """E2E: run with aspirate+heat+eis → all 7 KPI types extracted."""
        t0 = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=4)
        t3 = t2 + timedelta(seconds=2)
        run_id = _insert_run_with_steps_and_artifacts(
            [
                {
                    "step_key": "s0",
                    "primitive": "aspirate",
                    "params": {"volume_ul": 50.0},
                    "status": "succeeded",
                    "started_at": t0.isoformat(),
                    "ended_at": t1.isoformat(),
                    "artifact_payload": {"measured_volume_ul": 49.8, "ok": True},
                },
                {
                    "step_key": "s1",
                    "primitive": "heat",
                    "params": {"temp_c": 37.0},
                    "status": "succeeded",
                    "started_at": t1.isoformat(),
                    "ended_at": t2.isoformat(),
                    "artifact_payload": {"measured_temp_c": 37.1, "ok": True},
                },
                {
                    "step_key": "s2",
                    "primitive": "eis",
                    "params": {},
                    "status": "succeeded",
                    "started_at": t2.isoformat(),
                    "ended_at": t3.isoformat(),
                    "artifact_payload": {"impedance_ohm": 95.5, "ok": True},
                },
            ],
            started_at=t0.isoformat(),
            ended_at=t3.isoformat(),
        )

        kpis = extract_and_store_kpis(run_id)
        kpi_names = {k.kpi_name for k in kpis}

        # All 7 KPI types should be present
        assert "volume_accuracy_pct" in kpi_names
        assert "temp_accuracy_c" in kpi_names
        assert "impedance_ohm" in kpi_names
        assert "step_duration_s" in kpi_names
        assert "run_success_rate" in kpi_names
        assert "run_duration_s" in kpi_names
        assert "recovery_count" in kpi_names

        # Verify specific values
        vol = next(k for k in kpis if k.kpi_name == "volume_accuracy_pct")
        assert abs(vol.kpi_value - (49.8 / 50.0 * 100)) < 0.01

        temp = next(k for k in kpis if k.kpi_name == "temp_accuracy_c")
        assert abs(temp.kpi_value - 0.1) < 0.001

        imp = next(k for k in kpis if k.kpi_name == "impedance_ohm")
        assert abs(imp.kpi_value - 95.5) < 0.01

        # 3 step_duration_s values (one per step)
        dur_kpis = [k for k in kpis if k.kpi_name == "step_duration_s"]
        assert len(dur_kpis) == 3

        rate = next(k for k in kpis if k.kpi_name == "run_success_rate")
        assert rate.kpi_value == 1.0  # all 3 succeeded

        run_dur = next(k for k in kpis if k.kpi_name == "run_duration_s")
        assert abs(run_dur.kpi_value - 9.0) < 0.01  # 3+4+2 seconds

        rec = next(k for k in kpis if k.kpi_name == "recovery_count")
        assert rec.kpi_value == 0.0


# ===========================================================================
# 10. Electrochemistry KPI extractors
# ===========================================================================


class TestElectrochemOverpotential:
    """η@10mA/cm² overpotential extraction."""

    def test_direct_overpotential_field(self):
        """Extract overpotential_mv directly from artifact."""
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"overpotential_mv": 320.5}
        kpi = extract_overpotential(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert kpi.kpi_name == "overpotential_mv"
        assert abs(kpi.kpi_value - 320.5) < 0.001
        assert kpi.kpi_unit == "mV"

    def test_computed_from_potential(self):
        """Compute overpotential from potential_v and reference_potential_v."""
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"potential_v": 1.55, "reference_potential_v": 1.23}
        kpi = extract_overpotential(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 320.0) < 0.1  # (1.55 - 1.23) * 1000

    def test_default_reference_potential(self):
        """Uses OER standard 1.23V when reference not specified."""
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"potential_v": 1.53}
        kpi = extract_overpotential(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 300.0) < 0.1

    def test_none_when_no_data(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        assert extract_overpotential(step, {"id": "a1"}, {}) is None
        assert extract_overpotential(step, None, None) is None

    def test_deterministic(self):
        """Same artifact → same KPI value (100 extractions)."""
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"overpotential_mv": 310.0}
        results = [extract_overpotential(step, {"id": "a1"}, payload) for _ in range(100)]
        values = [r.kpi_value for r in results]
        assert len(set(values)) == 1  # all identical


class TestElectrochemCurrentDensity:
    """Current density mA/cm² extraction."""

    def test_direct_field(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"current_density_ma_cm2": 15.3}
        kpi = extract_current_density(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 15.3) < 0.001

    def test_computed_from_current_and_area(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"current_ma": 30.0, "electrode_area_cm2": 2.0}
        kpi = extract_current_density(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 15.0) < 0.001

    def test_none_when_area_zero(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"current_ma": 30.0, "electrode_area_cm2": 0}
        assert extract_current_density(step, {"id": "a1"}, payload) is None


class TestElectrochemCoulombicEfficiency:
    """Coulombic efficiency (CE) extraction."""

    def test_direct_field(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"coulombic_efficiency": 0.95}
        kpi = extract_coulombic_efficiency(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 0.95) < 0.0001

    def test_computed_from_charges(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"charge_discharge_c": 9.5, "charge_charge_c": 10.0}
        kpi = extract_coulombic_efficiency(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 0.95) < 0.0001

    def test_none_when_charge_in_zero(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"charge_discharge_c": 9.5, "charge_charge_c": 0}
        assert extract_coulombic_efficiency(step, {"id": "a1"}, payload) is None


class TestElectrochemStabilityDecay:
    """Stability decay % extraction."""

    def test_direct_field(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"stability_decay_pct": 5.2}
        kpi = extract_stability_decay(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 5.2) < 0.001

    def test_computed_from_currents(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"initial_current_ma": 100.0, "final_current_ma": 90.0}
        kpi = extract_stability_decay(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 10.0) < 0.001  # (1 - 90/100) * 100 = 10%

    def test_none_when_initial_zero(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"initial_current_ma": 0, "final_current_ma": 90.0}
        assert extract_stability_decay(step, {"id": "a1"}, payload) is None


class TestElectrochemChargePassed:
    """Total charge passed (Coulombs) extraction."""

    def test_direct_field(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"charge_passed_c": 45.6}
        kpi = extract_charge_passed(step, {"id": "a1"}, payload)
        assert kpi is not None
        assert abs(kpi.kpi_value - 45.6) < 0.001

    def test_computed_from_current_and_time(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        payload = {"current_ma": 100.0, "duration_s": 3600.0}
        kpi = extract_charge_passed(step, {"id": "a1"}, payload)
        assert kpi is not None
        # 100 mA × 3600 s / 1000 = 360 C
        assert abs(kpi.kpi_value - 360.0) < 0.001

    def test_none_when_no_data(self):
        step = {"id": "s1", "primitive": "squidstat.run_experiment", "params_json": "{}"}
        assert extract_charge_passed(step, {"id": "a1"}, {}) is None
