from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, TypeVar

from app.core.config import get_settings

T = TypeVar("T")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.object_store_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def run_txn(fn: Callable[[sqlite3.Connection], T]) -> T:
    with connection() as conn:
        try:
            conn.execute("BEGIN")
            result = fn(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def parse_json(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    return json.loads(raw)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def init_db() -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS campaigns (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        cadence_seconds INTEGER NOT NULL,
        protocol_json TEXT NOT NULL,
        inputs_json TEXT NOT NULL,
        policy_json TEXT NOT NULL,
        next_fire_at TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        campaign_id TEXT,
        trigger_type TEXT NOT NULL,
        trigger_payload_json TEXT NOT NULL,
        session_key TEXT NOT NULL,
        status TEXT NOT NULL,
        protocol_json TEXT NOT NULL,
        inputs_json TEXT NOT NULL,
        compiled_graph_json TEXT,
        graph_hash TEXT,
        policy_snapshot_json TEXT NOT NULL,
        rejection_reason TEXT,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        ended_at TEXT,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
    );

    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
    CREATE INDEX IF NOT EXISTS idx_runs_campaign ON runs(campaign_id);

    CREATE TABLE IF NOT EXISTS run_steps (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_key TEXT NOT NULL,
        primitive TEXT NOT NULL,
        params_json TEXT NOT NULL,
        depends_on_json TEXT NOT NULL,
        resources_json TEXT NOT NULL,
        status TEXT NOT NULL,
        attempt INTEGER NOT NULL DEFAULT 0,
        idempotency_key TEXT NOT NULL,
        started_at TEXT,
        ended_at TEXT,
        error TEXT,
        UNIQUE (run_id, step_key),
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_run_steps_run ON run_steps(run_id);
    CREATE INDEX IF NOT EXISTS idx_run_steps_status ON run_steps(status);

    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_id TEXT,
        kind TEXT NOT NULL,
        uri TEXT NOT NULL,
        checksum TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
        FOREIGN KEY (step_id) REFERENCES run_steps(id)
    );

    CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);

    CREATE TABLE IF NOT EXISTS provenance_events (
        id TEXT PRIMARY KEY,
        run_id TEXT,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        details_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_events_run ON provenance_events(run_id);

    CREATE TABLE IF NOT EXISTS resource_locks (
        resource_id TEXT PRIMARY KEY,
        owner_run_id TEXT,
        lease_until TEXT,
        fencing_token INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (owner_run_id) REFERENCES runs(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        approver TEXT NOT NULL,
        decision TEXT NOT NULL,
        reason TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);

    CREATE TABLE IF NOT EXISTS instrument_sessions (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        instrument_id TEXT NOT NULL,
        firmware_version TEXT NOT NULL,
        calibration_id TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_run ON instrument_sessions(run_id);

    -- Memory system tables (episodic / semantic / procedural)

    CREATE TABLE IF NOT EXISTS memory_episodes (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_key TEXT NOT NULL,
        primitive TEXT NOT NULL,
        params_json TEXT NOT NULL,
        outcome TEXT NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_mem_episodes_primitive ON memory_episodes(primitive);
    CREATE INDEX IF NOT EXISTS idx_mem_episodes_run ON memory_episodes(run_id);

    CREATE TABLE IF NOT EXISTS memory_semantic (
        primitive TEXT NOT NULL,
        param_name TEXT NOT NULL,
        mean REAL NOT NULL DEFAULT 0.0,
        stddev REAL NOT NULL DEFAULT 0.0,
        sample_count INTEGER NOT NULL DEFAULT 0,
        success_rate REAL NOT NULL DEFAULT 0.0,
        success_count INTEGER NOT NULL DEFAULT 0,
        total_count INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (primitive, param_name)
    );

    CREATE TABLE IF NOT EXISTS memory_procedures (
        id TEXT PRIMARY KEY,
        trigger_primitive TEXT NOT NULL,
        trigger_error_pattern TEXT NOT NULL,
        recipe_json TEXT NOT NULL,
        source TEXT NOT NULL,
        hit_count INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_mem_procedures_trigger ON memory_procedures(trigger_primitive);

    -- Metrics / KPI store

    CREATE TABLE IF NOT EXISTS run_kpis (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        step_id TEXT,
        kpi_name TEXT NOT NULL,
        kpi_value REAL,
        kpi_unit TEXT NOT NULL,
        kpi_schema_version TEXT NOT NULL,
        source_artifact_id TEXT,
        details_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
        FOREIGN KEY (step_id) REFERENCES run_steps(id),
        FOREIGN KEY (source_artifact_id) REFERENCES artifacts(id)
    );

    CREATE INDEX IF NOT EXISTS idx_run_kpis_run ON run_kpis(run_id);
    CREATE INDEX IF NOT EXISTS idx_run_kpis_name ON run_kpis(kpi_name);
    CREATE INDEX IF NOT EXISTS idx_run_kpis_version ON run_kpis(kpi_schema_version);

    -- Run reviews (LLM evaluator output)

    CREATE TABLE IF NOT EXISTS run_reviews (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL UNIQUE,
        score REAL NOT NULL,
        verdict TEXT NOT NULL,
        failure_attributions_json TEXT NOT NULL,
        improvements_json TEXT NOT NULL,
        model TEXT NOT NULL,
        review_schema_version TEXT NOT NULL,
        raw_response TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_run_reviews_run ON run_reviews(run_id);

    -- Batch candidate generation (parameter space exploration)

    CREATE TABLE IF NOT EXISTS batch_requests (
        id TEXT PRIMARY KEY,
        campaign_id TEXT,
        protocol_template_json TEXT NOT NULL,
        space_json TEXT NOT NULL,
        strategy TEXT NOT NULL,
        n_candidates INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
    );

    CREATE INDEX IF NOT EXISTS idx_batch_requests_campaign ON batch_requests(campaign_id);
    CREATE INDEX IF NOT EXISTS idx_batch_requests_status ON batch_requests(status);

    CREATE TABLE IF NOT EXISTS batch_candidates (
        id TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL,
        candidate_index INTEGER NOT NULL,
        params_json TEXT NOT NULL,
        origin TEXT NOT NULL,
        score REAL,
        selected_run_id TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (batch_id, candidate_index),
        FOREIGN KEY (batch_id) REFERENCES batch_requests(id) ON DELETE CASCADE,
        FOREIGN KEY (selected_run_id) REFERENCES runs(id)
    );

    CREATE INDEX IF NOT EXISTS idx_batch_candidates_batch ON batch_candidates(batch_id);

    -- Evolution Engine (Phase C5)

    CREATE TABLE IF NOT EXISTS evolution_proposals (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        proposal_type TEXT NOT NULL,
        change_summary TEXT NOT NULL,
        change_details_json TEXT NOT NULL,
        magnitude REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        auto_approve_reason TEXT,
        reviewed_by TEXT,
        reviewed_at TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(id)
    );

    CREATE INDEX IF NOT EXISTS idx_proposals_status ON evolution_proposals(status);
    CREATE INDEX IF NOT EXISTS idx_proposals_run ON evolution_proposals(run_id);

    CREATE TABLE IF NOT EXISTS evolved_priors (
        id TEXT PRIMARY KEY,
        primitive TEXT NOT NULL,
        param_name TEXT NOT NULL,
        evolved_min REAL NOT NULL,
        evolved_max REAL NOT NULL,
        confidence REAL NOT NULL,
        source_run_id TEXT NOT NULL,
        proposal_id TEXT,
        generation INTEGER NOT NULL DEFAULT 1,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY (source_run_id) REFERENCES runs(id),
        FOREIGN KEY (proposal_id) REFERENCES evolution_proposals(id)
    );

    CREATE INDEX IF NOT EXISTS idx_evolved_priors_active
        ON evolved_priors(primitive, param_name, is_active);

    CREATE TABLE IF NOT EXISTS protocol_templates (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        parent_template_id TEXT,
        protocol_json TEXT NOT NULL,
        source_run_id TEXT,
        score REAL,
        tags_json TEXT NOT NULL DEFAULT '[]',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (parent_template_id) REFERENCES protocol_templates(id),
        FOREIGN KEY (source_run_id) REFERENCES runs(id)
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_templates_name_version
        ON protocol_templates(name, version);

    -- Campaign initialization conversation sessions

    CREATE TABLE IF NOT EXISTS conversation_sessions (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'active',
        current_round INTEGER NOT NULL DEFAULT 1,
        slots_json TEXT NOT NULL DEFAULT '{}',
        validation_errors_json TEXT NOT NULL DEFAULT '{}',
        completed_rounds_json TEXT NOT NULL DEFAULT '[]',
        injection_pack_json TEXT,
        campaign_id TEXT,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
    );

    CREATE INDEX IF NOT EXISTS idx_conv_sessions_status
        ON conversation_sessions(status);
    """

    with connection() as conn:
        conn.executescript(schema)
        conn.commit()
