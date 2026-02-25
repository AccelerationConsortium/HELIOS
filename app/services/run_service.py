from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from typing import Any
import uuid

from app.core.constants import (
    RUN_STATUS_AWAITING_APPROVAL,
    RUN_STATUS_FAILED,
    RUN_STATUS_REJECTED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SCHEDULED,
)
from app.core.db import json_dumps, parse_json, row_to_dict, run_txn, utcnow_iso
from app.services.audit import record_event
from app.services.compiler import CompileError, compile_protocol
from app.services.safety import evaluate_preflight


class DomainError(ValueError):
    pass


def default_policy() -> dict[str, Any]:
    from app.services.safety import BATTERY_LAB_PRIMITIVES

    return {
        "max_temp_c": 95.0,
        "max_volume_ul": 1000.0,
        "allowed_primitives": list(BATTERY_LAB_PRIMITIVES),
        "require_human_approval": False,
    }


def create_campaign(
    *,
    name: str,
    cadence_seconds: int,
    protocol: dict[str, Any],
    inputs: dict[str, Any],
    policy_snapshot: dict[str, Any] | None,
    actor: str,
) -> dict[str, Any]:
    if cadence_seconds <= 0:
        raise DomainError("cadence_seconds must be positive")

    campaign_id = str(uuid.uuid4())
    now = utcnow_iso()
    policy = policy_snapshot or default_policy()

    def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
        conn.execute(
            """
            INSERT INTO campaigns (
                id, name, cadence_seconds, protocol_json, inputs_json, policy_json,
                next_fire_at, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                campaign_id,
                name,
                cadence_seconds,
                json_dumps(protocol),
                json_dumps(inputs),
                json_dumps(policy),
                now,
                now,
                now,
            ),
        )
        record_event(
            conn,
            run_id=None,
            actor=actor,
            action="campaign.created",
            details={"campaign_id": campaign_id, "name": name, "cadence_seconds": cadence_seconds},
        )
        row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        assert row is not None
        return _campaign_row(row)

    return run_txn(_txn)


def list_campaigns() -> list[dict[str, Any]]:
    def _txn(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM campaigns ORDER BY created_at DESC"
        ).fetchall()
        return [_campaign_row(r) for r in rows]

    return run_txn(_txn)


def _campaign_row(row: sqlite3.Row) -> dict[str, Any]:
    data = row_to_dict(row)
    assert data is not None
    data["protocol"] = parse_json(data.pop("protocol_json"), {})
    data["inputs"] = parse_json(data.pop("inputs_json"), {})
    data["policy_snapshot"] = parse_json(data.pop("policy_json"), {})
    data["is_active"] = bool(data["is_active"])
    return data


def create_run(
    *,
    trigger_type: str,
    trigger_payload: dict[str, Any],
    campaign_id: str | None,
    protocol: dict[str, Any],
    inputs: dict[str, Any],
    policy_snapshot: dict[str, Any] | None,
    actor: str,
    session_key: str | None = None,
) -> dict[str, Any]:
    policy = policy_snapshot or default_policy()

    try:
        compiled, graph_hash = compile_protocol(
            protocol=protocol,
            inputs=inputs,
            policy_snapshot=policy,
        )
    except CompileError as exc:
        raise DomainError(str(exc)) from exc

    safety = evaluate_preflight(compiled_graph=compiled, policy_snapshot=policy)
    now = utcnow_iso()
    run_id = str(uuid.uuid4())

    if not safety.allowed:
        status = RUN_STATUS_REJECTED
        rejection_reason = "; ".join(safety.violations)
    elif safety.requires_approval:
        status = RUN_STATUS_AWAITING_APPROVAL
        rejection_reason = None
    else:
        status = RUN_STATUS_SCHEDULED
        rejection_reason = None

    session = session_key or campaign_id or run_id

    def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
        # Orchestrator campaigns live in campaign_state, not campaigns.
        # Only reference campaign_id when it exists in campaigns to avoid FK violations.
        stored_campaign_id: str | None = None
        if campaign_id is not None:
            row = conn.execute(
                "SELECT 1 FROM campaigns WHERE id = ? LIMIT 1", (campaign_id,)
            ).fetchone()
            if row is not None:
                stored_campaign_id = campaign_id

        conn.execute(
            """
            INSERT INTO runs (
                id, campaign_id, trigger_type, trigger_payload_json, session_key,
                status, protocol_json, inputs_json, compiled_graph_json, graph_hash,
                policy_snapshot_json, rejection_reason, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stored_campaign_id,
                trigger_type,
                json_dumps(trigger_payload),
                session,
                status,
                json_dumps(protocol),
                json_dumps(inputs),
                json_dumps(compiled),
                graph_hash,
                json_dumps(policy),
                rejection_reason,
                actor,
                now,
                now,
            ),
        )

        for step in compiled["steps"]:
            step_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO run_steps (
                    id, run_id, step_key, primitive, params_json,
                    depends_on_json, resources_json, status, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    step_id,
                    run_id,
                    step["step_key"],
                    step["primitive"],
                    json_dumps(step.get("params", {})),
                    json_dumps(step.get("depends_on", [])),
                    json_dumps(step.get("resources", [])),
                    f"{run_id}:{step['step_key']}:0",
                ),
            )

        record_event(
            conn,
            run_id=run_id,
            actor=actor,
            action="run.created",
            details={
                "trigger_type": trigger_type,
                "session_key": session,
                "status": status,
                "graph_hash": graph_hash,
            },
        )

        if rejection_reason:
            record_event(
                conn,
                run_id=run_id,
                actor="safety-engine",
                action="run.rejected",
                details={"violations": safety.violations},
            )

        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert row is not None
        return _run_row(conn, row)

    return run_txn(_txn)


def approve_run(*, run_id: str, approver: str, reason: str | None) -> dict[str, Any]:
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> dict[str, Any]:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise DomainError("run not found")

        if run["status"] != RUN_STATUS_AWAITING_APPROVAL:
            raise DomainError("run is not awaiting approval")

        conn.execute(
            """
            INSERT INTO approvals (id, run_id, approver, decision, reason, created_at)
            VALUES (?, ?, ?, 'approved', ?, ?)
            """,
            (str(uuid.uuid4()), run_id, approver, reason, now),
        )
        conn.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
            (RUN_STATUS_SCHEDULED, now, run_id),
        )
        record_event(
            conn,
            run_id=run_id,
            actor=approver,
            action="run.approved",
            details={"reason": reason},
        )
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert row is not None
        return _run_row(conn, row)

    return run_txn(_txn)


def claim_schedulable_runs(limit: int = 8) -> list[str]:
    def _txn(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT id FROM runs WHERE status = ? ORDER BY created_at ASC LIMIT ?",
            (RUN_STATUS_SCHEDULED, limit),
        ).fetchall()
        run_ids: list[str] = []
        now = utcnow_iso()
        for row in rows:
            run_id = row["id"]
            cursor = conn.execute(
                """
                UPDATE runs SET status = ?, started_at = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (RUN_STATUS_RUNNING, now, now, run_id, RUN_STATUS_SCHEDULED),
            )
            if cursor.rowcount == 1:
                run_ids.append(run_id)
                record_event(
                    conn,
                    run_id=run_id,
                    actor="scheduler",
                    action="run.claimed",
                    details={"claimed_at": now},
                )
        return run_ids

    return run_txn(_txn)


def mark_run_failed_if_running(run_id: str, reason: str) -> None:
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT status FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] != RUN_STATUS_RUNNING:
            return

        conn.execute(
            "UPDATE runs SET status = ?, rejection_reason = ?, ended_at = ?, updated_at = ? WHERE id = ?",
            (RUN_STATUS_FAILED, reason, now, now, run_id),
        )
        record_event(
            conn,
            run_id=run_id,
            actor="scheduler",
            action="run.worker_failed",
            details={"reason": reason},
        )

    run_txn(_txn)


def get_run(run_id: str) -> dict[str, Any] | None:
    def _txn(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return _run_row(conn, row)

    return run_txn(_txn)


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    def _txn(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_run_row(conn, row) for row in rows]

    return run_txn(_txn)


def list_events(run_id: str) -> list[dict[str, Any]]:
    def _txn(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM provenance_events WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = row_to_dict(row)
            assert item is not None
            item["details"] = parse_json(item.pop("details_json"), {})
            out.append(item)
        return out

    return run_txn(_txn)


def _run_row(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    run = row_to_dict(row)
    assert run is not None
    run["trigger_payload"] = parse_json(run.pop("trigger_payload_json"), {})
    run["protocol"] = parse_json(run.pop("protocol_json"), {})
    run["inputs"] = parse_json(run.pop("inputs_json"), {})
    run["compiled_graph"] = parse_json(run.pop("compiled_graph_json"), {})
    run["policy_snapshot"] = parse_json(run.pop("policy_snapshot_json"), {})

    rows = conn.execute(
        "SELECT * FROM run_steps WHERE run_id = ? ORDER BY step_key ASC",
        (run["id"],),
    ).fetchall()

    steps: list[dict[str, Any]] = []
    for s in rows:
        item = row_to_dict(s)
        assert item is not None
        item["params"] = parse_json(item.pop("params_json"), {})
        item["depends_on"] = parse_json(item.pop("depends_on_json"), [])
        item["resources"] = parse_json(item.pop("resources_json"), [])
        steps.append(item)

    run["steps"] = steps
    return run


def trigger_due_campaigns(actor: str = "campaign-loop") -> list[str]:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    def _txn(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            """
            SELECT * FROM campaigns
            WHERE is_active = 1 AND next_fire_at <= ?
            ORDER BY next_fire_at ASC
            """,
            (now_iso,),
        ).fetchall()

        fired_run_ids: list[str] = []
        for row in rows:
            campaign = _campaign_row(row)
            run = _create_run_in_conn(
                conn,
                trigger_type="time",
                trigger_payload={"campaign_id": campaign["id"], "fired_at": now_iso},
                campaign_id=campaign["id"],
                protocol=campaign["protocol"],
                inputs=campaign["inputs"],
                policy_snapshot=campaign["policy_snapshot"],
                actor=actor,
                session_key=campaign["id"],
            )
            fired_run_ids.append(run["id"])

            next_fire = now + timedelta(seconds=int(campaign["cadence_seconds"]))
            conn.execute(
                "UPDATE campaigns SET next_fire_at = ?, updated_at = ? WHERE id = ?",
                (next_fire.isoformat(), now_iso, campaign["id"]),
            )
            record_event(
                conn,
                run_id=run["id"],
                actor=actor,
                action="campaign.fired",
                details={"campaign_id": campaign["id"], "next_fire_at": next_fire.isoformat()},
            )

        return fired_run_ids

    return run_txn(_txn)


def _create_run_in_conn(
    conn: sqlite3.Connection,
    *,
    trigger_type: str,
    trigger_payload: dict[str, Any],
    campaign_id: str | None,
    protocol: dict[str, Any],
    inputs: dict[str, Any],
    policy_snapshot: dict[str, Any] | None,
    actor: str,
    session_key: str | None,
) -> dict[str, Any]:
    policy = policy_snapshot or default_policy()
    compiled, graph_hash = compile_protocol(
        protocol=protocol,
        inputs=inputs,
        policy_snapshot=policy,
    )
    safety = evaluate_preflight(compiled_graph=compiled, policy_snapshot=policy)

    now = utcnow_iso()
    run_id = str(uuid.uuid4())
    if not safety.allowed:
        status = RUN_STATUS_REJECTED
        rejection_reason = "; ".join(safety.violations)
    elif safety.requires_approval:
        status = RUN_STATUS_AWAITING_APPROVAL
        rejection_reason = None
    else:
        status = RUN_STATUS_SCHEDULED
        rejection_reason = None

    session = session_key or campaign_id or run_id

    conn.execute(
        """
        INSERT INTO runs (
            id, campaign_id, trigger_type, trigger_payload_json, session_key,
            status, protocol_json, inputs_json, compiled_graph_json, graph_hash,
            policy_snapshot_json, rejection_reason, created_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            campaign_id,
            trigger_type,
            json_dumps(trigger_payload),
            session,
            status,
            json_dumps(protocol),
            json_dumps(inputs),
            json_dumps(compiled),
            graph_hash,
            json_dumps(policy),
            rejection_reason,
            actor,
            now,
            now,
        ),
    )

    for step in compiled["steps"]:
        step_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO run_steps (
                id, run_id, step_key, primitive, params_json,
                depends_on_json, resources_json, status, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                step_id,
                run_id,
                step["step_key"],
                step["primitive"],
                json_dumps(step.get("params", {})),
                json_dumps(step.get("depends_on", [])),
                json_dumps(step.get("resources", [])),
                f"{run_id}:{step['step_key']}:0",
            ),
        )

    record_event(
        conn,
        run_id=run_id,
        actor=actor,
        action="run.created",
        details={
            "trigger_type": trigger_type,
            "session_key": session,
            "status": status,
            "graph_hash": graph_hash,
        },
    )

    if rejection_reason:
        record_event(
            conn,
            run_id=run_id,
            actor="safety-engine",
            action="run.rejected",
            details={"violations": safety.violations},
        )

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    return _run_row(conn, row)


def create_run_from_trigger(
    *,
    trigger_type: str,
    trigger_payload: dict[str, Any],
    campaign_id: str | None,
    protocol: dict[str, Any] | None,
    inputs: dict[str, Any] | None,
    policy_snapshot: dict[str, Any] | None,
    actor: str,
    session_key: str | None,
) -> dict[str, Any]:
    if campaign_id:
        campaign = get_campaign(campaign_id)
        if campaign is None:
            raise DomainError("campaign not found")
        protocol = campaign["protocol"]
        inputs = inputs or campaign["inputs"]
        policy_snapshot = policy_snapshot or campaign["policy_snapshot"]

    if protocol is None or inputs is None:
        raise DomainError("protocol and inputs are required when campaign_id is not provided")

    return create_run(
        trigger_type=trigger_type,
        trigger_payload=trigger_payload,
        campaign_id=campaign_id,
        protocol=protocol,
        inputs=inputs,
        policy_snapshot=policy_snapshot,
        actor=actor,
        session_key=session_key,
    )


def get_campaign(campaign_id: str) -> dict[str, Any] | None:
    def _txn(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        if row is None:
            return None
        return _campaign_row(row)

    return run_txn(_txn)


def worker_load_run(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run is None:
        raise DomainError("run not found")
    return run


def worker_set_step_state(
    *,
    run_id: str,
    step_id: str,
    status: str,
    actor: str,
    error: str | None = None,
) -> None:
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> None:
        if status == "running":
            conn.execute(
                "UPDATE run_steps SET status = ?, started_at = ?, attempt = attempt + 1 WHERE id = ? AND run_id = ?",
                (status, now, step_id, run_id),
            )
        elif status in ("succeeded", "failed", "skipped"):
            conn.execute(
                "UPDATE run_steps SET status = ?, ended_at = ?, error = ? WHERE id = ? AND run_id = ?",
                (status, now, error, step_id, run_id),
            )
        else:
            conn.execute(
                "UPDATE run_steps SET status = ? WHERE id = ? AND run_id = ?",
                (status, step_id, run_id),
            )

        record_event(
            conn,
            run_id=run_id,
            actor=actor,
            action="step.state_changed",
            details={"step_id": step_id, "status": status, "error": error},
        )

    run_txn(_txn)


def worker_append_artifact(
    *,
    run_id: str,
    step_id: str,
    kind: str,
    uri: str,
    checksum: str,
    metadata: dict[str, Any],
) -> None:
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> None:
        artifact_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO artifacts (id, run_id, step_id, kind, uri, checksum, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                run_id,
                step_id,
                kind,
                uri,
                checksum,
                json_dumps(metadata),
                now,
            ),
        )
        record_event(
            conn,
            run_id=run_id,
            actor="worker",
            action="artifact.created",
            details={"artifact_id": artifact_id, "kind": kind, "uri": uri},
        )

    run_txn(_txn)


def worker_complete_run(
    *, run_id: str, final_status: str, actor: str, reason: str | None = None
) -> None:
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ?, updated_at = ?, rejection_reason = COALESCE(?, rejection_reason) WHERE id = ?",
            (final_status, now, now, reason, run_id),
        )
        record_event(
            conn,
            run_id=run_id,
            actor=actor,
            action="run.completed",
            details={"final_status": final_status, "reason": reason},
        )

    run_txn(_txn)


def worker_open_instrument_session(
    *,
    run_id: str,
    instrument_id: str,
    firmware_version: str,
    calibration_id: str,
) -> str:
    session_id = str(uuid.uuid4())
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> str:
        conn.execute(
            """
            INSERT INTO instrument_sessions (
                id, run_id, instrument_id, firmware_version, calibration_id, status, started_at
            ) VALUES (?, ?, ?, ?, ?, 'running', ?)
            """,
            (session_id, run_id, instrument_id, firmware_version, calibration_id, now),
        )
        record_event(
            conn,
            run_id=run_id,
            actor="worker",
            action="instrument_session.started",
            details={
                "instrument_session_id": session_id,
                "instrument_id": instrument_id,
                "firmware_version": firmware_version,
                "calibration_id": calibration_id,
            },
        )
        return session_id

    return run_txn(_txn)


def worker_close_instrument_session(*, run_id: str, session_id: str, status: str) -> None:
    now = utcnow_iso()

    def _txn(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE instrument_sessions SET status = ?, ended_at = ? WHERE id = ? AND run_id = ?",
            (status, now, session_id, run_id),
        )
        record_event(
            conn,
            run_id=run_id,
            actor="worker",
            action="instrument_session.ended",
            details={"instrument_session_id": session_id, "status": status},
        )

    run_txn(_txn)


def worker_list_steps(run_id: str) -> list[dict[str, Any]]:
    def _txn(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM run_steps WHERE run_id = ? ORDER BY step_key ASC",
            (run_id,),
        ).fetchall()
        steps: list[dict[str, Any]] = []
        for row in rows:
            item = row_to_dict(row)
            assert item is not None
            item["params"] = parse_json(item.pop("params_json"), {})
            item["depends_on"] = parse_json(item.pop("depends_on_json"), [])
            item["resources"] = parse_json(item.pop("resources_json"), [])
            steps.append(item)
        return steps

    return run_txn(_txn)


def worker_get_completed_step_keys(run_id: str) -> set[str]:
    def _txn(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute(
            "SELECT step_key FROM run_steps WHERE run_id = ? AND status = 'succeeded'",
            (run_id,),
        ).fetchall()
        return {row["step_key"] for row in rows}

    return run_txn(_txn)


def list_locks() -> list[dict[str, Any]]:
    def _txn(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM resource_locks ORDER BY resource_id ASC"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = row_to_dict(row)
            if item is not None:
                out.append(item)
        return out

    return run_txn(_txn)
