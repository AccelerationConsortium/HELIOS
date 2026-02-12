"""Campaign state persistence layer — checkpoint / resume for the orchestrator."""

from __future__ import annotations

import json
from typing import Any

from app.core.db import (
    connection,
    json_dumps,
    parse_json,
    row_to_dict,
    run_txn,
    utcnow_iso,
)


# ---------------------------------------------------------------------------
# Campaign-level CRUD
# ---------------------------------------------------------------------------

def create_campaign(
    campaign_id: str,
    input_data: dict[str, Any],
    direction: str = "minimize",
) -> None:
    """INSERT a new campaign_state row at 'planning' status."""
    now = utcnow_iso()

    def _insert(conn):
        conn.execute(
            """INSERT OR IGNORE INTO campaign_state
               (campaign_id, status, input_json, direction, created_at, updated_at)
               VALUES (?, 'planning', ?, ?, ?, ?)""",
            (campaign_id, json_dumps(input_data), direction, now, now),
        )

    run_txn(_insert)


def load_campaign(campaign_id: str) -> dict[str, Any] | None:
    """SELECT campaign_state and deserialise JSON columns."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
    d = row_to_dict(row)
    if d is None:
        return None
    # Deserialise JSON fields
    d["input"] = parse_json(d.pop("input_json"), {})
    d["plan"] = parse_json(d.pop("plan_json"), None)
    d["kpi_history"] = parse_json(d.pop("kpi_history_json"), [])
    d["all_kpis"] = parse_json(d.pop("all_kpis_json"), [])
    d["all_params"] = parse_json(d.pop("all_params_json"), [])
    d["all_rounds"] = parse_json(d.pop("all_rounds_json"), [])
    return d


def update_campaign_status(campaign_id: str, status: str, **kwargs: Any) -> None:
    """UPDATE campaign_state with status + arbitrary extra columns."""
    now = utcnow_iso()
    set_parts = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, now]

    for col, val in kwargs.items():
        set_parts.append(f"{col} = ?")
        values.append(val)

    values.append(campaign_id)

    def _update(conn):
        conn.execute(
            f"UPDATE campaign_state SET {', '.join(set_parts)} WHERE campaign_id = ?",
            values,
        )

    run_txn(_update)


def save_plan(
    campaign_id: str,
    plan_dict: dict[str, Any],
    total_rounds: int,
) -> None:
    """Checkpoint the campaign plan and set total_rounds."""
    now = utcnow_iso()

    def _save(conn):
        conn.execute(
            """UPDATE campaign_state
               SET plan_json = ?, total_rounds = ?, updated_at = ?
               WHERE campaign_id = ?""",
            (json_dumps(plan_dict), total_rounds, now, campaign_id),
        )

    run_txn(_save)


# ---------------------------------------------------------------------------
# Round-level checkpoint
# ---------------------------------------------------------------------------

def start_round(
    campaign_id: str,
    round_num: int,
    strategy: str,
    n_candidates: int,
    strategy_decision: dict[str, Any] | None = None,
) -> None:
    """INSERT a new campaign_rounds row and bump current_round."""
    now = utcnow_iso()

    def _start(conn):
        conn.execute(
            """INSERT OR REPLACE INTO campaign_rounds
               (campaign_id, round_number, status, strategy,
                strategy_decision_json, n_candidates_total, started_at)
               VALUES (?, ?, 'running', ?, ?, ?, ?)""",
            (
                campaign_id,
                round_num,
                strategy,
                json_dumps(strategy_decision) if strategy_decision else None,
                n_candidates,
                now,
            ),
        )
        conn.execute(
            """UPDATE campaign_state
               SET current_round = ?, updated_at = ?
               WHERE campaign_id = ?""",
            (round_num, now, campaign_id),
        )

    run_txn(_start)


def complete_round(
    campaign_id: str,
    round_num: int,
    batch_kpis: list[float],
    batch_params: list[dict[str, Any]],
) -> None:
    """Mark round completed and record batch results."""
    now = utcnow_iso()

    def _complete(conn):
        conn.execute(
            """UPDATE campaign_rounds
               SET status = 'completed',
                   batch_kpis_json = ?, batch_params_json = ?,
                   completed_at = ?
               WHERE campaign_id = ? AND round_number = ?""",
            (
                json_dumps(batch_kpis),
                json_dumps(batch_params),
                now,
                campaign_id,
                round_num,
            ),
        )

    run_txn(_complete)


def load_round_state(
    campaign_id: str, round_num: int
) -> dict[str, Any] | None:
    """Load a single round checkpoint."""
    with connection() as conn:
        row = conn.execute(
            """SELECT * FROM campaign_rounds
               WHERE campaign_id = ? AND round_number = ?""",
            (campaign_id, round_num),
        ).fetchone()
    d = row_to_dict(row)
    if d is None:
        return None
    d["strategy_decision"] = parse_json(d.pop("strategy_decision_json"), None)
    d["batch_kpis"] = parse_json(d.pop("batch_kpis_json"), [])
    d["batch_params"] = parse_json(d.pop("batch_params_json"), [])
    return d


# ---------------------------------------------------------------------------
# Candidate-level checkpoint
# ---------------------------------------------------------------------------

def start_candidate(
    campaign_id: str,
    round_num: int,
    idx: int,
    params: dict[str, Any],
    graph_hash: str | None = None,
) -> None:
    """INSERT a new campaign_candidates row."""
    now = utcnow_iso()

    def _start(conn):
        conn.execute(
            """INSERT OR REPLACE INTO campaign_candidates
               (campaign_id, round_number, candidate_index, status,
                params_json, graph_hash, started_at)
               VALUES (?, ?, ?, 'compiling', ?, ?, ?)""",
            (campaign_id, round_num, idx, json_dumps(params), graph_hash, now),
        )

    run_txn(_start)


def complete_candidate(
    campaign_id: str,
    round_num: int,
    idx: int,
    *,
    kpi: float | None = None,
    run_id: str | None = None,
    qc: str | None = None,
    status: str = "completed",
    error: str | None = None,
) -> None:
    """Update candidate with final results."""
    now = utcnow_iso()

    def _complete(conn):
        conn.execute(
            """UPDATE campaign_candidates
               SET status = ?, kpi_value = ?, run_id = ?,
                   qc_quality = ?, error = ?, completed_at = ?
               WHERE campaign_id = ? AND round_number = ? AND candidate_index = ?""",
            (status, kpi, run_id, qc, error, now, campaign_id, round_num, idx),
        )
        # Bump done count
        conn.execute(
            """UPDATE campaign_rounds
               SET n_candidates_done = n_candidates_done + 1
               WHERE campaign_id = ? AND round_number = ?""",
            (campaign_id, round_num),
        )

    run_txn(_complete)


def is_candidate_done(
    campaign_id: str,
    round_num: int,
    idx: int,
    graph_hash: str | None = None,
) -> bool:
    """Idempotent check — already completed/failed?

    Checks by PK first, then by graph_hash if provided.
    """
    with connection() as conn:
        row = conn.execute(
            """SELECT status FROM campaign_candidates
               WHERE campaign_id = ? AND round_number = ? AND candidate_index = ?""",
            (campaign_id, round_num, idx),
        ).fetchone()
        if row and row["status"] in ("completed", "failed"):
            return True

        if graph_hash:
            row = conn.execute(
                """SELECT status FROM campaign_candidates
                   WHERE campaign_id = ? AND graph_hash = ?
                   AND status IN ('completed', 'failed')
                   LIMIT 1""",
                (campaign_id, graph_hash),
            ).fetchone()
            return row is not None

    return False


def update_candidate_graph_hash(
    campaign_id: str,
    round_num: int,
    idx: int,
    graph_hash: str,
) -> None:
    """Set graph_hash after compilation (not known at start_candidate time)."""
    def _update(conn):
        conn.execute(
            """UPDATE campaign_candidates SET graph_hash = ?
               WHERE campaign_id = ? AND round_number = ? AND candidate_index = ?""",
            (graph_hash, campaign_id, round_num, idx),
        )

    run_txn(_update)


# ---------------------------------------------------------------------------
# KPI checkpoint (bulk state snapshot)
# ---------------------------------------------------------------------------

def checkpoint_kpi(
    campaign_id: str,
    kpi_history: list[float],
    all_kpis: list[float],
    all_params: list[dict[str, Any]],
    all_rounds: list[int],
    best_kpi: float | None,
    total_runs: int,
) -> None:
    """Snapshot accumulated KPI state to campaign_state."""
    now = utcnow_iso()

    def _ckpt(conn):
        conn.execute(
            """UPDATE campaign_state
               SET kpi_history_json = ?, all_kpis_json = ?,
                   all_params_json = ?, all_rounds_json = ?,
                   best_kpi = ?, total_runs = ?, updated_at = ?
               WHERE campaign_id = ?""",
            (
                json_dumps(kpi_history),
                json_dumps(all_kpis),
                json_dumps(all_params),
                json_dumps(all_rounds),
                best_kpi,
                total_runs,
                now,
                campaign_id,
            ),
        )

    run_txn(_ckpt)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def list_incomplete_campaigns() -> list[dict[str, Any]]:
    """Return campaigns in planning/running status (resumable)."""
    with connection() as conn:
        rows = conn.execute(
            """SELECT campaign_id, status, current_round, total_rounds,
                      best_kpi, created_at, updated_at
               FROM campaign_state
               WHERE status IN ('planning', 'running')
               ORDER BY updated_at DESC""",
        ).fetchall()
    return [dict(r) for r in rows]


def load_completed_candidates(
    campaign_id: str,
) -> dict[str, Any]:
    """Rebuild accumulated state from completed candidates for resume.

    Returns dict with keys: kpi_history, all_kpis, all_params, all_rounds,
    best_kpi, total_runs — matching the local vars in process().
    """
    with connection() as conn:
        # Load campaign-level snapshot (the latest checkpoint_kpi data)
        state = conn.execute(
            """SELECT kpi_history_json, all_kpis_json, all_params_json,
                      all_rounds_json, best_kpi, total_runs, direction
               FROM campaign_state WHERE campaign_id = ?""",
            (campaign_id,),
        ).fetchone()

    if state is None:
        return {
            "kpi_history": [],
            "all_kpis": [],
            "all_params": [],
            "all_rounds": [],
            "best_kpi": None,
            "total_runs": 0,
            "direction": "minimize",
        }

    return {
        "kpi_history": parse_json(state["kpi_history_json"], []),
        "all_kpis": parse_json(state["all_kpis_json"], []),
        "all_params": parse_json(state["all_params_json"], []),
        "all_rounds": parse_json(state["all_rounds_json"], []),
        "best_kpi": state["best_kpi"],
        "total_runs": state["total_runs"],
        "direction": state["direction"],
    }


def get_completed_rounds(campaign_id: str) -> list[int]:
    """Return list of round numbers that have status='completed'."""
    with connection() as conn:
        rows = conn.execute(
            """SELECT round_number FROM campaign_rounds
               WHERE campaign_id = ? AND status = 'completed'
               ORDER BY round_number""",
            (campaign_id,),
        ).fetchall()
    return [r["round_number"] for r in rows]
