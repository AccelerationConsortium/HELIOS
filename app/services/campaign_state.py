"""Campaign state persistence layer — checkpoint / resume for the orchestrator."""

from __future__ import annotations

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
    campaign_context: dict[str, Any] | None = None,
) -> None:
    """INSERT a new campaign_state row at 'planning' status."""
    now = utcnow_iso()

    def _insert(conn):
        conn.execute(
            """INSERT OR IGNORE INTO campaign_state
               (campaign_id, status, input_json, direction, campaign_context_json,
                created_at, updated_at)
               VALUES (?, 'planning', ?, ?, ?, ?, ?)""",
            (
                campaign_id,
                json_dumps(input_data),
                direction,
                json_dumps(campaign_context or {}),
                now,
                now,
            ),
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
    d["backend_failure_counts"] = parse_json(
        d.pop("backend_failure_counts_json", None), {}
    )
    d["all_failed_params"] = parse_json(d.pop("all_failed_params_json", None), [])
    d["bomcp_backend_state"] = parse_json(
        d.pop("bomcp_backend_state_json", None), None
    )
    d["campaign_context"] = parse_json(d.pop("campaign_context_json", None), {})
    d["failure_events"] = parse_json(d.pop("failure_events_json", None), [])
    d["latest_strategy_trace"] = parse_json(
        d.pop("latest_strategy_trace_json", None), None
    )
    d["backend_performance"] = parse_json(
        d.pop("backend_performance_json", None), {}
    )
    d["strategy_bandit"] = parse_json(d.pop("strategy_bandit_json", None), {})
    d["space_revisions"] = parse_json(d.pop("space_revisions_json", None), [])
    d["strategy_rewards"] = parse_json(d.pop("strategy_rewards_json", None), [])
    d["shadow_bandit_records"] = parse_json(
        d.pop("shadow_bandit_records_json", None), []
    )
    d["objective_transitions"] = parse_json(
        d.pop("objective_transitions_json", None), []
    )
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


def save_campaign_context(campaign_id: str, campaign_context: dict[str, Any]) -> None:
    """Persist scientific campaign context without touching loop state."""
    now = utcnow_iso()

    def _save(conn):
        conn.execute(
            """UPDATE campaign_state
               SET campaign_context_json = ?, updated_at = ?
               WHERE campaign_id = ?""",
            (json_dumps(campaign_context), now, campaign_id),
        )

    run_txn(_save)


def append_failure_event(campaign_id: str, failure_event: dict[str, Any]) -> None:
    """Append a typed failure event to campaign context history."""
    now = utcnow_iso()

    def _append(conn):
        row = conn.execute(
            "SELECT failure_events_json FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        events = parse_json(row["failure_events_json"] if row else None, [])
        events.append(failure_event)
        conn.execute(
            """UPDATE campaign_state
               SET failure_events_json = ?, updated_at = ?
               WHERE campaign_id = ?""",
            (json_dumps(events), now, campaign_id),
        )

    run_txn(_append)


def append_space_revision(campaign_id: str, space_revision: dict[str, Any]) -> None:
    """Append a proposed parameter-space revision for audit and future action."""
    now = utcnow_iso()

    def _append(conn):
        row = conn.execute(
            "SELECT space_revisions_json FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        revisions = parse_json(row["space_revisions_json"] if row else None, [])
        revisions.append(space_revision)
        conn.execute(
            """UPDATE campaign_state
               SET space_revisions_json = ?, updated_at = ?
               WHERE campaign_id = ?""",
            (json_dumps(revisions), now, campaign_id),
        )

    run_txn(_append)


def save_strategy_memory(
    campaign_id: str,
    *,
    backend_performance: dict[str, Any] | None = None,
    strategy_bandit: dict[str, Any] | None = None,
) -> None:
    """Persist backend credit assignment and contextual bandit state."""
    now = utcnow_iso()
    set_parts = ["updated_at = ?"]
    values: list[Any] = [now]
    if backend_performance is not None:
        set_parts.append("backend_performance_json = ?")
        values.append(json_dumps(backend_performance))
    if strategy_bandit is not None:
        set_parts.append("strategy_bandit_json = ?")
        values.append(json_dumps(strategy_bandit))
    values.append(campaign_id)

    def _save(conn):
        conn.execute(
            f"UPDATE campaign_state SET {', '.join(set_parts)} WHERE campaign_id = ?",
            values,
        )

    run_txn(_save)


def append_strategy_reward(campaign_id: str, reward: dict[str, Any]) -> None:
    """Append a logging-only strategy reward record."""
    _append_json_list(campaign_id, "strategy_rewards_json", reward)


def append_shadow_bandit_record(campaign_id: str, record: dict[str, Any]) -> None:
    """Append a logging-only shadow bandit evaluation record."""
    _append_json_list(campaign_id, "shadow_bandit_records_json", record)


def append_objective_transition(
    campaign_id: str,
    transition: dict[str, Any],
) -> None:
    """Append a proposed objective hierarchy transition."""
    _append_json_list(campaign_id, "objective_transitions_json", transition)


def _append_json_list(campaign_id: str, column: str, item: dict[str, Any]) -> None:
    now = utcnow_iso()

    def _append(conn):
        row = conn.execute(
            f"SELECT {column} FROM campaign_state WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        items = parse_json(row[column] if row else None, [])
        items.append(item)
        conn.execute(
            f"""UPDATE campaign_state
                SET {column} = ?, updated_at = ?
                WHERE campaign_id = ?""",
            (json_dumps(items), now, campaign_id),
        )

    run_txn(_append)


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
    backend_failure_counts: dict[str, int] | None = None,
    all_failed_params: list[dict[str, Any]] | None = None,
    bomcp_backend_state: dict[str, Any] | None = None,
    latest_strategy_trace: dict[str, Any] | None = None,
) -> None:
    """Snapshot accumulated KPI and dynamic strategy state to campaign_state."""
    now = utcnow_iso()

    def _ckpt(conn):
        set_parts = [
            "kpi_history_json = ?",
            "all_kpis_json = ?",
            "all_params_json = ?",
            "all_rounds_json = ?",
            "best_kpi = ?",
            "total_runs = ?",
            "updated_at = ?",
        ]
        values: list[Any] = [
            json_dumps(kpi_history),
            json_dumps(all_kpis),
            json_dumps(all_params),
            json_dumps(all_rounds),
            best_kpi,
            total_runs,
            now,
        ]
        if backend_failure_counts is not None:
            set_parts.append("backend_failure_counts_json = ?")
            values.append(json_dumps(backend_failure_counts))
        if all_failed_params is not None:
            set_parts.append("all_failed_params_json = ?")
            values.append(json_dumps(all_failed_params))
        if bomcp_backend_state is not None:
            set_parts.append("bomcp_backend_state_json = ?")
            values.append(json_dumps(bomcp_backend_state))
        if latest_strategy_trace is not None:
            set_parts.append("latest_strategy_trace_json = ?")
            values.append(json_dumps(latest_strategy_trace))

        values.append(campaign_id)
        conn.execute(
            f"""UPDATE campaign_state
                SET {', '.join(set_parts)}
                WHERE campaign_id = ?""",
            values,
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
                      all_rounds_json, best_kpi, total_runs, direction,
                      backend_failure_counts_json, all_failed_params_json,
                      bomcp_backend_state_json, campaign_context_json,
                      failure_events_json, latest_strategy_trace_json,
                      backend_performance_json, strategy_bandit_json,
                      space_revisions_json, strategy_rewards_json,
                      shadow_bandit_records_json, objective_transitions_json
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
            "backend_failure_counts": {},
            "all_failed_params": [],
            "bomcp_backend_state": None,
            "campaign_context": {},
            "failure_events": [],
            "latest_strategy_trace": None,
            "backend_performance": {},
            "strategy_bandit": {},
            "space_revisions": [],
            "strategy_rewards": [],
            "shadow_bandit_records": [],
            "objective_transitions": [],
        }

    return {
        "kpi_history": parse_json(state["kpi_history_json"], []),
        "all_kpis": parse_json(state["all_kpis_json"], []),
        "all_params": parse_json(state["all_params_json"], []),
        "all_rounds": parse_json(state["all_rounds_json"], []),
        "best_kpi": state["best_kpi"],
        "total_runs": state["total_runs"],
        "direction": state["direction"],
        "backend_failure_counts": parse_json(state["backend_failure_counts_json"], {}),
        "all_failed_params": parse_json(state["all_failed_params_json"], []),
        "bomcp_backend_state": parse_json(state["bomcp_backend_state_json"], None),
        "campaign_context": parse_json(state["campaign_context_json"], {}),
        "failure_events": parse_json(state["failure_events_json"], []),
        "latest_strategy_trace": parse_json(
            state["latest_strategy_trace_json"], None
        ),
        "backend_performance": parse_json(state["backend_performance_json"], {}),
        "strategy_bandit": parse_json(state["strategy_bandit_json"], {}),
        "space_revisions": parse_json(state["space_revisions_json"], []),
        "strategy_rewards": parse_json(state["strategy_rewards_json"], []),
        "shadow_bandit_records": parse_json(
            state["shadow_bandit_records_json"], []
        ),
        "objective_transitions": parse_json(
            state["objective_transitions_json"], []
        ),
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
