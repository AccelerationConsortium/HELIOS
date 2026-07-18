"""Append-only persistence for scored decisions (RLVR wedge, Phase A / T4).

Turns the in-memory ``CampaignDecisionAccounting`` (trace + outcome + verifiable
reward) into a durable trajectory row, and exports the accumulated rows as JSONL
for offline policy evaluation and group-relative ranking (Phases B/C).

    accounting ──► persist_campaign_trajectory() ──► decision_trajectories row
    decision_trajectories ──► export_trajectories_jsonl() ──► one JSON obj / line

Append-only by construction: every call inserts a fresh uuid-keyed row; nothing
is updated or deleted. Provenance is the point — a decision's score and the
per-signal verifier report that produced it must remain auditable forever.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from app.core import db
from app.services.decision_outcome import CampaignDecisionAccounting

__all__ = [
    "TRAJECTORY_SCHEMA_VERSION",
    "persist_campaign_trajectory",
    "persist_loop_trajectory",
    "load_trajectories",
    "export_trajectories_jsonl",
]

TRAJECTORY_SCHEMA_VERSION = "1"


def _insert_row(
    *,
    campaign_id: str,
    trace_id: str | None,
    round_index: int | None,
    layer: str,
    reward: Any,
    trajectory: dict[str, Any],
) -> str:
    """Insert one append-only trajectory row. Returns the new row id."""
    row_id = f"traj-{uuid4().hex}"
    verifier_report = [v.model_dump() for v in getattr(reward, "verifications", []) or []]

    def _insert(conn: Any) -> None:
        conn.execute(
            """
            INSERT INTO decision_trajectories (
                id, campaign_id, trace_id, round_index, layer,
                rubric_version, reward, process_reward, outcome_reward,
                verifier_report_json, trajectory_json,
                trajectory_schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                campaign_id,
                trace_id,
                round_index,
                layer,
                getattr(reward, "rubric_version", "v0.1_static"),
                reward.reward,
                getattr(reward, "process_reward", 0.0),
                getattr(reward, "outcome_reward", 0.0),
                db.json_dumps(verifier_report),
                db.json_dumps(trajectory),
                TRAJECTORY_SCHEMA_VERSION,
                db.utcnow_iso(),
            ),
        )

    db.run_txn(_insert)
    return row_id


def persist_loop_trajectory(
    campaign_id: str, round_index: int, reward: Any, *, state: dict | None = None
) -> str:
    """Persist a live loop-layer decision (LoopReward) as a trajectory row."""
    return _insert_row(
        campaign_id=campaign_id,
        trace_id=getattr(reward, "iteration_id", None),
        round_index=round_index,
        layer="loop",
        reward=reward,
        trajectory={
            "reward": reward.model_dump(mode="json"),
            "state": state or {},
        },
    )


def persist_campaign_trajectory(
    accounting: CampaignDecisionAccounting, *, layer: str = "campaign"
) -> str:
    """Insert one append-only trajectory row from a decision accounting bundle.

    Returns the new row id. Never updates or deletes.
    """
    outcome = accounting.outcome
    return _insert_row(
        campaign_id=outcome.campaign_id,
        trace_id=outcome.trace_id,
        round_index=outcome.round_index,
        layer=layer,
        reward=accounting.reward,
        trajectory={
            "trace": accounting.trace.model_dump(mode="json"),
            "outcome": outcome.model_dump(mode="json"),
            "reward": accounting.reward.model_dump(mode="json"),
        },
    )


def load_trajectories(campaign_id: str | None = None) -> list[dict[str, Any]]:
    """Return trajectory rows (optionally scoped to one campaign), oldest first."""
    with db.connection() as conn:
        if campaign_id is None:
            rows = conn.execute(
                "SELECT * FROM decision_trajectories ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM decision_trajectories "
                "WHERE campaign_id = ? ORDER BY created_at, id",
                (campaign_id,),
            ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["verifier_report"] = db.parse_json(
            record.pop("verifier_report_json", None), []
        )
        record["trajectory"] = db.parse_json(
            record.pop("trajectory_json", None), {}
        )
        result.append(record)
    return result


def export_trajectories_jsonl(campaign_id: str | None = None) -> str:
    """Serialize trajectories as JSONL (one JSON object per line)."""
    rows = load_trajectories(campaign_id)
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows)
