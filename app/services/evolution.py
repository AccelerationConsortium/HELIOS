"""Evolution Engine — prior tightening, protocol templates, human gate.

Three pillars of concrete data-structure evolution triggered by review outcomes:

1. **Prior Tightening**: Narrow parameter bounds based on Welford stats (mean ± k*stddev)
2. **Protocol Templates**: Versioned template library from high-scoring runs
3. **Human Gate**: All mutations go through proposals (auto-approve small, require human for large)

All operations are advisory — wrapped in try/except, never block run completion.
NO LLM in the critical path — all evolution logic is rule-based.

Pipeline:
1. ``start_evolution_listener()`` subscribes to ``run.reviewed`` events
2. ``process_review_event(run_id)`` fetches review → evolve_priors → maybe_create_template
3. All changes go through ``create_evolution_proposal()`` with auto-approve rules
4. ``candidate_gen.sample_prior_guided()`` reads evolved_priors for tightened bounds
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3 as _sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso
from app.services.audit import record_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema versioning & thresholds
# ---------------------------------------------------------------------------

EVOLUTION_SCHEMA_VERSION = "1"

# Thresholds for auto-approve vs human gate
AUTO_APPROVE_MAGNITUDE = 0.3  # changes < 30% magnitude → auto-approve
PRIOR_TIGHTEN_MIN_SCORE = 70.0  # only tighten priors from score >= 70
TEMPLATE_CREATE_MIN_SCORE = 80.0  # only create templates from score >= 80
PRIOR_K_STDDEV = 2.0  # tighten to mean ± k*stddev
MIN_SAMPLE_COUNT = 5  # need ≥5 samples before evolving priors

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolvedPrior:
    """A tightened parameter bound derived from run outcomes."""

    id: str
    primitive: str
    param_name: str
    evolved_min: float
    evolved_max: float
    confidence: float
    source_run_id: str
    proposal_id: str | None
    generation: int
    is_active: bool


@dataclass(frozen=True)
class EvolutionProposal:
    """A proposed mutation to the system — may require human approval."""

    id: str
    run_id: str
    proposal_type: str  # "prior_tightening" | "template_creation"
    change_summary: str
    change_details: dict[str, Any]
    magnitude: float  # 0.0-1.0
    status: str  # "pending" | "approved" | "rejected" | "auto_approved"


# ---------------------------------------------------------------------------
# Pillar 1: Prior Tightening
# ---------------------------------------------------------------------------


def _compute_tightened_bounds(
    primitive: str, param_name: str, k: float = PRIOR_K_STDDEV
) -> tuple[float, float, float] | None:
    """Query memory_semantic and compute tightened bounds.

    Returns (evolved_min, evolved_max, confidence) or None if insufficient data.
    """
    from app.services.memory import get_param_priors

    prior = get_param_priors(primitive, param_name)
    if prior is None or prior.sample_count < MIN_SAMPLE_COUNT:
        return None

    stddev = prior.stddev if prior.stddev > 0 else abs(prior.mean * 0.1)
    evolved_min = prior.mean - k * stddev
    evolved_max = prior.mean + k * stddev

    # Confidence is based on sample count — more samples = higher confidence
    confidence = min(1.0, prior.sample_count / 20.0)

    return evolved_min, evolved_max, confidence


def _calc_prior_magnitude(
    old_min: float, old_max: float, new_min: float, new_max: float
) -> float:
    """Calculate magnitude of a prior tightening change.

    Magnitude = 1 - (new_range / old_range), capped [0, 1].
    Higher = bigger change.
    """
    old_range = abs(old_max - old_min)
    new_range = abs(new_max - new_min)
    if old_range <= 0:
        return 0.0
    magnitude = 1.0 - (new_range / old_range)
    return max(0.0, min(1.0, magnitude))


def evolve_priors(run_id: str, review_data: dict[str, Any]) -> list[str]:
    """Pillar 1: Compute tightened bounds from review + memory_semantic.

    Only tightens priors when verdict == "passed" and score >= threshold.
    Returns list of created proposal IDs.
    """
    score = review_data.get("score", 0)
    verdict = review_data.get("verdict", "")

    if score < PRIOR_TIGHTEN_MIN_SCORE or verdict != "passed":
        return []

    # Find improvements with category "parameter" to identify target params
    improvements = review_data.get("improvements", [])
    # Also look at failure_attributions for primitives we can tighten
    attributions = review_data.get("failure_attributions", [])

    # Gather all primitives from the run's steps
    run_data = _get_run_steps(run_id)
    if not run_data:
        return []

    proposal_ids: list[str] = []

    for step in run_data:
        primitive = step.get("primitive", "")
        params = parse_json(step.get("params_json"), {})

        for param_name, value in params.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue

            bounds = _compute_tightened_bounds(primitive, param_name)
            if bounds is None:
                continue

            evolved_min, evolved_max, confidence = bounds

            # Get current active prior to compute magnitude
            current = get_active_evolved_prior(primitive, param_name)
            if current is not None:
                old_min, old_max = current.evolved_min, current.evolved_max
                generation = current.generation + 1
            else:
                # Use a wide default range based on the computed bounds
                old_min = evolved_min - abs(evolved_max - evolved_min)
                old_max = evolved_max + abs(evolved_max - evolved_min)
                generation = 1

            magnitude = _calc_prior_magnitude(old_min, old_max, evolved_min, evolved_max)

            # Create proposal
            proposal_id = create_evolution_proposal(
                run_id=run_id,
                proposal_type="prior_tightening",
                change_summary=f"Tighten {primitive}.{param_name} to [{evolved_min:.4f}, {evolved_max:.4f}]",
                change_details={
                    "primitive": primitive,
                    "param_name": param_name,
                    "evolved_min": evolved_min,
                    "evolved_max": evolved_max,
                    "confidence": confidence,
                    "generation": generation,
                    "old_min": old_min,
                    "old_max": old_max,
                },
                magnitude=magnitude,
            )
            proposal_ids.append(proposal_id)

    return proposal_ids


def _get_run_steps(run_id: str) -> list[dict[str, Any]]:
    """Fetch run steps for prior evolution analysis."""

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT step_key, primitive, params_json, status "
            "FROM run_steps WHERE run_id = ? AND status = 'succeeded'",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    try:
        return run_txn(_txn)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pillar 2: Protocol Template Library
# ---------------------------------------------------------------------------


def _next_template_version(name: str, conn: _sqlite3.Connection) -> int:
    """Query max version for name, return +1."""
    row = conn.execute(
        "SELECT MAX(version) as max_v FROM protocol_templates WHERE name = ?",
        (name,),
    ).fetchone()
    current = row["max_v"] if row and row["max_v"] is not None else 0
    return current + 1


def maybe_create_template(run_id: str, review_data: dict[str, Any]) -> str | None:
    """Pillar 2: If score >= threshold, snapshot protocol as versioned template.

    Returns proposal_id or None if skipped.
    """
    score = review_data.get("score", 0)
    if score < TEMPLATE_CREATE_MIN_SCORE:
        return None

    # Fetch the run's protocol
    protocol = _get_run_protocol(run_id)
    if protocol is None:
        return None

    # Generate a template name from the run's campaign or a default
    campaign_id = _get_run_campaign_id(run_id)
    name = f"auto-{campaign_id}" if campaign_id else f"auto-{run_id[:8]}"

    magnitude = 1.0 - (score / 100.0)  # Higher score = lower magnitude

    proposal_id = create_evolution_proposal(
        run_id=run_id,
        proposal_type="template_creation",
        change_summary=f"Create template '{name}' from run with score {score:.1f}",
        change_details={
            "name": name,
            "protocol": protocol,
            "score": score,
            "campaign_id": campaign_id,
        },
        magnitude=magnitude,
    )
    return proposal_id


def _get_run_protocol(run_id: str) -> dict[str, Any] | None:
    """Fetch the protocol JSON for a run."""

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT protocol_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return parse_json(row["protocol_json"], None)

    try:
        return run_txn(_txn)
    except Exception:
        return None


def _get_run_campaign_id(run_id: str) -> str | None:
    """Fetch the campaign_id for a run."""

    def _txn(conn: _sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT campaign_id FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        return row["campaign_id"] if row else None

    try:
        return run_txn(_txn)
    except Exception:
        return None


def create_template(
    *,
    name: str,
    protocol: dict[str, Any],
    parent_template_id: str | None = None,
    tags: list[str] | None = None,
    source_run_id: str | None = None,
    score: float | None = None,
    created_by: str = "system",
) -> dict[str, Any]:
    """Create a protocol template manually (API-driven)."""
    now = utcnow_iso()
    template_id = str(uuid.uuid4())

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any]:
        version = _next_template_version(name, conn)
        conn.execute(
            "INSERT INTO protocol_templates "
            "(id, name, version, parent_template_id, protocol_json, "
            "source_run_id, score, tags_json, is_active, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (
                template_id,
                name,
                version,
                parent_template_id,
                json_dumps(protocol),
                source_run_id,
                score,
                json_dumps(tags or []),
                created_by,
                now,
            ),
        )
        return {
            "id": template_id,
            "name": name,
            "version": version,
            "parent_template_id": parent_template_id,
            "protocol": protocol,
            "source_run_id": source_run_id,
            "score": score,
            "tags": tags or [],
            "is_active": True,
            "created_by": created_by,
            "created_at": now,
        }

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Pillar 3: Human Gate (Evolution Proposals)
# ---------------------------------------------------------------------------


def _should_auto_approve(
    proposal_type: str, magnitude: float
) -> tuple[bool, str | None]:
    """Rule engine: determine if a proposal can be auto-approved.

    Returns (should_auto_approve, reason).
    """
    if magnitude < AUTO_APPROVE_MAGNITUDE:
        return True, f"magnitude {magnitude:.2f} < threshold {AUTO_APPROVE_MAGNITUDE}"
    if proposal_type == "template_creation" and magnitude < 0.5:
        return True, "template from high-score run"
    return False, None


def create_evolution_proposal(
    *,
    run_id: str,
    proposal_type: str,
    change_summary: str,
    change_details: dict[str, Any],
    magnitude: float,
) -> str:
    """Create a proposal and apply auto-approve rules.

    Returns the proposal_id.
    """
    now = utcnow_iso()
    proposal_id = str(uuid.uuid4())

    auto_approve, reason = _should_auto_approve(proposal_type, magnitude)

    def _txn(conn: _sqlite3.Connection) -> None:
        status = "auto_approved" if auto_approve else "pending"
        conn.execute(
            "INSERT INTO evolution_proposals "
            "(id, run_id, proposal_type, change_summary, change_details_json, "
            "magnitude, status, auto_approve_reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                proposal_id,
                run_id,
                proposal_type,
                change_summary,
                json_dumps(change_details),
                magnitude,
                status,
                reason,
                now,
            ),
        )
        record_event(
            conn,
            run_id=run_id,
            actor="evolution",
            action="evolution.proposal_created",
            details={
                "proposal_id": proposal_id,
                "proposal_type": proposal_type,
                "magnitude": magnitude,
                "status": status,
            },
        )
        # If auto-approved, apply immediately
        if auto_approve:
            _apply_proposal(proposal_id, conn)

    run_txn(_txn)
    return proposal_id


def approve_proposal(proposal_id: str, reviewer: str, reason: str | None = None) -> dict[str, Any]:
    """Human approves a pending proposal → apply changes."""
    now = utcnow_iso()

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM evolution_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Proposal not found: {proposal_id}")
        if row["status"] != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is '{row['status']}', expected 'pending'"
            )

        conn.execute(
            "UPDATE evolution_proposals SET status = 'approved', "
            "reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (reviewer, now, proposal_id),
        )
        record_event(
            conn,
            run_id=row["run_id"],
            actor=reviewer,
            action="evolution.proposal_approved",
            details={"proposal_id": proposal_id, "reason": reason},
        )
        _apply_proposal(proposal_id, conn)

        # Re-read the updated proposal
        updated = conn.execute(
            "SELECT * FROM evolution_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        result = dict(updated)
        result["change_details"] = parse_json(result.pop("change_details_json"), {})
        return result

    return run_txn(_txn)


def reject_proposal(proposal_id: str, reviewer: str, reason: str | None = None) -> dict[str, Any]:
    """Human rejects a pending proposal."""
    now = utcnow_iso()

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM evolution_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Proposal not found: {proposal_id}")
        if row["status"] != "pending":
            raise ValueError(
                f"Proposal {proposal_id} is '{row['status']}', expected 'pending'"
            )

        conn.execute(
            "UPDATE evolution_proposals SET status = 'rejected', "
            "reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (reviewer, now, proposal_id),
        )
        record_event(
            conn,
            run_id=row["run_id"],
            actor=reviewer,
            action="evolution.proposal_rejected",
            details={"proposal_id": proposal_id, "reason": reason},
        )

        updated = conn.execute(
            "SELECT * FROM evolution_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        result = dict(updated)
        result["change_details"] = parse_json(result.pop("change_details_json"), {})
        return result

    return run_txn(_txn)


def _apply_proposal(proposal_id: str, conn: _sqlite3.Connection) -> None:
    """Execute the actual data changes for an approved/auto-approved proposal."""
    row = conn.execute(
        "SELECT * FROM evolution_proposals WHERE id = ?",
        (proposal_id,),
    ).fetchone()
    if row is None:
        return

    proposal_type = row["proposal_type"]
    details = parse_json(row["change_details_json"], {})
    run_id = row["run_id"]
    now = utcnow_iso()

    if proposal_type == "prior_tightening":
        primitive = details["primitive"]
        param_name = details["param_name"]

        # Deactivate previous active prior for this (primitive, param_name)
        conn.execute(
            "UPDATE evolved_priors SET is_active = 0 "
            "WHERE primitive = ? AND param_name = ? AND is_active = 1",
            (primitive, param_name),
        )

        # Insert new evolved prior
        prior_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO evolved_priors "
            "(id, primitive, param_name, evolved_min, evolved_max, confidence, "
            "source_run_id, proposal_id, generation, is_active, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                prior_id,
                primitive,
                param_name,
                details["evolved_min"],
                details["evolved_max"],
                details["confidence"],
                run_id,
                proposal_id,
                details.get("generation", 1),
                now,
            ),
        )

    elif proposal_type == "template_creation":
        template_id = str(uuid.uuid4())
        name = details["name"]
        version = _next_template_version(name, conn)

        conn.execute(
            "INSERT INTO protocol_templates "
            "(id, name, version, parent_template_id, protocol_json, "
            "source_run_id, score, tags_json, is_active, created_by, created_at) "
            "VALUES (?, ?, ?, NULL, ?, ?, ?, '[]', 1, 'evolution', ?)",
            (
                template_id,
                name,
                version,
                json_dumps(details["protocol"]),
                run_id,
                details.get("score"),
                now,
            ),
        )


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


def get_active_evolved_prior(
    primitive: str, param_name: str
) -> EvolvedPrior | None:
    """Return the latest active evolved prior for a (primitive, param_name)."""

    def _txn(conn: _sqlite3.Connection) -> EvolvedPrior | None:
        row = conn.execute(
            "SELECT * FROM evolved_priors "
            "WHERE primitive = ? AND param_name = ? AND is_active = 1 "
            "ORDER BY generation DESC LIMIT 1",
            (primitive, param_name),
        ).fetchone()
        if row is None:
            return None
        return EvolvedPrior(
            id=row["id"],
            primitive=row["primitive"],
            param_name=row["param_name"],
            evolved_min=row["evolved_min"],
            evolved_max=row["evolved_max"],
            confidence=row["confidence"],
            source_run_id=row["source_run_id"],
            proposal_id=row["proposal_id"],
            generation=row["generation"],
            is_active=bool(row["is_active"]),
        )

    return run_txn(_txn)


def list_evolved_priors(
    primitive: str | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """List evolved priors with optional filters."""

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if primitive:
            conditions.append("primitive = ?")
            params.append(primitive)
        if active_only:
            conditions.append("is_active = 1")

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM evolved_priors WHERE {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    return run_txn(_txn)


def get_template(template_id: str) -> dict[str, Any] | None:
    """Return a single protocol template."""

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM protocol_templates WHERE id = ?",
            (template_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["protocol"] = parse_json(item.pop("protocol_json"), {})
        item["tags"] = parse_json(item.pop("tags_json"), [])
        item["is_active"] = bool(item["is_active"])
        return item

    return run_txn(_txn)


def list_templates(
    name: str | None = None,
    is_active: bool | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List protocol templates with optional filters."""

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if name:
            conditions.append("name = ?")
            params.append(name)
        if is_active is not None:
            conditions.append("is_active = ?")
            params.append(1 if is_active else 0)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM protocol_templates WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["protocol"] = parse_json(item.pop("protocol_json"), {})
            item["tags"] = parse_json(item.pop("tags_json"), [])
            item["is_active"] = bool(item["is_active"])
            result.append(item)
        return result

    return run_txn(_txn)


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    """Return a single evolution proposal."""

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM evolution_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["change_details"] = parse_json(item.pop("change_details_json"), {})
        return item

    return run_txn(_txn)


def list_proposals(
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List evolution proposals with optional filters."""

    def _txn(conn: _sqlite3.Connection) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM evolution_proposals WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["change_details"] = parse_json(item.pop("change_details_json"), {})
            result.append(item)
        return result

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Main entry — process a review event
# ---------------------------------------------------------------------------


def process_review_event(run_id: str) -> None:
    """Main entry: fetch review → evolve_priors → maybe_create_template.

    Called by the evolution listener when a ``run.reviewed`` event arrives.
    Advisory — never raises.
    """
    from app.services.reviewer import get_run_review

    review = get_run_review(run_id)
    if review is None:
        logger.debug("No review found for run %s — skipping evolution", run_id)
        return

    # Build review_data dict for evolution functions
    review_data = {
        "score": review["score"],
        "verdict": review["verdict"],
        "improvements": review.get("improvements", []),
        "failure_attributions": review.get("failure_attributions", []),
    }

    # Pillar 1: Prior tightening
    try:
        proposal_ids = evolve_priors(run_id, review_data)
        if proposal_ids:
            logger.debug(
                "Created %d prior evolution proposals for run %s",
                len(proposal_ids),
                run_id,
            )
    except Exception:
        logger.warning("Prior evolution failed for run %s", run_id, exc_info=True)

    # Pillar 2: Template creation
    try:
        template_proposal = maybe_create_template(run_id, review_data)
        if template_proposal:
            logger.debug(
                "Created template proposal %s for run %s",
                template_proposal,
                run_id,
            )
    except Exception:
        logger.warning("Template creation failed for run %s", run_id, exc_info=True)


# ---------------------------------------------------------------------------
# Event listener — async
# ---------------------------------------------------------------------------

_listener_task: asyncio.Task[None] | None = None


async def _on_run_reviewed(run_id: str) -> None:
    """Process a reviewed run. Advisory — never blocks."""
    try:
        process_review_event(run_id)
        logger.debug("Evolution processed for run %s", run_id)
    except Exception:
        logger.warning("Evolution failed for run %s", run_id, exc_info=True)


async def start_evolution_listener(bus: Any) -> Any:
    """Subscribe to the event bus and process run.reviewed events.

    Returns the Subscription handle for cleanup.
    """
    global _listener_task

    sub = await bus.subscribe(run_id=None)  # global subscription

    async def _listen() -> None:
        async for event in sub:
            if event.action == "run.reviewed":
                run_id = event.run_id
                if run_id:
                    await _on_run_reviewed(run_id)

    _listener_task = asyncio.create_task(_listen())
    return sub


async def stop_evolution_listener(sub: Any, bus: Any) -> None:
    """Cancel the evolution listener and unsubscribe."""
    global _listener_task

    sub.cancel()
    await bus.unsubscribe(sub)

    if _listener_task is not None:
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None
