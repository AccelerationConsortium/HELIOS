"""Governance Layer — Saga Coordinator (compensating transactions).

Implements the Saga pattern for the multi-layer pipeline:

    L3 (InverseDesignAgent) → L2 (DesignAgent) → L1 (CompilerAgent) → L0 (Hardware)

Each layer registers a step when it begins work.  If a failure is detected
at any layer, rollback() triggers compensation from that layer back up to L3,
ultimately emitting a RevisionRequest so InverseDesignAgent can re-design
with tighter constraints informed by the failure context.

Key invariants
--------------
- Steps are committed in order (L3 → L2 → L1 → L0, step_order 0, 1, 2, 3).
- Rollback reverses order (L0 → L3, highest step_order first).
- A saga that has already been compensated cannot be rolled back again.
- All saga state is persisted in governance_saga_steps for durability.

Usage
-----
saga_id = coordinator.begin_saga(campaign_id, round_number)

# At each layer:
step = coordinator.register_step(saga_id, ..., layer="L3", agent_name="inverse_design_agent")
coordinator.commit_step(saga_id, step.id)

# If a failure is detected at L2:
revision_req = coordinator.rollback(
    saga_id,
    from_layer="L2",
    reason="DesignAgent rejected all candidates",
    failed_claims=failing_claims,
)
# Pass revision_req back to InverseDesignAgent
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.core.db import connection, json_dumps, utcnow_iso
from app.governance.schemas import (
    Claim,
    RevisionRequest,
    SagaStep,
    SagaStepStatus,
)

logger = logging.getLogger(__name__)

# Layer ordering (lower index = closer to L3)
_LAYER_ORDER: list[str] = ["L3", "L2", "L1", "L0"]


def _layer_index(layer: str) -> int:
    try:
        return _LAYER_ORDER.index(layer)
    except ValueError:
        return len(_LAYER_ORDER)


# ---------------------------------------------------------------------------
# SagaCoordinator
# ---------------------------------------------------------------------------

class SagaCoordinator:
    """Manages compensating transaction chains across pipeline layers.

    One coordinator instance is shared across all campaigns; isolation is
    maintained by saga_id + campaign_id keys.
    """

    # ----------------------------------------------------------------- begin

    def begin_saga(self, campaign_id: str, round_number: int) -> str:
        """Start a new saga for a campaign round.

        Returns the saga_id to be passed to subsequent register/commit calls.
        """
        saga_id = f"saga-{uuid.uuid4().hex[:10]}"
        logger.debug(
            "Saga begun: %s (campaign=%s round=%d)",
            saga_id,
            campaign_id,
            round_number,
        )
        return saga_id

    # --------------------------------------------------------------- register

    def register_step(
        self,
        saga_id: str,
        campaign_id: str,
        round_number: int,
        layer: str,
        agent_name: str,
        compensation_payload: dict[str, Any] | None = None,
    ) -> SagaStep:
        """Register a pipeline step under this saga.

        Args:
            saga_id:              Saga identifier from begin_saga().
            campaign_id:          Campaign this saga belongs to.
            round_number:         Current campaign round.
            layer:                "L0" | "L1" | "L2" | "L3"
            agent_name:           Agent handling this layer.
            compensation_payload: Data needed to undo this step's effects
                                  (e.g. candidate IDs, parameter snapshots).

        Returns:
            The SagaStep in "pending" state.
        """
        step_id = f"ss-{uuid.uuid4().hex[:10]}"
        step_order = _layer_index(layer)
        payload = compensation_payload or {}

        step = SagaStep(
            id=step_id,
            saga_id=saga_id,
            layer=layer,
            agent_name=agent_name,
            step_order=step_order,
            status="pending",
            compensation_payload=payload,
        )

        try:
            with connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO governance_saga_steps (
                        id, saga_id, campaign_id, round_number,
                        layer, agent_name, step_order, status,
                        compensation_payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        step.id,
                        saga_id,
                        campaign_id,
                        round_number,
                        layer,
                        agent_name,
                        step_order,
                        "pending",
                        json_dumps(payload),
                        step.created_at,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("register_step failed saga=%s layer=%s: %s", saga_id, layer, exc)
            raise

        logger.debug(
            "Saga step registered: saga=%s layer=%s agent=%s order=%d",
            saga_id,
            layer,
            agent_name,
            step_order,
        )
        return step

    # ----------------------------------------------------------------- commit

    def commit_step(self, saga_id: str, step_id: str) -> None:
        """Mark a step as committed (its effects are now in place)."""
        now = utcnow_iso()
        try:
            with connection() as conn:
                conn.execute(
                    """
                    UPDATE governance_saga_steps
                    SET status = 'committed', committed_at = ?
                    WHERE id = ? AND saga_id = ?
                    """,
                    (now, step_id, saga_id),
                )
                conn.commit()
        except Exception as exc:
            logger.error(
                "commit_step failed saga=%s step=%s: %s", saga_id, step_id, exc
            )
            raise

        logger.debug("Saga step committed: saga=%s step=%s", saga_id, step_id)

    # --------------------------------------------------------------- rollback

    def rollback(
        self,
        saga_id: str,
        from_layer: str,
        reason: str,
        failed_claims: list[Claim],
    ) -> RevisionRequest:
        """Roll back committed steps from *from_layer* back to L3.

        Steps are compensated in reverse order (highest step_order first).
        For each committed step, status transitions:
            committed → compensating → compensated

        Returns a RevisionRequest to be passed to InverseDesignAgent (L3),
        containing the failure context so it can revise candidates.
        """
        # Load all committed steps for this saga at or above from_layer
        from_order = _layer_index(from_layer)
        steps = self._load_committed_steps(saga_id, from_order)

        if not steps:
            logger.warning(
                "Saga rollback for %s found no committed steps at/above %s",
                saga_id,
                from_layer,
            )

        # Compensate in reverse order
        for step in sorted(steps, key=lambda s: s.step_order, reverse=True):
            self._compensate_step(step)

        # Derive suggested constraints from failed claim patterns
        suggested_constraints = self._derive_constraints(failed_claims)

        # Load campaign_id + round_number from the first step
        campaign_id, round_number = self._load_saga_context(saga_id)

        revision_req = RevisionRequest(
            saga_id=saga_id,
            campaign_id=campaign_id,
            round_number=round_number,
            failed_layer=from_layer,
            failure_summary=reason,
            failed_claims=tuple(failed_claims),
            suggested_constraints=suggested_constraints,
        )

        logger.warning(
            "Saga rollback complete: saga=%s from_layer=%s reason=%r "
            "steps_compensated=%d",
            saga_id,
            from_layer,
            reason,
            len(steps),
        )
        return revision_req

    # --------------------------------------------------------------- queries

    def get_steps(self, saga_id: str) -> list[SagaStep]:
        """Return all steps for this saga ordered by step_order."""
        try:
            with connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM governance_saga_steps
                    WHERE saga_id = ?
                    ORDER BY step_order ASC
                    """,
                    (saga_id,),
                ).fetchall()
            return [self._row_to_step(r) for r in rows]
        except Exception as exc:
            logger.error("get_steps failed saga=%s: %s", saga_id, exc)
            return []

    # --------------------------------------------------------------- internal

    def _load_committed_steps(
        self, saga_id: str, from_order: int
    ) -> list[SagaStep]:
        """Load committed steps at or above *from_order* (i.e. L3 and earlier)."""
        try:
            with connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM governance_saga_steps
                    WHERE saga_id = ?
                      AND status = 'committed'
                      AND step_order <= ?
                    ORDER BY step_order ASC
                    """,
                    (saga_id, from_order),
                ).fetchall()
            return [self._row_to_step(r) for r in rows]
        except Exception as exc:
            logger.error(
                "_load_committed_steps failed saga=%s: %s", saga_id, exc
            )
            return []

    def _compensate_step(self, step: SagaStep) -> None:
        """Transition one step through compensating → compensated."""
        now = utcnow_iso()
        try:
            with connection() as conn:
                conn.execute(
                    """
                    UPDATE governance_saga_steps
                    SET status = 'compensating'
                    WHERE id = ?
                    """,
                    (step.id,),
                )
                conn.commit()

            # Execute compensation logic (currently a structured log;
            # real implementations would call agent.compensate())
            logger.info(
                "Compensating saga step: saga=%s layer=%s agent=%s payload=%s",
                step.saga_id,
                step.layer,
                step.agent_name,
                step.compensation_payload,
            )

            with connection() as conn:
                conn.execute(
                    """
                    UPDATE governance_saga_steps
                    SET status = 'compensated', compensated_at = ?
                    WHERE id = ?
                    """,
                    (now, step.id),
                )
                conn.commit()
        except Exception as exc:
            logger.error(
                "_compensate_step failed for step %s: %s", step.id, exc
            )
            with connection() as conn:
                conn.execute(
                    "UPDATE governance_saga_steps SET status = 'failed' WHERE id = ?",
                    (step.id,),
                )
                conn.commit()

    def _load_saga_context(self, saga_id: str) -> tuple[str, int]:
        """Return (campaign_id, round_number) from the first registered step."""
        try:
            with connection() as conn:
                row = conn.execute(
                    """
                    SELECT campaign_id, round_number
                    FROM governance_saga_steps
                    WHERE saga_id = ?
                    ORDER BY step_order ASC
                    LIMIT 1
                    """,
                    (saga_id,),
                ).fetchone()
            if row:
                return str(row["campaign_id"]), int(row["round_number"])
        except Exception as exc:
            logger.error("_load_saga_context failed saga=%s: %s", saga_id, exc)
        return "unknown", 0

    @staticmethod
    def _derive_constraints(failed_claims: list[Claim]) -> dict[str, Any]:
        """Build suggested_constraints from the failed claims.

        For numeric claims, suggests tightening the upper bound to 90% of
        the failing value as a conservative starting point.
        """
        constraints: dict[str, Any] = {}
        for claim in failed_claims:
            if claim.param_value is not None:
                constraints[claim.param_name] = {
                    "suggested_max": claim.param_value * 0.9,
                    "reason": f"Failed at {claim.param_value}; tighten to 90%",
                    "source_claim_id": claim.id,
                }
            elif claim.param_value_str is not None:
                constraints[claim.param_name] = {
                    "exclude_value": claim.param_value_str,
                    "reason": f"Categorical value {claim.param_value_str!r} caused failure",
                    "source_claim_id": claim.id,
                }
        return constraints

    @staticmethod
    def _row_to_step(row: Any) -> SagaStep:
        import json
        payload: dict[str, Any] = json.loads(
            row["compensation_payload_json"] or "{}"
        )
        return SagaStep(
            id=row["id"],
            saga_id=row["saga_id"],
            layer=row["layer"],
            agent_name=row["agent_name"],
            step_order=int(row["step_order"]),
            status=row["status"],
            compensation_payload=payload,
            committed_at=row["committed_at"],
            compensated_at=row["compensated_at"],
            created_at=row["created_at"],
        )
