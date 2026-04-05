"""Governance Layer — Circuit Breaker (cascade failure prevention).

Implements a stateful, three-tier circuit breaker that prevents cascading
errors from propagating from L3 all the way to L0 hardware.

Breaker states (analogous to the classic circuit breaker pattern):
  closed    → normal operation; checks are evaluated
  open      → tripped; all gates return blocked_hard immediately
  half_open → one probe request allowed to test if the issue is resolved

Three-tier baseline strategy
-----------------------------
Tier 1 (round 0)     : Hard bounds check — physics feasibility only.
                        Uses SafetyEnvelope + DimensionDef min/max.
Tier 2 (round 1–2)   : Normalized position check — warn if claim lands
                        in the outer 5% of the declared search space.
Tier 3 (round ≥ 3)   : Statistical outlier check + amplification factor.
                        Uses historical claim values for μ/σ calculation.

The breaker trips when:
  - amplification_factor  > AMPLIFICATION_HARD_THRESHOLD  → blocked_hard
  - amplification_factor  > AMPLIFICATION_SOFT_THRESHOLD  → blocked_soft
  - any hard_block violation is present                   → blocked_hard
  - any soft_block violation is present                   → blocked_soft

State is persisted to governance_circuit_breaker so it survives restarts.
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.db import connection, json_dumps, utcnow_iso
from app.governance.claim_verifier import ClaimVerifier
from app.governance.claim_tracker import ClaimTracker
from app.governance.schemas import (
    BreakCondition,
    Claim,
    CircuitBreakerState,
    GovernanceDecision,
    GovernanceVerdict,
    PolicyViolation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_AMPLIFICATION_SOFT = 5.0
_AMPLIFICATION_HARD = 10.0


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Campaign-scoped circuit breaker with persistent state.

    One breaker instance can manage all campaigns; state is keyed by
    campaign_id in the governance_circuit_breaker table.
    """

    def __init__(self) -> None:
        self._verifier = ClaimVerifier()
        self._tracker = ClaimTracker()

    # ----------------------------------------------------------------- public

    def check(
        self,
        campaign_id: str,
        round_number: int,
        claims: list[Claim],
        safety_envelope: dict[str, Any],
        dim_defs: list[dict[str, Any]],
        kpi_history: list[float],
    ) -> GovernanceDecision:
        """Evaluate *claims* through the three-tier pipeline.

        Returns a single GovernanceDecision representing the worst verdict
        across all claims evaluated.

        Args:
            campaign_id:     Campaign being evaluated.
            round_number:    Current round index (0-based).
            claims:          Claims emitted by agents this round.
            safety_envelope: SafetyEnvelope.model_dump() from TaskContract.
            dim_defs:        List of DimensionDef.model_dump() dicts.
            kpi_history:     Best KPI per previous round (for context).
        """
        # Fast path: breaker is already open
        state = self.get_state(campaign_id)
        if state == "open":
            return GovernanceDecision(
                verdict="blocked_hard",
                triggered_by="open_breaker",
                violations=(),
                warnings=(),
            )

        dim_index = {d["param_name"]: d for d in dim_defs}
        all_violations: list[PolicyViolation] = []
        all_warnings: list[PolicyViolation] = []
        worst_amp = 0.0
        worst_sigma: float | None = None
        triggering_claim: Claim | None = None

        for claim in claims:
            dim_def = dim_index.get(claim.param_name)
            param_history = self._tracker.get_param_history_stats(
                campaign_id, claim.param_name
            )

            violations = self._verifier.verify(
                claim=claim,
                safety_envelope=safety_envelope,
                dim_def=dim_def,
                param_history=param_history,
                round_number=round_number,
            )

            amp = self._tracker.compute_amplification_factor(claim, campaign_id)
            if amp > worst_amp:
                worst_amp = amp
                triggering_claim = claim

            mean, std = param_history
            if (
                mean is not None
                and std is not None
                and claim.param_value is not None
            ):
                sigma = abs(claim.param_value - mean) / std
                if worst_sigma is None or sigma > worst_sigma:
                    worst_sigma = sigma

            for v in violations:
                (all_violations if v.severity != "warning" else all_warnings).append(v)

        # Add amplification-based violations
        if worst_amp >= _AMPLIFICATION_HARD and round_number >= 3:
            all_violations.append(
                PolicyViolation(
                    rule_id="cascade.amplification_hard",
                    message=(
                        f"Claim amplification factor {worst_amp:.2f} "
                        f"exceeds hard threshold {_AMPLIFICATION_HARD}"
                    ),
                    severity="hard_block",
                    observed=worst_amp,
                    threshold=_AMPLIFICATION_HARD,
                )
            )
        elif worst_amp >= _AMPLIFICATION_SOFT and round_number >= 3:
            all_violations.append(
                PolicyViolation(
                    rule_id="cascade.amplification_soft",
                    message=(
                        f"Claim amplification factor {worst_amp:.2f} "
                        f"exceeds soft threshold {_AMPLIFICATION_SOFT}"
                    ),
                    severity="soft_block",
                    observed=worst_amp,
                    threshold=_AMPLIFICATION_SOFT,
                )
            )

        verdict = self._derive_verdict(all_violations, all_warnings)
        decision = GovernanceDecision(
            verdict=verdict,
            triggered_by="circuit_breaker",
            violations=tuple(all_violations),
            warnings=tuple(all_warnings),
            claim=triggering_claim,
            amplification_factor=worst_amp if worst_amp > 0 else None,
            sigma_deviation=worst_sigma,
        )

        # Update breaker state based on verdict
        if verdict == "blocked_hard":
            cond = BreakCondition(
                reason=all_violations[0].message if all_violations else "hard_block",
                amplification_factor=worst_amp,
                sigma_deviation=worst_sigma,
                claim_id=triggering_claim.id if triggering_claim else None,
            )
            self._trip(campaign_id, cond)
        elif verdict in ("approved", "approved_with_warning"):
            self._record_success(campaign_id)

        return decision

    def get_state(self, campaign_id: str) -> CircuitBreakerState:
        """Return the current breaker state for *campaign_id*."""
        try:
            with connection() as conn:
                row = conn.execute(
                    "SELECT state FROM governance_circuit_breaker "
                    "WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchone()
            if row is None:
                return "closed"
            return row["state"]  # type: ignore[return-value]
        except Exception as exc:
            logger.error("get_state failed for %s: %s", campaign_id, exc)
            return "closed"

    def reset(self, campaign_id: str) -> None:
        """Manually reset the breaker to closed (e.g. after human review)."""
        self._upsert_state(
            campaign_id,
            state="closed",
            trip_reason=None,
            trip_condition=None,
            tripped_at=None,
            reset_at=utcnow_iso(),
            consecutive_failures=0,
        )
        logger.info("Circuit breaker reset for campaign %s", campaign_id)

    def half_open(self, campaign_id: str) -> None:
        """Move breaker to half-open to allow one probe request."""
        self._upsert_state(
            campaign_id,
            state="half_open",
            trip_reason=None,
            trip_condition=None,
            tripped_at=None,
            reset_at=None,
            consecutive_failures=None,
        )
        logger.info("Circuit breaker half-opened for campaign %s", campaign_id)

    # --------------------------------------------------------------- internal

    def _trip(self, campaign_id: str, condition: BreakCondition) -> None:
        self._upsert_state(
            campaign_id,
            state="open",
            trip_reason=condition.reason,
            trip_condition=condition.to_dict(),
            tripped_at=utcnow_iso(),
            reset_at=None,
            consecutive_failures=None,  # increment handled in upsert
        )
        logger.warning(
            "Circuit breaker TRIPPED for campaign %s: %s (amp=%.2f)",
            campaign_id,
            condition.reason,
            condition.amplification_factor,
        )

    def _record_success(self, campaign_id: str) -> None:
        """Reset consecutive failure count on a clean pass."""
        try:
            with connection() as conn:
                conn.execute(
                    """
                    UPDATE governance_circuit_breaker
                    SET consecutive_failures = 0, updated_at = ?
                    WHERE campaign_id = ?
                    """,
                    (utcnow_iso(), campaign_id),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("_record_success update skipped: %s", exc)

    def _upsert_state(
        self,
        campaign_id: str,
        *,
        state: CircuitBreakerState,
        trip_reason: str | None,
        trip_condition: dict[str, Any] | None,
        tripped_at: str | None,
        reset_at: str | None,
        consecutive_failures: int | None,
    ) -> None:
        now = utcnow_iso()
        try:
            with connection() as conn:
                existing = conn.execute(
                    "SELECT consecutive_failures FROM governance_circuit_breaker "
                    "WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchone()

                if existing is None:
                    failures = 1 if state == "open" else 0
                    conn.execute(
                        """
                        INSERT INTO governance_circuit_breaker
                            (campaign_id, state, trip_reason, trip_condition_json,
                             tripped_at, reset_at, consecutive_failures, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            campaign_id,
                            state,
                            trip_reason,
                            json_dumps(trip_condition) if trip_condition else None,
                            tripped_at,
                            reset_at,
                            failures,
                            now,
                        ),
                    )
                else:
                    current_failures = int(existing["consecutive_failures"])
                    if consecutive_failures is None:
                        # Increment on trip, keep on others
                        new_failures = (
                            current_failures + 1 if state == "open" else current_failures
                        )
                    else:
                        new_failures = consecutive_failures

                    conn.execute(
                        """
                        UPDATE governance_circuit_breaker
                        SET state = ?, trip_reason = ?, trip_condition_json = ?,
                            tripped_at = COALESCE(?, tripped_at),
                            reset_at = COALESCE(?, reset_at),
                            consecutive_failures = ?,
                            updated_at = ?
                        WHERE campaign_id = ?
                        """,
                        (
                            state,
                            trip_reason,
                            json_dumps(trip_condition) if trip_condition else None,
                            tripped_at,
                            reset_at,
                            new_failures,
                            now,
                            campaign_id,
                        ),
                    )

                conn.commit()
        except Exception as exc:
            logger.error("_upsert_state failed for %s: %s", campaign_id, exc)

    @staticmethod
    def _derive_verdict(
        violations: list[PolicyViolation],
        warnings: list[PolicyViolation],
    ) -> GovernanceVerdict:
        """Derive the most severe verdict from collected violations/warnings."""
        if any(v.severity == "hard_block" for v in violations):
            return "blocked_hard"
        if any(v.severity == "soft_block" for v in violations):
            return "blocked_soft"
        if warnings:
            return "approved_with_warning"
        return "approved"
