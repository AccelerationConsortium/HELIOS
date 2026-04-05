"""Governance Layer — Claim Tracker (Event Sourcing / Lineage DAG).

Persists claims emitted by agents and reconstructs their causal ancestry.
Analogous to Event Sourcing: every claim is an immutable fact appended to
the governance_claims table; edges are encoded as parent_trace_ids.

Public API
----------
ClaimTracker.record()                  : persist a new claim
ClaimTracker.get_lineage()             : full ancestor chain + amplification
ClaimTracker.compute_amplification_factor() : cascade risk metric
init_governance_schema()               : create governance DB tables
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from app.core.db import connection, json_dumps, utcnow_iso
from app.governance.schemas import Claim, ClaimLineage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — governance tables (idempotent; run at startup)
# ---------------------------------------------------------------------------

_GOVERNANCE_DDL = """
CREATE TABLE IF NOT EXISTS governance_claims (
    id                    TEXT PRIMARY KEY,
    campaign_id           TEXT NOT NULL,
    round_number          INTEGER NOT NULL,
    trace_id              TEXT NOT NULL UNIQUE,
    emitting_agent        TEXT NOT NULL,
    param_name            TEXT NOT NULL,
    param_value           REAL,
    param_value_str       TEXT,
    source_type           TEXT NOT NULL,
    source_ref            TEXT NOT NULL DEFAULT '',
    confidence            REAL NOT NULL DEFAULT 1.0,
    parent_trace_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gov_claims_campaign
    ON governance_claims(campaign_id, round_number);
CREATE INDEX IF NOT EXISTS idx_gov_claims_trace
    ON governance_claims(trace_id);
CREATE INDEX IF NOT EXISTS idx_gov_claims_param
    ON governance_claims(campaign_id, param_name);

CREATE TABLE IF NOT EXISTS governance_decisions (
    id                   TEXT PRIMARY KEY,
    campaign_id          TEXT NOT NULL,
    round_number         INTEGER NOT NULL,
    claim_id             TEXT,
    verdict              TEXT NOT NULL,
    triggered_by         TEXT NOT NULL,
    violations_json      TEXT NOT NULL DEFAULT '[]',
    warnings_json        TEXT NOT NULL DEFAULT '[]',
    amplification_factor REAL,
    sigma_deviation      REAL,
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gov_decisions_campaign
    ON governance_decisions(campaign_id, round_number);

CREATE TABLE IF NOT EXISTS governance_circuit_breaker (
    campaign_id           TEXT PRIMARY KEY,
    state                 TEXT NOT NULL DEFAULT 'closed',
    trip_reason           TEXT,
    trip_condition_json   TEXT,
    tripped_at            TEXT,
    reset_at              TEXT,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS governance_saga_steps (
    id                       TEXT PRIMARY KEY,
    saga_id                  TEXT NOT NULL,
    campaign_id              TEXT NOT NULL,
    round_number             INTEGER NOT NULL,
    layer                    TEXT NOT NULL,
    agent_name               TEXT NOT NULL,
    step_order               INTEGER NOT NULL,
    status                   TEXT NOT NULL DEFAULT 'pending',
    compensation_payload_json TEXT NOT NULL DEFAULT '{}',
    committed_at             TEXT,
    compensated_at           TEXT,
    created_at               TEXT NOT NULL,
    UNIQUE (saga_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_saga_steps_saga
    ON governance_saga_steps(saga_id, step_order);
CREATE INDEX IF NOT EXISTS idx_saga_steps_campaign
    ON governance_saga_steps(campaign_id, round_number);
"""


def init_governance_schema() -> None:
    """Create all governance tables if they do not exist.

    Idempotent — safe to call on every startup.
    """
    with connection() as conn:
        conn.executescript(_GOVERNANCE_DDL)
        conn.commit()
    logger.info("Governance schema initialised")


# ---------------------------------------------------------------------------
# ClaimTracker
# ---------------------------------------------------------------------------

class ClaimTracker:
    """Persistent lineage DAG for agent-emitted claims.

    Design
    ------
    - Each Claim maps to one row in governance_claims.
    - Parent references are stored as a JSON array of trace_ids
      (not a separate edges table) to keep queries simple.
    - Ancestor traversal is iterative BFS to avoid recursion limits on
      deep chains.
    """

    # ------------------------------------------------------------------ write

    def record(
        self,
        campaign_id: str,
        round_number: int,
        claim: Claim,
    ) -> None:
        """Persist *claim* to governance_claims.

        Silently skips duplicate trace_ids (idempotent).
        """
        try:
            with connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO governance_claims (
                        id, campaign_id, round_number, trace_id,
                        emitting_agent, param_name,
                        param_value, param_value_str,
                        source_type, source_ref, confidence,
                        parent_trace_ids_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim.id,
                        campaign_id,
                        round_number,
                        claim.trace_id,
                        claim.emitting_agent,
                        claim.param_name,
                        claim.param_value,
                        claim.param_value_str,
                        claim.source_type,
                        claim.source_ref,
                        claim.confidence,
                        json_dumps(list(claim.parent_trace_ids)),
                        claim.created_at,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("Failed to record claim %s: %s", claim.id, exc)
            raise

        logger.debug(
            "Claim recorded: id=%s param=%s value=%s agent=%s round=%d",
            claim.id,
            claim.param_name,
            claim.display_value(),
            claim.emitting_agent,
            round_number,
        )

    def record_decision(
        self,
        campaign_id: str,
        round_number: int,
        decision: "GovernanceDecision",  # noqa: F821 — forward ref OK at runtime
    ) -> None:
        """Persist a governance decision record."""
        from app.governance.schemas import GovernanceDecision  # local to avoid circulars

        try:
            with connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO governance_decisions (
                        id, campaign_id, round_number,
                        claim_id, verdict, triggered_by,
                        violations_json, warnings_json,
                        amplification_factor, sigma_deviation,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision.id,
                        campaign_id,
                        round_number,
                        decision.claim.id if decision.claim else None,
                        decision.verdict,
                        decision.triggered_by,
                        json_dumps([v.to_dict() for v in decision.violations]),
                        json_dumps([w.to_dict() for w in decision.warnings]),
                        decision.amplification_factor,
                        decision.sigma_deviation,
                        decision.created_at,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("Failed to record governance decision %s: %s", decision.id, exc)

    # ------------------------------------------------------------------ read

    def get_ancestors(self, trace_id: str) -> list[Claim]:
        """Return the full ancestor chain ordered root → immediate parent.

        Uses iterative BFS to handle arbitrary depth without stack overflow.
        """
        visited: set[str] = set()
        queue: list[str] = [trace_id]
        ordered: list[Claim] = []

        while queue:
            current_tid = queue.pop(0)
            if current_tid in visited:
                continue
            visited.add(current_tid)

            row = self._fetch_by_trace_id(current_tid)
            if row is None:
                continue

            claim = self._row_to_claim(row)
            ordered.append(claim)

            parent_tids: list[str] = json.loads(
                row["parent_trace_ids_json"] or "[]"
            )
            queue.extend(t for t in parent_tids if t not in visited)

        # Reverse so root comes first; drop the claim itself (last element)
        ordered.reverse()
        if ordered and ordered[-1].trace_id == trace_id:
            ordered = ordered[:-1]
        return ordered

    def count_downstream_refs(self, trace_id: str) -> int:
        """Count how many governance_claims rows list *trace_id* as a parent."""
        try:
            with connection() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM governance_claims
                    WHERE parent_trace_ids_json LIKE ?
                    """,
                    (f'%"{trace_id}"%',),
                ).fetchone()
            return int(row["cnt"]) if row else 0
        except Exception as exc:
            logger.error("count_downstream_refs failed for %s: %s", trace_id, exc)
            return 0

    def get_param_history_stats(
        self,
        campaign_id: str,
        param_name: str,
    ) -> tuple[float | None, float | None]:
        """Return (mean, std) of historical numeric values for this param.

        Returns (None, None) when fewer than 3 data points exist.
        """
        try:
            with connection() as conn:
                rows = conn.execute(
                    """
                    SELECT param_value
                    FROM governance_claims
                    WHERE campaign_id = ?
                      AND param_name = ?
                      AND param_value IS NOT NULL
                    ORDER BY round_number ASC
                    """,
                    (campaign_id, param_name),
                ).fetchall()
        except Exception as exc:
            logger.error("get_param_history_stats failed: %s", exc)
            return None, None

        if len(rows) < 3:
            return None, None

        values = [float(r["param_value"]) for r in rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        return mean, std if std > 0 else None

    def compute_amplification_factor(
        self,
        claim: Claim,
        campaign_id: str,
    ) -> float:
        """Compute amplification_factor = downstream_refs × |σ_deviation|.

        Falls back to raw downstream_ref_count when history or numeric
        value is unavailable.
        """
        downstream = self.count_downstream_refs(claim.trace_id)
        if downstream == 0:
            return 0.0

        if claim.param_value is None:
            # Categorical claim: no σ calculation — use ref count alone
            return float(downstream)

        mean, std = self.get_param_history_stats(campaign_id, claim.param_name)
        if mean is None or std is None:
            return float(downstream)

        sigma_dev = abs(claim.param_value - mean) / std
        return downstream * sigma_dev

    def get_lineage(
        self,
        claim: Claim,
        campaign_id: str,
    ) -> ClaimLineage:
        """Build the full ClaimLineage for *claim*."""
        ancestors = self.get_ancestors(claim.trace_id)
        downstream = self.count_downstream_refs(claim.trace_id)
        amp = self.compute_amplification_factor(claim, campaign_id)

        return ClaimLineage(
            claim_id=claim.id,
            claim=claim,
            ancestor_claims=tuple(ancestors),
            causal_depth=len(ancestors),
            downstream_ref_count=downstream,
            amplification_factor=amp,
        )

    # ---------------------------------------------------------------- helpers

    def _fetch_by_trace_id(self, trace_id: str) -> sqlite3.Row | None:
        try:
            with connection() as conn:
                return conn.execute(
                    "SELECT * FROM governance_claims WHERE trace_id = ?",
                    (trace_id,),
                ).fetchone()
        except Exception as exc:
            logger.error("DB fetch failed for trace_id=%s: %s", trace_id, exc)
            return None

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        parent_tids: list[str] = json.loads(
            row["parent_trace_ids_json"] or "[]"
        )
        return Claim(
            id=row["id"],
            param_name=row["param_name"],
            param_value=row["param_value"],
            param_value_str=row["param_value_str"],
            source_type=row["source_type"],
            source_ref=row["source_ref"] or "",
            emitting_agent=row["emitting_agent"],
            trace_id=row["trace_id"],
            confidence=float(row["confidence"]),
            parent_trace_ids=tuple(parent_tids),
            created_at=row["created_at"],
        )
