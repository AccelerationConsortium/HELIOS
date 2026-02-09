"""Failure Signature Schema — structured, machine-readable failure patterns.

Upgrades reviewer output from natural-language root-cause text to typed,
machine-readable ``FailureSignature`` objects.  Each signature carries:

* ``failure_type`` — what went wrong (from an exhaustive domain enum)
* ``severity`` — CRITICAL / HIGH / MEDIUM / LOW / BYPASS
* ``likely_cause`` — probable root cause (from a curated set)
* ``retryable`` — whether the step is worth retrying
* ``recommended_patch`` — optional remediation action with parameters

Classification pipeline:
1. ``classify_failure()`` — rule-based regex matcher against known patterns
2. ``_PATCH_LIBRARY`` — maps (failure_type, likely_cause) to remediation
3. ``classify_run_failures()`` — batch classification for all failed steps
4. ``summarize_failure_signatures()`` — aggregate statistics

All operations are advisory — wrapped in try/except, never block.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

#: Exhaustive failure types for the electrochemistry / liquid-handling domain.
FAILURE_TYPES: frozenset[str] = frozenset({
    "volume_delivery_failure",
    "temperature_deviation",
    "temperature_overshoot",
    "impedance_anomaly",
    "electrode_degradation",
    "electrolyte_contamination",
    "tip_shortage",
    "liquid_insufficient",
    "deck_conflict",
    "instrument_disconnection",
    "instrument_timeout",
    "sensor_drift",
    "file_missing",
    "protocol_sequence_error",
    "safety_limit_exceeded",
    "unknown",
})

#: Common root causes mapped across failure types.
LIKELY_CAUSES: frozenset[str] = frozenset({
    "tip_clog",
    "tip_missing",
    "insufficient_liquid",
    "thermal_runaway",
    "heater_malfunction",
    "electrode_fouling",
    "bubble_formation",
    "contaminated_solution",
    "connection_lost",
    "power_interruption",
    "sensor_calibration_drift",
    "file_system_error",
    "parameter_out_of_range",
    "hardware_limit",
    "software_error",
    "unknown",
})

#: Allowed severity levels (superset of reviewer's CRITICAL/BYPASS).
SEVERITIES: frozenset[str] = frozenset({
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "BYPASS",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommendedPatch:
    """A machine-readable remediation action."""

    action: str  # e.g. "replace_tip_and_retry", "recalibrate_sensor"
    params: dict[str, Any]  # action-specific parameters
    description: str  # human-readable summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "action": self.action,
            "params": dict(self.params),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecommendedPatch:
        """Deserialize from a dict."""
        return cls(
            action=str(data.get("action", "")),
            params=dict(data.get("params", {})),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class FailureSignature:
    """Structured, machine-readable failure pattern."""

    failure_type: str  # from FAILURE_TYPES
    severity: str  # from SEVERITIES
    likely_cause: str  # from LIKELY_CAUSES
    retryable: bool
    recommended_patch: RecommendedPatch | None
    step_key: str  # which step failed
    primitive: str  # which primitive
    error_message: str  # original error string
    confidence: float  # 0.0 - 1.0, classification confidence
    tags: tuple[str, ...]  # additional metadata tags

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "failure_type": self.failure_type,
            "severity": self.severity,
            "likely_cause": self.likely_cause,
            "retryable": self.retryable,
            "recommended_patch": (
                self.recommended_patch.to_dict()
                if self.recommended_patch is not None
                else None
            ),
            "step_key": self.step_key,
            "primitive": self.primitive,
            "error_message": self.error_message,
            "confidence": self.confidence,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureSignature:
        """Deserialize from a dict."""
        patch_data = data.get("recommended_patch")
        patch = (
            RecommendedPatch.from_dict(patch_data)
            if patch_data is not None
            else None
        )
        tags_raw = data.get("tags", ())
        return cls(
            failure_type=str(data.get("failure_type", "unknown")),
            severity=str(data.get("severity", "HIGH")),
            likely_cause=str(data.get("likely_cause", "unknown")),
            retryable=bool(data.get("retryable", False)),
            recommended_patch=patch,
            step_key=str(data.get("step_key", "")),
            primitive=str(data.get("primitive", "")),
            error_message=str(data.get("error_message", "")),
            confidence=float(data.get("confidence", 0.0)),
            tags=tuple(str(t) for t in tags_raw),
        )


# ---------------------------------------------------------------------------
# Recommended Patch Library
# ---------------------------------------------------------------------------

#: Maps (failure_type, likely_cause) to a RecommendedPatch.
_PATCH_LIBRARY: dict[tuple[str, str], RecommendedPatch] = {
    ("tip_shortage", "tip_missing"): RecommendedPatch(
        action="replace_tip_rack",
        params={"max_retries": 1},
        description="Replace depleted tip rack and retry",
    ),
    ("volume_delivery_failure", "tip_clog"): RecommendedPatch(
        action="replace_tip_and_retry",
        params={"max_retries": 2},
        description="Drop current tip, pick new tip, retry aspiration",
    ),
    ("temperature_deviation", "heater_malfunction"): RecommendedPatch(
        action="recalibrate_and_retry",
        params={"wait_s": 60, "max_retries": 1},
        description="Wait for thermal equilibrium and retry",
    ),
    ("impedance_anomaly", "electrode_fouling"): RecommendedPatch(
        action="clean_electrode_and_retry",
        params={"cleaning_cycles": 3, "max_retries": 1},
        description="Run electrode cleaning cycles and retry measurement",
    ),
    ("instrument_disconnection", "connection_lost"): RecommendedPatch(
        action="reconnect_and_retry",
        params={"wait_s": 5, "max_retries": 3},
        description="Wait, reconnect instrument, retry step",
    ),
    ("instrument_timeout", "connection_lost"): RecommendedPatch(
        action="reconnect_and_retry",
        params={"wait_s": 10, "max_retries": 2},
        description="Wait, reconnect, retry with extended timeout",
    ),
    ("sensor_drift", "sensor_calibration_drift"): RecommendedPatch(
        action="recalibrate_sensor",
        params={"calibration_type": "auto"},
        description="Run automatic sensor recalibration",
    ),
    ("liquid_insufficient", "insufficient_liquid"): RecommendedPatch(
        action="refill_and_retry",
        params={"max_retries": 1},
        description="Alert for liquid refill, then retry",
    ),
    ("file_missing", "file_system_error"): RecommendedPatch(
        action="log_and_skip",
        params={},
        description="Log missing file warning and skip artifact upload",
    ),
}


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

# Each rule: (error_pattern_regex, primitive_match_or_None,
#              failure_type, likely_cause, severity, retryable)
_CLASSIFICATION_RULES: list[tuple[str, str | None, str, str, str, bool]] = [
    # --- Volume / pipetting failures ---
    (
        r"(?i)no tips? available",
        "robot.pick_up_tip",
        "tip_shortage", "tip_missing", "CRITICAL", True,
    ),
    (
        r"(?i)insufficient liquid",
        "robot.aspirate",
        "liquid_insufficient", "insufficient_liquid", "HIGH", True,
    ),
    (
        r"(?i)volume.*(error|mismatch|deviation)",
        "robot.aspirate",
        "volume_delivery_failure", "tip_clog", "HIGH", True,
    ),
    (
        r"(?i)volume.*(error|mismatch|deviation)",
        "robot.dispense",
        "volume_delivery_failure", "tip_clog", "HIGH", True,
    ),
    # --- Temperature failures ---
    (
        r"(?i)temp.*(overshoot|exceeded)",
        "heat",
        "temperature_overshoot", "thermal_runaway", "CRITICAL", False,
    ),
    (
        r"(?i)temp.*(deviation|accuracy|drift)",
        "heat",
        "temperature_deviation", "heater_malfunction", "MEDIUM", True,
    ),
    # --- Electrochemistry failures ---
    (
        r"(?i)impedance.*(anomal|spike|out.of.range)",
        "squidstat",
        "impedance_anomaly", "electrode_fouling", "HIGH", True,
    ),
    (
        r"(?i)electrode.*(degrad|fail)",
        None,
        "electrode_degradation", "electrode_fouling", "CRITICAL", False,
    ),
    # --- Connection failures ---
    (
        r"(?i)(disconnect|connection.*(lost|error|refused))",
        None,
        "instrument_disconnection", "connection_lost", "CRITICAL", True,
    ),
    (
        r"(?i)time.?out",
        None,
        "instrument_timeout", "connection_lost", "HIGH", True,
    ),
    # --- Deck / labware failures ---
    (
        r"(?i)(deck|slot).*(conflict|occupied)",
        "robot.load_labware",
        "deck_conflict", "software_error", "HIGH", False,
    ),
    # --- File failures ---
    (
        r"(?i)file.*(not found|missing)",
        None,
        "file_missing", "file_system_error", "MEDIUM", True,
    ),
    # --- Safety ---
    (
        r"(?i)safety.*(limit|violation|exceeded)",
        None,
        "safety_limit_exceeded", "hardware_limit", "CRITICAL", False,
    ),
    (
        r"(?i)sensor.*(drift|calibration)",
        None,
        "sensor_drift", "sensor_calibration_drift", "MEDIUM", True,
    ),
]

# Pre-compile regexes for performance.
_COMPILED_RULES: list[tuple[re.Pattern[str], str | None, str, str, str, bool]] = [
    (re.compile(pattern), prim, ftype, cause, sev, retry)
    for pattern, prim, ftype, cause, sev, retry in _CLASSIFICATION_RULES
]


def _lookup_patch(
    failure_type: str,
    likely_cause: str,
) -> RecommendedPatch | None:
    """Return the recommended patch for a (failure_type, likely_cause) pair."""
    return _PATCH_LIBRARY.get((failure_type, likely_cause))


def classify_failure(
    step_key: str,
    primitive: str,
    error_message: str,
) -> FailureSignature:
    """Classify a step failure into a structured FailureSignature.

    Uses rule-based pattern matching against known failure patterns.
    Falls back to ``"unknown"`` type if no rule matches.

    Parameters
    ----------
    step_key:
        Identifier for the failed step.
    primitive:
        The primitive that was executing when the failure occurred.
    error_message:
        The original error text from the step.

    Returns
    -------
    FailureSignature
    """
    for compiled_re, prim_match, ftype, cause, severity, retryable in _COMPILED_RULES:
        # If the rule specifies a primitive, the step's primitive must contain it.
        if prim_match is not None and prim_match not in primitive:
            continue

        if compiled_re.search(error_message):
            patch = _lookup_patch(ftype, cause)
            return FailureSignature(
                failure_type=ftype,
                severity=severity,
                likely_cause=cause,
                retryable=retryable,
                recommended_patch=patch,
                step_key=step_key,
                primitive=primitive,
                error_message=error_message,
                confidence=0.85,
                tags=("rule-matched",),
            )

    # --- Fallback: unknown failure ---
    return FailureSignature(
        failure_type="unknown",
        severity="HIGH",
        likely_cause="unknown",
        retryable=False,
        recommended_patch=None,
        step_key=step_key,
        primitive=primitive,
        error_message=error_message,
        confidence=0.2,
        tags=("unmatched",),
    )


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


def classify_run_failures(
    steps: list[dict[str, Any]],
) -> list[FailureSignature]:
    """Classify all failed steps in a run.

    Parameters
    ----------
    steps:
        List of step dicts with ``step_key``, ``primitive``, ``status``,
        and ``error`` fields.

    Returns
    -------
    list[FailureSignature]
        One signature per failed step (status != ``"succeeded"``).
    """
    signatures: list[FailureSignature] = []
    for step in steps:
        status = step.get("status", "")
        if status == "succeeded":
            continue

        step_key = str(step.get("step_key", ""))
        primitive = str(step.get("primitive", ""))
        error = str(step.get("error", "") or "")

        try:
            sig = classify_failure(step_key, primitive, error)
            signatures.append(sig)
        except Exception:
            logger.debug(
                "Failed to classify step %s — skipping", step_key, exc_info=True,
            )

    return signatures


def summarize_failure_signatures(
    signatures: list[FailureSignature],
) -> dict[str, Any]:
    """Generate an aggregate summary of failure signatures.

    Returns
    -------
    dict with keys:
        ``total_failures``, ``by_type``, ``by_severity``,
        ``retryable_count``, ``non_retryable_count``, ``unique_causes``.
    """
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    retryable_count = 0
    non_retryable_count = 0
    causes: set[str] = set()

    for sig in signatures:
        by_type[sig.failure_type] = by_type.get(sig.failure_type, 0) + 1
        by_severity[sig.severity] = by_severity.get(sig.severity, 0) + 1
        if sig.retryable:
            retryable_count += 1
        else:
            non_retryable_count += 1
        causes.add(sig.likely_cause)

    return {
        "total_failures": len(signatures),
        "by_type": by_type,
        "by_severity": by_severity,
        "retryable_count": retryable_count,
        "non_retryable_count": non_retryable_count,
        "unique_causes": sorted(causes),
    }


# ---------------------------------------------------------------------------
# Cross-run failure pattern learning layer
# ---------------------------------------------------------------------------

import sqlite3
import uuid

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso


def _ensure_failure_learning_tables(conn: sqlite3.Connection) -> None:
    """Lazily create the failure learning tables if they do not exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS failure_frequency (
            id TEXT PRIMARY KEY,
            failure_type TEXT NOT NULL,
            likely_cause TEXT NOT NULL,
            primitive TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            last_seen_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            run_ids_json TEXT NOT NULL DEFAULT '[]',
            UNIQUE(failure_type, likely_cause, primitive)
        );

        CREATE TABLE IF NOT EXISTS failure_chains (
            id TEXT PRIMARY KEY,
            predecessor_type TEXT NOT NULL,
            predecessor_cause TEXT NOT NULL,
            successor_type TEXT NOT NULL,
            successor_cause TEXT NOT NULL,
            co_occurrence_count INTEGER NOT NULL DEFAULT 1,
            confidence REAL NOT NULL DEFAULT 0.0,
            last_seen_at TEXT NOT NULL,
            UNIQUE(predecessor_type, predecessor_cause, successor_type, successor_cause)
        );
        """
    )


# ---------------------------------------------------------------------------
# Learned-pattern data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailurePattern:
    """A learned pattern from cross-run failure analysis."""

    failure_type: str
    likely_cause: str
    primitive: str
    occurrence_count: int
    first_seen_at: str
    last_seen_at: str
    run_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "failure_type": self.failure_type,
            "likely_cause": self.likely_cause,
            "primitive": self.primitive,
            "occurrence_count": self.occurrence_count,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "run_ids": list(self.run_ids),
        }


@dataclass(frozen=True)
class FailureChain:
    """A causal chain between two failure types."""

    predecessor_type: str
    predecessor_cause: str
    successor_type: str
    successor_cause: str
    co_occurrence_count: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "predecessor_type": self.predecessor_type,
            "predecessor_cause": self.predecessor_cause,
            "successor_type": self.successor_type,
            "successor_cause": self.successor_cause,
            "co_occurrence_count": self.co_occurrence_count,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def record_failure(
    conn: sqlite3.Connection,
    signature: FailureSignature,
    run_id: str,
) -> None:
    """UPSERT a single failure into ``failure_frequency``.

    Increments ``occurrence_count``, updates ``last_seen_at``, and appends
    *run_id* to the stored ``run_ids_json`` array.
    """
    _ensure_failure_learning_tables(conn)
    now = utcnow_iso()

    existing = conn.execute(
        "SELECT id, occurrence_count, run_ids_json "
        "FROM failure_frequency "
        "WHERE failure_type = ? AND likely_cause = ? AND primitive = ?",
        (signature.failure_type, signature.likely_cause, signature.primitive),
    ).fetchone()

    if existing is not None:
        run_ids: list[str] = parse_json(existing["run_ids_json"], [])
        if run_id not in run_ids:
            run_ids.append(run_id)
        conn.execute(
            "UPDATE failure_frequency "
            "SET occurrence_count = occurrence_count + 1, "
            "    last_seen_at = ?, "
            "    run_ids_json = ? "
            "WHERE id = ?",
            (now, json_dumps(run_ids), existing["id"]),
        )
    else:
        row_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO failure_frequency "
            "(id, failure_type, likely_cause, primitive, occurrence_count, "
            " last_seen_at, first_seen_at, run_ids_json) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (
                row_id,
                signature.failure_type,
                signature.likely_cause,
                signature.primitive,
                now,
                now,
                json_dumps([run_id]),
            ),
        )


def record_failure_chains(
    conn: sqlite3.Connection,
    signatures: list[FailureSignature],
    run_id: str,  # noqa: ARG001  — reserved for future per-run chain tracking
) -> None:
    """Record causal chains between consecutive failure signatures.

    For each consecutive pair of *signatures* (ordered by ``step_key``),
    inserts or updates a row in ``failure_chains``.  Confidence is computed
    as ``co_occurrence_count / total_occurrence_of_predecessor``.
    """
    if len(signatures) < 2:
        return

    _ensure_failure_learning_tables(conn)
    now = utcnow_iso()

    # Sort signatures by step_key for consistent ordering.
    ordered = sorted(signatures, key=lambda s: s.step_key)

    for pred, succ in zip(ordered, ordered[1:]):
        existing = conn.execute(
            "SELECT id, co_occurrence_count "
            "FROM failure_chains "
            "WHERE predecessor_type = ? AND predecessor_cause = ? "
            "  AND successor_type = ? AND successor_cause = ?",
            (
                pred.failure_type,
                pred.likely_cause,
                succ.failure_type,
                succ.likely_cause,
            ),
        ).fetchone()

        if existing is not None:
            new_co = existing["co_occurrence_count"] + 1
            conn.execute(
                "UPDATE failure_chains "
                "SET co_occurrence_count = ?, last_seen_at = ? "
                "WHERE id = ?",
                (new_co, now, existing["id"]),
            )
        else:
            row_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO failure_chains "
                "(id, predecessor_type, predecessor_cause, "
                " successor_type, successor_cause, "
                " co_occurrence_count, confidence, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, 1, 0.0, ?)",
                (
                    row_id,
                    pred.failure_type,
                    pred.likely_cause,
                    succ.failure_type,
                    succ.likely_cause,
                    now,
                ),
            )

        # Recompute confidence for all chains starting from this predecessor.
        pred_total_row = conn.execute(
            "SELECT occurrence_count FROM failure_frequency "
            "WHERE failure_type = ? AND likely_cause = ?",
            (pred.failure_type, pred.likely_cause),
        ).fetchone()
        pred_total = pred_total_row["occurrence_count"] if pred_total_row else 1

        conn.execute(
            "UPDATE failure_chains "
            "SET confidence = CAST(co_occurrence_count AS REAL) / ? "
            "WHERE predecessor_type = ? AND predecessor_cause = ?",
            (max(pred_total, 1), pred.failure_type, pred.likely_cause),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def learn_from_run(run_id: str) -> None:
    """Classify failures for *run_id* and record patterns + chains.

    This is the primary entry point for cross-run learning.  It fetches the
    run's steps from the database, classifies failures, and persists the
    learned patterns.  All operations are advisory — errors are logged and
    never propagated.
    """
    try:

        def _inner(conn: sqlite3.Connection) -> None:
            rows = conn.execute(
                "SELECT step_key, primitive, status, error "
                "FROM run_steps WHERE run_id = ? ORDER BY step_key ASC",
                (run_id,),
            ).fetchall()
            steps = [dict(r) for r in rows]

            signatures = classify_run_failures(steps)
            if not signatures:
                return

            for sig in signatures:
                record_failure(conn, sig, run_id)

            record_failure_chains(conn, signatures, run_id)

        run_txn(_inner)
    except Exception:
        logger.debug(
            "Failure learning for run %s failed — skipping",
            run_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_frequent_failures(min_count: int = 3) -> list[FailurePattern]:
    """Return failure patterns with at least *min_count* occurrences.

    Results are ordered by ``occurrence_count`` descending.
    """
    try:
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            rows = conn.execute(
                "SELECT failure_type, likely_cause, primitive, "
                "       occurrence_count, first_seen_at, last_seen_at, "
                "       run_ids_json "
                "FROM failure_frequency "
                "WHERE occurrence_count >= ? "
                "ORDER BY occurrence_count DESC",
                (min_count,),
            ).fetchall()

            return [
                FailurePattern(
                    failure_type=row["failure_type"],
                    likely_cause=row["likely_cause"],
                    primitive=row["primitive"],
                    occurrence_count=row["occurrence_count"],
                    first_seen_at=row["first_seen_at"],
                    last_seen_at=row["last_seen_at"],
                    run_ids=tuple(parse_json(row["run_ids_json"], [])),
                )
                for row in rows
            ]
    except Exception:
        logger.debug("get_frequent_failures failed — returning empty", exc_info=True)
        return []


def get_failure_chains(min_confidence: float = 0.3) -> list[FailureChain]:
    """Return failure chains with at least *min_confidence*.

    Results are ordered by ``confidence`` descending.
    """
    try:
        with connection() as conn:
            _ensure_failure_learning_tables(conn)
            rows = conn.execute(
                "SELECT predecessor_type, predecessor_cause, "
                "       successor_type, successor_cause, "
                "       co_occurrence_count, confidence "
                "FROM failure_chains "
                "WHERE confidence >= ? "
                "ORDER BY confidence DESC",
                (min_confidence,),
            ).fetchall()

            return [
                FailureChain(
                    predecessor_type=row["predecessor_type"],
                    predecessor_cause=row["predecessor_cause"],
                    successor_type=row["successor_type"],
                    successor_cause=row["successor_cause"],
                    co_occurrence_count=row["co_occurrence_count"],
                    confidence=row["confidence"],
                )
                for row in rows
            ]
    except Exception:
        logger.debug("get_failure_chains failed — returning empty", exc_info=True)
        return []


def predict_failures(
    current_signatures: list[FailureSignature],
) -> list[FailureSignature]:
    """Predict what might fail next based on learned failure chains.

    For each signature in *current_signatures*, looks up chains where it is
    the predecessor and returns synthetic ``FailureSignature`` objects with
    the tag ``"predicted"`` and the chain's confidence score.
    """
    if not current_signatures:
        return []

    predictions: list[FailureSignature] = []
    seen: set[tuple[str, str]] = set()

    try:
        with connection() as conn:
            _ensure_failure_learning_tables(conn)

            for sig in current_signatures:
                rows = conn.execute(
                    "SELECT successor_type, successor_cause, confidence "
                    "FROM failure_chains "
                    "WHERE predecessor_type = ? AND predecessor_cause = ? "
                    "  AND confidence >= 0.1 "
                    "ORDER BY confidence DESC",
                    (sig.failure_type, sig.likely_cause),
                ).fetchall()

                for row in rows:
                    key = (row["successor_type"], row["successor_cause"])
                    if key in seen:
                        continue
                    seen.add(key)

                    predictions.append(
                        FailureSignature(
                            failure_type=row["successor_type"],
                            severity="MEDIUM",
                            likely_cause=row["successor_cause"],
                            retryable=True,
                            recommended_patch=_lookup_patch(
                                row["successor_type"],
                                row["successor_cause"],
                            ),
                            step_key="",
                            primitive="",
                            error_message="",
                            confidence=float(row["confidence"]),
                            tags=("predicted",),
                        )
                    )
    except Exception:
        logger.debug("predict_failures failed — returning empty", exc_info=True)

    return predictions


def get_failure_learning_summary() -> dict[str, Any]:
    """Return a high-level summary of the failure learning state.

    Keys: ``total_patterns``, ``top_failures`` (top 5 by count),
    ``chain_count``, ``prediction_accuracy`` (placeholder ``0.0``).
    """
    try:
        with connection() as conn:
            _ensure_failure_learning_tables(conn)

            total_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_frequency",
            ).fetchone()
            total_patterns = total_row["cnt"] if total_row else 0

            top_rows = conn.execute(
                "SELECT failure_type, likely_cause, primitive, occurrence_count "
                "FROM failure_frequency "
                "ORDER BY occurrence_count DESC "
                "LIMIT 5",
            ).fetchall()
            top_failures = [
                {
                    "failure_type": r["failure_type"],
                    "likely_cause": r["likely_cause"],
                    "primitive": r["primitive"],
                    "occurrence_count": r["occurrence_count"],
                }
                for r in top_rows
            ]

            chain_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM failure_chains",
            ).fetchone()
            chain_count = chain_row["cnt"] if chain_row else 0

            return {
                "total_patterns": total_patterns,
                "top_failures": top_failures,
                "chain_count": chain_count,
                "prediction_accuracy": 0.0,
            }
    except Exception:
        logger.debug(
            "get_failure_learning_summary failed — returning defaults",
            exc_info=True,
        )
        return {
            "total_patterns": 0,
            "top_failures": [],
            "chain_count": 0,
            "prediction_accuracy": 0.0,
        }
