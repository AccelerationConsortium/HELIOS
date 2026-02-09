"""LLM-based run reviewer — scores, attributes failures, suggests improvements.

Narrowly scoped: NO new experiment proposals or new plans.
Only: scoring (0-100), failure attribution (based on run state),
structured improvement suggestions.

Pipeline:
1. ``build_review_system_prompt()`` assembles constrained evaluation prompt
2. ``format_run_for_review()`` builds structured run summary for LLM
3. ``review_run()`` calls LLM -> ``parse_review_response()`` -> stores in run_reviews
4. Event listener triggers on ``run.completed``

All operations are advisory — wrapped in try/except, never block
run completion.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.db import json_dumps, parse_json, run_txn, utcnow_iso
from app.services.audit import record_event
from app.services.llm_gateway import LLMMessage, LLMProvider, get_llm_provider
from app.services.metrics import get_run_kpis
from app.services.run_service import get_run, list_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

REVIEW_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ReviewParseError(ValueError):
    """Raised when the LLM response cannot be parsed into a valid review."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureAttribution:
    """A single failure attributed to a specific step."""

    step_key: str
    primitive: str
    error: str
    severity: str  # "CRITICAL" | "BYPASS"
    root_cause: str  # LLM-generated


@dataclass(frozen=True)
class Improvement:
    """A structured improvement suggestion."""

    category: str  # "calibration" | "parameter" | "sequence" | "recovery" | "environment"
    target: str  # primitive or step_key
    suggestion: str


@dataclass(frozen=True)
class RunReview:
    """Complete review of a single run."""

    score: float  # 0-100
    verdict: str  # "passed" | "failed" | "degraded"
    failure_attributions: list[FailureAttribution]
    improvements: list[Improvement]
    model: str
    raw_response: str


# ---------------------------------------------------------------------------
# System prompt (tightly constrained)
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM_PROMPT = """\
You are a laboratory experiment run evaluator. Your ONLY job is to:
1. SCORE the run (0-100) based on success rate, KPI quality, and execution efficiency
2. ATTRIBUTE failures to specific steps with root cause analysis
3. SUGGEST structured improvements

You must NOT propose new experiments or generate new plans.
You must NOT modify parameters or suggest different experimental designs.
You must NOT suggest new protocols or alternative experiment workflows.

Output ONLY a JSON object with this exact schema:
{
  "score": <number 0-100>,
  "verdict": "passed" | "failed" | "degraded",
  "failure_attributions": [
    {
      "step_key": "<step identifier>",
      "primitive": "<primitive name>",
      "error": "<error message from step>",
      "severity": "CRITICAL" | "BYPASS",
      "root_cause": "<your root cause analysis>"
    }
  ],
  "improvements": [
    {
      "category": "calibration" | "parameter" | "sequence" | "recovery" | "environment",
      "target": "<primitive or step_key>",
      "suggestion": "<specific actionable suggestion>"
    }
  ]
}

Scoring rubric:
- 90-100: All steps succeeded, KPIs within expected range
- 70-89: Minor issues (bypassed failures, slightly off KPIs)
- 50-69: Significant issues (recovery needed, KPIs degraded)
- 0-49: Major failures (critical step failures, run aborted)

Verdict rules:
- "passed": score >= 70 and no CRITICAL failures
- "degraded": score >= 50 and at most BYPASS-level failures
- "failed": score < 50 or any unrecoverable CRITICAL failure

failure_attributions array must be empty if no steps failed.
improvements array must have at least one entry if score < 90.
Respond ONLY with the JSON object. Do not include any other text outside the JSON.
"""


def build_review_system_prompt() -> str:
    """Return the constrained system prompt for run evaluation."""
    return _REVIEW_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Run data formatting for LLM
# ---------------------------------------------------------------------------


def format_run_for_review(
    run: dict[str, Any],
    kpis: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    """Build a structured run summary for the LLM reviewer."""
    sections: list[str] = []

    # Run overview
    sections.append("## Run Overview")
    sections.append(f"- Run ID: {run['id']}")
    sections.append(f"- Status: {run['status']}")
    sections.append(f"- Started: {run.get('started_at', 'N/A')}")
    sections.append(f"- Ended: {run.get('ended_at', 'N/A')}")
    if run.get("rejection_reason"):
        sections.append(f"- Rejection Reason: {run['rejection_reason']}")

    # Steps summary
    steps = run.get("steps", [])
    sections.append(f"\n## Steps ({len(steps)} total)")
    for step in steps:
        status_marker = "OK" if step["status"] == "succeeded" else step["status"].upper()
        line = f"- [{status_marker}] {step['step_key']}: {step['primitive']}"
        if step.get("error"):
            line += f" | error: {step['error']}"
        sections.append(line)

    # KPIs
    if kpis:
        sections.append(f"\n## KPIs ({len(kpis)} values)")
        for kpi in kpis:
            entry = f"- {kpi['kpi_name']}: {kpi['kpi_value']} {kpi['kpi_unit']}"
            if kpi.get("step_id"):
                entry += f" (step: {kpi['step_id']})"
            sections.append(entry)

    # Recovery events
    recovery_events = [e for e in events if e.get("action", "").startswith("recovery.")]
    if recovery_events:
        sections.append(f"\n## Recovery Events ({len(recovery_events)})")
        for evt in recovery_events:
            details = evt.get("details", {})
            sections.append(f"- {evt['action']}: {json.dumps(details)}")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

_VALID_VERDICTS = frozenset({"passed", "failed", "degraded"})
_VALID_SEVERITIES = frozenset({"CRITICAL", "BYPASS"})
_VALID_CATEGORIES = frozenset({"calibration", "parameter", "sequence", "recovery", "environment"})


def parse_review_response(raw: str) -> RunReview:
    """Extract a ``RunReview`` from raw LLM text.

    Supports:
    - Raw JSON object at the top level
    - JSON wrapped in a ```json ... ``` code block
    """
    text = raw.strip()

    # Try to extract from code block first
    match = _CODE_BLOCK_RE.search(text)
    json_text = match.group(1).strip() if match else text

    # Fallback: find first { ... last }
    if not json_text.startswith("{"):
        brace_start = json_text.find("{")
        brace_end = json_text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            json_text = json_text[brace_start : brace_end + 1]
        else:
            raise ReviewParseError(f"No JSON object found in LLM response: {text[:200]}")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ReviewParseError(f"Invalid JSON in LLM response: {exc}") from exc

    if not isinstance(data, dict):
        raise ReviewParseError(f"Expected JSON object, got {type(data).__name__}")

    # --- Validate score ---
    score = data.get("score")
    if not isinstance(score, (int, float)) or score < 0 or score > 100:
        raise ReviewParseError(f"'score' must be a number 0-100, got {score!r}")

    # --- Validate verdict ---
    verdict = data.get("verdict")
    if verdict not in _VALID_VERDICTS:
        raise ReviewParseError(
            f"'verdict' must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}"
        )

    # --- Parse failure_attributions ---
    raw_attrs = data.get("failure_attributions", [])
    if not isinstance(raw_attrs, list):
        raise ReviewParseError("'failure_attributions' must be an array")
    attributions: list[FailureAttribution] = []
    for i, attr in enumerate(raw_attrs):
        if not isinstance(attr, dict):
            raise ReviewParseError(f"failure_attributions[{i}]: expected object")
        severity = attr.get("severity", "CRITICAL")
        if severity not in _VALID_SEVERITIES:
            severity = "CRITICAL"  # fail-safe default
        attributions.append(
            FailureAttribution(
                step_key=str(attr.get("step_key", f"unknown-{i}")),
                primitive=str(attr.get("primitive", "unknown")),
                error=str(attr.get("error", "")),
                severity=severity,
                root_cause=str(attr.get("root_cause", "")),
            )
        )

    # --- Parse improvements ---
    raw_imps = data.get("improvements", [])
    if not isinstance(raw_imps, list):
        raise ReviewParseError("'improvements' must be an array")
    improvements: list[Improvement] = []
    for i, imp in enumerate(raw_imps):
        if not isinstance(imp, dict):
            raise ReviewParseError(f"improvements[{i}]: expected object")
        category = imp.get("category", "parameter")
        if category not in _VALID_CATEGORIES:
            category = "parameter"  # fail-safe default
        improvements.append(
            Improvement(
                category=category,
                target=str(imp.get("target", "")),
                suggestion=str(imp.get("suggestion", "")),
            )
        )

    return RunReview(
        score=float(score),
        verdict=verdict,
        failure_attributions=attributions,
        improvements=improvements,
        model="unknown",  # caller sets actual model
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Core review + storage
# ---------------------------------------------------------------------------


async def review_run(
    run_id: str,
    *,
    provider: LLMProvider | None = None,
) -> RunReview:
    """Score, attribute failures, and suggest improvements for a completed run.

    Parameters
    ----------
    run_id:
        The run to review.
    provider:
        Optional LLM provider override (default: ``get_llm_provider()``).

    Returns
    -------
    RunReview

    Raises
    ------
    LLMError
        If the LLM call fails.
    ReviewParseError
        If the LLM response cannot be parsed.
    ValueError
        If run not found.
    """
    # 1. Load all run data
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    kpis = get_run_kpis(run_id)
    events = list_events(run_id)

    # 2. Build evaluation context
    system = build_review_system_prompt()
    user_content = format_run_for_review(run, kpis, events)

    # 3. Call LLM
    llm = provider or get_llm_provider()
    response = await llm.complete(
        messages=[LLMMessage(role="user", content=user_content)],
        system=system,
    )

    # 4. Parse structured response
    review = parse_review_response(response.content)
    # Replace model with actual model from LLM response
    review = RunReview(
        score=review.score,
        verdict=review.verdict,
        failure_attributions=review.failure_attributions,
        improvements=review.improvements,
        model=response.model,
        raw_response=response.content,
    )

    # 5. Store in run_reviews table + record provenance event
    _store_review(run_id, review)

    return review


def _store_review(run_id: str, review: RunReview) -> None:
    """Persist review to run_reviews table and record provenance event."""
    import sqlite3 as _sqlite3

    now = utcnow_iso()
    review_id = str(uuid.uuid4())

    def _txn(conn: _sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO run_reviews "
            "(id, run_id, score, verdict, failure_attributions_json, "
            "improvements_json, model, review_schema_version, raw_response, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                review_id,
                run_id,
                review.score,
                review.verdict,
                json_dumps([
                    {
                        "step_key": a.step_key,
                        "primitive": a.primitive,
                        "error": a.error,
                        "severity": a.severity,
                        "root_cause": a.root_cause,
                    }
                    for a in review.failure_attributions
                ]),
                json_dumps([
                    {
                        "category": i.category,
                        "target": i.target,
                        "suggestion": i.suggestion,
                    }
                    for i in review.improvements
                ]),
                review.model,
                REVIEW_SCHEMA_VERSION,
                review.raw_response,
                now,
            ),
        )
        record_event(
            conn,
            run_id=run_id,
            actor="reviewer",
            action="run.reviewed",
            details={
                "review_id": review_id,
                "score": review.score,
                "verdict": review.verdict,
                "failure_count": len(review.failure_attributions),
                "improvement_count": len(review.improvements),
            },
        )

    run_txn(_txn)


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def get_run_review(run_id: str) -> dict[str, Any] | None:
    """Return the review for a given run, or None if not yet reviewed."""
    import sqlite3 as _sqlite3

    def _txn(conn: _sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM run_reviews WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["failure_attributions"] = parse_json(
            item.pop("failure_attributions_json"), [],
        )
        item["improvements"] = parse_json(item.pop("improvements_json"), [])
        return item

    return run_txn(_txn)


# ---------------------------------------------------------------------------
# Event listener — async write path
# ---------------------------------------------------------------------------

_listener_task: asyncio.Task[None] | None = None


async def _on_run_completed(run_id: str) -> None:
    """Review a completed run. Advisory — never blocks."""
    try:
        review = await review_run(run_id)
        logger.debug(
            "Review completed for run %s: score=%.1f verdict=%s",
            run_id,
            review.score,
            review.verdict,
        )
    except Exception:
        logger.warning("Review failed for run %s", run_id, exc_info=True)


async def start_reviewer_listener(bus: Any) -> Any:
    """Subscribe to the event bus and process run.completed events.

    Returns the Subscription handle for cleanup.
    """
    global _listener_task

    sub = await bus.subscribe(run_id=None)  # global subscription

    async def _listen() -> None:
        async for event in sub:
            if event.action == "run.completed":
                run_id = event.run_id
                if run_id:
                    await _on_run_completed(run_id)

    _listener_task = asyncio.create_task(_listen())
    return sub


async def stop_reviewer_listener(sub: Any, bus: Any) -> None:
    """Cancel the reviewer listener and unsubscribe."""
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
