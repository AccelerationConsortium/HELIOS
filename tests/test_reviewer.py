"""Tests for the LLM-based run reviewer (Phase C3: Reviewer Agent).

Covers:
- System prompt constraints (scoring rubric, no new experiments, JSON output)
- Run data formatting for LLM (format_run_for_review)
- Response parsing (valid JSON, code blocks, failed reviews, invalid input)
- Core review_run workflow (successful run, failed run, nonexistent run)
- DB storage (persistence, read path, schema version, UNIQUE constraint, provenance event)
- Failure isolation (_on_run_completed handles errors gracefully)
- Event listener (EventBus → run.completed → listener processes)
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_reviewer_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "reviewer_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, json_dumps, utcnow_iso  # noqa: E402
from app.services.llm_gateway import LLMError, MockProvider  # noqa: E402
from app.services.reviewer import (  # noqa: E402
    REVIEW_SCHEMA_VERSION,
    FailureAttribution,
    Improvement,
    ReviewParseError,
    RunReview,
    _on_run_completed,
    build_review_system_prompt,
    format_run_for_review,
    get_run_review,
    parse_review_response,
    review_run,
    start_reviewer_listener,
    stop_reviewer_listener,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    # Clean tables between tests
    with connection() as conn:
        conn.execute("DELETE FROM evolved_priors")
        conn.execute("DELETE FROM evolution_proposals")
        conn.execute("DELETE FROM protocol_templates")
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM run_reviews")
        conn.execute("DELETE FROM run_kpis")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM provenance_events")
        conn.execute("DELETE FROM run_steps")
        conn.execute("DELETE FROM runs")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_run_with_steps(
    steps: list[dict],
    run_status: str = "succeeded",
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> str:
    """Insert a run + run_steps directly into DB. Returns run_id."""
    run_id = str(uuid.uuid4())
    now = utcnow_iso()
    start = started_at or now
    end = ended_at or now

    with connection() as conn:
        conn.execute(
            "INSERT INTO runs "
            "(id, campaign_id, trigger_type, trigger_payload_json, session_key, "
            "status, protocol_json, inputs_json, compiled_graph_json, graph_hash, "
            "policy_snapshot_json, created_by, created_at, updated_at, started_at, ended_at) "
            "VALUES (?, NULL, 'manual', '{}', ?, ?, '{}', '{}', '{}', 'h', '{}', 'test', ?, ?, ?, ?)",
            (run_id, run_id, run_status, now, now, start, end),
        )
        for step in steps:
            step_id = step.get("id", str(uuid.uuid4()))
            conn.execute(
                "INSERT INTO run_steps "
                "(id, run_id, step_key, primitive, params_json, depends_on_json, "
                "resources_json, status, idempotency_key, started_at, ended_at, error) "
                "VALUES (?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?, ?, ?)",
                (
                    step_id,
                    run_id,
                    step["step_key"],
                    step["primitive"],
                    json_dumps(step.get("params", {})),
                    step.get("status", "succeeded"),
                    f"{run_id}:{step['step_key']}:0",
                    step.get("started_at"),
                    step.get("ended_at"),
                    step.get("error"),
                ),
            )
        conn.commit()
    return run_id


# ---------------------------------------------------------------------------
# Sample LLM JSON responses
# ---------------------------------------------------------------------------

SAMPLE_REVIEW_JSON = json.dumps({
    "score": 92.5,
    "verdict": "passed",
    "failure_attributions": [],
    "improvements": [
        {
            "category": "parameter",
            "target": "aspirate",
            "suggestion": "Consider reducing aspiration speed for viscous liquids",
        }
    ],
})

FAILED_REVIEW_JSON = json.dumps({
    "score": 35.0,
    "verdict": "failed",
    "failure_attributions": [
        {
            "step_key": "s2",
            "primitive": "aspirate",
            "error": "tip missing",
            "severity": "CRITICAL",
            "root_cause": "Tip rack was not loaded before aspiration step",
        },
        {
            "step_key": "s3",
            "primitive": "heat",
            "error": "temperature overshoot",
            "severity": "BYPASS",
            "root_cause": "Heater PID tuning too aggressive for small volumes",
        },
    ],
    "improvements": [
        {
            "category": "sequence",
            "target": "s2",
            "suggestion": "Add tip check step before aspiration",
        },
        {
            "category": "calibration",
            "target": "heat",
            "suggestion": "Reduce PID gains for volumes under 50uL",
        },
    ],
})

DEGRADED_REVIEW_JSON = json.dumps({
    "score": 65.0,
    "verdict": "degraded",
    "failure_attributions": [
        {
            "step_key": "s1",
            "primitive": "dispense",
            "error": "partial dispense",
            "severity": "BYPASS",
            "root_cause": "Air bubble in tip caused volume deficit",
        },
    ],
    "improvements": [
        {
            "category": "recovery",
            "target": "dispense",
            "suggestion": "Add air gap removal step before dispensing",
        },
    ],
})


# ===========================================================================
# 1. System Prompt Tests
# ===========================================================================


class TestSystemPrompt:
    """Verify the reviewer system prompt is correctly constrained."""

    def test_contains_scoring_rubric(self):
        prompt = build_review_system_prompt()
        assert "0-100" in prompt
        assert "90-100" in prompt
        assert "0-49" in prompt

    def test_forbids_new_experiments(self):
        prompt = build_review_system_prompt()
        assert "must NOT propose new experiments" in prompt

    def test_requires_json_output(self):
        prompt = build_review_system_prompt()
        assert '"score"' in prompt
        assert '"verdict"' in prompt
        assert '"failure_attributions"' in prompt
        assert '"improvements"' in prompt


# ===========================================================================
# 2. Format Run for Review Tests
# ===========================================================================


class TestFormatRun:
    """Verify format_run_for_review builds correct structured text."""

    def test_includes_run_status(self):
        run = {
            "id": "r1",
            "status": "succeeded",
            "started_at": "2024-01-01T00:00:00Z",
            "ended_at": "2024-01-01T00:10:00Z",
            "steps": [],
        }
        result = format_run_for_review(run, [], [])
        assert "succeeded" in result
        assert "r1" in result

    def test_includes_failed_step_errors(self):
        run = {
            "id": "r2",
            "status": "failed",
            "steps": [
                {"step_key": "s1", "primitive": "robot.home", "status": "succeeded"},
                {
                    "step_key": "s2",
                    "primitive": "aspirate",
                    "status": "failed",
                    "error": "tip missing",
                },
            ],
        }
        result = format_run_for_review(run, [], [])
        assert "FAILED" in result
        assert "tip missing" in result
        assert "s2" in result

    def test_includes_kpi_values(self):
        run = {"id": "r3", "status": "succeeded", "steps": []}
        kpis = [
            {
                "kpi_name": "volume_accuracy_pct",
                "kpi_value": 99.7,
                "kpi_unit": "pct",
                "step_id": None,
            },
        ]
        result = format_run_for_review(run, kpis, [])
        assert "volume_accuracy_pct" in result
        assert "99.7" in result


# ===========================================================================
# 3. Parse Response Tests
# ===========================================================================


class TestParseResponse:
    """Verify parse_review_response handles various LLM output formats."""

    def test_valid_json(self):
        review = parse_review_response(SAMPLE_REVIEW_JSON)
        assert review.score == 92.5
        assert review.verdict == "passed"
        assert len(review.failure_attributions) == 0
        assert len(review.improvements) == 1
        assert review.improvements[0].category == "parameter"

    def test_json_in_code_block(self):
        wrapped = f"```json\n{SAMPLE_REVIEW_JSON}\n```"
        review = parse_review_response(wrapped)
        assert review.score == 92.5
        assert review.verdict == "passed"

    def test_failed_review_parse(self):
        review = parse_review_response(FAILED_REVIEW_JSON)
        assert review.score == 35.0
        assert review.verdict == "failed"
        assert len(review.failure_attributions) == 2
        assert review.failure_attributions[0].severity == "CRITICAL"
        assert review.failure_attributions[1].severity == "BYPASS"
        assert len(review.improvements) == 2

    def test_invalid_json_raises(self):
        with pytest.raises(ReviewParseError, match="Invalid JSON"):
            parse_review_response("{not valid json}")

    def test_missing_score_raises(self):
        bad = json.dumps({"verdict": "passed", "failure_attributions": [], "improvements": []})
        with pytest.raises(ReviewParseError, match="score"):
            parse_review_response(bad)

    def test_invalid_verdict_raises(self):
        bad = json.dumps({
            "score": 80,
            "verdict": "unknown_verdict",
            "failure_attributions": [],
            "improvements": [],
        })
        with pytest.raises(ReviewParseError, match="verdict"):
            parse_review_response(bad)

    def test_score_out_of_range_raises(self):
        bad = json.dumps({
            "score": 150,
            "verdict": "passed",
            "failure_attributions": [],
            "improvements": [],
        })
        with pytest.raises(ReviewParseError, match="score"):
            parse_review_response(bad)


# ===========================================================================
# 4. Core review_run Tests
# ===========================================================================


class TestReviewRun:
    """Verify the full review_run workflow with MockProvider."""

    @pytest.mark.anyio
    async def test_successful_run_review(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
            {"step_key": "s1", "primitive": "aspirate", "status": "succeeded"},
        ])
        provider = MockProvider(responses=[SAMPLE_REVIEW_JSON])
        review = await review_run(run_id, provider=provider)

        assert review.score == 92.5
        assert review.verdict == "passed"
        assert review.model == "mock-model"
        assert provider.call_count == 1
        # Verify system prompt was passed to LLM
        assert "score" in provider.last_call["system"].lower()

    @pytest.mark.anyio
    async def test_failed_run_review(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
            {
                "step_key": "s1",
                "primitive": "aspirate",
                "status": "failed",
                "error": "tip missing",
            },
            {"step_key": "s2", "primitive": "heat", "status": "skipped"},
        ])
        provider = MockProvider(responses=[FAILED_REVIEW_JSON])
        review = await review_run(run_id, provider=provider)

        assert review.score == 35.0
        assert review.verdict == "failed"
        assert len(review.failure_attributions) == 2
        assert len(review.improvements) == 2
        # Verify run data was passed as user message
        user_content = provider.last_call["messages"][0].content
        assert "tip missing" in user_content

    @pytest.mark.anyio
    async def test_nonexistent_run_raises(self):
        provider = MockProvider(responses=[SAMPLE_REVIEW_JSON])
        with pytest.raises(ValueError, match="Run not found"):
            await review_run("nonexistent-id", provider=provider)


# ===========================================================================
# 5. Storage Tests
# ===========================================================================


class TestStorage:
    """Verify review persistence and read path."""

    @pytest.mark.anyio
    async def test_review_persisted_to_db(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
        ])
        provider = MockProvider(responses=[SAMPLE_REVIEW_JSON])
        await review_run(run_id, provider=provider)

        stored = get_run_review(run_id)
        assert stored is not None
        assert stored["score"] == 92.5
        assert stored["verdict"] == "passed"
        assert stored["run_id"] == run_id

    @pytest.mark.anyio
    async def test_read_path_returns_none_for_unreviewed(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
        ])
        assert get_run_review(run_id) is None

    @pytest.mark.anyio
    async def test_schema_version_stored(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
        ])
        provider = MockProvider(responses=[SAMPLE_REVIEW_JSON])
        await review_run(run_id, provider=provider)

        stored = get_run_review(run_id)
        assert stored["review_schema_version"] == REVIEW_SCHEMA_VERSION

    @pytest.mark.anyio
    async def test_unique_constraint_prevents_duplicate(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
        ])
        provider = MockProvider(responses=[SAMPLE_REVIEW_JSON, SAMPLE_REVIEW_JSON])
        await review_run(run_id, provider=provider)

        # Second review for same run should raise (UNIQUE constraint)
        with pytest.raises(Exception):
            await review_run(run_id, provider=provider)

    @pytest.mark.anyio
    async def test_provenance_event_recorded(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
        ])
        provider = MockProvider(responses=[SAMPLE_REVIEW_JSON])
        await review_run(run_id, provider=provider)

        with connection() as conn:
            row = conn.execute(
                "SELECT * FROM provenance_events WHERE run_id = ? AND action = 'run.reviewed'",
                (run_id,),
            ).fetchone()
        assert row is not None
        details = json.loads(row["details_json"])
        assert "review_id" in details
        assert details["score"] == 92.5
        assert details["verdict"] == "passed"

    @pytest.mark.anyio
    async def test_failure_attributions_and_improvements_stored(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
            {"step_key": "s1", "primitive": "aspirate", "status": "failed", "error": "tip missing"},
        ])
        provider = MockProvider(responses=[FAILED_REVIEW_JSON])
        await review_run(run_id, provider=provider)

        stored = get_run_review(run_id)
        assert len(stored["failure_attributions"]) == 2
        assert stored["failure_attributions"][0]["severity"] == "CRITICAL"
        assert stored["failure_attributions"][0]["root_cause"] != ""
        assert len(stored["improvements"]) == 2
        assert stored["improvements"][0]["category"] == "sequence"


# ===========================================================================
# 6. Failure Isolation Tests
# ===========================================================================


class TestFailureIsolation:
    """Verify advisory wrapper — reviewer errors never block."""

    @pytest.mark.anyio
    async def test_on_run_completed_bad_run_id(self):
        """_on_run_completed with nonexistent run should log, not raise."""
        # Should complete without exception
        await _on_run_completed("nonexistent-run-id")

    @pytest.mark.anyio
    async def test_on_run_completed_llm_error(self):
        """_on_run_completed handles LLM failure gracefully."""
        run_id = _insert_run_with_steps([
            {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
        ])
        # Default provider in test env is MockProvider with no responses → LLMError
        # _on_run_completed should catch and log
        await _on_run_completed(run_id)
        # Verify no review was stored
        assert get_run_review(run_id) is None


# ===========================================================================
# 7. Event Listener Tests
# ===========================================================================


class TestEventListener:
    """Verify EventBus integration with reviewer listener."""

    @pytest.mark.anyio
    async def test_listener_processes_run_completed(self):
        """EventBus → run.completed → listener triggers review."""
        from app.services.event_bus import EventBus, EventMessage

        bus = EventBus()
        await bus.start()

        try:
            sub = await start_reviewer_listener(bus)

            # Create a run that the listener will try to review
            run_id = _insert_run_with_steps([
                {"step_key": "s0", "primitive": "robot.home", "status": "succeeded"},
            ])

            # Publish event — the listener will try to review
            # (It will fail because the default MockProvider has no responses,
            #  but it should handle the error gracefully without raising)
            event = EventMessage(
                id=str(uuid.uuid4()),
                run_id=run_id,
                actor="test",
                action="run.completed",
                details={},
                created_at=utcnow_iso(),
            )
            bus.publish(event)

            # Give the listener time to process
            await asyncio.sleep(0.1)

            await stop_reviewer_listener(sub, bus)
        finally:
            await bus.stop()
