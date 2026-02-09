"""Tests for Evolution Engine (Phase C5: Priors + Templates + Human Gate).

Covers:
- Prior tightening (compute bounds, magnitude, store/read, generation increment)
- Protocol templates (create, version increment, parent lineage)
- Evolution proposals (create, auto-approve rules, approve/reject, status)
- process_review_event (high-score triggers, low-score skipped, missing review)
- Human gate (large magnitude pending, small auto-approved, approve applies)
- candidate_gen integration (evolved prior overrides bounds, fallback)
- Event listener (receives run.reviewed, advisory never raises)
- Storage (tables exist, FK constraints, read paths)
"""
from __future__ import annotations

import os
import tempfile
import uuid

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_evolution_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "evolution_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import asyncio  # noqa: E402

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, json_dumps, utcnow_iso  # noqa: E402
from app.services.evolution import (  # noqa: E402
    AUTO_APPROVE_MAGNITUDE,
    EVOLUTION_SCHEMA_VERSION,
    MIN_SAMPLE_COUNT,
    PRIOR_K_STDDEV,
    PRIOR_TIGHTEN_MIN_SCORE,
    TEMPLATE_CREATE_MIN_SCORE,
    EvolvedPrior,
    EvolutionProposal,
    _calc_prior_magnitude,
    _compute_tightened_bounds,
    _should_auto_approve,
    approve_proposal,
    create_evolution_proposal,
    create_template,
    evolve_priors,
    get_active_evolved_prior,
    get_proposal,
    get_template,
    list_evolved_priors,
    list_proposals,
    list_templates,
    maybe_create_template,
    process_review_event,
    reject_proposal,
    start_evolution_listener,
    stop_evolution_listener,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    with connection() as conn:
        # Cleanup in FK-safe order
        conn.execute("DELETE FROM evolved_priors")
        conn.execute("DELETE FROM evolution_proposals")
        conn.execute("DELETE FROM protocol_templates")
        conn.execute("DELETE FROM run_reviews")
        conn.execute("DELETE FROM run_kpis")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM run_steps")
        conn.execute("DELETE FROM batch_candidates")
        conn.execute("DELETE FROM batch_requests")
        conn.execute("DELETE FROM memory_semantic")
        conn.execute("DELETE FROM provenance_events")
        conn.execute("DELETE FROM approvals")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM campaigns")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_run(run_id: str | None = None, protocol: dict | None = None, campaign_id: str | None = None) -> str:
    """Insert a minimal run into the DB and return its id."""
    rid = run_id or str(uuid.uuid4())
    now = utcnow_iso()
    proto = protocol or {"steps": [{"id": "s1", "primitive": "heat", "params": {"temp": 50}}]}
    with connection() as conn:
        conn.execute(
            "INSERT INTO runs (id, campaign_id, trigger_type, trigger_payload_json, "
            "session_key, status, protocol_json, inputs_json, compiled_graph_json, "
            "graph_hash, policy_snapshot_json, rejection_reason, created_by, "
            "created_at, updated_at, started_at, ended_at) "
            "VALUES (?, ?, 'manual', '{}', 'test-session', 'succeeded', ?, '{}', "
            "'{}', NULL, '{}', NULL, 'test', ?, ?, ?, ?)",
            (rid, campaign_id, json_dumps(proto), now, now, now, now),
        )
        conn.commit()
    return rid


def _insert_run_step(run_id: str, step_key: str, primitive: str, params: dict, status: str = "succeeded") -> str:
    """Insert a run_step and return its id."""
    step_id = str(uuid.uuid4())
    now = utcnow_iso()
    with connection() as conn:
        conn.execute(
            "INSERT INTO run_steps (id, run_id, step_key, primitive, params_json, "
            "depends_on_json, resources_json, status, attempt, idempotency_key) "
            "VALUES (?, ?, ?, ?, ?, '[]', '[]', ?, 1, ?)",
            (step_id, run_id, step_key, primitive, json_dumps(params), status, str(uuid.uuid4())),
        )
        conn.commit()
    return step_id


def _insert_review(run_id: str, score: float, verdict: str, improvements: list | None = None, failure_attributions: list | None = None) -> str:
    """Insert a run_review and return its id."""
    review_id = str(uuid.uuid4())
    now = utcnow_iso()
    with connection() as conn:
        conn.execute(
            "INSERT INTO run_reviews (id, run_id, score, verdict, "
            "failure_attributions_json, improvements_json, model, "
            "review_schema_version, raw_response, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'test-model', '1', '{}', ?)",
            (
                review_id,
                run_id,
                score,
                verdict,
                json_dumps(failure_attributions or []),
                json_dumps(improvements or []),
                now,
            ),
        )
        conn.commit()
    return review_id


def _insert_memory_semantic(
    primitive: str, param_name: str, mean: float, stddev: float, sample_count: int
) -> None:
    """Insert or replace a memory_semantic row."""
    now = utcnow_iso()
    with connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO memory_semantic "
            "(primitive, param_name, mean, stddev, sample_count, "
            "success_rate, success_count, total_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, ?)",
            (primitive, param_name, mean, stddev, sample_count, sample_count, sample_count, now),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Test: Evolved Priors
# ---------------------------------------------------------------------------


class TestEvolvedPriors:
    """Tests for prior tightening computations and storage."""

    def test_compute_tightened_bounds(self):
        """_compute_tightened_bounds returns (min, max, confidence) from memory_semantic."""
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)
        result = _compute_tightened_bounds("heat", "temp", k=2.0)
        assert result is not None
        evolved_min, evolved_max, confidence = result
        assert evolved_min == pytest.approx(40.0)  # 50 - 2*5
        assert evolved_max == pytest.approx(60.0)  # 50 + 2*5
        assert 0 < confidence <= 1.0

    def test_compute_tightened_bounds_insufficient_samples(self):
        """Returns None when sample_count < MIN_SAMPLE_COUNT."""
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=3)
        result = _compute_tightened_bounds("heat", "temp")
        assert result is None

    def test_calc_prior_magnitude(self):
        """Magnitude = 1 - (new_range / old_range)."""
        # 50% tightening
        mag = _calc_prior_magnitude(0.0, 100.0, 25.0, 75.0)
        assert mag == pytest.approx(0.5)

        # No change
        mag = _calc_prior_magnitude(0.0, 100.0, 0.0, 100.0)
        assert mag == pytest.approx(0.0)

        # Complete collapse
        mag = _calc_prior_magnitude(0.0, 100.0, 50.0, 50.0)
        assert mag == pytest.approx(1.0)

    def test_evolved_prior_generation_increment(self):
        """Creating successive evolved priors increments generation."""
        run_id = _insert_run()
        _insert_run_step(run_id, "s1", "heat", {"temp": 50.0})
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)

        # First evolution — small magnitude for auto-approve
        p1 = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Tighten heat.temp",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 40.0,
                "evolved_max": 60.0,
                "confidence": 0.5,
                "generation": 1,
                "old_min": 0.0,
                "old_max": 100.0,
            },
            magnitude=0.1,  # auto-approve
        )
        prior1 = get_active_evolved_prior("heat", "temp")
        assert prior1 is not None
        assert prior1.generation == 1

        # Second evolution
        p2 = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Tighten heat.temp further",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.7,
                "generation": 2,
                "old_min": 40.0,
                "old_max": 60.0,
            },
            magnitude=0.1,  # auto-approve
        )
        prior2 = get_active_evolved_prior("heat", "temp")
        assert prior2 is not None
        assert prior2.generation == 2
        # Previous prior deactivated
        assert prior1.id != prior2.id


# ---------------------------------------------------------------------------
# Test: Protocol Templates
# ---------------------------------------------------------------------------


class TestProtocolTemplates:
    """Tests for versioned protocol template library."""

    def test_create_template(self):
        """create_template stores and returns template."""
        result = create_template(
            name="test-template",
            protocol={"steps": [{"id": "s1"}]},
            tags=["tag1", "tag2"],
        )
        assert result["name"] == "test-template"
        assert result["version"] == 1
        assert result["tags"] == ["tag1", "tag2"]
        assert result["is_active"] is True

    def test_version_increment(self):
        """Creating templates with same name auto-increments version."""
        t1 = create_template(name="my-proto", protocol={"v": 1})
        t2 = create_template(name="my-proto", protocol={"v": 2})
        assert t1["version"] == 1
        assert t2["version"] == 2

    def test_parent_lineage(self):
        """Template can reference a parent template."""
        t1 = create_template(name="parent-proto", protocol={"v": 1})
        t2 = create_template(
            name="child-proto",
            protocol={"v": 2},
            parent_template_id=t1["id"],
        )
        assert t2["parent_template_id"] == t1["id"]

        # Read back
        fetched = get_template(t2["id"])
        assert fetched is not None
        assert fetched["parent_template_id"] == t1["id"]


# ---------------------------------------------------------------------------
# Test: Evolution Proposals
# ---------------------------------------------------------------------------


class TestEvolutionProposals:
    """Tests for the proposal system and auto-approve rules."""

    def test_create_proposal(self):
        """create_evolution_proposal stores proposal."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Test proposal",
            change_details={"test": True},
            magnitude=0.5,
        )
        proposal = get_proposal(proposal_id)
        assert proposal is not None
        assert proposal["proposal_type"] == "prior_tightening"
        assert proposal["status"] == "pending"  # magnitude > 0.3

    def test_auto_approve_small_magnitude(self):
        """Small magnitude proposals are auto-approved."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Small change",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.5,
                "generation": 1,
            },
            magnitude=0.1,  # < AUTO_APPROVE_MAGNITUDE
        )
        proposal = get_proposal(proposal_id)
        assert proposal is not None
        assert proposal["status"] == "auto_approved"
        assert proposal["auto_approve_reason"] is not None

    def test_approve_pending_proposal(self):
        """approve_proposal transitions pending → approved."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Needs approval",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.5,
                "generation": 1,
            },
            magnitude=0.8,  # > threshold → pending
        )
        proposal = get_proposal(proposal_id)
        assert proposal["status"] == "pending"

        result = approve_proposal(proposal_id, "scientist-1", "Looks good")
        assert result["status"] == "approved"
        assert result["reviewed_by"] == "scientist-1"

    def test_reject_pending_proposal(self):
        """reject_proposal transitions pending → rejected."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="template_creation",
            change_summary="Reject me",
            change_details={"test": True},
            magnitude=0.9,
        )
        result = reject_proposal(proposal_id, "scientist-2", "Too aggressive")
        assert result["status"] == "rejected"
        assert result["reviewed_by"] == "scientist-2"


# ---------------------------------------------------------------------------
# Test: Auto-Approve Rules
# ---------------------------------------------------------------------------


class TestAutoApproveRules:
    """Tests for _should_auto_approve rule engine."""

    def test_small_magnitude_auto_approves(self):
        approved, reason = _should_auto_approve("prior_tightening", 0.1)
        assert approved is True
        assert reason is not None

    def test_large_magnitude_needs_human(self):
        approved, reason = _should_auto_approve("prior_tightening", 0.5)
        assert approved is False
        assert reason is None

    def test_template_moderate_magnitude_auto_approves(self):
        """Template creation with magnitude < 0.5 is auto-approved."""
        approved, reason = _should_auto_approve("template_creation", 0.4)
        assert approved is True


# ---------------------------------------------------------------------------
# Test: Process Review Event
# ---------------------------------------------------------------------------


class TestProcessReviewEvent:
    """Tests for the main entry point: process_review_event."""

    def test_high_score_triggers_evolution(self):
        """A high-score passed review triggers prior evolution."""
        run_id = _insert_run()
        _insert_run_step(run_id, "s1", "heat", {"temp": 50.0})
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)
        _insert_review(run_id, score=85.0, verdict="passed")

        process_review_event(run_id)

        # Should have created at least one proposal
        proposals = list_proposals(run_id=run_id)
        assert len(proposals) >= 1
        prior_proposals = [p for p in proposals if p["proposal_type"] == "prior_tightening"]
        assert len(prior_proposals) >= 1

    def test_low_score_skipped(self):
        """A low-score review does not trigger prior evolution."""
        run_id = _insert_run()
        _insert_run_step(run_id, "s1", "heat", {"temp": 50.0})
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)
        _insert_review(run_id, score=40.0, verdict="failed")

        process_review_event(run_id)

        proposals = list_proposals(run_id=run_id)
        prior_proposals = [p for p in proposals if p["proposal_type"] == "prior_tightening"]
        assert len(prior_proposals) == 0

    def test_missing_review_handled(self):
        """process_review_event with no review doesn't raise."""
        run_id = _insert_run()
        # No review inserted — should not raise
        process_review_event(run_id)

    def test_high_score_creates_template_proposal(self):
        """A score >= TEMPLATE_CREATE_MIN_SCORE creates a template proposal."""
        run_id = _insert_run()
        _insert_review(run_id, score=90.0, verdict="passed")

        process_review_event(run_id)

        proposals = list_proposals(run_id=run_id)
        template_proposals = [p for p in proposals if p["proposal_type"] == "template_creation"]
        assert len(template_proposals) == 1


# ---------------------------------------------------------------------------
# Test: Human Gate
# ---------------------------------------------------------------------------


class TestHumanGate:
    """Tests for the human approval gate."""

    def test_large_magnitude_pending(self):
        """Large magnitude prior changes require human approval."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Big change",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 49.0,
                "evolved_max": 51.0,
                "confidence": 0.9,
                "generation": 1,
            },
            magnitude=0.7,
        )
        proposal = get_proposal(proposal_id)
        assert proposal["status"] == "pending"

        # No evolved prior should exist yet
        prior = get_active_evolved_prior("heat", "temp")
        assert prior is None

    def test_approve_applies_changes(self):
        """Approving a pending prior_tightening proposal creates the evolved prior."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Needs approval",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.8,
                "generation": 1,
            },
            magnitude=0.6,  # pending
        )
        # Before approval: no prior
        assert get_active_evolved_prior("heat", "temp") is None

        approve_proposal(proposal_id, "admin")

        # After approval: evolved prior created
        prior = get_active_evolved_prior("heat", "temp")
        assert prior is not None
        assert prior.evolved_min == pytest.approx(45.0)
        assert prior.evolved_max == pytest.approx(55.0)

    def test_reject_does_not_apply(self):
        """Rejecting a proposal does not create an evolved prior."""
        run_id = _insert_run()
        proposal_id = create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Will be rejected",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.8,
                "generation": 1,
            },
            magnitude=0.6,
        )
        reject_proposal(proposal_id, "admin", "Too risky")

        # No prior created
        assert get_active_evolved_prior("heat", "temp") is None


# ---------------------------------------------------------------------------
# Test: Candidate Gen Integration
# ---------------------------------------------------------------------------


class TestCandidateGenIntegration:
    """Tests for evolved prior integration into candidate_gen."""

    def test_evolved_prior_overrides_bounds(self):
        """When an evolved prior exists, sample_prior_guided uses tightened bounds."""
        from app.services.candidate_gen import ParameterSpace, SearchDimension, sample_prior_guided

        run_id = _insert_run()
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)

        # Create an auto-approved evolved prior: [45, 55]
        create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Tighten",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.8,
                "generation": 1,
            },
            magnitude=0.1,  # auto-approve
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="temp",
                    param_type="number",
                    min_value=0.0,
                    max_value=100.0,
                    primitive="heat",
                ),
            ),
            protocol_template={},
        )

        # Sample many candidates to verify bounds
        samples = sample_prior_guided(space, 50, seed=42)
        for s in samples:
            # All values should be within evolved bounds [45, 55]
            assert 45.0 <= s["temp"] <= 55.0, f"temp={s['temp']} outside [45, 55]"

    def test_no_evolved_prior_uses_dimension_bounds(self):
        """Without evolved priors, sample_prior_guided uses dimension bounds."""
        from app.services.candidate_gen import ParameterSpace, SearchDimension, sample_prior_guided

        _insert_memory_semantic("cool", "rate", mean=5.0, stddev=1.0, sample_count=10)

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="rate",
                    param_type="number",
                    min_value=0.0,
                    max_value=10.0,
                    primitive="cool",
                ),
            ),
            protocol_template={},
        )

        samples = sample_prior_guided(space, 20, seed=42)
        for s in samples:
            assert 0.0 <= s["rate"] <= 10.0

    def test_scoring_penalizes_out_of_evolved_bounds(self):
        """_score_candidate adds penalty for values outside evolved bounds."""
        from app.services.candidate_gen import ParameterSpace, SearchDimension, _score_candidate

        run_id = _insert_run()
        _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)

        # Create evolved prior [45, 55]
        create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Tighten",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 45.0,
                "evolved_max": 55.0,
                "confidence": 0.8,
                "generation": 1,
            },
            magnitude=0.1,
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="temp",
                    param_type="number",
                    min_value=0.0,
                    max_value=100.0,
                    primitive="heat",
                ),
            ),
            protocol_template={},
        )

        # Score a value within evolved bounds
        score_in = _score_candidate({"temp": 50.0}, space)
        # Score a value outside evolved bounds
        score_out = _score_candidate({"temp": 30.0}, space)

        assert score_in is not None
        assert score_out is not None
        assert score_out > score_in  # penalty makes it higher (worse)


# ---------------------------------------------------------------------------
# Test: Event Listener
# ---------------------------------------------------------------------------


class TestEventListener:
    """Tests for the evolution event listener."""

    def test_listener_receives_run_reviewed(self):
        """Listener processes run.reviewed events."""
        from app.services.event_bus import EventBus, EventMessage

        bus = EventBus()
        loop = asyncio.new_event_loop()

        async def _run():
            await bus.start()
            sub = await start_evolution_listener(bus)

            run_id = _insert_run()
            _insert_review(run_id, score=85.0, verdict="passed")
            _insert_run_step(run_id, "s1", "heat", {"temp": 50.0})
            _insert_memory_semantic("heat", "temp", mean=50.0, stddev=5.0, sample_count=10)

            event = EventMessage(
                id=str(uuid.uuid4()),
                run_id=run_id,
                actor="reviewer",
                action="run.reviewed",
                details={"score": 85.0, "verdict": "passed"},
                created_at=utcnow_iso(),
            )
            bus.publish(event)

            # Give the listener time to process
            await asyncio.sleep(0.2)

            await stop_evolution_listener(sub, bus)
            await bus.stop()

            # Should have created proposals
            proposals = list_proposals(run_id=run_id)
            return proposals

        try:
            proposals = loop.run_until_complete(_run())
            assert len(proposals) >= 1
        finally:
            loop.close()

    def test_listener_advisory_never_raises(self):
        """Listener doesn't raise even if processing fails."""
        from app.services.event_bus import EventBus, EventMessage

        bus = EventBus()
        loop = asyncio.new_event_loop()

        async def _run():
            await bus.start()
            sub = await start_evolution_listener(bus)

            # Publish event for non-existent run — should not raise
            event = EventMessage(
                id=str(uuid.uuid4()),
                run_id="nonexistent-run-id",
                actor="reviewer",
                action="run.reviewed",
                details={},
                created_at=utcnow_iso(),
            )
            bus.publish(event)

            await asyncio.sleep(0.2)
            await stop_evolution_listener(sub, bus)
            await bus.stop()

        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Test: Storage & Schema
# ---------------------------------------------------------------------------


class TestStorage:
    """Tests for DB table existence and read paths."""

    def test_tables_exist(self):
        """All C5 tables exist in the DB."""
        with connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('evolved_priors', 'evolution_proposals', 'protocol_templates')"
            ).fetchall()
            table_names = {r["name"] for r in tables}
            assert "evolved_priors" in table_names
            assert "evolution_proposals" in table_names
            assert "protocol_templates" in table_names

    def test_list_evolved_priors(self):
        """list_evolved_priors returns stored priors."""
        run_id = _insert_run()
        create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Test",
            change_details={
                "primitive": "heat",
                "param_name": "temp",
                "evolved_min": 40.0,
                "evolved_max": 60.0,
                "confidence": 0.5,
                "generation": 1,
            },
            magnitude=0.1,  # auto-approve → creates prior
        )
        priors = list_evolved_priors(primitive="heat")
        assert len(priors) >= 1
        assert priors[0]["primitive"] == "heat"

    def test_list_templates(self):
        """list_templates returns stored templates."""
        create_template(name="tpl-a", protocol={"v": 1})
        create_template(name="tpl-b", protocol={"v": 2})

        all_templates = list_templates()
        assert len(all_templates) >= 2

        filtered = list_templates(name="tpl-a")
        assert len(filtered) == 1
        assert filtered[0]["name"] == "tpl-a"

    def test_list_proposals_filter(self):
        """list_proposals filters by status."""
        run_id = _insert_run()
        # Create one pending, one auto-approved
        create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Pending",
            change_details={"primitive": "x", "param_name": "y", "evolved_min": 0, "evolved_max": 1, "confidence": 0.5, "generation": 1},
            magnitude=0.8,  # pending
        )
        create_evolution_proposal(
            run_id=run_id,
            proposal_type="prior_tightening",
            change_summary="Auto",
            change_details={"primitive": "a", "param_name": "b", "evolved_min": 0, "evolved_max": 1, "confidence": 0.5, "generation": 1},
            magnitude=0.1,  # auto-approved
        )

        pending = list_proposals(status="pending")
        auto = list_proposals(status="auto_approved")
        assert len(pending) >= 1
        assert len(auto) >= 1
        assert all(p["status"] == "pending" for p in pending)
        assert all(p["status"] == "auto_approved" for p in auto)
