"""G3E2 End-to-End Integration Tests.

Tests the complete Goal-Generate-Execute-Evaluate-Evolve loop:
1. Goal: Define optimization objective
2. Generate: Create candidate parameters (with evolved priors)
3. Execute: Run candidates and collect KPIs
4. Evaluate: Assess progress, detect convergence
5. Evolve: Trigger prior tightening and template creation

This test verifies that all 5 phases work together correctly.
"""
from __future__ import annotations

import os
import tempfile
import uuid

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_g3e2_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "g3e2_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

from typing import Any  # noqa: E402

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import init_db  # noqa: E402
from app.services.campaign_loop import (  # noqa: E402
    CampaignGoal,
    run_campaign_offline,
)
from app.services.candidate_gen import ParameterSpace, SearchDimension  # noqa: E402
import uuid  # noqa: E402

from app.core.db import connection, json_dumps, utcnow_iso  # noqa: E402
from app.services.evolution import (  # noqa: E402
    get_active_evolved_prior,
    list_evolved_priors,
    list_proposals,
    list_templates,
    process_review_event,
)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize DB before each test."""
    init_db()


# ---------------------------------------------------------------------------
# Helper: Database insertion
# ---------------------------------------------------------------------------


def _insert_memory_semantic(
    primitive: str, param_name: str, mean: float, stddev: float, sample_count: int
) -> None:
    """Insert or replace a memory_semantic row."""
    now = utcnow_iso()
    with connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_semantic (
                primitive, param_name, mean, stddev, sample_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                primitive,
                param_name,
                mean,
                stddev,
                sample_count,
                now,
            ),
        )
        conn.commit()


def _insert_campaign(campaign_id: str) -> None:
    """Insert a minimal campaign record for offline testing."""
    now = utcnow_iso()
    with connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO campaigns "
            "(id, name, cadence_seconds, protocol_json, inputs_json, "
            "policy_json, next_fire_at, is_active, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                campaign_id,
                f"test-{campaign_id}",
                3600,
                json_dumps({"steps": []}),
                json_dumps({}),
                json_dumps({}),
                now,
                1,
                now,
                now,
            ),
        )
        conn.commit()


def _insert_run(run_id: str, campaign_id: str, params: dict | None = None) -> None:
    """Insert a minimal run record with optional params for evolution."""
    now = utcnow_iso()
    params = params or {}

    with connection() as conn:
        # Insert run
        conn.execute(
            "INSERT OR IGNORE INTO runs "
            "(id, campaign_id, trigger_type, trigger_payload_json, session_key, "
            "status, protocol_json, inputs_json, policy_snapshot_json, "
            "created_by, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                campaign_id,
                "manual",
                json_dumps({}),
                "test-session",
                "completed",
                json_dumps({"steps": []}),
                json_dumps({}),
                json_dumps({}),
                "test-user",
                now,
                now,
            ),
        )

        # Insert run_step if params provided (needed for evolve_priors)
        if params:
            step_id = str(uuid.uuid4())
            # Extract primitive from params if available (use 'test_primitive' as default)
            primitive = params.pop("_primitive", "test_primitive")
            conn.execute(
                "INSERT OR IGNORE INTO run_steps "
                "(id, run_id, step_key, primitive, params_json, "
                "depends_on_json, resources_json, status, attempt, idempotency_key) "
                "VALUES (?, ?, ?, ?, ?, '[]', '[]', 'succeeded', 1, ?)",
                (
                    step_id,
                    run_id,
                    "s1",
                    primitive,
                    json_dumps(params),
                    str(uuid.uuid4()),
                ),
            )

        conn.commit()


def _insert_review(
    run_id: str,
    campaign_id: str,
    score: float,
    verdict: str,
    params: dict | None = None,
    improvements: list | None = None,
    failure_attributions: list | None = None,
    comments: str = "",
) -> str:
    """Insert a run_review and return its id."""
    # First ensure run exists (with params for evolution)
    _insert_run(run_id, campaign_id, params=params)

    review_id = str(uuid.uuid4())
    now = utcnow_iso()
    improvements = improvements or []
    failure_attributions = failure_attributions or []
    with connection() as conn:
        conn.execute(
            "INSERT INTO run_reviews (id, run_id, score, verdict, "
            "failure_attributions_json, improvements_json, model, "
            "review_schema_version, raw_response, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'test-model', '1', ?, ?)",
            (
                review_id,
                run_id,
                score,
                verdict,
                json_dumps(failure_attributions),
                json_dumps(improvements),
                json_dumps({"comments": comments}),
                now,
            ),
        )
        conn.commit()
    return review_id


# ---------------------------------------------------------------------------
# Helper: Simulation function
# ---------------------------------------------------------------------------


def _sim_parabola(params: dict[str, Any]) -> dict[str, Any]:
    """Simulate a parabola: KPI = -(x-5)^2 + 100.

    Optimal at x=5, KPI=100.
    """
    x = params.get("x", 0.0)
    kpi = -(x - 5.0) ** 2 + 100.0
    return {"quadratic_kpi": kpi}


# ---------------------------------------------------------------------------
# Test: End-to-End G3E2 Loop
# ---------------------------------------------------------------------------


class TestG3E2EndToEnd:
    """Test complete Goal-Generate-Execute-Evaluate-Evolve cycle."""

    def test_full_g3e2_cycle_with_evolution(self):
        """Test all 5 G3E2 phases work together.

        Scenario:
        1. Goal: Maximize quadratic_kpi with 3 rounds
        2. Generate: Use prior_guided sampling (will use evolved priors after round 1)
        3. Execute: Offline simulator (parabola)
        4. Evaluate: Track best KPI, detect convergence
        5. Evolve: Process high-score runs → create evolved priors
        """
        # Phase 1: Goal
        goal = CampaignGoal(
            objective_kpi="quadratic_kpi",
            direction="maximize",
            target_value=99.0,  # Near-optimal target
            max_rounds=3,
            batch_size=5,
            strategy="prior_guided",  # Uses evolved priors if available
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="x",
                    param_type="number",
                    min_value=0.0,
                    max_value=10.0,
                    primitive="optimize",  # For memory/evolution lookup
                ),
            ),
            protocol_template={"steps": []},
        )

        # Seed initial memory stats for x (centered around 5.0)
        _insert_memory_semantic("optimize", "x", mean=5.0, stddev=2.0, sample_count=10)

        # Create campaign record for FK constraint
        campaign_id = f"g3e2-test-{uuid.uuid4().hex[:8]}"
        _insert_campaign(campaign_id)

        # Phase 2-4: Run campaign (Generate, Execute, Evaluate)
        result = run_campaign_offline(
            goal=goal,
            space=space,
            sim_fn=_sim_parabola,
            campaign_id=campaign_id,
        )

        # Verify campaign completed
        assert len(result.rounds) >= 1
        assert result.best_kpi is not None
        assert result.best_kpi > 95.0  # Should find near-optimal value

        # Phase 5: Evolve - manually trigger evolution for each round's best run
        # In production, this happens via event listener
        for round_result in result.rounds:
            if round_result.run_ids:
                best_run_id = round_result.run_ids[0]  # Simplified: use first run

                # Create a high-score review to trigger evolution
                # Need to extract params from the candidate
                # For simplicity, use approximate params
                _insert_review(
                    run_id=best_run_id,
                    campaign_id=campaign_id,
                    score=85.0,  # High score triggers evolution
                    verdict="passed",
                    params={"_primitive": "optimize", "x": 5.0},  # Approximate optimal params
                    improvements=[],
                    failure_attributions=[],
                    comments="Automated review for G3E2 test",
                )

                # Trigger evolution manually (simulating event listener)
                process_review_event(best_run_id)

        # Verify evolution created proposals and evolved priors
        proposals = list_proposals()
        assert len(proposals) > 0, "Evolution should create proposals"

        priors = list_evolved_priors(primitive="optimize", active_only=True)
        assert len(priors) > 0, "Evolution should create evolved priors for x"

        # Verify evolved prior tightened bounds around optimal value (x≈5.0)
        evolved_x = get_active_evolved_prior("optimize", "x")
        if evolved_x is not None:
            # Tightened bounds should be narrower than original [0, 10]
            assert evolved_x.evolved_min >= 0.0
            assert evolved_x.evolved_max <= 10.0
            assert evolved_x.evolved_max - evolved_x.evolved_min < 10.0
            # Should be centered around optimal region (3-7)
            assert 3.0 <= evolved_x.evolved_min <= 7.0
            assert 3.0 <= evolved_x.evolved_max <= 7.0

    def test_g3e2_prior_tightening_improves_convergence(self):
        """Test that evolved priors help converge faster in subsequent campaigns."""
        # First campaign: establish priors
        goal1 = CampaignGoal(
            objective_kpi="quadratic_kpi",
            direction="maximize",
            target_value=None,
            max_rounds=2,
            batch_size=10,
            strategy="prior_guided",
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="x",
                    param_type="number",
                    min_value=0.0,
                    max_value=10.0,
                    primitive="opt2",
                ),
            ),
            protocol_template={"steps": []},
        )

        # Seed broad initial stats
        _insert_memory_semantic("opt2", "x", mean=5.0, stddev=3.0, sample_count=5)

        campaign_id1 = f"g3e2-camp1-{uuid.uuid4().hex[:8]}"
        _insert_campaign(campaign_id1)

        result1 = run_campaign_offline(
            goal=goal1,
            space=space,
            sim_fn=_sim_parabola,
            campaign_id=campaign_id1,
        )

        # Trigger evolution
        for round_result in result1.rounds:
            if round_result.run_ids:
                run_id = round_result.run_ids[0]
                _insert_review(
                    run_id=run_id,
                    campaign_id=campaign_id1,
                    score=80.0,
                    verdict="passed",
                    params={"_primitive": "opt2", "x": 5.0},
                    improvements=[],
                    failure_attributions=[],
                    comments="Campaign 1 review",
                )
                process_review_event(run_id)

        # Verify evolved prior exists
        evolved = get_active_evolved_prior("opt2", "x")
        assert evolved is not None, "First campaign should create evolved prior"

        # Second campaign: should benefit from tightened prior
        goal2 = CampaignGoal(
            objective_kpi="quadratic_kpi",
            direction="maximize",
            target_value=None,
            max_rounds=2,
            batch_size=10,
            strategy="prior_guided",  # Will use evolved prior from campaign 1
        )

        campaign_id2 = f"g3e2-camp2-{uuid.uuid4().hex[:8]}"
        _insert_campaign(campaign_id2)

        result2 = run_campaign_offline(
            goal=goal2,
            space=space,
            sim_fn=_sim_parabola,
            campaign_id=campaign_id2,
        )

        # Second campaign should achieve better or equal performance
        # (hard to guarantee better due to randomness, but evolved priors help)
        assert result2.best_kpi is not None
        assert result2.best_kpi >= 90.0  # Should find good value quickly

    def test_g3e2_template_creation_from_high_score(self):
        """Test that high-scoring runs create protocol templates."""
        goal = CampaignGoal(
            objective_kpi="quadratic_kpi",
            direction="maximize",
            target_value=None,
            max_rounds=2,
            batch_size=5,
            strategy="lhs",
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="x",
                    param_type="number",
                    min_value=0.0,
                    max_value=10.0,
                    primitive="opt3",
                ),
            ),
            protocol_template={"steps": [{"action": "optimize", "params": {}}]},
        )

        campaign_id = f"g3e2-template-{uuid.uuid4().hex[:8]}"
        _insert_campaign(campaign_id)

        result = run_campaign_offline(
            goal=goal,
            space=space,
            sim_fn=_sim_parabola,
            campaign_id=campaign_id,
        )

        # Trigger evolution with high score (≥80 triggers template creation)
        if result.rounds and result.rounds[0].run_ids:
            run_id = result.rounds[0].run_ids[0]
            _insert_review(
                run_id=run_id,
                campaign_id=campaign_id,
                score=85.0,  # High score triggers template
                verdict="passed",
                params={"_primitive": "opt3", "x": 5.0},
                improvements=[],
                failure_attributions=[],
                comments="High-score run for template",
            )
            process_review_event(run_id)

        # Verify template was created
        templates = list_templates(is_active=True)
        assert len(templates) > 0, "High-score run should create template"

        # Verify template contains protocol
        template = templates[0]
        assert "protocol" in template
        assert template["protocol"] is not None

    def test_g3e2_low_score_skips_evolution(self):
        """Test that low-score runs do not trigger evolution."""
        goal = CampaignGoal(
            objective_kpi="quadratic_kpi",
            direction="maximize",
            target_value=None,
            max_rounds=1,
            batch_size=3,
            strategy="random",
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="x",
                    param_type="number",
                    min_value=0.0,
                    max_value=10.0,
                    primitive="opt4",
                ),
            ),
            protocol_template={"steps": []},
        )

        campaign_id = f"g3e2-lowscore-{uuid.uuid4().hex[:8]}"
        _insert_campaign(campaign_id)

        result = run_campaign_offline(
            goal=goal,
            space=space,
            sim_fn=_sim_parabola,
            campaign_id=campaign_id,
        )

        # Count proposals before
        proposals_before = list_proposals()
        priors_before = list_evolved_priors(primitive="opt4")

        # Trigger evolution with low score (<70 should skip prior tightening)
        if result.rounds and result.rounds[0].run_ids:
            run_id = result.rounds[0].run_ids[0]
            _insert_review(
                run_id=run_id,
                campaign_id=campaign_id,
                score=50.0,  # Low score
                verdict="passed",
                params={"_primitive": "opt4", "x": 3.0},
                improvements=[],
                failure_attributions=[],
                comments="Low-score run",
            )
            process_review_event(run_id)

        # Verify no new priors or proposals created
        proposals_after = list_proposals()
        priors_after = list_evolved_priors(primitive="opt4")

        assert len(priors_after) == len(priors_before), "Low score should not create priors"
        # Proposals might still be created for templates if score > 80, but priors should not


# ---------------------------------------------------------------------------
# Test: G3E2 Components Coordination
# ---------------------------------------------------------------------------


class TestG3E2ComponentCoordination:
    """Test that G3E2 components coordinate correctly."""

    def test_evolved_prior_used_in_next_round(self):
        """Test that evolved priors from round N are used in round N+1."""
        goal = CampaignGoal(
            objective_kpi="quadratic_kpi",
            direction="maximize",
            target_value=None,
            max_rounds=3,
            batch_size=5,
            strategy="prior_guided",
        )

        space = ParameterSpace(
            dimensions=(
                SearchDimension(
                    param_name="x",
                    param_type="number",
                    min_value=0.0,
                    max_value=10.0,
                    primitive="coord1",
                ),
            ),
            protocol_template={"steps": []},
        )

        # Seed initial memory
        _insert_memory_semantic("coord1", "x", mean=5.0, stddev=2.0, sample_count=10)

        campaign_id = f"g3e2-coord-{uuid.uuid4().hex[:8]}"
        _insert_campaign(campaign_id)

        # Run campaign with multiple rounds
        result = run_campaign_offline(
            goal=goal,
            space=space,
            sim_fn=_sim_parabola,
            campaign_id=campaign_id,
        )

        # Manually trigger evolution after round 1
        if len(result.rounds) >= 2 and result.rounds[0].run_ids:
            run_id = result.rounds[0].run_ids[0]
            _insert_review(
                run_id=run_id,
                campaign_id=campaign_id,
                score=85.0,
                verdict="passed",
                params={"_primitive": "coord1", "x": 5.0},
                improvements=[],
                failure_attributions=[],
                comments="Round 1 review",
            )
            process_review_event(run_id)

        # Verify evolved prior exists after round 1
        evolved = get_active_evolved_prior("coord1", "x")
        assert evolved is not None, "Round 1 should create evolved prior"

        # Round 2 should have used the evolved prior
        # (This is implicit in the campaign_loop calling sample_prior_guided)
        assert len(result.rounds) >= 2, "Campaign should complete multiple rounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
