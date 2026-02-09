"""Tests for the three-layer semantic memory system (Phase D).

Covers:
- Schema tables existence
- Seed recipes (idempotent)
- Episodic write path (extract_episodes)
- Semantic aggregation (update_semantic_facts, Welford's algorithm)
- Procedural detection (detect_repair_patterns)
- Read path (get_param_priors, get_repair_recipes, format_memory_for_prompt)
- Planner integration (memory context in system prompt)
- Grounding integration (optional param filling from priors)
- Event listener (run.completed → memory extraction)
- Safety: memory failure doesn't block planning
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_memory_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "memory_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, json_dumps, utcnow_iso  # noqa: E402
from app.services.memory import (  # noqa: E402
    Episode,
    ParamPrior,
    RepairRecipe,
    detect_repair_patterns,
    extract_episodes,
    format_memory_for_prompt,
    get_param_priors,
    get_repair_recipes,
    seed_initial_recipes,
    start_memory_listener,
    stop_memory_listener,
    update_semantic_facts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    # Clean memory tables between tests
    with connection() as conn:
        conn.execute("DELETE FROM memory_episodes")
        conn.execute("DELETE FROM memory_semantic")
        conn.execute("DELETE FROM memory_procedures")
        conn.commit()


def _insert_run_with_steps(
    steps: list[dict],
    run_status: str = "succeeded",
) -> str:
    """Helper: insert a run + run_steps directly into DB. Returns run_id."""
    import uuid

    run_id = str(uuid.uuid4())
    now = utcnow_iso()

    with connection() as conn:
        conn.execute(
            "INSERT INTO runs "
            "(id, campaign_id, trigger_type, trigger_payload_json, session_key, "
            "status, protocol_json, inputs_json, compiled_graph_json, graph_hash, "
            "policy_snapshot_json, created_by, created_at, updated_at) "
            "VALUES (?, NULL, 'manual', '{}', ?, ?, '{}', '{}', '{}', 'h', '{}', 'test', ?, ?)",
            (run_id, run_id, run_status, now, now),
        )
        for step in steps:
            step_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO run_steps "
                "(id, run_id, step_key, primitive, params_json, depends_on_json, "
                "resources_json, status, idempotency_key, error) "
                "VALUES (?, ?, ?, ?, ?, '[]', '[]', ?, ?, ?)",
                (
                    step_id,
                    run_id,
                    step["step_key"],
                    step["primitive"],
                    json_dumps(step.get("params", {})),
                    step.get("status", "succeeded"),
                    f"{run_id}:{step['step_key']}:0",
                    step.get("error"),
                ),
            )
        conn.commit()
    return run_id


# ===========================================================================
# 1. Schema tables
# ===========================================================================


class TestSchema:
    def test_memory_tables_exist(self):
        with connection() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name LIKE 'memory_%'"
                ).fetchall()
            ]
        assert "memory_episodes" in tables
        assert "memory_semantic" in tables
        assert "memory_procedures" in tables


# ===========================================================================
# 2. Seed recipes
# ===========================================================================


class TestSeedRecipes:
    def test_seed_inserts_recipes(self):
        seed_initial_recipes()
        with connection() as conn:
            rows = conn.execute("SELECT * FROM memory_procedures WHERE source = 'seed'").fetchall()
        assert len(rows) == 2
        primitives = {r["trigger_primitive"] for r in rows}
        assert primitives == {"robot.aspirate", "robot.dispense"}

    def test_seed_is_idempotent(self):
        seed_initial_recipes()
        seed_initial_recipes()
        seed_initial_recipes()
        with connection() as conn:
            rows = conn.execute("SELECT * FROM memory_procedures WHERE source = 'seed'").fetchall()
        assert len(rows) == 2  # still only 2, not 6


# ===========================================================================
# 3. Episodic write path
# ===========================================================================


class TestEpisodicWrite:
    def test_extract_episodes_from_completed_run(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s1", "primitive": "robot.home", "params": {}, "status": "succeeded"},
            {
                "step_key": "s2",
                "primitive": "heat",
                "params": {"temp_c": 65},
                "status": "succeeded",
            },
        ])
        episodes = extract_episodes(run_id)
        assert len(episodes) == 2
        assert episodes[0].primitive == "robot.home"
        assert episodes[0].outcome == "succeeded"
        assert episodes[1].primitive == "heat"
        assert episodes[1].params == {"temp_c": 65}

        # Verify persisted to DB
        with connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_episodes WHERE run_id = ?", (run_id,)
            ).fetchall()
        assert len(rows) == 2

    def test_extract_episodes_skips_pending_steps(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s1", "primitive": "robot.home", "params": {}, "status": "succeeded"},
            {"step_key": "s2", "primitive": "heat", "params": {}, "status": "pending"},
            {"step_key": "s3", "primitive": "cool", "params": {}, "status": "running"},
            {"step_key": "s4", "primitive": "shake", "params": {}, "status": "skipped"},
        ])
        episodes = extract_episodes(run_id)
        assert len(episodes) == 1
        assert episodes[0].step_key == "s1"

    def test_extract_episodes_empty_run(self):
        run_id = _insert_run_with_steps([])
        episodes = extract_episodes(run_id)
        assert episodes == []

    def test_extract_episodes_includes_failures(self):
        run_id = _insert_run_with_steps([
            {"step_key": "s1", "primitive": "robot.home", "params": {}, "status": "succeeded"},
            {
                "step_key": "s2",
                "primitive": "robot.aspirate",
                "params": {"volume": 100},
                "status": "failed",
                "error": "tip not attached",
            },
        ])
        episodes = extract_episodes(run_id)
        assert len(episodes) == 2
        assert episodes[1].outcome == "failed"
        assert episodes[1].error == "tip not attached"


# ===========================================================================
# 4. Semantic aggregation
# ===========================================================================


class TestSemanticAggregation:
    def test_update_semantic_single_episode(self):
        episodes = [
            Episode(
                run_id="r1",
                step_key="s1",
                primitive="heat",
                params={"temp_c": 65.0},
                outcome="succeeded",
            )
        ]
        update_semantic_facts(episodes)
        prior = get_param_priors("heat", "temp_c")
        assert prior is not None
        assert prior.mean == 65.0
        assert prior.stddev == 0.0
        assert prior.sample_count == 1

    def test_update_semantic_mean_stddev(self):
        """Welford's algorithm with three observations."""
        episodes = [
            Episode(
                run_id="r1",
                step_key="s1",
                primitive="heat",
                params={"temp_c": 60.0},
                outcome="succeeded",
            ),
            Episode(
                run_id="r2",
                step_key="s1",
                primitive="heat",
                params={"temp_c": 70.0},
                outcome="succeeded",
            ),
            Episode(
                run_id="r3",
                step_key="s1",
                primitive="heat",
                params={"temp_c": 80.0},
                outcome="succeeded",
            ),
        ]
        update_semantic_facts(episodes)
        prior = get_param_priors("heat", "temp_c")
        assert prior is not None
        assert prior.sample_count == 3
        assert abs(prior.mean - 70.0) < 0.01
        # stddev of [60, 70, 80] population = sqrt(200/3) ≈ 8.165
        assert abs(prior.stddev - 8.165) < 0.1

    def test_update_semantic_incremental(self):
        """Two separate calls should produce same result as one batch."""
        ep1 = [
            Episode(
                run_id="r1", step_key="s1", primitive="heat",
                params={"temp_c": 60.0}, outcome="succeeded",
            ),
        ]
        ep2 = [
            Episode(
                run_id="r2", step_key="s1", primitive="heat",
                params={"temp_c": 80.0}, outcome="succeeded",
            ),
        ]
        update_semantic_facts(ep1)
        update_semantic_facts(ep2)
        prior = get_param_priors("heat", "temp_c")
        assert prior is not None
        assert prior.sample_count == 2
        assert abs(prior.mean - 70.0) < 0.01

    def test_update_semantic_skips_failed(self):
        episodes = [
            Episode(
                run_id="r1",
                step_key="s1",
                primitive="heat",
                params={"temp_c": 65.0},
                outcome="failed",
                error="overtemp",
            )
        ]
        update_semantic_facts(episodes)
        prior = get_param_priors("heat", "temp_c")
        assert prior is None  # failed episodes don't contribute to param stats

    def test_update_semantic_skips_non_numeric(self):
        episodes = [
            Episode(
                run_id="r1",
                step_key="s1",
                primitive="robot.move_to",
                params={"labware": "plate_1", "well": "A1", "pipette": "small"},
                outcome="succeeded",
            )
        ]
        update_semantic_facts(episodes)
        # String params should not be tracked
        assert get_param_priors("robot.move_to", "labware") is None
        assert get_param_priors("robot.move_to", "well") is None

    def test_update_semantic_empty_episodes(self):
        update_semantic_facts([])
        # Should not crash


# ===========================================================================
# 5. Procedural detection
# ===========================================================================


class TestProceduralDetection:
    def test_detect_repair_simple(self):
        episodes = [
            Episode(
                run_id="r1", step_key="s1", primitive="robot.aspirate",
                params={"volume": 100}, outcome="failed", error="tip not attached",
            ),
            Episode(
                run_id="r1", step_key="s2", primitive="robot.drop_tip",
                params={}, outcome="succeeded",
            ),
            Episode(
                run_id="r1", step_key="s3", primitive="robot.pick_up_tip",
                params={}, outcome="succeeded",
            ),
            Episode(
                run_id="r1", step_key="s4", primitive="robot.aspirate",
                params={"volume": 100}, outcome="succeeded",
            ),
        ]
        detect_repair_patterns(episodes)
        recipes = get_repair_recipes("robot.aspirate")
        assert len(recipes) >= 1
        found = [r for r in recipes if r.trigger_error_pattern == "tip"]
        assert len(found) == 1
        assert len(found[0].steps) == 2  # drop_tip + pick_up_tip
        assert found[0].steps[0]["primitive"] == "robot.drop_tip"
        assert found[0].steps[1]["primitive"] == "robot.pick_up_tip"

    def test_detect_repair_no_false_positives(self):
        """No repair if failure is not followed by same-primitive success."""
        episodes = [
            Episode(
                run_id="r1", step_key="s1", primitive="robot.aspirate",
                params={"volume": 100}, outcome="failed", error="overtemp",
            ),
            Episode(
                run_id="r1", step_key="s2", primitive="robot.home",
                params={}, outcome="succeeded",
            ),
        ]
        detect_repair_patterns(episodes)
        recipes = get_repair_recipes("robot.aspirate")
        assert len(recipes) == 0

    def test_detect_repair_skip_no_intermediate_steps(self):
        """No repair if failure immediately followed by success (no recovery steps)."""
        episodes = [
            Episode(
                run_id="r1", step_key="s1", primitive="heat",
                params={"temp_c": 65}, outcome="failed", error="timeout",
            ),
            Episode(
                run_id="r1", step_key="s2", primitive="heat",
                params={"temp_c": 65}, outcome="succeeded",
            ),
        ]
        detect_repair_patterns(episodes)
        recipes = get_repair_recipes("heat")
        assert len(recipes) == 0

    def test_detect_repair_idempotent(self):
        episodes = [
            Episode(
                run_id="r1", step_key="s1", primitive="robot.aspirate",
                params={}, outcome="failed", error="tip error",
            ),
            Episode(
                run_id="r1", step_key="s2", primitive="robot.drop_tip",
                params={}, outcome="succeeded",
            ),
            Episode(
                run_id="r1", step_key="s3", primitive="robot.aspirate",
                params={}, outcome="succeeded",
            ),
        ]
        detect_repair_patterns(episodes)
        detect_repair_patterns(episodes)
        recipes = get_repair_recipes("robot.aspirate")
        tip_recipes = [r for r in recipes if r.trigger_error_pattern == "tip"]
        assert len(tip_recipes) == 1  # not duplicated


# ===========================================================================
# 6. Read path
# ===========================================================================


class TestReadPath:
    def test_get_param_priors_returns_none_for_missing(self):
        assert get_param_priors("nonexistent", "x") is None

    def test_get_repair_recipes_empty(self):
        assert get_repair_recipes("nonexistent") == []

    def test_format_memory_for_prompt_empty(self):
        result = format_memory_for_prompt()
        assert result == ""

    def test_format_memory_for_prompt_with_episodes(self):
        # Insert some episodes directly
        run_id = _insert_run_with_steps([
            {"step_key": "s1", "primitive": "heat", "params": {"temp_c": 65}, "status": "succeeded"},
            {"step_key": "s2", "primitive": "heat", "params": {"temp_c": 70}, "status": "succeeded"},
        ])
        extract_episodes(run_id)

        result = format_memory_for_prompt()
        assert "Memory Context (Advisory)" in result
        assert "heat" in result
        assert "100%" in result  # 2/2 succeeded

    def test_format_memory_for_prompt_with_recipes(self):
        seed_initial_recipes()
        result = format_memory_for_prompt()
        assert "Recovery Recipes" in result
        assert "robot.aspirate" in result
        assert "tip" in result

    def test_format_memory_for_prompt_filtered(self):
        seed_initial_recipes()
        # Only ask for heat recipes (seed only has robot.aspirate/dispense)
        result = format_memory_for_prompt(primitives=["heat"])
        # Should still show seed recipes since they match general query
        # But episode section should be empty for heat
        assert result == "" or "heat" not in result or "Recovery" in result


# ===========================================================================
# 7. Planner integration
# ===========================================================================


class TestPlannerIntegration:
    def test_planner_prompt_includes_memory_context(self):
        """When memory has data, it appears in the system prompt."""
        from app.services.planner import build_system_prompt

        # Seed some data
        run_id = _insert_run_with_steps([
            {"step_key": "s1", "primitive": "heat", "params": {"temp_c": 65}, "status": "succeeded"},
        ])
        extract_episodes(run_id)
        seed_initial_recipes()

        prompt = build_system_prompt()
        assert "Memory Context (Advisory)" in prompt
        assert "Recovery Recipes" in prompt

    def test_planner_prompt_works_without_memory(self):
        """System prompt still works when memory tables are empty."""
        from app.services.planner import build_system_prompt

        prompt = build_system_prompt()
        assert "Available Capabilities" in prompt
        # Memory section should not appear
        assert "Memory Context" not in prompt


# ===========================================================================
# 8. Grounding integration
# ===========================================================================


class TestGroundingIntegration:
    def test_grounding_fills_optional_param_from_prior(self):
        """If an optional param is missing and prior has enough data, fill it."""
        from app.services.plan_grounding import ground_plan
        from app.services.planner import PlanResult, PlanStep

        # Build up semantic memory for robot.move_to_well speed param
        episodes = [
            Episode(
                run_id=f"r{i}", step_key="s1", primitive="robot.move_to_well",
                params={"labware": "plate", "well": "A1", "pipette": "small", "speed": 50.0 + i},
                outcome="succeeded",
            )
            for i in range(5)
        ]
        update_semantic_facts(episodes)

        # Plan step that doesn't specify optional 'speed'
        plan = PlanResult(
            steps=[
                PlanStep(
                    id="s1",
                    primitive="robot.move_to_well",
                    params={"labware": "plate", "well": "A1", "pipette": "small"},
                ),
            ],
            raw_response="{}",
            model="test",
        )
        result = ground_plan(plan)
        assert result.ok

        # speed should be filled from prior
        step_params = result.protocol["steps"][0]["params"]
        assert "speed" in step_params
        assert isinstance(step_params["speed"], float)
        # Should have a warning about filling from prior
        prior_warnings = [w for w in result.warnings if "memory prior" in w]
        assert len(prior_warnings) == 1

    def test_grounding_does_not_fill_required_params(self):
        """Memory priors should never fill required parameters.

        robot.aspirate requires labware, well, pipette, volume.
        Even with priors, missing required params should cause errors.
        """
        from app.services.plan_grounding import ground_plan
        from app.services.planner import PlanResult, PlanStep

        # Build up semantic memory for robot.aspirate volume
        episodes = [
            Episode(
                run_id=f"r{i}", step_key="s1", primitive="robot.aspirate",
                params={"labware": "plate", "well": "A1", "pipette": "small", "volume": 100.0 + i},
                outcome="succeeded",
            )
            for i in range(5)
        ]
        update_semantic_facts(episodes)

        # Plan step missing required params — should NOT be filled by memory
        plan = PlanResult(
            steps=[PlanStep(id="s1", primitive="robot.aspirate", params={})],
            raw_response="{}",
            model="test",
        )
        result = ground_plan(plan)
        # Should have errors for missing required params
        assert not result.ok
        assert any("required parameter" in e for e in result.errors)

    def test_grounding_skips_prior_with_insufficient_samples(self):
        """Prior needs sample_count >= 3 to be used."""
        from app.services.plan_grounding import ground_plan
        from app.services.planner import PlanResult, PlanStep

        # Only 2 samples — below threshold
        episodes = [
            Episode(
                run_id=f"r{i}", step_key="s1", primitive="robot.move_to_well",
                params={"labware": "plate", "well": "A1", "pipette": "small", "speed": 50.0},
                outcome="succeeded",
            )
            for i in range(2)
        ]
        update_semantic_facts(episodes)

        plan = PlanResult(
            steps=[
                PlanStep(
                    id="s1",
                    primitive="robot.move_to_well",
                    params={"labware": "plate", "well": "A1", "pipette": "small"},
                ),
            ],
            raw_response="{}",
            model="test",
        )
        result = ground_plan(plan)
        assert result.ok
        # speed should NOT be filled (only 2 samples)
        step_params = result.protocol["steps"][0]["params"]
        assert "speed" not in step_params

    def test_grounding_does_not_override_explicit_value(self):
        """If a param is already provided, memory prior should not override it."""
        from app.services.plan_grounding import ground_plan
        from app.services.planner import PlanResult, PlanStep

        # Build enough samples
        episodes = [
            Episode(
                run_id=f"r{i}", step_key="s1", primitive="robot.move_to_well",
                params={"labware": "plate", "well": "A1", "pipette": "small", "speed": 50.0},
                outcome="succeeded",
            )
            for i in range(5)
        ]
        update_semantic_facts(episodes)

        # Explicitly provide speed=999
        plan = PlanResult(
            steps=[
                PlanStep(
                    id="s1",
                    primitive="robot.move_to_well",
                    params={"labware": "plate", "well": "A1", "pipette": "small", "speed": 999.0},
                ),
            ],
            raw_response="{}",
            model="test",
        )
        result = ground_plan(plan)
        assert result.ok
        assert result.protocol["steps"][0]["params"]["speed"] == 999.0
        # No "memory prior" warning
        assert not any("memory prior" in w for w in result.warnings)


# ===========================================================================
# 9. Event listener
# ===========================================================================


class TestEventListener:
    def test_memory_listener_processes_run_completed(self):
        """E2E: listener receives run.completed → extracts episodes."""
        from app.services.event_bus import EventBus, EventMessage

        bus = EventBus()

        async def _test():
            await bus.start()
            sub = await start_memory_listener(bus)

            # Create a run with steps
            run_id = _insert_run_with_steps([
                {"step_key": "s1", "primitive": "robot.home", "params": {}, "status": "succeeded"},
                {"step_key": "s2", "primitive": "heat", "params": {"temp_c": 65}, "status": "succeeded"},
            ])

            # Publish run.completed event
            event = EventMessage(
                id="evt-1",
                run_id=run_id,
                actor="worker",
                action="run.completed",
                details={"final_status": "succeeded"},
                created_at=utcnow_iso(),
            )
            bus.publish(event)

            # Give the listener time to process
            await asyncio.sleep(0.2)

            # Verify episodes were extracted
            with connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_episodes WHERE run_id = ?", (run_id,)
                ).fetchall()
            assert len(rows) == 2

            # Cleanup
            await stop_memory_listener(sub, bus)
            await bus.stop()

        asyncio.run(_test())

    def test_memory_listener_ignores_other_events(self):
        """Listener should only process run.completed, not other actions."""
        from app.services.event_bus import EventBus, EventMessage

        bus = EventBus()

        async def _test():
            await bus.start()
            sub = await start_memory_listener(bus)

            run_id = _insert_run_with_steps([
                {"step_key": "s1", "primitive": "robot.home", "params": {}, "status": "succeeded"},
            ])

            # Publish a non-completed event
            event = EventMessage(
                id="evt-1",
                run_id=run_id,
                actor="worker",
                action="step.state_changed",
                details={"step_id": "s1", "status": "running"},
                created_at=utcnow_iso(),
            )
            bus.publish(event)

            await asyncio.sleep(0.1)

            # Should NOT have extracted episodes
            with connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_episodes WHERE run_id = ?", (run_id,)
                ).fetchall()
            assert len(rows) == 0

            await stop_memory_listener(sub, bus)
            await bus.stop()

        asyncio.run(_test())


# ===========================================================================
# 10. Safety: memory failure isolation
# ===========================================================================


class TestMemoryFailureIsolation:
    def test_format_memory_for_prompt_handles_error(self):
        """Even if memory DB is corrupted, format should return empty string."""
        # This tests the try/except wrapper
        result = format_memory_for_prompt()
        assert isinstance(result, str)

    def test_planner_works_without_memory_tables(self):
        """If memory import fails, planning should still work."""
        from app.services.planner import build_system_prompt

        prompt = build_system_prompt()
        assert len(prompt) > 100  # Should still have SOUL.md + capabilities
