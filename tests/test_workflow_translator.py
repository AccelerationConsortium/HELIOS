"""Tests for the battery-lab workflow → OTbot protocol translator."""
from __future__ import annotations

import pytest

from app.services.workflow_translator import (
    TranslationError,
    translate_battery_workflow,
)
from app.services.compiler import compile_protocol
from app.services.safety import BATTERY_LAB_PRIMITIVES


# ---------------------------------------------------------------------------
# Fixtures: minimal workflow fragments
# ---------------------------------------------------------------------------

def _setup_phase() -> dict:
    """A simple sequential setup phase."""
    return {
        "phase_name": "setup",
        "steps": [
            {
                "step_id": "setup_001",
                "action": "robot.home",
                "params": {},
            },
            {
                "step_id": "setup_002",
                "action": "robot.load_pipettes",
                "params": {"small_mount": "left", "large_mount": "right"},
            },
            {
                "step_id": "setup_003",
                "action": "robot.set_lights",
                "params": {"on": True},
            },
        ],
    }


def _parallel_phase() -> dict:
    """A phase with two parallel threads."""
    return {
        "phase_name": "preparation",
        "parallel_threads": [
            {
                "thread_name": "fill_wash",
                "steps": [
                    {
                        "step_id": "prep_w_001",
                        "action": "plc.dispense_ml",
                        "params": {"pump": 1, "volume_ml": 100.0},
                    },
                    {
                        "step_id": "prep_w_002",
                        "action": "wait",
                        "params": {"duration_seconds": 5},
                    },
                ],
            },
            {
                "thread_name": "electrode_prep",
                "steps": [
                    {
                        "step_id": "prep_e_001",
                        "action": "robot.pick_up_tip",
                        "params": {
                            "labware": "tip_rack",
                            "well": "A1",
                            "pipette": "p1000_single_gen2",
                        },
                    },
                    {
                        "step_id": "prep_e_002",
                        "action": "robot.drop_tip",
                        "params": {"pipette": "p1000_single_gen2", "drop_in_trash": True},
                    },
                ],
            },
        ],
    }


def _post_phase() -> dict:
    """A simple post-processing phase after parallel."""
    return {
        "phase_name": "post",
        "steps": [
            {
                "step_id": "post_001",
                "action": "log",
                "params": {"message": "all done"},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSequentialPhase:
    def test_translates_to_flat_steps(self):
        wf = {"phases": [_setup_phase()]}
        result = translate_battery_workflow(wf)

        assert "steps" in result
        assert len(result["steps"]) == 3

    def test_first_step_has_no_deps(self):
        wf = {"phases": [_setup_phase()]}
        result = translate_battery_workflow(wf)

        assert result["steps"][0]["depends_on"] == []

    def test_chain_dependencies(self):
        """Each step depends on the previous one."""
        wf = {"phases": [_setup_phase()]}
        result = translate_battery_workflow(wf)

        assert result["steps"][1]["depends_on"] == ["setup_001"]
        assert result["steps"][2]["depends_on"] == ["setup_002"]

    def test_preserves_action_as_primitive(self):
        wf = {"phases": [_setup_phase()]}
        result = translate_battery_workflow(wf)

        assert result["steps"][0]["primitive"] == "robot.home"
        assert result["steps"][1]["primitive"] == "robot.load_pipettes"

    def test_preserves_params(self):
        wf = {"phases": [_setup_phase()]}
        result = translate_battery_workflow(wf)

        assert result["steps"][1]["params"]["small_mount"] == "left"

    def test_resources_auto_mapped(self):
        wf = {"phases": [_setup_phase()]}
        result = translate_battery_workflow(wf)

        # robot.* → ot2-robot
        assert "ot2-robot" in result["steps"][0]["resources"]


class TestParallelPhase:
    def test_parallel_threads_produce_steps(self):
        wf = {"phases": [_parallel_phase()]}
        result = translate_battery_workflow(wf)

        # 2 steps in thread 1 + 2 steps in thread 2 = 4
        assert len(result["steps"]) == 4

    def test_first_steps_of_each_thread_have_no_deps(self):
        """When the parallel phase is the first phase, thread heads have no deps."""
        wf = {"phases": [_parallel_phase()]}
        result = translate_battery_workflow(wf)

        head_keys = {s["step_key"] for s in result["steps"] if s["depends_on"] == []}
        assert "prep_w_001" in head_keys
        assert "prep_e_001" in head_keys

    def test_intra_thread_chaining(self):
        """Steps within the same thread are chained sequentially."""
        wf = {"phases": [_parallel_phase()]}
        result = translate_battery_workflow(wf)

        by_key = {s["step_key"]: s for s in result["steps"]}
        assert by_key["prep_w_002"]["depends_on"] == ["prep_w_001"]
        assert by_key["prep_e_002"]["depends_on"] == ["prep_e_001"]

    def test_cross_thread_no_deps(self):
        """Steps in different threads should NOT depend on each other."""
        wf = {"phases": [_parallel_phase()]}
        result = translate_battery_workflow(wf)

        by_key = {s["step_key"]: s for s in result["steps"]}
        # prep_e_001 should NOT depend on any prep_w step
        assert "prep_w_001" not in by_key["prep_e_001"]["depends_on"]
        assert "prep_w_002" not in by_key["prep_e_001"]["depends_on"]


class TestCrossPhase:
    def test_sequential_then_sequential(self):
        """Second phase's first step depends on first phase's last step."""
        wf = {"phases": [_setup_phase(), _post_phase()]}
        result = translate_battery_workflow(wf)

        post_step = [s for s in result["steps"] if s["step_key"] == "post_001"][0]
        assert post_step["depends_on"] == ["setup_003"]

    def test_sequential_then_parallel(self):
        """All thread heads depend on the previous phase's tail."""
        wf = {"phases": [_setup_phase(), _parallel_phase()]}
        result = translate_battery_workflow(wf)

        by_key = {s["step_key"]: s for s in result["steps"]}
        # Both thread heads should depend on setup_003
        assert by_key["prep_w_001"]["depends_on"] == ["setup_003"]
        assert by_key["prep_e_001"]["depends_on"] == ["setup_003"]

    def test_parallel_then_sequential_join_barrier(self):
        """Next phase's first step depends on ALL thread tails (join)."""
        wf = {"phases": [_parallel_phase(), _post_phase()]}
        result = translate_battery_workflow(wf)

        post_step = [s for s in result["steps"] if s["step_key"] == "post_001"][0]
        # Should depend on both thread tails
        assert "prep_w_002" in post_step["depends_on"]
        assert "prep_e_002" in post_step["depends_on"]

    def test_three_phases_full_chain(self):
        """setup → parallel → post should produce correct dependency graph."""
        wf = {"phases": [_setup_phase(), _parallel_phase(), _post_phase()]}
        result = translate_battery_workflow(wf)

        assert len(result["steps"]) == 3 + 4 + 1  # 8 steps total
        by_key = {s["step_key"]: s for s in result["steps"]}

        # setup chain
        assert by_key["setup_001"]["depends_on"] == []
        assert by_key["setup_002"]["depends_on"] == ["setup_001"]
        assert by_key["setup_003"]["depends_on"] == ["setup_002"]

        # parallel threads fork from setup_003
        assert by_key["prep_w_001"]["depends_on"] == ["setup_003"]
        assert by_key["prep_e_001"]["depends_on"] == ["setup_003"]

        # post joins both thread tails
        assert set(by_key["post_001"]["depends_on"]) == {"prep_w_002", "prep_e_002"}


class TestCompilerCompatibility:
    def test_translated_protocol_compiles(self):
        """Translated result can be fed to compile_protocol without error."""
        wf = {"phases": [_setup_phase(), _parallel_phase(), _post_phase()]}
        protocol = translate_battery_workflow(wf)
        policy = {
            "allowed_primitives": list(BATTERY_LAB_PRIMITIVES),
            "max_temp_c": 95.0,
            "max_volume_ul": 1000.0,
        }

        compiled, graph_hash = compile_protocol(
            protocol=protocol, inputs={}, policy_snapshot=policy,
        )

        assert "steps" in compiled
        assert len(compiled["steps"]) == 8
        assert isinstance(graph_hash, str)
        assert len(graph_hash) == 64  # SHA-256 hex digest

    def test_graph_hash_deterministic(self):
        """Same workflow always produces the same graph_hash."""
        wf = {"phases": [_setup_phase()]}
        protocol = translate_battery_workflow(wf)
        policy = {"allowed_primitives": list(BATTERY_LAB_PRIMITIVES)}

        _, h1 = compile_protocol(protocol=protocol, inputs={}, policy_snapshot=policy)
        _, h2 = compile_protocol(protocol=protocol, inputs={}, policy_snapshot=policy)
        assert h1 == h2


class TestErrorHandling:
    def test_empty_phases_raises(self):
        with pytest.raises(TranslationError, match="non-empty"):
            translate_battery_workflow({"phases": []})

    def test_missing_phases_raises(self):
        with pytest.raises(TranslationError, match="non-empty"):
            translate_battery_workflow({})

    def test_phase_without_steps_or_threads_raises(self):
        with pytest.raises(TranslationError, match="neither"):
            translate_battery_workflow({"phases": [{"phase_name": "bad"}]})

    def test_step_missing_action_raises(self):
        wf = {
            "phases": [
                {
                    "phase_name": "bad",
                    "steps": [{"step_id": "s1", "params": {}}],
                }
            ]
        }
        with pytest.raises(TranslationError, match="missing 'action'"):
            translate_battery_workflow(wf)

    def test_auto_generates_step_key_when_missing(self):
        """Steps without step_id get auto-generated keys."""
        wf = {
            "phases": [
                {
                    "phase_name": "auto",
                    "steps": [{"action": "wait", "params": {"duration_seconds": 1}}],
                }
            ]
        }
        result = translate_battery_workflow(wf)
        assert result["steps"][0]["step_key"] == "auto_000"


class TestResourceMapping:
    def test_robot_primitives(self):
        wf = {"phases": [{"phase_name": "r", "steps": [
            {"step_id": "r1", "action": "robot.home", "params": {}}
        ]}]}
        result = translate_battery_workflow(wf)
        assert result["steps"][0]["resources"] == ["ot2-robot"]

    def test_plc_primitives(self):
        wf = {"phases": [{"phase_name": "p", "steps": [
            {"step_id": "p1", "action": "plc.dispense_ml", "params": {"pump": 1, "volume_ml": 10}}
        ]}]}
        result = translate_battery_workflow(wf)
        assert result["steps"][0]["resources"] == ["plc-controller"]

    def test_relay_primitives(self):
        wf = {"phases": [{"phase_name": "rl", "steps": [
            {"step_id": "rl1", "action": "relay.switch_to", "params": {"channel": 0}}
        ]}]}
        result = translate_battery_workflow(wf)
        assert result["steps"][0]["resources"] == ["relay-controller"]

    def test_wait_falls_back(self):
        wf = {"phases": [{"phase_name": "w", "steps": [
            {"step_id": "w1", "action": "wait", "params": {"duration_seconds": 1}}
        ]}]}
        result = translate_battery_workflow(wf)
        assert result["steps"][0]["resources"] == ["lab-controller"]
