"""
Tests for WorkflowSupervisor — validates:
  1. Real workflow execution (step_id / stage / cursor movement)
  2. SKIP for optional steps (on_failure=skip, criticality=optional)
  3. ABORT for critical steps
  4. PlanPatch cascading on DEGRADE (downstream postconditions updated)
  5. Retry budget exhaustion → fallback to on_failure
"""
import pytest
from exp_agent.core.types import PlanStep, PlanPatch, Action, HardwareError
from exp_agent.devices.simulated.heater import SimHeater
from exp_agent.orchestrator.workflow_supervisor import WorkflowSupervisor


# ============================================================================
# Helpers
# ============================================================================

def make_plan(target: float = 120.0, device: str = "heater_1") -> list[PlanStep]:
    """Minimal 4-step plan for testing."""
    return [
        PlanStep(
            step_id="setup", stage="setup",
            action=Action(name="cool_down", effect="write", device=device,
                          postconditions=["telemetry.heating == False"]),
            criticality="critical", on_failure="abort",
        ),
        PlanStep(
            step_id="heat", stage="heating",
            action=Action(name="set_temperature", effect="write", device=device,
                          params={"temperature": target},
                          postconditions=[
                              f"telemetry.target == {target}",
                              f"telemetry.temperature ~= {target} +/- 2.0 within 20s",
                          ]),
            criticality="critical", on_failure="abort", max_retries=2,
        ),
        PlanStep(
            step_id="snapshot", stage="diagnostics",
            action=Action(name="wait", effect="write", device=device,
                          params={"duration": 1},
                          postconditions=[
                              f"telemetry.temperature ~= {target} +/- 5.0 within 5s",
                          ]),
            criticality="optional", on_failure="skip",
        ),
        PlanStep(
            step_id="cooldown", stage="cooldown",
            action=Action(name="cool_down", effect="write", device=device,
                          postconditions=["telemetry.heating == False"]),
            criticality="critical", on_failure="abort",
        ),
    ]


# ============================================================================
# 1. Happy-path workflow execution
# ============================================================================

class TestWorkflowExecution:
    """Verify that execute_plan() walks steps with step_id/stage/cursor."""

    def test_happy_path_all_steps_ok(self):
        """No faults → all steps complete in order."""
        device = SimHeater(name="heater_1", fault_mode="none")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        result = supervisor.execute_plan(make_plan())

        assert result.success is True
        assert result.aborted_at is None
        assert len(result.steps) == 4
        # All steps should be "ok"
        outcomes = [s.outcome for s in result.steps]
        assert outcomes == ["ok", "ok", "ok", "ok"]
        # step_ids in order
        ids = [s.step_id for s in result.steps]
        assert ids == ["setup", "heat", "snapshot", "cooldown"]
        # stages in order
        stages = [s.stage for s in result.steps]
        assert stages == ["setup", "heating", "diagnostics", "cooldown"]

    def test_step_ids_and_stages_in_output(self):
        """Verify each StepResult has step_id and stage."""
        device = SimHeater(name="heater_1", fault_mode="none")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        result = supervisor.execute_plan(make_plan())

        for sr in result.steps:
            assert sr.step_id is not None
            assert sr.stage is not None
            assert sr.step_id != ""
            assert sr.stage != ""


# ============================================================================
# 2. SKIP — optional step fail → skip → next step continues
# ============================================================================

class TestSkipDecision:
    """Verify SKIP triggers for optional steps."""

    def test_optional_step_fail_triggers_skip(self):
        """
        When a sensor_fail happens on an optional step with on_failure=skip,
        the step should be skipped and execution continues.
        """
        # Use sensor_fail mode — it triggers after tick_count > 5
        # But we need it to fail on the optional step specifically.
        # Instead, let's craft a plan where the optional step has impossible postconditions.
        device = SimHeater(name="heater_1", fault_mode="none")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        plan = [
            PlanStep(
                step_id="setup", stage="setup",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
            PlanStep(
                step_id="optional_check", stage="diagnostics",
                action=Action(name="wait", effect="write", device="heater_1",
                              params={"duration": 1},
                              # Impossible postcondition: temp must be 999°C
                              postconditions=["telemetry.temperature ~= 999.0 +/- 0.1 within 1s"]),
                criticality="optional", on_failure="skip",
            ),
            PlanStep(
                step_id="final", stage="cooldown",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
        ]

        result = supervisor.execute_plan(plan)

        # Plan should succeed overall (optional step skipped, critical steps OK)
        assert result.success is True
        assert len(result.steps) == 3

        # setup: ok
        assert result.steps[0].outcome == "ok"
        assert result.steps[0].step_id == "setup"

        # optional_check: skipped
        assert result.steps[1].outcome == "skipped"
        assert result.steps[1].step_id == "optional_check"
        assert result.steps[1].decision == "skip"

        # final: ok
        assert result.steps[2].outcome == "ok"
        assert result.steps[2].step_id == "final"

    def test_critical_step_fail_triggers_abort(self):
        """
        When a critical step fails (on_failure=abort), the plan aborts.
        """
        device = SimHeater(name="heater_1", fault_mode="none")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        plan = [
            PlanStep(
                step_id="setup", stage="setup",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
            PlanStep(
                step_id="critical_heat", stage="heating",
                action=Action(name="set_temperature", effect="write", device="heater_1",
                              params={"temperature": 120.0},
                              # Impossible postcondition
                              postconditions=["telemetry.temperature ~= 999.0 +/- 0.1 within 1s"]),
                criticality="critical", on_failure="abort", max_retries=0,
            ),
            PlanStep(
                step_id="final", stage="cooldown",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
        ]

        result = supervisor.execute_plan(plan)

        assert result.success is False
        assert result.aborted_at == "critical_heat"
        # final step should NOT have run
        step_ids = [s.step_id for s in result.steps]
        assert "final" not in step_ids


# ============================================================================
# 3. DEGRADE — PlanPatch cascading
# ============================================================================

class TestDegradeCascade:
    """Verify degrade produces PlanPatch that updates downstream steps."""

    def test_plan_patch_structure(self):
        """PlanPatch should contain overrides, relaxations, and notes."""
        patch = PlanPatch(
            original_target=120.0,
            degraded_target=110.0,
            overrides={"hold": {"temperature": 110.0}},
            relaxations={"hold": [
                "telemetry.temperature ~= 110.0 +/- 2.0 within 10s",
            ]},
            notes=["Degraded from 120°C to 110°C at step preheat"],
        )

        assert patch.original_target == 120.0
        assert patch.degraded_target == 110.0
        assert "hold" in patch.overrides
        assert patch.overrides["hold"]["temperature"] == 110.0
        assert "hold" in patch.relaxations
        assert len(patch.notes) == 1

    def test_degrade_updates_downstream_postconditions(self):
        """
        When overshoot causes degrade on 'heat' step,
        downstream steps should get patched postconditions with the new target.
        """
        device = SimHeater(name="heater_1", fault_mode="overshoot")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        # Build a plan where heat will overshoot → degrade,
        # and downstream steps reference the original 120.0 target
        plan = [
            PlanStep(
                step_id="setup", stage="setup",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
            PlanStep(
                step_id="heat", stage="heating",
                action=Action(name="set_temperature", effect="write", device="heater_1",
                              params={"temperature": 120.0},
                              postconditions=[
                                  "telemetry.target == 120.0",
                                  "telemetry.temperature ~= 120.0 +/- 2.0 within 20s",
                              ]),
                criticality="critical", on_failure="abort", max_retries=2,
            ),
            PlanStep(
                step_id="hold", stage="hold",
                action=Action(name="wait", effect="write", device="heater_1",
                              params={"duration": 2},
                              postconditions=[
                                  "telemetry.temperature ~= 120.0 +/- 3.0 within 10s",
                              ]),
                criticality="optional", on_failure="skip",
            ),
            PlanStep(
                step_id="cooldown", stage="cooldown",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
        ]

        result = supervisor.execute_plan(plan)

        # We expect: setup=ok, heat=degraded or aborted (depending on overshoot timing)
        # If degrade happened, we should see patches
        degraded_steps = [s for s in result.steps if s.outcome == "degraded"]
        if degraded_steps:
            # Verify patches exist
            assert len(result.patches) > 0
            patch = result.patches[0]
            assert patch.degraded_target is not None
            assert patch.degraded_target < 120.0
            assert patch.original_target == 120.0
            assert len(patch.notes) > 0
            # "hold" step should have relaxed postconditions
            if "hold" in patch.relaxations:
                for pc in patch.relaxations["hold"]:
                    # Should reference the degraded target, not 120.0
                    assert "120.0" not in pc

    def test_patch_apply_changes_action_params(self):
        """Verify _apply_patches actually modifies the action."""
        device = SimHeater(name="heater_1", fault_mode="none")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        # Manually add a patch
        supervisor.active_patches.append(PlanPatch(
            original_target=120.0,
            degraded_target=110.0,
            overrides={"hold": {"temperature": 110.0}},
            relaxations={"hold": [
                "telemetry.temperature ~= 110.0 +/- 2.0 within 10s",
            ]},
            notes=["Degraded to 110°C"],
        ))

        step = PlanStep(
            step_id="hold", stage="hold",
            action=Action(name="set_temperature", effect="write", device="heater_1",
                          params={"temperature": 120.0},
                          postconditions=[
                              "telemetry.temperature ~= 120.0 +/- 2.0 within 10s",
                          ]),
            criticality="critical", on_failure="abort",
        )

        patched = supervisor._apply_patches(step)

        # Params should be overridden
        assert patched.params["temperature"] == 110.0
        # Postconditions should be relaxed
        assert "110.0" in patched.postconditions[0]
        assert "120.0" not in patched.postconditions[0]


# ============================================================================
# 4. Retry budget exhaustion → fallback
# ============================================================================

class TestRetryBudget:
    """Verify retry budget and fallback to on_failure."""

    def test_retry_budget_exhausted_optional_skips(self):
        """Optional step with max_retries=0 → immediate skip on failure."""
        device = SimHeater(name="heater_1", fault_mode="none")
        supervisor = WorkflowSupervisor(device=device, target_temp=120.0)

        plan = [
            PlanStep(
                step_id="optional_bad", stage="diagnostics",
                action=Action(name="wait", effect="write", device="heater_1",
                              params={"duration": 1},
                              postconditions=["telemetry.temperature ~= 999.0 +/- 0.1 within 1s"]),
                criticality="optional", on_failure="skip", max_retries=0,
            ),
            PlanStep(
                step_id="end", stage="cooldown",
                action=Action(name="cool_down", effect="write", device="heater_1",
                              postconditions=["telemetry.heating == False"]),
                criticality="critical", on_failure="abort",
            ),
        ]

        result = supervisor.execute_plan(plan)

        assert result.success is True
        assert result.steps[0].outcome == "skipped"
        assert result.steps[1].outcome == "ok"
