"""
Workflow manager for coordinating experiment loops and step execution.

This module manages the overall experiment workflow, including loop control,
step dependencies, and coordination between recovery policies and execution.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..recovery.workflow_policy import (
    WorkflowContext,
    ExperimentLoop,
    WorkflowStep,
    WorkflowPhase,
    StepDependency,
    WorkflowRecoveryPolicy,
)
from ..core.types import Decision, HardwareError, DeviceState, Action
from ..recovery.recovery_agent import RecoveryAgent
from ..executor.guarded_executor import GuardedExecutor


@dataclass
class LoopResult:
    """Result of a single experiment loop."""

    loop_id: int
    status: str  # "completed", "failed", "aborted", "skipped"
    completed_steps: List[str]
    failed_steps: Dict[str, str]
    skipped_steps: List[str]
    duration: float
    reason: Optional[str] = None


class WorkflowManager:
    """Manages experiment workflows and loops."""

    def __init__(
        self,
        workflow_steps: List[WorkflowStep],
        recovery_policy: Optional[WorkflowRecoveryPolicy] = None,
    ):
        self.workflow_steps = workflow_steps
        self.recovery_policy = recovery_policy or WorkflowRecoveryPolicy()
        self.device_executor = GuardedExecutor()
        self.basic_recovery = RecoveryAgent()

        # Loop management
        self.current_loop: Optional[ExperimentLoop] = None
        self.loop_history: List[LoopResult] = []
        self.max_loops = 10  # Maximum experiment loops

        # Context
        self.context = WorkflowContext(
            current_loop=ExperimentLoop(loop_id=0), workflow_steps=workflow_steps
        )

    def run_experiment(self, max_loops: int = 5) -> List[LoopResult]:
        """
        Run the complete experiment with multiple loops.

        Args:
            max_loops: Maximum number of experiment loops to attempt

        Returns:
            List of loop results
        """
        print("=== Starting Experiment Workflow ===")
        print(f"Total steps: {len(self.workflow_steps)}")
        print(f"Max loops: {max_loops}")

        self.max_loops = max_loops
        results = []

        for loop_num in range(max_loops):
            print(f"\n--- Starting Loop {loop_num + 1}/{max_loops} ---")

            # Create new loop
            self.current_loop = ExperimentLoop(loop_id=loop_num + 1)
            self.context.current_loop = self.current_loop
            self.context.step_history = []

            # Run the loop
            result = self._run_single_loop()
            results.append(result)

            print(f"Loop {loop_num + 1} completed with status: {result.status}")

            # Check if we should continue
            if result.status in ["failed", "aborted"]:
                if not self._should_continue_after_failure(result, loop_num + 1):
                    break
            elif result.status == "completed":
                print("Experiment completed successfully!")
                break

        print(f"\n=== Experiment Finished ===")
        print(f"Total loops executed: {len(results)}")
        successful_loops = sum(1 for r in results if r.status == "completed")
        print(f"Successful loops: {successful_loops}")

        return results

    def _run_single_loop(self) -> LoopResult:
        """Run a single experiment loop."""
        start_time = time.time()
        loop = self.current_loop
        if not loop:
            return LoopResult(
                loop_id=0,
                status="failed",
                completed_steps=[],
                failed_steps={},
                skipped_steps=[],
                duration=0.0,
                reason="No active loop",
            )

        try:
            step_index = 0
            while step_index < len(self.workflow_steps):
                step = self.workflow_steps[step_index]
                loop.current_step_index = step_index

                print(
                    f"Executing step {step_index + 1}/{len(self.workflow_steps)}: {step.name}"
                )

                # Check if we can execute this step
                if not loop.can_proceed_to_next_step(step):
                    reason = f"Cannot proceed to step '{step.name}' due to unmet dependencies"
                    print(f"Skipping step: {reason}")
                    loop.skipped_steps.add(step.name)
                    step_index += 1
                    continue

                # Execute the step
                success, error = self._execute_step(step)

                if success:
                    print(f"✓ Step '{step.name}' completed successfully")
                    loop.completed_steps.add(step.name)
                    self.context.step_history.append(f"completed:{step.name}")
                    step_index += 1

                else:
                    # Step failed - make recovery decision
                    if error:
                        print(f"✗ Step '{step.name}' failed: {error.message}")
                        loop.failed_steps[step.name] = error.message

                        decision = self.recovery_policy.make_decision(
                            error,
                            {},
                            self.context,  # TODO: pass actual device states
                        )
                    else:
                        # This shouldn't happen, but handle gracefully
                        print(f"✗ Step '{step.name}' failed: Unknown error")
                        loop.failed_steps[step.name] = "Unknown error"
                        from ..core.types import Decision, DecisionType

                        decision = Decision(
                            kind="abort", rationale="Unknown error occurred", actions=[]
                        )

                    print(f"Recovery decision: {decision.kind} - {decision.rationale}")

                    if decision.kind == "retry":
                        # Retry the same step
                        if self._can_retry_step(step):
                            print("Retrying step...")
                            continue
                        else:
                            print("Max retries exceeded, skipping step")
                            loop.skipped_steps.add(step.name)
                            step_index += 1

                    elif decision.kind == "skip":
                        # Skip this step
                        print("Skipping step and continuing...")
                        loop.skipped_steps.add(step.name)
                        step_index += 1

                    elif decision.kind == "abort":
                        # Abort this loop
                        loop.status = "aborted"
                        break

                    elif decision.kind == "degrade":
                        # Try degraded execution
                        print("Attempting degraded execution...")
                        success, error = self._execute_degraded_step(
                            step, decision.actions
                        )
                        if success:
                            loop.completed_steps.add(f"{step.name}(degraded)")
                            step_index += 1
                        else:
                            loop.skipped_steps.add(step.name)
                            step_index += 1

            # Loop completed
            if loop.status != "aborted":
                if len(loop.completed_steps) >= len(
                    [s for s in self.workflow_steps if not s.can_skip]
                ):
                    loop.status = "completed"
                else:
                    loop.status = "failed"

        except Exception as e:
            loop.status = "failed"
            print(f"Unexpected error in loop: {e}")

        duration = time.time() - start_time

        return LoopResult(
            loop_id=loop.loop_id,
            status=loop.status,
            completed_steps=list(loop.completed_steps),
            failed_steps=loop.failed_steps.copy(),
            skipped_steps=list(loop.skipped_steps),
            duration=duration,
        )

    def _execute_step(self, step: WorkflowStep) -> tuple[bool, Optional[HardwareError]]:
        """Execute a single workflow step."""
        try:
            # Here we would execute the actual step actions
            # For now, simulate execution
            print(f"  Executing actions for step '{step.name}'...")

            # Simulate some steps failing
            import random

            if (
                step.name == "sample_prep_2" and random.random() < 0.6
            ):  # 60% failure rate
                # Simulate sample prep failure
                raise HardwareError(
                    device="heater_1",
                    type="sample_contamination",
                    severity="high",
                    message="Sample preparation failed - contamination detected",
                )
            elif (
                step.name == "measurement" and random.random() < 0.3
            ):  # 30% failure rate
                # Simulate measurement failure
                raise HardwareError(
                    device="sensor_1",
                    type="timeout",
                    severity="medium",
                    message="Measurement timed out",
                )

            # Simulate successful execution
            time.sleep(0.5)  # Simulate execution time
            return True, None

        except HardwareError as e:
            return False, e
        except Exception as e:
            error = HardwareError(
                device="workflow_manager",
                type="execution_error",
                severity="high",
                message=f"Step execution failed: {e}",
            )
            return False, error

    def _execute_degraded_step(
        self, step: WorkflowStep, recovery_actions: List[Action]
    ) -> tuple[bool, Optional[HardwareError]]:
        """Execute a step in degraded mode."""
        try:
            print(f"  Executing degraded step '{step.name}'...")
            # Execute recovery actions first
            for action in recovery_actions:
                print(f"    Recovery action: {action.name}")

            # Then attempt the step with reduced requirements
            time.sleep(0.3)  # Faster degraded execution
            return True, None

        except Exception as e:
            error = HardwareError(
                device="workflow_manager",
                type="degraded_execution_failed",
                severity="medium",
                message=f"Degraded execution failed: {e}",
            )
            return False, error

    def _can_retry_step(self, step: WorkflowStep) -> bool:
        """Check if a step can be retried."""
        # Simple retry limit - could be more sophisticated
        return (
            len([h for h in self.context.step_history if f"retry:{step.name}" in h])
            < step.max_retries
        )

    def _should_continue_after_failure(
        self, result: LoopResult, current_loop_num: int
    ) -> bool:
        """Decide whether to continue after a loop failure."""
        if current_loop_num >= self.max_loops:
            return False

        # Continue if we had some success in previous loops
        successful_loops = sum(1 for r in self.loop_history if r.status == "completed")

        # Continue if at least 30% of steps completed in this loop
        completion_rate = len(result.completed_steps) / len(self.workflow_steps)

        return successful_loops > 0 or completion_rate > 0.3

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "current_loop": self.current_loop.loop_id if self.current_loop else None,
            "total_loops_completed": len(self.loop_history),
            "workflow_steps": len(self.workflow_steps),
            "status": self.current_loop.status if self.current_loop else "not_started",
        }


# Example workflow definition
def create_sample_prep_workflow() -> List[WorkflowStep]:
    """Create a sample preparation workflow."""
    return [
        WorkflowStep(
            name="setup_equipment",
            phase=WorkflowPhase.SETUP,
            devices=["heater_1", "pump_1"],
            dependency=StepDependency.CRITICAL,
            can_skip=False,
            preconditions=["equipment_powered_on"],
            postconditions=["equipment_ready"],
        ),
        WorkflowStep(
            name="sample_prep_1",
            phase=WorkflowPhase.SAMPLE_PREP,
            devices=["heater_1"],
            dependency=StepDependency.HARD,
            can_skip=False,
            preconditions=["equipment_ready"],
            postconditions=["sample_prepared"],
        ),
        WorkflowStep(
            name="sample_prep_2",
            phase=WorkflowPhase.SAMPLE_PREP,
            devices=["heater_1", "pump_1"],
            dependency=StepDependency.SOFT,
            can_skip=True,
            preconditions=["sample_prepared"],
            postconditions=["sample_conditioned"],
        ),
        WorkflowStep(
            name="measurement",
            phase=WorkflowPhase.MEASUREMENT,
            devices=["sensor_1"],
            dependency=StepDependency.HARD,
            can_skip=False,
            preconditions=["sample_conditioned"],
            postconditions=["measurement_complete"],
        ),
        WorkflowStep(
            name="analysis",
            phase=WorkflowPhase.ANALYSIS,
            devices=["analyzer_1"],
            dependency=StepDependency.SOFT,
            can_skip=True,
            preconditions=["measurement_complete"],
            postconditions=["analysis_complete"],
        ),
        WorkflowStep(
            name="cleanup",
            phase=WorkflowPhase.CLEANUP,
            devices=["heater_1", "pump_1"],
            dependency=StepDependency.NONE,
            can_skip=False,
            preconditions=[],
            postconditions=["equipment_clean"],
        ),
    ]
