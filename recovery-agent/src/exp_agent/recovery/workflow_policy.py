"""
Workflow-aware error recovery policies for the Experiment Agent.

This module extends the recovery system to consider workflow steps, dependencies,
and experiment loops when making recovery decisions.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from ..core.types import Decision, DecisionType, HardwareError, DeviceState


class WorkflowPhase(Enum):
    """Experiment workflow phases."""

    SETUP = "setup"  # 实验准备阶段
    SAMPLE_PREP = "sample_prep"  # 样品制备阶段
    MEASUREMENT = "measurement"  # 测量阶段
    ANALYSIS = "analysis"  # 分析阶段
    CLEANUP = "cleanup"  # 清理阶段


class StepDependency(Enum):
    """Types of step dependencies."""

    NONE = "none"  # 无依赖
    SOFT = "soft"  # 软依赖（可跳过）
    HARD = "hard"  # 硬依赖（必须成功）
    CRITICAL = "critical"  # 关键步骤（失败则终止实验）


@dataclass
class WorkflowStep:
    """Represents a step in the experiment workflow."""

    name: str
    phase: WorkflowPhase
    devices: List[str]  # Devices involved in this step
    dependency: StepDependency = StepDependency.NONE
    can_skip: bool = False
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes default
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    def is_critical(self) -> bool:
        """Check if this step is critical for the experiment."""
        return self.dependency == StepDependency.CRITICAL

    def can_be_skipped(self) -> bool:
        """Check if this step can be safely skipped."""
        return self.can_skip or self.dependency == StepDependency.SOFT


@dataclass
class ExperimentLoop:
    """Represents a complete experiment loop/cycle."""

    loop_id: int
    current_step_index: int = 0
    completed_steps: Set[str] = field(default_factory=set)
    skipped_steps: Set[str] = field(default_factory=set)
    failed_steps: Dict[str, str] = field(
        default_factory=dict
    )  # step_name -> error_reason
    status: str = "running"  # running, completed, failed, aborted

    def can_proceed_to_next_step(self, next_step: WorkflowStep) -> bool:
        """Check if we can proceed to the next step."""
        if next_step.dependency == StepDependency.NONE:
            return True

        if next_step.dependency == StepDependency.SOFT:
            return True  # Can proceed even if previous steps failed

        if next_step.dependency == StepDependency.HARD:
            # Check if all prerequisite steps completed successfully
            required_phases = self._get_required_phases(next_step.phase)
            for phase in required_phases:
                phase_steps = [
                    s for s in self.completed_steps if s.startswith(f"{phase.value}_")
                ]
                if not phase_steps:
                    return False
            return True

        if next_step.dependency == StepDependency.CRITICAL:
            # Must have all previous phases completed
            return len(self.completed_steps) == len(
                self._get_all_previous_phases(next_step.phase)
            )

        return False

    def _get_required_phases(self, current_phase: WorkflowPhase) -> List[WorkflowPhase]:
        """Get phases required for the current phase."""
        phase_order = [
            WorkflowPhase.SETUP,
            WorkflowPhase.SAMPLE_PREP,
            WorkflowPhase.MEASUREMENT,
            WorkflowPhase.ANALYSIS,
            WorkflowPhase.CLEANUP,
        ]

        current_index = phase_order.index(current_phase)
        if current_phase == WorkflowPhase.SAMPLE_PREP:
            return [WorkflowPhase.SETUP]
        elif current_phase == WorkflowPhase.MEASUREMENT:
            return [WorkflowPhase.SETUP, WorkflowPhase.SAMPLE_PREP]
        elif current_phase == WorkflowPhase.ANALYSIS:
            return [
                WorkflowPhase.SETUP,
                WorkflowPhase.SAMPLE_PREP,
                WorkflowPhase.MEASUREMENT,
            ]
        elif current_phase == WorkflowPhase.CLEANUP:
            return []  # Cleanup can run regardless
        return []

    def _get_all_previous_phases(
        self, current_phase: WorkflowPhase
    ) -> List[WorkflowPhase]:
        """Get all phases before the current one."""
        phase_order = [
            WorkflowPhase.SETUP,
            WorkflowPhase.SAMPLE_PREP,
            WorkflowPhase.MEASUREMENT,
            WorkflowPhase.ANALYSIS,
            WorkflowPhase.CLEANUP,
        ]
        current_index = phase_order.index(current_phase)
        return phase_order[:current_index]


@dataclass
class WorkflowContext:
    """Context information for workflow-aware decision making."""

    current_loop: ExperimentLoop
    workflow_steps: List[WorkflowStep]
    step_history: List[str] = field(default_factory=list)  # Recent step executions
    loop_history: List[ExperimentLoop] = field(default_factory=list)  # Previous loops

    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get the currently executing step."""
        if 0 <= self.current_loop.current_step_index < len(self.workflow_steps):
            return self.workflow_steps[self.current_loop.current_step_index]
        return None

    def get_next_step(self) -> Optional[WorkflowStep]:
        """Get the next step in the workflow."""
        next_index = self.current_loop.current_step_index + 1
        if next_index < len(self.workflow_steps):
            return self.workflow_steps[next_index]
        return None


class WorkflowRecoveryPolicy:
    """Policy for workflow-aware error recovery."""

    def __init__(self):
        self.step_failure_counts: Dict[str, int] = {}
        self.phase_failure_counts: Dict[str, int] = {}

    def make_decision(
        self,
        error: HardwareError,
        device_states: Dict[str, DeviceState],
        context: WorkflowContext,
    ) -> Decision:
        """
        Make a recovery decision considering workflow context.

        Args:
            error: The hardware error that occurred
            device_states: Current states of all devices
            context: Workflow execution context

        Returns:
            Decision: Recovery decision with rationale
        """

        current_step = context.get_current_step()
        if not current_step:
            # No current step - this shouldn't happen
            return Decision(
                kind="abort", rationale="No active workflow step", actions=[]
            )

        # Track failure counts
        self._update_failure_counts(error, current_step)

        # Analyze error impact on workflow
        impact = self._analyze_workflow_impact(error, current_step, context)

        if impact == "critical_failure":
            # Critical step failed - abort entire experiment
            return Decision(
                kind="abort",
                rationale=f"Critical step '{current_step.name}' failed: {error.message}. Aborting experiment.",
                actions=self._get_cleanup_actions(current_step.devices),
            )

        elif impact == "can_skip_step":
            # Non-critical step failed but can be skipped
            next_step = context.get_next_step()
            if next_step and context.current_loop.can_proceed_to_next_step(next_step):
                return Decision(
                    kind="skip",
                    rationale=f"Step '{current_step.name}' failed but can be skipped. Proceeding to '{next_step.name}'.",
                    actions=[],
                )
            else:
                return Decision(
                    kind="abort",
                    rationale=f"Cannot proceed after skipping step '{current_step.name}'.",
                    actions=self._get_cleanup_actions(current_step.devices),
                )

        elif impact == "retry_step":
            # Can retry the current step
            if (
                self.step_failure_counts.get(current_step.name, 0)
                < current_step.max_retries
            ):
                return Decision(
                    kind="retry",
                    rationale=f"Retrying step '{current_step.name}' (attempt {self.step_failure_counts[current_step.name] + 1}/{current_step.max_retries})",
                    actions=[],
                )
            else:
                # Max retries exceeded - decide whether to skip or abort
                return self._decide_after_max_retries(current_step, context)

        elif impact == "degrade_and_continue":
            # Try degraded operation
            return Decision(
                kind="degrade",
                rationale=f"Step '{current_step.name}' failed, attempting degraded operation.",
                actions=self._get_degraded_actions(current_step, error),
            )

        elif impact == "skip_to_next_loop":
            # Skip remaining steps and start next experiment loop
            return Decision(
                kind="abort",  # Abort current loop
                rationale=f"Step '{current_step.name}' failed in {current_step.phase.value} phase. Skipping to next experiment loop.",
                actions=self._get_cleanup_actions(current_step.devices),
            )

        elif impact == "can_skip_step":
            # Non-critical step failed but can be skipped
            next_step = context.get_next_step()
            if next_step and context.current_loop.can_proceed_to_next_step(next_step):
                return Decision(
                    kind="skip",
                    rationale=f"Step '{current_step.name}' failed but can be skipped. Proceeding to '{next_step.name}'.",
                    actions=[],
                )
            else:
                return Decision(
                    kind="abort",
                    rationale=f"Cannot proceed after skipping step '{current_step.name}'.",
                    actions=self._get_cleanup_actions(current_step.devices),
                )

        elif impact == "retry_step":
            # Can retry the current step
            if (
                self.step_failure_counts.get(current_step.name, 0)
                < current_step.max_retries
            ):
                return Decision(
                    kind="retry",
                    rationale=f"Retrying step '{current_step.name}' (attempt {self.step_failure_counts[current_step.name] + 1}/{current_step.max_retries})",
                    actions=[],
                )
            else:
                # Max retries exceeded - decide whether to skip or abort
                return self._decide_after_max_retries(current_step, context)

        elif impact == "degrade_and_continue":
            # Try degraded operation
            return Decision(
                kind="degrade",
                rationale=f"Step '{current_step.name}' failed, attempting degraded operation.",
                actions=self._get_degraded_actions(current_step, error),
            )

        elif impact == "skip_to_next_loop":
            # Skip remaining steps and start next experiment loop
            return Decision(
                kind="abort",  # Abort current loop
                rationale=f"Step '{current_step.name}' failed in {current_step.phase.value} phase. Skipping to next experiment loop.",
                actions=self._get_cleanup_actions(current_step.devices),
            )

        # Default fallback
        return Decision(
            kind="abort",
            rationale=f"Unexpected error in step '{current_step.name}': {error.message}",
            actions=self._get_cleanup_actions(current_step.devices),
        )

        # Default fallback
        return Decision(
            kind=DecisionType.ABORT,
            rationale=f"Unexpected error in step '{current_step.name}': {error.message}",
            actions=self._get_cleanup_actions(current_step.devices),
        )

    def _analyze_workflow_impact(
        self, error: HardwareError, step: WorkflowStep, context: WorkflowContext
    ) -> str:
        """Analyze how the error impacts the workflow."""

        # Check if it's a critical step
        if step.is_critical():
            return "critical_failure"

        # Check error severity and type
        if error.severity == "high":
            if step.phase == WorkflowPhase.SAMPLE_PREP:
                # Sample prep failures often allow skipping to next loop
                return "skip_to_next_loop"
            elif step.phase in [WorkflowPhase.MEASUREMENT, WorkflowPhase.ANALYSIS]:
                # Measurement/analysis failures might be recoverable
                if error.type in ["timeout", "communication_error"]:
                    return "retry_step"
                else:
                    return "can_skip_step"
            else:
                return "critical_failure"

        elif error.severity == "medium":
            # Medium severity - often recoverable
            if step.can_be_skipped():
                return "can_skip_step"
            else:
                return "retry_step"

        else:  # low severity
            return "retry_step"

    def _update_failure_counts(self, error: HardwareError, step: WorkflowStep):
        """Update failure tracking counts."""
        step_key = step.name
        phase_key = step.phase.value

        self.step_failure_counts[step_key] = (
            self.step_failure_counts.get(step_key, 0) + 1
        )
        self.phase_failure_counts[phase_key] = (
            self.phase_failure_counts.get(phase_key, 0) + 1
        )

    def _decide_after_max_retries(
        self, step: WorkflowStep, context: WorkflowContext
    ) -> Decision:
        """Decide what to do after maximum retries are exceeded."""

        if step.can_be_skipped():
            next_step = context.get_next_step()
            if next_step and context.current_loop.can_proceed_to_next_step(next_step):
                return Decision(
                    kind="skip",
                    rationale=f"Max retries exceeded for '{step.name}', skipping to next step.",
                    actions=[],
                )

        # Check if we can skip to next experiment loop
        if step.phase in [WorkflowPhase.SAMPLE_PREP, WorkflowPhase.MEASUREMENT]:
            return Decision(
                kind="abort",
                rationale=f"Max retries exceeded for '{step.name}' in {step.phase.value}. Skipping to next loop.",
                actions=self._get_cleanup_actions(step.devices),
            )

        # Default: abort
        return Decision(
            kind="abort",
            rationale=f"Max retries exceeded for '{step.name}', aborting.",
            actions=self._get_cleanup_actions(step.devices),
        )

        # Check if we can skip to next experiment loop
        if step.phase in [WorkflowPhase.SAMPLE_PREP, WorkflowPhase.MEASUREMENT]:
            return Decision(
                kind=DecisionType.ABORT,
                rationale=f"Max retries exceeded for '{step.name}' in {step.phase.value}. Skipping to next loop.",
                actions=self._get_cleanup_actions(step.devices),
            )

        # Default: abort
        return Decision(
            kind=DecisionType.ABORT,
            rationale=f"Max retries exceeded for '{step.name}', aborting.",
            actions=self._get_cleanup_actions(step.devices),
        )

    def _get_cleanup_actions(self, devices: List[str]) -> List[Any]:
        """Get cleanup actions for the specified devices."""
        from ..core.types import Action

        actions = []
        for device in devices:
            actions.append(Action(name="cool_down", effect="write", device=device))
        return actions

    def _get_degraded_actions(
        self, step: WorkflowStep, error: HardwareError
    ) -> List[Any]:
        """Get actions for degraded operation mode."""
        from ..core.types import Action

        if step.phase == WorkflowPhase.MEASUREMENT and "temperature" in error.context:
            # For temperature-related errors, try lower temperature
            return [
                Action(name="cool_down", effect="write", device=step.devices[0]),
                Action(
                    name="set_temperature",
                    effect="write",
                    params={"temperature": 100.0},
                    device=step.devices[0],
                ),
            ]

        return self._get_cleanup_actions(step.devices)
