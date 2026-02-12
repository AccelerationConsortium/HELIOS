"""
WorkflowSupervisor — the real execute_plan() entry point.

Walks a list of PlanSteps, executing each through GuardedExecutor.
On failure, runs the full recovery pipeline:
    classify → signature → decide
then moves the workflow cursor accordingly:
    retry → re-execute same step
    skip  → advance cursor, log skip
    degrade → apply PlanPatch to downstream steps, advance
    abort → safe shutdown, stop

Safety Integration (Phase 1):
    Pre-flight safety gate checks experiment safety before execution.
    If SafetyAgent is provided and returns "deny", workflow is blocked.
    SafetyPacket constraints are used for runtime action validation.
"""
import time
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any, TYPE_CHECKING

from ..core.types import (
    PlanStep, PlanPatch, Action, Decision, ExecutionState,
    HardwareError, DeviceState,
)
from ..core.safety_types import (
    SafetyPacket, ExperimentSummary, ChemicalInfo, GateDecision,
)
from ..devices.simulated.heater import SimHeater
from ..executor.guarded_executor import GuardedExecutor
from ..recovery.recovery_agent import RecoveryAgent

if TYPE_CHECKING:
    from ..safety.agent import SafetyAgent


@dataclass
class StepResult:
    step_id: str
    stage: str
    outcome: Literal["ok", "skipped", "degraded", "aborted"]
    error_type: Optional[str] = None
    decision: Optional[str] = None
    rationale: Optional[str] = None
    patch: Optional[PlanPatch] = None


@dataclass
class PlanResult:
    success: bool
    steps: List[StepResult] = field(default_factory=list)
    patches: List[PlanPatch] = field(default_factory=list)
    aborted_at: Optional[str] = None
    # Safety integration
    safety_gate_decision: Optional[GateDecision] = None
    safety_gate_rationale: Optional[str] = None
    safety_packet: Optional[SafetyPacket] = None


class WorkflowSupervisor:
    """
    Executes a plan of PlanSteps against a SimHeater device.

    Features:
      1. Step-by-step cursor with stage/step_id logging
      2. Full recovery pipeline on failure (classify → signature → decide)
      3. SKIP for optional steps (on_failure == "skip")
      4. PlanPatch propagation on DEGRADE (updates downstream postconditions)
      5. Pre-flight safety gate (when SafetyAgent is provided)
      6. Runtime safety constraint checking (Phase 2)

    Safety Integration:
      If a SafetyAgent is provided, execute_plan() will first run a pre-flight
      safety assessment. The workflow will be blocked if the gate decision is "deny".
      Constraints from the SafetyPacket can be used for runtime validation.
    """

    def __init__(
        self,
        device: SimHeater,
        target_temp: float = 120.0,
        safety_agent: Optional["SafetyAgent"] = None,
        experiment_summary: Optional[ExperimentSummary] = None,
    ):
        """Initialize WorkflowSupervisor.

        Args:
            device: The device to execute the plan on.
            target_temp: Target temperature for the experiment.
            safety_agent: Optional SafetyAgent for pre-flight assessment.
            experiment_summary: Optional experiment summary for safety assessment.
                               If safety_agent is provided but experiment_summary is not,
                               a default summary will be generated from the plan.
        """
        self.device = device
        self.target_temp = target_temp
        self.safety_agent = safety_agent
        self.experiment_summary = experiment_summary
        self.executor = GuardedExecutor()
        self.recovery = RecoveryAgent()
        self.state = ExecutionState(devices={device.name: device.read_state()})
        self.history: List[DeviceState] = []
        self.active_patches: List[PlanPatch] = []
        # Safety packet from pre-flight assessment
        self.safety_packet: Optional[SafetyPacket] = None

    def _configure_safety_runtime(self, packet: SafetyPacket) -> None:
        """Configure runtime components with SafetyPacket constraints.

        This propagates the SafetyPacket to RecoveryAgent for veto power
        on chemical safety events.
        """
        self.safety_packet = packet
        # Propagate to RecoveryAgent for chemical safety veto
        self.recovery.set_safety_packet(packet)

    # ------------------------------------------------------------------
    # Pre-flight Safety Gate
    # ------------------------------------------------------------------
    async def _run_safety_gate(self, plan: List[PlanStep]) -> Optional[SafetyPacket]:
        """Run pre-flight safety assessment if SafetyAgent is configured.

        Args:
            plan: The workflow plan to assess.

        Returns:
            SafetyPacket if assessment was run, None if no safety agent configured.

        Raises:
            SafetyGateError: If gate decision is "deny".
        """
        if self.safety_agent is None:
            return None

        # Build experiment summary if not provided
        summary = self.experiment_summary
        if summary is None:
            summary = self._build_experiment_summary(plan)

        print("\n" + "=" * 66)
        print("  PRE-FLIGHT SAFETY GATE")
        print("=" * 66)
        print(f"  Assessing experiment safety...")
        if summary.chemicals:
            print(f"  Chemicals: {[c.name for c in summary.chemicals]}")
        print(f"  Parameters: {summary.parameters}")

        # Run assessment
        packet = await self.safety_agent.assess(summary)
        self._configure_safety_runtime(packet)

        print(f"\n  Gate Decision: {packet.gate_decision.upper()}")
        print(f"  Rationale: {packet.gate_rationale}")
        print(f"  Risk Level: {packet.overall_risk_level}")

        if packet.hazards:
            print(f"\n  Hazards Identified: {len(packet.hazards)}")
            for h in packet.hazards[:3]:  # Show first 3
                print(f"    - {h.chemical_name}: {', '.join(h.ghs_codes[:3])}")

        if packet.constraints:
            print(f"\n  Safety Constraints: {len(packet.constraints)}")
            for c in packet.constraints[:3]:  # Show first 3
                print(f"    - [{c.type}] {c.description}")

        if packet.ppe:
            print(f"\n  Required PPE: {len(packet.ppe)} items")

        print("=" * 66)

        return packet

    def _build_experiment_summary(self, plan: List[PlanStep]) -> ExperimentSummary:
        """Build experiment summary from plan for safety assessment."""
        # Extract parameters from plan steps
        parameters: Dict[str, Any] = {
            "temperature": self.target_temp,
        }

        procedure_steps: List[str] = []
        for step in plan:
            desc = step.description or f"{step.action.name} ({step.stage})"
            procedure_steps.append(desc)
            # Extract additional parameters from action params
            for key, val in step.action.params.items():
                if key not in parameters:
                    parameters[key] = val

        return ExperimentSummary(
            title=f"Experiment on {self.device.name}",
            chemicals=[],  # Would need to be provided externally
            procedure_steps=procedure_steps,
            parameters=parameters,
            equipment=[self.device.name],
        )

    def execute_plan_sync(self, plan: List[PlanStep]) -> PlanResult:
        """Synchronous wrapper for execute_plan (for backwards compatibility)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.execute_plan_async(plan)
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def execute_plan(self, plan: List[PlanStep]) -> PlanResult:
        """Execute a workflow plan, returning structured results.

        This is the synchronous entry point. For async usage, use execute_plan_async().

        If a SafetyAgent is configured, this will run pre-flight safety assessment.
        The workflow will be blocked if the gate decision is "deny".
        """
        import asyncio

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - can't use run_until_complete
            # Create a new thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.execute_plan_async(plan)
                )
                return future.result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            return asyncio.run(self.execute_plan_async(plan))

    async def execute_plan_async(self, plan: List[PlanStep]) -> PlanResult:
        """Execute a workflow plan asynchronously, returning structured results.

        This is the async entry point that supports SafetyAgent integration.
        """
        result = PlanResult(success=False)

        # --- Pre-flight Safety Gate ---
        if self.safety_agent is not None:
            try:
                packet = await self._run_safety_gate(plan)
                if packet:
                    result.safety_packet = packet
                    result.safety_gate_decision = packet.gate_decision
                    result.safety_gate_rationale = packet.gate_rationale

                    if packet.gate_decision == "deny":
                        print("\n  ✗ WORKFLOW BLOCKED BY SAFETY GATE")
                        print(f"    Reason: {packet.gate_rationale}")
                        result.success = False
                        return result

                    if packet.gate_decision == "allow_with_constraints":
                        print("\n  ⚠ WORKFLOW ALLOWED WITH CONSTRAINTS")
                        print(f"    {len(packet.constraints)} constraint(s) will be enforced")

            except Exception as e:
                print(f"\n  ⚠ SAFETY GATE ERROR: {e}")
                print("    Proceeding without safety assessment (fail-open)")
                result.safety_gate_rationale = f"Assessment failed: {e}"

        # --- Continue with normal execution ---
        cursor = 0
        retry_budget: Dict[str, int] = {}  # step_id -> retries used

        print("=" * 66)
        print("  WORKFLOW SUPERVISOR — execute_plan()")
        print("=" * 66)
        print(f"  Steps: {len(plan)}  |  Device: {self.device.name}")
        print(f"  Target: {self.target_temp}°C")
        if self.safety_packet:
            print(f"  Safety: {self.safety_packet.gate_decision} | Risk: {self.safety_packet.overall_risk_level}")
        print("=" * 66)

        while cursor < len(plan):
            step = plan[cursor]
            retries_used = retry_budget.get(step.step_id, 0)

            print(f"\n{'─' * 60}")
            print(f"  [{cursor+1}/{len(plan)}]  step_id={step.step_id}  stage={step.stage}")
            print(f"           criticality={step.criticality}  on_failure={step.on_failure}")
            if step.description:
                print(f"           {step.description}")
            print(f"{'─' * 60}")

            # --- Apply active patches to this step ---
            patched_action = self._apply_patches(step)

            # --- Observe ---
            try:
                self.device.tick()
                dev_state = self.device.read_state()
                self.state.devices[self.device.name] = dev_state
                self.history.append(dev_state)
                if len(self.history) > 10:
                    self.history.pop(0)
            except HardwareError as obs_err:
                print(f"  ✗ OBSERVE FAILED  step_id={step.step_id}  error={obs_err.type}")
                # Observation failure — feed into recovery with step semantics
                if step.on_failure == "skip" and step.criticality == "optional":
                    print(f"  CURSOR: skip (observation error on optional step)")
                    result.steps.append(StepResult(
                        step_id=step.step_id, stage=step.stage,
                        outcome="skipped", error_type=obs_err.type,
                        decision="skip",
                        rationale="Observation error on optional step → skip",
                    ))
                    cursor += 1
                    continue
                else:
                    print(f"  CURSOR: abort (observation error on critical step)")
                    result.steps.append(StepResult(
                        step_id=step.step_id, stage=step.stage,
                        outcome="aborted", error_type=obs_err.type,
                        decision="abort",
                        rationale=f"Observation error: {obs_err.message}",
                    ))
                    result.aborted_at = step.step_id
                    self._shutdown()
                    result.patches = self.active_patches
                    return result

            # --- Execute ---
            print(f"  ACTION: {patched_action.name} {patched_action.params}")
            try:
                # Pass SafetyPacket for runtime constraint checking
                self.executor.execute(
                    self.device, patched_action, self.state,
                    safety_packet=self.safety_packet
                )
                # Success
                print(f"  ✓ STEP OK  step_id={step.step_id}")
                result.steps.append(StepResult(
                    step_id=step.step_id,
                    stage=step.stage,
                    outcome="ok",
                ))
                retry_budget.pop(step.step_id, None)
                self.recovery.reset_retry_counts()
                cursor += 1

            except HardwareError as err:
                print(f"  ✗ STEP FAILED  step_id={step.step_id}  error={err.type}")

                # --- Full recovery pipeline ---
                decision = self.recovery.decide(
                    state=dev_state,
                    error=err,
                    history=self.history,
                    last_action=patched_action,
                    stage=step.stage,
                )
                print(f"  DECISION: {decision.kind}  — {decision.rationale}")

                # --- Cursor logic ---
                if decision.kind == "abort":
                    # Check step's on_failure override
                    if step.on_failure == "skip" and step.criticality == "optional":
                        print(f"  CURSOR: skip (on_failure override for optional step)")
                        result.steps.append(StepResult(
                            step_id=step.step_id, stage=step.stage,
                            outcome="skipped", error_type=err.type,
                            decision="skip", rationale="Optional step abort → skip override",
                        ))
                        cursor += 1
                        continue

                    print(f"  CURSOR: abort — stopping plan")
                    self._execute_recovery_actions(decision.actions)
                    result.steps.append(StepResult(
                        step_id=step.step_id, stage=step.stage,
                        outcome="aborted", error_type=err.type,
                        decision="abort", rationale=decision.rationale,
                    ))
                    result.aborted_at = step.step_id
                    self._shutdown()
                    result.patches = self.active_patches
                    return result

                elif decision.kind == "retry":
                    retries_used += 1
                    retry_budget[step.step_id] = retries_used
                    if retries_used > step.max_retries:
                        # Exceeded budget — fall back to step's on_failure
                        print(f"  CURSOR: retry budget exhausted ({retries_used}/{step.max_retries})")
                        if step.on_failure == "skip":
                            print(f"  CURSOR: skip (on_failure fallback)")
                            result.steps.append(StepResult(
                                step_id=step.step_id, stage=step.stage,
                                outcome="skipped", error_type=err.type,
                                decision="skip",
                                rationale=f"Retry budget exhausted, on_failure=skip",
                            ))
                            cursor += 1
                        else:
                            print(f"  CURSOR: abort (on_failure fallback)")
                            self._execute_recovery_actions(decision.actions)
                            result.steps.append(StepResult(
                                step_id=step.step_id, stage=step.stage,
                                outcome="aborted", error_type=err.type,
                                decision="abort",
                                rationale=f"Retry budget exhausted, on_failure=abort",
                            ))
                            result.aborted_at = step.step_id
                            self._shutdown()
                            result.patches = self.active_patches
                            return result
                    else:
                        print(f"  CURSOR: retry ({retries_used}/{step.max_retries})")
                        self._execute_recovery_actions(decision.actions)
                        # Do NOT advance cursor — re-execute same step
                        continue

                elif decision.kind == "skip":
                    print(f"  CURSOR: skip  step_id={step.step_id}")
                    result.steps.append(StepResult(
                        step_id=step.step_id, stage=step.stage,
                        outcome="skipped", error_type=err.type,
                        decision="skip", rationale=decision.rationale,
                    ))
                    cursor += 1

                elif decision.kind == "degrade":
                    print(f"  CURSOR: degrade  step_id={step.step_id}")
                    self._execute_recovery_actions(decision.actions)

                    # Build PlanPatch for downstream steps
                    patch = self._build_patch(step, decision, plan, cursor)
                    if patch:
                        self.active_patches.append(patch)
                        print(f"  PATCH: {len(patch.notes)} note(s), "
                              f"{len(patch.overrides)} override(s), "
                              f"{len(patch.relaxations)} relaxation(s)")
                        for note in patch.notes:
                            print(f"    → {note}")

                    result.steps.append(StepResult(
                        step_id=step.step_id, stage=step.stage,
                        outcome="degraded", error_type=err.type,
                        decision="degrade", rationale=decision.rationale,
                        patch=patch,
                    ))
                    cursor += 1

            time.sleep(0.3)

        # Plan completed (cursor walked all steps without abort)
        result.success = result.aborted_at is None
        result.patches = self.active_patches

        print("\n" + "=" * 66)
        if result.success:
            print("  PLAN COMPLETE — all steps processed")
        else:
            print(f"  PLAN ABORTED at step {result.aborted_at}")
        print("=" * 66)
        self._print_summary(result)
        self._shutdown()
        return result

    # ------------------------------------------------------------------
    # Patch helpers
    # ------------------------------------------------------------------
    def _apply_patches(self, step: PlanStep) -> Action:
        """Apply active PlanPatches to a step's action, return patched copy."""
        action = step.action
        # Make a mutable copy of params
        patched_params = dict(action.params)
        patched_postconditions = list(action.postconditions)

        for patch in self.active_patches:
            # Override params
            if step.step_id in patch.overrides:
                for key, val in patch.overrides[step.step_id].items():
                    old = patched_params.get(key)
                    patched_params[key] = val
                    print(f"  PATCH APPLIED: {step.step_id}.{key}: {old} → {val}")

            # Relax postconditions
            if step.step_id in patch.relaxations:
                patched_postconditions = patch.relaxations[step.step_id]
                print(f"  PATCH APPLIED: {step.step_id} postconditions relaxed")

        return Action(
            name=action.name,
            effect=action.effect,
            params=patched_params,
            irreversible=action.irreversible,
            preconditions=action.preconditions,
            postconditions=patched_postconditions,
            safety_constraints=action.safety_constraints,
            device=action.device,
        )

    def _build_patch(
        self, step: PlanStep, decision: Decision,
        plan: List[PlanStep], cursor: int,
    ) -> Optional[PlanPatch]:
        """Build a PlanPatch when degrade happens."""
        # Extract degraded target from decision actions
        degraded_target = None
        for act in decision.actions:
            if act.name == "set_temperature" and "temperature" in act.params:
                degraded_target = act.params["temperature"]
                break

        if degraded_target is None:
            return None

        original_target = self.target_temp
        patch = PlanPatch(
            original_target=original_target,
            degraded_target=degraded_target,
            notes=[
                f"Degraded from {original_target}°C to {degraded_target}°C at step {step.step_id}",
                f"Downstream postconditions relaxed. Results may be compromised/anomalous.",
            ],
        )

        # Scan downstream steps for temperature-dependent params & postconditions
        for future_step in plan[cursor + 1:]:
            fa = future_step.action

            # Override temperature params
            if "temperature" in fa.params:
                patch.overrides[future_step.step_id] = {
                    "temperature": degraded_target,
                }

            # Relax postconditions mentioning the old target
            new_postconditions = []
            relaxed = False
            for pc in fa.postconditions:
                if str(original_target) in pc:
                    new_pc = pc.replace(str(original_target), str(degraded_target))
                    new_postconditions.append(new_pc)
                    relaxed = True
                else:
                    new_postconditions.append(pc)
            if relaxed:
                patch.relaxations[future_step.step_id] = new_postconditions

        # Update our own tracked target
        self.target_temp = degraded_target
        return patch

    # ------------------------------------------------------------------
    # Recovery / shutdown helpers
    # ------------------------------------------------------------------
    def _execute_recovery_actions(self, actions: List[Action]):
        if not actions:
            return
        print(f"  RECOVERY: executing {len(actions)} action(s)")
        for act in actions:
            try:
                # Pass SafetyPacket for runtime constraint checking
                self.executor.execute(
                    self.device, act, self.state,
                    safety_packet=self.safety_packet
                )
            except HardwareError as e:
                print(f"  RECOVERY FAILED: {e.message}")
                break

    def _shutdown(self):
        print("\n  SHUTDOWN: safe shutdown sequence")
        try:
            cool = Action(name="cool_down", effect="write", device=self.device.name,
                          postconditions=["telemetry.heating == False", "status == idle"])
            # Pass SafetyPacket for runtime constraint checking
            self.executor.execute(
                self.device, cool, self.state,
                safety_packet=self.safety_packet
            )
        except Exception as e:
            print(f"  SHUTDOWN WARNING: {e}")
        print("  SHUTDOWN: complete")

    def _print_summary(self, result: PlanResult):
        ok = sum(1 for s in result.steps if s.outcome == "ok")
        skipped = sum(1 for s in result.steps if s.outcome == "skipped")
        degraded = sum(1 for s in result.steps if s.outcome == "degraded")
        aborted = sum(1 for s in result.steps if s.outcome == "aborted")
        print(f"\n  SUMMARY:  ok={ok}  skipped={skipped}  degraded={degraded}  aborted={aborted}")
        if result.patches:
            print(f"  PATCHES:  {len(result.patches)} degrade patch(es) applied")
        for sr in result.steps:
            tag = {"ok": "✓", "skipped": "⊘", "degraded": "↓", "aborted": "✗"}[sr.outcome]
            extra = f"  [{sr.decision}: {sr.rationale}]" if sr.decision else ""
            print(f"    {tag} {sr.step_id} ({sr.stage}){extra}")
