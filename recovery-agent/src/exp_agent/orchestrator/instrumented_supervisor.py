"""
Instrumented Supervisor - Full log-decision-recovery pipeline integration.

This supervisor wraps the standard execution loop with comprehensive
structured logging, enabling:
- Complete audit trail of all decisions
- Error forensics with full context
- Performance analysis
- Replay capability
"""
import time
from pathlib import Path
from typing import Literal, Optional, List

from ..core.types import Action, ExecutionState, HardwareError, Decision, DeviceState
from ..devices.simulated.heater import SimHeater
from ..executor.guarded_executor import GuardedExecutor
from ..recovery.recovery_agent import RecoveryAgent
from ..recovery.policy import classify_error, analyze_signature
from ..logging.pipeline import (
    PipelineLogger,
    MemoryBackend,
    FileBackend,
    ConsoleBackend,
    TrailAnalyzer,
    LogLevel,
)
from ..logging.anomaly import AnomalyPacket


class InstrumentedSupervisor:
    """
    Supervisor with full logging instrumentation.

    The log-decision-recovery pipeline flow:

    1. OBSERVE: Device tick → read state → log telemetry
    2. ERROR DETECTED: Exception → log_error_detected (starts correlation)
    3. CLASSIFY: Error profile → log_error_classified
    4. ANALYZE: Signature detection → log_signature_analyzed
    5. DECIDE: Recovery decision → log_decision_made
    6. RECOVER: Execute actions → log_recovery_action(s)
    7. COMPLETE: Recovery result → log_recovery_completed (ends correlation)

    All events in steps 2-7 share a correlation_id for traceability.
    """

    def __init__(
        self,
        target_temp: float,
        fault_mode: Literal["none", "random", "timeout", "overshoot", "sensor_fail"],
        experiment_id: str = None,
        log_dir: Path = None,
        log_level: LogLevel = LogLevel.INFO,
    ):
        # Core components
        self.device = SimHeater(name="heater_1", fault_mode=fault_mode)
        self.executor = GuardedExecutor()
        self.recovery = RecoveryAgent()

        self.target_temp = target_temp
        self.fault_mode = fault_mode
        self.state = ExecutionState(devices={self.device.name: self.device.read_state()})

        # Retry management
        self.retry_counts = {}
        self.MAX_RETRIES = 3
        self.history: List[DeviceState] = []

        # Logging pipeline
        self.logger = PipelineLogger(experiment_id=experiment_id)
        self._setup_logging(log_dir, log_level)

        # Memory backend for analysis
        self.memory_backend = MemoryBackend()
        self.logger.add_backend(self.memory_backend)

    def _setup_logging(self, log_dir: Path, log_level: LogLevel) -> None:
        """Configure logging backends."""
        # Console output
        self.logger.add_backend(ConsoleBackend(min_level=log_level))

        # File persistence (optional)
        if log_dir:
            self.logger.add_backend(
                FileBackend(log_dir, self.logger.experiment_id)
            )

    def run(self) -> bool:
        """
        Execute the experiment with full logging.

        Returns True if experiment completed successfully.
        """
        self.logger.log_experiment_started(
            target_temp=self.target_temp,
            fault_mode=self.fault_mode,
            config={
                "max_retries": self.MAX_RETRIES,
                "max_steps": 50,
            }
        )

        # Plan
        plan = [
            Action(
                name="set_temperature",
                effect="write",
                params={"temperature": self.target_temp},
                device=self.device.name,
                postconditions=[
                    f"telemetry.target == {self.target_temp}",
                    f"telemetry.temperature ~= {self.target_temp} +/- 2.0 within 20s"
                ]
            ),
            Action(
                name="wait",
                effect="write",
                params={"duration": 5},
                device=self.device.name
            )
        ]

        step_index = 0
        max_steps = 50
        current_step = 0
        success = False

        while current_step < max_steps:
            current_step += 1
            self.logger.set_step(current_step)

            # =================================================================
            # 1. OBSERVE
            # =================================================================
            try:
                self.device.tick()
                dev_state = self.device.read_state()
                self.state.devices[self.device.name] = dev_state

                # Update history
                self.history.append(dev_state)
                if len(self.history) > 10:
                    self.history.pop(0)

                # Log telemetry
                self.logger.log_telemetry_update(dev_state, len(self.history))

            except HardwareError as e:
                # Handle observation error through pipeline
                decision = self._handle_error_pipeline(e, dev_state=None)

                if decision.kind == "abort":
                    self._perform_abort(decision)
                    break

                if decision.actions:
                    self._execute_recovery_pipeline(decision)
                continue

            # =================================================================
            # 2. CHECK PLAN COMPLETION
            # =================================================================
            if step_index >= len(plan):
                current_temp = dev_state.telemetry.get("temperature", 0)
                self.logger.log_experiment_completed(
                    success=True,
                    total_steps=current_step,
                    final_temp=current_temp
                )
                success = True
                break

            # =================================================================
            # 3. EXECUTE NEXT ACTION
            # =================================================================
            next_action = plan[step_index]
            self.logger.log_action_proposed(next_action)

            try:
                start_time = time.time()
                self.logger.log_action_started(next_action)

                self.executor.execute(self.device, next_action, self.state)

                duration_ms = (time.time() - start_time) * 1000
                self.logger.log_action_completed(next_action, duration_ms)

                # Success - reset retries
                self.retry_counts = {}
                step_index += 1

            except HardwareError as e:
                self.logger.log_action_failed(next_action, e)

                if not self._check_retry_budget(e):
                    self.logger.log_shutdown("Retry budget exceeded", safe=True)
                    self.shutdown()
                    break

                # Full pipeline: error → classify → analyze → decide → recover
                decision = self._handle_error_pipeline(e, last_action=next_action)

                if decision.kind == "abort":
                    self._perform_abort(decision)
                    break

                # Execute recovery
                self._execute_recovery_pipeline(decision)

                # Update plan progress based on decision
                if decision.kind in ["skip", "degrade"]:
                    step_index += 1
                # "retry" keeps step_index the same

            time.sleep(0.5)

        self.logger.flush()
        return success

    def _handle_error_pipeline(
        self,
        error: HardwareError,
        dev_state: DeviceState = None,
        last_action: Action = None,
    ) -> Decision:
        """
        Full error handling pipeline with logging.

        Pipeline:
        1. log_error_detected (starts correlation)
        2. log_error_classified
        3. log_signature_analyzed
        4. log_decision_made

        Returns the Decision for the caller to execute.
        """
        # Get current state if not provided
        if dev_state is None:
            dev_state = self.state.devices.get(self.device.name)

        # 1. Error detected - starts correlation context
        corr_id = self.logger.log_error_detected(error, dev_state)

        # 2. Classify error
        profile = classify_error(error)
        self.logger.log_error_classified(
            error,
            unsafe=profile.unsafe,
            recoverable=profile.recoverable,
            strategy=profile.default_strategy
        )

        # 3. Analyze signature
        sig = analyze_signature(self.history)
        self.logger.log_signature_analyzed(
            mode=sig.mode,
            confidence=sig.confidence,
            features=sig.details,
        )

        # 4. Make decision
        decision = self.recovery.decide(
            state=dev_state,
            error=error,
            history=self.history,
            last_action=last_action
        )

        retry_count = self.retry_counts.get(error.type, 0)
        self.logger.log_decision_made(decision, error, retry_count)

        # Phase 2: log LLM advisory proposal if present
        if getattr(self.recovery, "last_llm_proposal", None) is not None:
            self.logger.log_llm_proposal(self.recovery.last_llm_proposal, error)

        # Phase 3 groundwork: emit an anomaly packet for downstream analysis
        try:
            packet = AnomalyPacket(
                packet_id=f"anom_{corr_id}",
                error=error.model_dump(),
                signature=sig,
                baseline_decision=decision,
                llm_proposal=getattr(self.recovery, "last_llm_proposal", None),
                telemetry_window=list(self.history),
                tags=["hardware_error", error.type],
                notes={"retry_count": retry_count},
            )
            self.logger.log_anomaly_packet(packet.model_dump(), device_name=error.device)
        except Exception:
            pass

        return decision

    def _execute_recovery_pipeline(self, decision: Decision) -> bool:
        """
        Execute recovery actions with logging.

        Pipeline:
        1. log_recovery_started
        2. log_recovery_action (for each action)
        3. log_recovery_completed
        """
        if not decision.actions:
            return True

        self.logger.log_recovery_started(decision)

        start_time = time.time()
        success = True
        total = len(decision.actions)

        for i, action in enumerate(decision.actions):
            action_start = time.time()
            try:
                self.executor.execute(self.device, action, self.state)
                duration_ms = (time.time() - action_start) * 1000
                self.logger.log_recovery_action(action, i, total, True, duration_ms)

            except HardwareError as e:
                duration_ms = (time.time() - action_start) * 1000
                self.logger.log_recovery_action(action, i, total, False, duration_ms)
                success = False
                break

        total_duration_ms = (time.time() - start_time) * 1000
        self.logger.log_recovery_completed(decision, success, total_duration_ms)

        return success

    def _check_retry_budget(self, error: HardwareError) -> bool:
        """Check and update retry budget."""
        err_key = error.type
        current = self.retry_counts.get(err_key, 0)
        self.retry_counts[err_key] = current + 1
        return self.retry_counts[err_key] <= self.MAX_RETRIES

    def _perform_abort(self, decision: Decision) -> None:
        """Perform abort with logging."""
        self.logger.log_shutdown(decision.rationale, safe=False)
        self._execute_recovery_pipeline(decision)
        self.shutdown()

    def shutdown(self) -> None:
        """Safe shutdown with logging."""
        self.logger.log_shutdown("Initiating safe shutdown", safe=True)

        actions = [
            Action(
                name="cool_down",
                effect="write",
                device=self.device.name,
                postconditions=["telemetry.heating == False", "status == idle"]
            )
        ]

        for i, action in enumerate(actions):
            try:
                self.executor.execute(self.device, action, self.state)
            except Exception as e:
                pass  # Best effort

        final_state = self.device.read_state()
        safe = final_state.telemetry.get("heating") is False

        self.logger.log_experiment_completed(
            success=safe,
            total_steps=self.logger._step_number,
            final_temp=final_state.telemetry.get("temperature")
        )

    def get_analyzer(self) -> TrailAnalyzer:
        """Get analyzer for decision trail analysis."""
        return TrailAnalyzer(self.memory_backend)

    def get_decision_summary(self):
        """Get summary of all decisions made during experiment."""
        return self.get_analyzer().summarize_decisions()


# ============================================================================
# Example Usage
# ============================================================================

def run_instrumented_experiment():
    """
    Example: Run an instrumented experiment and analyze results.
    """
    print("=" * 60)
    print("LOG-DECISION-RECOVERY PIPELINE DEMO")
    print("=" * 60)

    # Run with overshoot fault to trigger recovery
    supervisor = InstrumentedSupervisor(
        target_temp=120.0,
        fault_mode="overshoot",
        experiment_id="demo_001",
        log_level=LogLevel.INFO,
    )

    success = supervisor.run()

    print("\n" + "=" * 60)
    print("DECISION ANALYSIS")
    print("=" * 60)

    # Analyze decisions
    summary = supervisor.get_decision_summary()
    print(f"\nTotal decisions: {summary.get('total_decisions', 0)}")
    print(f"By kind: {summary.get('by_kind', {})}")
    print(f"By error type: {summary.get('by_error_type', {})}")
    print(f"By signature: {summary.get('by_signature', {})}")
    print(f"Success rate: {summary.get('success_rate', 0):.1%}")

    # Get detailed trails
    analyzer = supervisor.get_analyzer()
    trails = analyzer.get_all_trails()

    if trails:
        print(f"\n--- First Decision Trail ---")
        trail = trails[0]
        print(f"Correlation ID: {trail.correlation_id}")
        print(f"Error: {trail.error_type} ({trail.error_severity})")
        print(f"Signature: {trail.signature_mode} (conf={trail.signature_confidence:.2f})")
        print(f"Decision: {trail.decision_kind}")
        print(f"Rationale: {trail.decision_rationale}")
        print(f"Actions: {trail.recovery_actions}")
        print(f"Success: {trail.recovery_success}")
        print(f"Duration: {trail.total_duration_ms:.0f}ms")

    return success


if __name__ == "__main__":
    run_instrumented_experiment()
