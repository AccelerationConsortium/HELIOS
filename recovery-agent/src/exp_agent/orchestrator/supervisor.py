import time
from typing import Literal

from ..core.types import Action, ExecutionState, HardwareError, Decision
from ..devices.simulated.heater import SimHeater
from ..executor.guarded_executor import GuardedExecutor
from ..recovery.recovery_agent import RecoveryAgent

class Supervisor:
    def __init__(self, target_temp: float, fault_mode: Literal["none", "random", "timeout", "overshoot", "sensor_fail"]):
        self.device = SimHeater(name="heater_1", fault_mode=fault_mode)
        self.executor = GuardedExecutor()
        self.recovery = RecoveryAgent()
        
        self.target_temp = target_temp
        self.state = ExecutionState(devices={self.device.name: self.device.read_state()})
        
        # Retry Budget
        self.retry_counts = {} 
        self.MAX_RETRIES = 3
        self.history = [] # For signature analysis

    # ... (run method updates below) ...



    def run(self):
        print("=== Supervisor Loop Started ===")
        print(f"Goal: Heat to {self.target_temp}°C")
        print(f"Postconditions enabled: checking device state using Polling (PostCheck) module")

        # Initial Plan
        # Note: We use the new DSL: "key ~= value +/- tol within X s"
        plan = [
            Action(
                name="set_temperature", 
                effect="write", 
                params={"temperature": self.target_temp}, 
                device=self.device.name,
                # "telemetry.temperature" needs to reach target... this takes time!
                # If we put a postcondition on the `set` action, it will block until reached.
                # Let's allow 20 steps (approx 10s if we tick fast? no, sleep is 0.5s)
                # Let's say within 20s.
                postconditions=[
                    f"telemetry.target == {self.target_temp}", 
                    f"telemetry.temperature ~= {self.target_temp} +/- 2.0 within 20s"
                ]
            ),
            # Wait is less needed now if we verify above, but can render "holding time"
            Action(name="wait", effect="write", params={"duration": 5}, device=self.device.name)
        ]
        
        step_index = 0
        max_steps = 50 
        current_step = 0
        
        while current_step < max_steps:
            current_step += 1
            print(f"\n--- Step {current_step} ---")
            
            # 1. Observe
            try:
                self.device.tick()
                dev_state = self.device.read_state()
                self.state.devices[self.device.name] = dev_state
                
                # History Tracking (Sliding Window)
                self.history.append(dev_state)
                if len(self.history) > 10: self.history.pop(0)
                
            except HardwareError as e:
                print(f"[Supervisor] Observation Error: {e.message}")
                if self._check_retry_budget(e):
                    decision = self.handle_error(e)
                else:
                    decision = Decision(kind="abort", rationale="Retry limit exceeded", actions=[Action(name="cool_down", effect="write")])

                if decision.kind == "abort":
                    self.perform_abort(decision)
                    break
                
                if decision.actions:
                     self.execute_recovery_actions(decision.actions)
                continue

            # 2. Check Plan Completion
            if step_index >= len(plan):
                # Double check stability
                current_temp = dev_state.telemetry.get("temperature", 0)
                print(f"Plan finished. Current Temp: {current_temp:.1f}")
                break

            # 3. Pick Next Action
            next_action = plan[step_index]
            
            # 4. Try Execute
            print(f"ActionProposed: {next_action.name} {next_action.params}")
            try:
                self.executor.execute(self.device, next_action, self.state)
                # Success
                self.retry_counts = {} 
                step_index += 1
            except HardwareError as e:
                print(f"ActionFailed: {e.message}")
                
                if not self._check_retry_budget(e):
                    print(f"*** Retry budget exceeded for {e.type} ***")
                    self.shutdown()
                    break

                decision = self.handle_error(e)
                print(f"RecoveryDecision: {decision.kind} -> {decision.rationale}")
                
                if decision.kind == "abort":
                    self.perform_abort(decision)
                    break
                
                # Execute recovery/mitigation actions (Guarded)
                self.execute_recovery_actions(decision.actions)

                if decision.kind == "skip":
                    step_index += 1
                elif decision.kind == "degrade":
                    # For MVP, assume the recovery fixed it or we just move on logic
                    # If we degraded, we assume the 'set_temp' logic is handled by recovery actions
                    step_index += 1
                elif decision.kind == "retry":
                    # If actions were empty, it means 'retry the original plan step'
                    # which happens naturally by NOT incrementing step_index.
                    print("Retrying current step...")
            
            time.sleep(0.5)

    def _check_retry_budget(self, error: HardwareError) -> bool:
        err_key = error.type
        current = self.retry_counts.get(err_key, 0)
        self.retry_counts[err_key] = current + 1
        return self.retry_counts[err_key] <= self.MAX_RETRIES

    def perform_abort(self, decision: Decision):
        print(f"!!! ABORTING RUN: {decision.rationale} !!!")
        # Run decision actions then shutdown
        self.execute_recovery_actions(decision.actions)
        self.shutdown()

    def shutdown(self):
        print("Performing Safe Shutdown Sequence...")
        actions = [
            Action(name="cool_down", effect="write", device=self.device.name, 
                   postconditions=["telemetry.heating == False", "status == idle"])
        ]
        
        for i, action in enumerate(actions):
            try:
                print(f"[Shutdown] Executing {action.name}...")
                self.executor.execute(self.device, action, self.state)
            except Exception as e:
                print(f"[Shutdown] Warning: Step {i} failed ({e}). Continuing best-effort.")
        
        # Terminal Invariant Check
        final_state = self.device.read_state()
        if final_state.telemetry.get("heating") is not False:
            print("CRITICAL: Shutdown failed to reach safe terminal state (Heating is ON)!")
            # In real life: notify operator, cut power via PDU
        else:
            print("Shutdown complete. Terminal state verified safe.")

    def handle_error(self, error: HardwareError) -> Decision:
        return self.recovery.decide(self.state.devices[self.device.name], error, history=self.history)

    def execute_recovery_actions(self, actions: list[Action]):
        if not actions: return
        print(f">>> Executing {len(actions)} Recovery Actions (Guarded)...")
        for action in actions:
            try:
                self.executor.execute(self.device, action, self.state)
            except HardwareError as e:
                print(f"CRITICAL: Recovery action failed! {e.message}")
                # If recovery fails, we might just stop modifying simulation and let loop handle it, 
                # or consume 'retry' budget in next observation? 
                # For safety, let's stop this batch.
                break
