import time
from typing import List
from ..core.types import DeviceState, HardwareError, Action, Severity
from ..core.predicates import ParsedPredicate

class PostCheck:
    def __init__(self, device):
        self.device = device

    def verify(self, action: Action):
        if not action.postconditions:
            return

        print(f"[PostCheck] Verifying {len(action.postconditions)} conditions for {action.name}")
        
        # Parse all conditions
        conditions = [ParsedPredicate(p) for p in action.postconditions]
        
        # Group by max timeout needed
        # Logic: We poll until ALL satisfied or global/local timeout
        start_time = time.time()
        
        # Determine strict timeout
        # If any condition has a specific 'within X', use the max of those, else default 0 (immediate)
        # However, if 'set_temperature' is called, physically it takes time. 
        # But 'set_temperature' action usually returns when command is sent.
        # Stabilizing is often a separate 'wait' or 'monitor' step.
        # IF the user puts "temp ~= 120 within 60s" on the set_temp action, they expect blocking verification.
        
        max_timeout = 0.0
        for c in conditions:
            if c.timeout > max_timeout:
                max_timeout = c.timeout
        
        # Default poll interval
        poll_interval = 0.5

        trace = [] # Trace of (time, telemetry)
        
        while True:
            # Read fresh state
            current_state = self.device.read_state()
            elapsed = time.time() - start_time
            trace.append({
                "time": f"{elapsed:.1f}s", 
                "temp": current_state.telemetry.get("temperature"),
                "status": current_state.status
            })
            
            # Check all
            all_passed = True
            failed_reasons = []
            
            for cond in conditions:
                if not cond.check(current_state):
                    all_passed = False
                    failed_reasons.append(f"{cond.condition.describe()} (Got: {cond.condition._get_value(current_state, cond.condition.key)})")
            
            # Check Invariants / Safety (Preempts timeout)
            try:
                self.check_safety_invariants(current_state)
            except HardwareError as e:
                # Re-raise immediately to break the wait
                e.context["polling_trace"] = trace[-5:] # Last 5
                raise e

            if all_passed:
                print(f"[PostCheck] All conditions met.")
                return
            
            if elapsed >= max_timeout:
                # Timeout occurred
                raise HardwareError(
                    device=self.device.name,
                    type="postcondition_failed",
                    severity="high",
                    message=f"Postconditions failed after {max_timeout}s: {'; '.join(failed_reasons)}",
                    action=action.name,
                    context={
                        "state": str(current_state), 
                        "elapsed": elapsed,
                        "polling_trace": trace[-10:] # Last 10
                    }
                )
            
            # Wait and retry
            time.sleep(poll_interval)

    def check_safety_invariants(self, state: DeviceState):
        """Hardcoded safety checks that must hold true at all times."""
        # TODO: Move these to a configurable policy or Invariant class
        
        # 1. Max Temp invariant
        temp = state.telemetry.get("temperature")
        if temp is not None and temp > 130.0:
             raise HardwareError(
                device=state.name,
                type="safety_violation",
                severity="high",
                message=f"Safety invariant violated: Temp {temp:.1f} > 130.0",
                when=str(time.time()),
                context={"temperature": temp, "invariant": "max_temp_130"}
            )
            
        # 2. Sensor health invariant
        # If telemetry is empty or has error flags
        # (Simplified: assuming if we got here, read_state worked, but we check values)
        if temp == -999:
             raise HardwareError(
                device=state.name,
                type="sensor_fail",
                severity="high",
                message="Safety invariant violated: Sensor invalid",
                when=str(time.time())
            )
