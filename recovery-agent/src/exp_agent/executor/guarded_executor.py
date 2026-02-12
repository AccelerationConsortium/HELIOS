"""
GuardedExecutor — 3-layer execution with safety integration.

Provides guardrails around action execution:
1. Pre-check: Preconditions validation
2. Safety check: Constraint verification (including SafetyPacket constraints)
3. Post-verify: Postcondition confirmation

Phase 2 Safety Integration:
- check_safety() now accepts optional SafetyPacket for chemical safety constraints
- Chemical safety violations raise HardwareError with appropriate type
- Integrates with safety checker for comprehensive action validation
"""

from typing import Optional, TYPE_CHECKING
from ..core.types import Action, ExecutionState, HardwareError, DeviceState

if TYPE_CHECKING:
    from ..core.safety_types import SafetyPacket


class GuardedExecutor:
    """Executes actions with pre/safety/post guardrails.

    Safety Integration:
        When a SafetyPacket is provided to check_safety(), the executor will:
        1. Check action against SafetyPacket constraints
        2. Verify current telemetry against safety thresholds
        3. Block actions that would violate chemical safety rules

    Example:
        ```python
        executor = GuardedExecutor()
        executor.execute(device, action, state, safety_packet=packet)
        ```
    """

    def __init__(self):
        pass

    def check_preconditions(self, state: ExecutionState, action: Action):
        """Check if preconditions for action are met.

        MVP: check simplistic predicates
        e.g. "is_idle" or "temp < 50" if encoded in strings
        For now, we assume simple validity
        """
        pass

    def check_safety(
        self,
        state: ExecutionState,
        action: Action,
        safety_packet: Optional["SafetyPacket"] = None
    ):
        """Check if action violates safety envelope.

        This method performs two types of safety checks:
        1. Built-in device safety limits (e.g., max temperature 130°C)
        2. Chemical safety constraints from SafetyPacket (if provided)

        Args:
            state: Current execution state with device telemetry.
            action: The action to validate.
            safety_packet: Optional SafetyPacket from pre-flight assessment.

        Raises:
            HardwareError: If safety violation detected.
                - type="safety_violation" for device limits
                - type="chemical_threshold_exceeded" for chemical safety
        """
        # --- 1. Built-in device safety limits ---
        if action.name == "set_temperature":
            temp = action.params.get("temperature", 0)
            if temp > 130:
                raise HardwareError(
                    device="executor",
                    type="safety_violation",
                    severity="high",
                    message=f"Proposed temperature {temp} exceeds safety limit 130",
                    when="pre_check",
                    action=action.name
                )

        # --- 2. Chemical safety constraints from SafetyPacket ---
        if safety_packet is not None:
            self._check_safety_packet_constraints(state, action, safety_packet)

    def _check_safety_packet_constraints(
        self,
        state: ExecutionState,
        action: Action,
        packet: "SafetyPacket"
    ):
        """Check action against SafetyPacket constraints.

        This implements the runtime safety overlay from plan.md section 3(2).
        """
        from ..safety.checker import check_action_safety

        # Get device state for telemetry
        device_state = None
        if action.device and action.device in state.devices:
            device_state = state.devices[action.device]
        else:
            # Use first device if action doesn't specify
            if state.devices:
                device_state = next(iter(state.devices.values()))

        if device_state is None:
            # No device state available, skip check
            return

        # Run safety check
        result = check_action_safety(action, packet, device_state)

        if result.result == "block":
            # Determine error type based on what was violated
            error_type = "safety_violation"
            if result.violated_thresholds:
                # Check if any threshold is chemical-safety related
                for t in result.violated_thresholds:
                    if t.severity == "critical":
                        error_type = "chemical_threshold_exceeded"
                        break

            # Build detailed message
            message_parts = [result.rationale]
            if result.violated_constraints:
                constraints_desc = [c.description for c in result.violated_constraints[:2]]
                message_parts.append(f"Violated constraints: {', '.join(constraints_desc)}")
            if result.alternative_actions:
                message_parts.append(f"Alternatives: {', '.join(result.alternative_actions[:3])}")

            raise HardwareError(
                device=action.device or "executor",
                type=error_type,
                severity="high",
                message=" | ".join(message_parts),
                when="safety_check",
                action=action.name,
                context={
                    "violated_constraints": [c.type for c in result.violated_constraints],
                    "violated_thresholds": [t.variable for t in result.violated_thresholds],
                    "alternative_actions": result.alternative_actions,
                    "triggered_playbooks": result.triggered_playbooks,
                }
            )

        elif result.result == "require_human":
            # Log warning but don't block - human intervention flagged
            print(f"  ⚠ SAFETY WARNING: Action {action.name} requires human verification")
            print(f"    Reason: {result.rationale}")
            if result.triggered_playbooks:
                print(f"    Triggered scenarios: {result.triggered_playbooks}")

    def verify_postconditions(self, state: ExecutionState, action: Action, device):
        """Verify postconditions after action execution."""
        from .post_check import PostCheck
        checker = PostCheck(device)
        checker.verify(action)

        # Update state after check
        state.devices[device.name] = device.read_state()

    def _check_predicate(self, state, condition):
        """Legacy/Internal removed in favor of PostCheck."""
        pass

    def execute(
        self,
        device,
        action: Action,
        state: ExecutionState,
        safety_packet: Optional["SafetyPacket"] = None
    ):
        """Execute an action on a device with guardrails.

        Args:
            device: The device to execute on.
            action: The action to execute.
            state: Current execution state.
            safety_packet: Optional SafetyPacket for chemical safety validation.

        Raises:
            HardwareError: If any guardrail check fails.
        """
        print(f"[Executor] Verifying action: {action.name} {action.params}...")

        # 1. Pre-execution checks
        self.check_preconditions(state, action)
        self.check_safety(state, action, safety_packet)

        # 2. Execution
        print(f"[Executor] Executing on device {device.name}...")
        try:
            device.execute(action)
        except Exception as e:
            if isinstance(e, HardwareError):
                raise e
            else:
                raise HardwareError(
                    device=device.name,
                    type="driver_error",
                    severity="high",
                    message=str(e),
                    action=action.name,
                    context={"original_error": str(e)}
                )

        # 3. Post-execution checks
        # Must pass device to read fresh state
        self.verify_postconditions(state, action, device)
        print(f"[Executor] Action {action.name} completed and verified.")
