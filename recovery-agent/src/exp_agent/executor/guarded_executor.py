"""
GuardedExecutor — 3-layer execution with safety integration and pre-flight validation.

Provides guardrails around action execution:
1. Pre-check: Preconditions validation
2. Safety check: Constraint verification (including SafetyPacket constraints)
3. Post-verify: Postcondition confirmation

Phase 2 Safety Integration:
- check_safety() accepts an optional SafetyPacket for chemical safety constraints
- Chemical safety violations raise HardwareError with appropriate type
- Integrates with the safety checker for comprehensive action validation

Execution Harness Hardening (this revision):
- Two-phase validation: pre-flight (no side effects) + execution.
- execute(..., validate_only=True) returns a PreflightResult
  ({status, violations, warnings}) WITHOUT dispatching to hardware.
  This catches the majority of failures before any costly/irreversible
  hardware commit, and enables dry-run and safe retry budgeting upstream
  (e.g. circuit breaker integration).
- Structured, JSON-compatible logging replaces ad-hoc print() calls.
- Dependency injection (logger, safety-check callable) keeps the class
  unit-testable with no global state.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Action, DeviceState, ExecutionState, HardwareError

if TYPE_CHECKING:
    from ..core.safety_types import ActionSafetyCheck, SafetyPacket


# ---------------------------------------------------------------------------
# Protocols & dependency-injection types
# ---------------------------------------------------------------------------


class Device(Protocol):
    """Minimal device contract the executor depends on.

    Declared as a Protocol so tests can inject fakes without inheritance.
    """

    name: str

    def execute(self, action: Action) -> Any: ...

    def read_state(self) -> DeviceState: ...


# Signature of the injectable safety-check function (matches
# exp_agent.safety.checker.check_action_safety).
SafetyCheckFn = Callable[[Action, "SafetyPacket", DeviceState], "ActionSafetyCheck"]


# ---------------------------------------------------------------------------
# Pre-flight result model
# ---------------------------------------------------------------------------

ValidationStatus = Literal["ok", "blocked", "requires_human"]
ViolationStage = Literal["precondition", "safety_check"]


class Violation(BaseModel):
    """A single blocking issue found during pre-flight validation."""

    model_config = ConfigDict(frozen=False)

    stage: ViolationStage
    type: str
    severity: str = "high"
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class Warning_(BaseModel):
    """A non-blocking advisory surfaced during pre-flight validation."""

    model_config = ConfigDict(frozen=False)

    stage: ViolationStage
    type: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class PreflightResult(BaseModel):
    """Outcome of a side-effect-free validation pass over an action.

    status:
        - "ok"             → safe to dispatch.
        - "blocked"        → at least one hard violation; must not dispatch.
        - "requires_human" → no hard block, but human verification flagged.
    """

    model_config = ConfigDict(frozen=False)

    status: ValidationStatus
    action: str
    device: Optional[str] = None
    violations: List[Violation] = Field(default_factory=list)
    warnings: List[Warning_] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class GuardedExecutor:
    """Executes actions with pre/safety/post guardrails.

    Two-phase validation:
        ``preflight(...)`` (or ``execute(..., validate_only=True)``) runs the
        pre-check and safety layers and returns a :class:`PreflightResult`
        WITHOUT touching hardware. ``execute(...)`` runs the full pipeline and
        dispatches to the device, raising :class:`HardwareError` on the first
        hard failure (preserving the original fail-fast contract).

    Example:
        ```python
        executor = GuardedExecutor()

        # Dry-run / pre-flight (no side effects):
        result = executor.execute(device, action, state, validate_only=True)
        if result.ok:
            executor.execute(device, action, state, safety_packet=packet)
        ```
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        safety_check_fn: Optional[SafetyCheckFn] = None,
        max_safe_temperature: float = 130.0,
    ) -> None:
        """Args:
        logger: Injectable structured logger. Defaults to a module logger.
        safety_check_fn: Injectable SafetyPacket checker. Defaults to
            ``exp_agent.safety.checker.check_action_safety`` (imported lazily
            to avoid a hard import cycle and to keep tests light).
        max_safe_temperature: Built-in hard temperature ceiling (°C).
        """
        self._log = logger or logging.getLogger("exp_agent.executor.guarded")
        self._safety_check_fn = safety_check_fn
        self._max_safe_temperature = max_safe_temperature

    # -- safety-check resolution -------------------------------------------

    def _resolve_safety_check_fn(self) -> SafetyCheckFn:
        if self._safety_check_fn is not None:
            return self._safety_check_fn
        from ..safety.checker import check_action_safety

        self._safety_check_fn = check_action_safety
        return self._safety_check_fn

    # -- precondition layer -------------------------------------------------

    def check_preconditions(
        self, state: ExecutionState, action: Action
    ) -> List[Violation]:
        """Validate that the action's declared preconditions hold.

        Returns a list of Violations (empty == passed). This collect-style
        return lets the pre-flight phase aggregate all problems; the execution
        phase converts the first one into a raised HardwareError.
        """
        violations: List[Violation] = []
        if not action.preconditions:
            return violations

        from ..core.predicates import ParsedPredicate

        device_state = self._resolve_device_state(state, action)
        if device_state is None:
            violations.append(
                Violation(
                    stage="precondition",
                    type="missing_device_state",
                    message=(
                        f"No device state available to verify preconditions "
                        f"for action '{action.name}'"
                    ),
                    context={"requested_device": action.device},
                )
            )
            return violations

        for raw in action.preconditions:
            predicate = ParsedPredicate(raw)
            if not predicate.check(device_state):
                got = predicate.condition._get_value(
                    device_state, predicate.condition.key
                )
                violations.append(
                    Violation(
                        stage="precondition",
                        type="precondition_failed",
                        message=(
                            f"Precondition not met: {predicate.condition.describe()} "
                            f"(got: {got})"
                        ),
                        context={
                            "predicate": raw,
                            "device": device_state.name,
                            "observed": got,
                        },
                    )
                )
        return violations

    # -- safety layer -------------------------------------------------------

    def check_safety(
        self,
        state: ExecutionState,
        action: Action,
        safety_packet: Optional["SafetyPacket"] = None,
    ) -> tuple[List[Violation], List[Warning_]]:
        """Evaluate the action against the safety envelope (no side effects).

        Performs two kinds of checks:
        1. Built-in device safety limits (e.g. max temperature ceiling).
        2. Chemical safety constraints from a SafetyPacket (if provided).

        Returns ``(violations, warnings)``. The execution path turns the first
        violation into a raised HardwareError; the pre-flight path returns them
        verbatim. ``require_human`` outcomes are surfaced as warnings (matching
        the original non-blocking behaviour).
        """
        violations: List[Violation] = []
        warnings: List[Warning_] = []

        # --- 1. Built-in device safety limits ---
        if action.name == "set_temperature":
            temp = action.params.get("temperature", 0)
            if temp > self._max_safe_temperature:
                violations.append(
                    Violation(
                        stage="safety_check",
                        type="safety_violation",
                        severity="high",
                        message=(
                            f"Proposed temperature {temp} exceeds safety limit "
                            f"{self._max_safe_temperature}"
                        ),
                        context={
                            "proposed_temperature": temp,
                            "limit": self._max_safe_temperature,
                            "when": "pre_check",
                        },
                    )
                )

        # --- 2. Chemical safety constraints from SafetyPacket ---
        if safety_packet is not None:
            packet_violations, packet_warnings = self._check_safety_packet_constraints(
                state, action, safety_packet
            )
            violations.extend(packet_violations)
            warnings.extend(packet_warnings)

        return violations, warnings

    def _check_safety_packet_constraints(
        self,
        state: ExecutionState,
        action: Action,
        packet: "SafetyPacket",
    ) -> tuple[List[Violation], List[Warning_]]:
        """Check action against SafetyPacket constraints.

        Implements the runtime safety overlay from plan.md section 3(2).
        Returns collected violations/warnings rather than raising, so the
        same logic serves both the pre-flight and execution phases.
        """
        violations: List[Violation] = []
        warnings: List[Warning_] = []

        device_state = self._resolve_device_state(state, action)
        if device_state is None:
            # No telemetry to validate against; skip (parity with prior code).
            return violations, warnings

        check = self._resolve_safety_check_fn()
        result = check(action, packet, device_state)

        if result.result == "block":
            error_type = "safety_violation"
            for t in result.violated_thresholds:
                if t.severity == "critical":
                    error_type = "chemical_threshold_exceeded"
                    break

            message_parts = [result.rationale]
            if result.violated_constraints:
                constraints_desc = [
                    c.description for c in result.violated_constraints[:2]
                ]
                message_parts.append(
                    f"Violated constraints: {', '.join(constraints_desc)}"
                )
            if result.alternative_actions:
                message_parts.append(
                    f"Alternatives: {', '.join(result.alternative_actions[:3])}"
                )

            violations.append(
                Violation(
                    stage="safety_check",
                    type=error_type,
                    severity="high",
                    message=" | ".join(message_parts),
                    context={
                        "violated_constraints": [
                            c.type for c in result.violated_constraints
                        ],
                        "violated_thresholds": [
                            t.variable for t in result.violated_thresholds
                        ],
                        "alternative_actions": result.alternative_actions,
                        "triggered_playbooks": result.triggered_playbooks,
                    },
                )
            )

        elif result.result == "require_human":
            warnings.append(
                Warning_(
                    stage="safety_check",
                    type="requires_human_verification",
                    message=(
                        f"Action {action.name} requires human verification: "
                        f"{result.rationale}"
                    ),
                    context={"triggered_playbooks": result.triggered_playbooks},
                )
            )

        return violations, warnings

    # -- pre-flight (phase 1, no side effects) ------------------------------

    def preflight(
        self,
        device: Device,
        action: Action,
        state: ExecutionState,
        safety_packet: Optional["SafetyPacket"] = None,
    ) -> PreflightResult:
        """Run pre-check + safety layers without dispatching to hardware.

        Returns a PreflightResult ({status, violations, warnings}). This is the
        side-effect-free half of the two-phase validation and is what backs
        ``execute(..., validate_only=True)``.
        """
        device_name = action.device or getattr(device, "name", None)
        self._log.info(
            "preflight.start",
            extra={
                "event": "preflight.start",
                "action": action.name,
                "params": action.params,
                "device": device_name,
                "safety_packet": safety_packet is not None,
            },
        )

        violations: List[Violation] = []
        warnings: List[Warning_] = []

        violations.extend(self.check_preconditions(state, action))

        safety_violations, safety_warnings = self.check_safety(
            state, action, safety_packet
        )
        violations.extend(safety_violations)
        warnings.extend(safety_warnings)

        if violations:
            status: ValidationStatus = "blocked"
        elif warnings:
            status = "requires_human"
        else:
            status = "ok"

        result = PreflightResult(
            status=status,
            action=action.name,
            device=device_name,
            violations=violations,
            warnings=warnings,
        )

        self._log.info(
            "preflight.complete",
            extra={
                "event": "preflight.complete",
                "action": action.name,
                "device": device_name,
                "status": status,
                "violation_count": len(violations),
                "warning_count": len(warnings),
                "violations": [v.model_dump() for v in violations],
                "warnings": [w.model_dump() for w in warnings],
            },
        )
        return result

    # -- post-condition layer (phase 2 only) --------------------------------

    def verify_postconditions(
        self, state: ExecutionState, action: Action, device: Device
    ) -> None:
        """Verify postconditions after action execution."""
        from .post_check import PostCheck

        checker = PostCheck(device)
        checker.verify(action)

        # Update state after check
        state.devices[device.name] = device.read_state()

    # -- full pipeline ------------------------------------------------------

    def execute(
        self,
        device: Device,
        action: Action,
        state: ExecutionState,
        safety_packet: Optional["SafetyPacket"] = None,
        validate_only: bool = False,
    ) -> Optional[PreflightResult]:
        """Execute an action on a device with guardrails.

        Args:
            device: The device to execute on.
            action: The action to execute.
            state: Current execution state.
            safety_packet: Optional SafetyPacket for chemical safety validation.
            validate_only: When True, run only the pre-flight (pre-check +
                safety) layers and return a PreflightResult WITHOUT dispatching
                to hardware or running post-conditions. Enables dry-run and
                cheap pre-commit validation.

        Returns:
            PreflightResult when ``validate_only=True``; otherwise ``None`` on
            successful execution.

        Raises:
            HardwareError: In execution mode, if any guardrail check fails or
                the device dispatch errors.
        """
        if validate_only:
            return self.preflight(device, action, state, safety_packet)

        device_name = getattr(device, "name", action.device or "executor")
        self._log.info(
            "execute.verify",
            extra={
                "event": "execute.verify",
                "action": action.name,
                "params": action.params,
                "device": device_name,
            },
        )

        # 1. Pre-execution checks (reuse the pre-flight result for fail-fast).
        preflight = self.preflight(device, action, state, safety_packet)
        if preflight.violations:
            first = preflight.violations[0]
            self._log.error(
                "execute.blocked",
                extra={
                    "event": "execute.blocked",
                    "action": action.name,
                    "device": device_name,
                    "violation": first.model_dump(),
                },
            )
            raise HardwareError(
                device=action.device or device_name,
                type=first.type,
                severity="high",
                message=first.message,
                when=first.stage,
                action=action.name,
                context=first.context,
            )
        for warning in preflight.warnings:
            self._log.warning(
                "execute.warning",
                extra={
                    "event": "execute.warning",
                    "action": action.name,
                    "device": device_name,
                    "warning": warning.model_dump(),
                },
            )

        # 2. Execution
        self._log.info(
            "execute.dispatch",
            extra={
                "event": "execute.dispatch",
                "action": action.name,
                "device": device_name,
            },
        )
        try:
            device.execute(action)
        except HardwareError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap driver errors uniformly
            self._log.error(
                "execute.driver_error",
                extra={
                    "event": "execute.driver_error",
                    "action": action.name,
                    "device": device_name,
                    "error": str(exc),
                },
            )
            raise HardwareError(
                device=device_name,
                type="driver_error",
                severity="high",
                message=str(exc),
                action=action.name,
                context={"original_error": str(exc)},
            ) from exc

        # 3. Post-execution checks (reads fresh device state)
        self.verify_postconditions(state, action, device)
        self._log.info(
            "execute.complete",
            extra={
                "event": "execute.complete",
                "action": action.name,
                "device": device_name,
            },
        )
        return None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_device_state(
        state: ExecutionState, action: Action
    ) -> Optional[DeviceState]:
        """Resolve the relevant DeviceState for an action.

        Prefers the action's named device; falls back to the first device in
        state (parity with the original packet-constraint logic).
        """
        if action.device and action.device in state.devices:
            return state.devices[action.device]
        if state.devices:
            return next(iter(state.devices.values()))
        return None
