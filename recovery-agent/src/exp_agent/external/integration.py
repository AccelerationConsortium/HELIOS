"""
Integration layer connecting external executors with the agent's Action system.

This module provides:
1. ExternalAction - Extended action type for external operations
2. ExternalDevice - A "device" that wraps external tools/APIs
3. ActionRouter - Routes actions to appropriate executors

Examples:
    # Create an external device for CLI tools
    cli_device = ExternalDevice(
        name="lab_tools",
        executors={
            "cli": CLIExecutor(allowed_commands=["labctl", "spectra-cli"]),
        }
    )

    # Execute a CLI action through the device
    action = ExternalAction(
        name="run_analysis",
        effect="write",
        executor_type="cli",
        external_config={
            "command": "labctl",
            "args": ["analyze", "--sample", "S001"]
        }
    )
    cli_device.execute(action)

    # Create an API-backed device
    api_device = ExternalDevice(
        name="lims_api",
        executors={
            "api": APIExecutor(base_url="https://lims.lab.com/api/v1")
        }
    )

    # Query sample status via API
    action = ExternalAction(
        name="get_sample",
        effect="read",
        executor_type="api",
        external_config={
            "url": "/samples/S001",
            "method": "GET"
        }
    )
    result = api_device.execute(action)
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..core.types import Action, DeviceState, HardwareError
from ..devices.base import Device
from .base import ExternalExecutor, ExecutionResult, ExecutionStatus, ExecutionError
from .cli import CLIExecutor, CLIAction
from .script import ScriptExecutor, ScriptAction, ScriptType
from .api import APIExecutor, APIAction, HTTPMethod


@dataclass
class ExternalAction:
    """Extended action type for external operations.

    Bridges the agent's Action model with external executors.
    """
    name: str
    effect: str  # "read" or "write"
    executor_type: str  # "cli", "script", or "api"
    external_config: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retries: int = 0

    def to_core_action(self) -> Action:
        """Convert to core Action type."""
        return Action(
            name=self.name,
            effect=self.effect,
            params={
                **self.params,
                "_external": {
                    "executor_type": self.executor_type,
                    "config": self.external_config,
                }
            }
        )

    @classmethod
    def from_core_action(cls, action: Action) -> Optional["ExternalAction"]:
        """Create from core Action if it has external config."""
        external = action.params.get("_external")
        if not external:
            return None

        params = {k: v for k, v in action.params.items() if k != "_external"}
        return cls(
            name=action.name,
            effect=action.effect,
            executor_type=external["executor_type"],
            external_config=external.get("config", {}),
            params=params,
        )


class ActionRouter:
    """Routes actions to appropriate external executors.

    Converts ExternalAction to executor-specific action types
    and handles the execution.
    """

    def __init__(self, executors: Dict[str, ExternalExecutor]):
        self.executors = executors

    def _build_cli_action(self, action: ExternalAction) -> CLIAction:
        """Build CLIAction from ExternalAction config."""
        config = action.external_config
        return CLIAction(
            name=action.name,
            command=config.get("command", ""),
            args=config.get("args", []),
            env=config.get("env", {}),
            working_dir=config.get("working_dir"),
            stdin=config.get("stdin"),
            timeout_seconds=action.timeout_seconds,
            retries=action.retries,
        )

    def _build_script_action(self, action: ExternalAction) -> ScriptAction:
        """Build ScriptAction from ExternalAction config."""
        config = action.external_config
        script_type = config.get("script_type", "python")
        return ScriptAction(
            name=action.name,
            script_path=config.get("script_path"),
            script_type=ScriptType(script_type),
            inline_code=config.get("inline_code"),
            args=config.get("args", []),
            env=config.get("env", {}),
            working_dir=config.get("working_dir"),
            timeout_seconds=action.timeout_seconds,
            retries=action.retries,
        )

    def _build_api_action(self, action: ExternalAction) -> APIAction:
        """Build APIAction from ExternalAction config."""
        config = action.external_config
        method = config.get("method", "GET")
        return APIAction(
            name=action.name,
            url=config.get("url", ""),
            method=HTTPMethod(method),
            headers=config.get("headers", {}),
            query_params=config.get("query_params", {}),
            json_body=config.get("json_body"),
            form_data=config.get("form_data"),
            graphql_query=config.get("graphql_query"),
            graphql_variables=config.get("graphql_variables"),
            timeout_seconds=action.timeout_seconds,
            retries=action.retries,
        )

    async def route(self, action: ExternalAction) -> ExecutionResult:
        """Route action to appropriate executor.

        Args:
            action: The external action to execute

        Returns:
            ExecutionResult from the executor

        Raises:
            ValueError: If executor type is unknown
            ExecutionError: If execution fails
        """
        executor = self.executors.get(action.executor_type)
        if not executor:
            raise ValueError(
                f"Unknown executor type: {action.executor_type}. "
                f"Available: {list(self.executors.keys())}"
            )

        # Build appropriate action type
        if action.executor_type == "cli":
            typed_action = self._build_cli_action(action)
        elif action.executor_type == "script":
            typed_action = self._build_script_action(action)
        elif action.executor_type == "api":
            typed_action = self._build_api_action(action)
        else:
            raise ValueError(f"Unsupported executor type: {action.executor_type}")

        return await executor.execute(typed_action)


class ExternalDevice(Device):
    """A device that wraps external tools, scripts, and APIs.

    This allows external operations to be used within the agent's
    standard device/action execution model.

    The device can have multiple executors for different types
    of external operations.
    """

    def __init__(
        self,
        name: str,
        executors: Optional[Dict[str, ExternalExecutor]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        """Initialize external device.

        Args:
            name: Device name
            executors: Dict of executor_type -> executor instance
            initial_state: Initial telemetry state
        """
        super().__init__(name)
        self.executors = executors or {}
        self.router = ActionRouter(self.executors)
        self._state = initial_state or {}
        self._last_result: Optional[ExecutionResult] = None
        self._healthy = True

    def add_executor(self, name: str, executor: ExternalExecutor) -> None:
        """Add an executor to the device."""
        self.executors[name] = executor
        self.router = ActionRouter(self.executors)

    def read_state(self) -> DeviceState:
        """Read current device state."""
        telemetry = {
            **self._state,
            "healthy": self._healthy,
            "executor_count": len(self.executors),
        }
        if self._last_result:
            telemetry["last_status"] = self._last_result.status.value
            telemetry["last_duration_ms"] = self._last_result.duration_ms

        return DeviceState(
            name=self.name,
            status="idle" if self._healthy else "error",
            telemetry=telemetry,
        )

    def execute(self, action: Action) -> None:
        """Execute an action on this device.

        Args:
            action: The action to execute (must be ExternalAction or
                    Action with _external params)

        Raises:
            HardwareError: If execution fails
        """
        # Convert to ExternalAction if needed
        if isinstance(action, ExternalAction):
            ext_action = action
        else:
            ext_action = ExternalAction.from_core_action(action)
            if not ext_action:
                raise HardwareError(
                    device=self.name,
                    type="invalid_action",
                    severity="high",
                    message=f"Action {action.name} is not an external action",
                    action=action.name,
                )

        # Execute asynchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self.router.route(ext_action))
            self._last_result = result

            if not result.success:
                self._healthy = False
                raise HardwareError(
                    device=self.name,
                    type="external_error",
                    severity="medium",
                    message=result.error or "External execution failed",
                    action=action.name,
                    context={
                        "status": result.status.value,
                        "exit_code": result.exit_code,
                        "output": str(result.output)[:500],
                    }
                )

            # Update state with result
            self._state["last_output"] = result.output
            self._healthy = True

        except ExecutionError as e:
            self._healthy = False
            raise HardwareError(
                device=self.name,
                type="execution_error",
                severity="high",
                message=str(e),
                action=action.name,
                context={"result": e.result.to_dict() if e.result else None}
            )

    def health(self) -> bool:
        """Check device health."""
        return self._healthy


def create_cli_device(
    name: str,
    allowed_commands: Optional[List[str]] = None,
    working_dir: Optional[str] = None,
) -> ExternalDevice:
    """Create a device for CLI tool execution.

    Args:
        name: Device name
        allowed_commands: Whitelist of allowed commands
        working_dir: Default working directory

    Returns:
        ExternalDevice configured for CLI execution
    """
    return ExternalDevice(
        name=name,
        executors={
            "cli": CLIExecutor(
                allowed_commands=allowed_commands,
                default_working_dir=working_dir,
            )
        }
    )


def create_script_device(
    name: str,
    python_venv: Optional[str] = None,
    allowed_paths: Optional[List[str]] = None,
) -> ExternalDevice:
    """Create a device for script execution.

    Args:
        name: Device name
        python_venv: Path to Python virtual environment
        allowed_paths: Whitelist of allowed script directories

    Returns:
        ExternalDevice configured for script execution
    """
    return ExternalDevice(
        name=name,
        executors={
            "script": ScriptExecutor(
                python_venv=python_venv,
                allowed_paths=allowed_paths,
            )
        }
    )


def create_api_device(
    name: str,
    base_url: str,
    default_headers: Optional[Dict[str, str]] = None,
    allowed_domains: Optional[List[str]] = None,
) -> ExternalDevice:
    """Create a device for API calls.

    Args:
        name: Device name
        base_url: Base URL for API
        default_headers: Default headers (e.g., auth)
        allowed_domains: Whitelist of allowed domains

    Returns:
        ExternalDevice configured for API calls
    """
    return ExternalDevice(
        name=name,
        executors={
            "api": APIExecutor(
                base_url=base_url,
                default_headers=default_headers,
                allowed_domains=allowed_domains,
            )
        }
    )


def create_hybrid_device(
    name: str,
    cli_config: Optional[Dict[str, Any]] = None,
    script_config: Optional[Dict[str, Any]] = None,
    api_config: Optional[Dict[str, Any]] = None,
) -> ExternalDevice:
    """Create a device with multiple executor types.

    Args:
        name: Device name
        cli_config: CLIExecutor configuration
        script_config: ScriptExecutor configuration
        api_config: APIExecutor configuration

    Returns:
        ExternalDevice with multiple executors
    """
    executors = {}

    if cli_config is not None:
        executors["cli"] = CLIExecutor(**cli_config)

    if script_config is not None:
        executors["script"] = ScriptExecutor(**script_config)

    if api_config is not None:
        executors["api"] = APIExecutor(**api_config)

    return ExternalDevice(name=name, executors=executors)
