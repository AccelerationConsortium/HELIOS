"""
CLI tool executor for running external command-line tools.

Examples:
    # Run a simple command
    executor = CLIExecutor()
    result = await executor.execute(CLIAction(
        name="list_files",
        command="ls",
        args=["-la", "/tmp"]
    ))

    # Run with environment variables
    result = await executor.execute(CLIAction(
        name="run_analysis",
        command="python",
        args=["analyze.py", "--input", "data.csv"],
        env={"PYTHONPATH": "/custom/path"},
        working_dir="/project"
    ))

    # Pipe input to command
    result = await executor.execute(CLIAction(
        name="filter_data",
        command="grep",
        args=["pattern"],
        stdin="line1\\nline2\\npattern_match\\n"
    ))
"""
import asyncio
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import (
    BaseAction,
    ExternalExecutor,
    ExecutionResult,
    ExecutionStatus,
    ExecutionError,
)


@dataclass
class CLIAction(BaseAction):
    """Action for CLI tool execution.

    Attributes:
        command: The command to execute (can be full path or command name)
        args: Command arguments
        env: Additional environment variables (merged with current env)
        working_dir: Working directory for execution
        stdin: Input to send to command's stdin
        capture_stderr: Whether to capture stderr separately
        shell: Whether to run through shell (use with caution)
    """
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    stdin: Optional[str] = None
    capture_stderr: bool = True
    shell: bool = False

    def validate(self) -> None:
        super().validate()
        if not self.command:
            raise ValueError("Command is required")

    @property
    def full_command(self) -> str:
        """Full command string for logging."""
        if self.args:
            return f"{self.command} {' '.join(shlex.quote(a) for a in self.args)}"
        return self.command


class CLIExecutor(ExternalExecutor):
    """Executor for CLI tools using asyncio subprocess.

    Features:
    - Async subprocess execution
    - Timeout handling with process termination
    - Environment variable management
    - Working directory support
    - Stdin piping
    - Stdout/stderr capture

    Security considerations:
    - Commands are NOT run through shell by default
    - Use allowed_commands to whitelist executables
    - Paths are validated before execution
    """

    def __init__(
        self,
        name: str = "cli",
        allowed_commands: Optional[List[str]] = None,
        default_env: Optional[Dict[str, str]] = None,
        default_working_dir: Optional[str] = None,
    ):
        """Initialize CLI executor.

        Args:
            name: Executor name for logging
            allowed_commands: Optional whitelist of allowed commands
            default_env: Default environment variables
            default_working_dir: Default working directory
        """
        super().__init__(name)
        self.allowed_commands = allowed_commands
        self.default_env = default_env or {}
        self.default_working_dir = default_working_dir

    def _resolve_command(self, command: str) -> str:
        """Resolve command to full path if possible."""
        # If it's already a path, validate it
        if "/" in command or "\\" in command:
            path = Path(command)
            if not path.exists():
                raise ExecutionError(f"Command not found: {command}")
            if not os.access(path, os.X_OK):
                raise ExecutionError(f"Command not executable: {command}")
            return str(path.resolve())

        # Try to find in PATH
        resolved = shutil.which(command)
        if resolved:
            return resolved

        # Return as-is, let subprocess handle it
        return command

    def _check_allowed(self, command: str) -> None:
        """Check if command is in allowed list."""
        if self.allowed_commands is None:
            return

        cmd_name = Path(command).name
        if cmd_name not in self.allowed_commands and command not in self.allowed_commands:
            raise ExecutionError(
                f"Command '{cmd_name}' not in allowed list: {self.allowed_commands}"
            )

    async def _execute_impl(self, action: BaseAction) -> ExecutionResult:
        """Execute CLI command."""
        if not isinstance(action, CLIAction):
            raise TypeError(f"Expected CLIAction, got {type(action)}")

        # Resolve and validate command
        command = self._resolve_command(action.command)
        self._check_allowed(command)

        # Build environment
        env = os.environ.copy()
        env.update(self.default_env)
        env.update(action.env)

        # Determine working directory
        cwd = action.working_dir or self.default_working_dir

        # Build command list
        cmd = [command] + action.args

        try:
            if action.shell:
                # Shell mode - join command for shell execution
                process = await asyncio.create_subprocess_shell(
                    action.full_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if action.capture_stderr else asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.PIPE if action.stdin else None,
                    env=env,
                    cwd=cwd,
                )
            else:
                # Direct execution (safer)
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if action.capture_stderr else asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.PIPE if action.stdin else None,
                    env=env,
                    cwd=cwd,
                )

            # Wait for completion with timeout
            stdin_bytes = action.stdin.encode() if action.stdin else None

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_bytes),
                    timeout=action.timeout_seconds
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Command timed out after {action.timeout_seconds}s",
                    exit_code=-1,
                    metadata={"command": action.full_command}
                )

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Determine success
            exit_code = process.returncode
            success = exit_code == 0

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                output=stdout_str,
                error=stderr_str if not success else None,
                exit_code=exit_code,
                metadata={
                    "command": action.full_command,
                    "stderr": stderr_str if success else None,
                    "working_dir": cwd,
                }
            )

        except FileNotFoundError:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Command not found: {command}",
                exit_code=-1,
                metadata={"command": action.full_command}
            )
        except PermissionError:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Permission denied: {command}",
                exit_code=-1,
                metadata={"command": action.full_command}
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                exit_code=-1,
                metadata={"command": action.full_command, "exception": type(e).__name__}
            )


# Convenience functions for common patterns
async def run_command(
    command: str,
    args: Optional[List[str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ExecutionResult:
    """Run a CLI command with default settings.

    Args:
        command: Command to run
        args: Command arguments
        timeout: Timeout in seconds
        **kwargs: Additional CLIAction parameters

    Returns:
        ExecutionResult
    """
    executor = CLIExecutor()
    action = CLIAction(
        name=f"run_{command}",
        command=command,
        args=args or [],
        timeout_seconds=timeout,
        **kwargs
    )
    return await executor.execute(action)


async def run_pipeline(
    commands: List[Dict[str, Any]],
    stop_on_failure: bool = True
) -> List[ExecutionResult]:
    """Run a pipeline of CLI commands.

    Args:
        commands: List of command dicts with 'command' and optional 'args'
        stop_on_failure: Whether to stop on first failure

    Returns:
        List of ExecutionResults
    """
    executor = CLIExecutor()
    results = []

    for cmd_spec in commands:
        action = CLIAction(
            name=cmd_spec.get("name", f"cmd_{len(results)}"),
            command=cmd_spec["command"],
            args=cmd_spec.get("args", []),
            timeout_seconds=cmd_spec.get("timeout", 30.0),
            env=cmd_spec.get("env", {}),
            working_dir=cmd_spec.get("working_dir"),
            stdin=cmd_spec.get("stdin"),
        )

        result = await executor.execute(action)
        results.append(result)

        if not result.success and stop_on_failure:
            break

    return results
