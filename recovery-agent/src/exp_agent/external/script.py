"""
Script executor for running local scripts (Python, shell, etc.).

Examples:
    # Run a Python script
    executor = ScriptExecutor()
    result = await executor.execute(ScriptAction(
        name="analyze_data",
        script_path="/path/to/analyze.py",
        script_type=ScriptType.PYTHON,
        args=["--input", "data.csv"]
    ))

    # Run inline Python code
    result = await executor.execute(ScriptAction(
        name="quick_calc",
        script_type=ScriptType.PYTHON,
        inline_code="print(2 + 2)"
    ))

    # Run shell script
    result = await executor.execute(ScriptAction(
        name="setup",
        script_path="/path/to/setup.sh",
        script_type=ScriptType.SHELL
    ))

    # Run with virtual environment
    executor = ScriptExecutor(python_venv="/path/to/.venv")
    result = await executor.execute(ScriptAction(
        name="ml_train",
        script_path="train.py",
        script_type=ScriptType.PYTHON
    ))
"""
import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseAction,
    ExternalExecutor,
    ExecutionResult,
    ExecutionStatus,
    ExecutionError,
)
from .cli import CLIExecutor, CLIAction


class ScriptType(str, Enum):
    """Supported script types."""
    PYTHON = "python"
    SHELL = "shell"
    BASH = "bash"
    POWERSHELL = "powershell"
    NODE = "node"


# Interpreters for each script type
INTERPRETERS = {
    ScriptType.PYTHON: [sys.executable, "python3", "python"],
    ScriptType.SHELL: ["sh"],
    ScriptType.BASH: ["bash"],
    ScriptType.POWERSHELL: ["pwsh", "powershell"],
    ScriptType.NODE: ["node", "nodejs"],
}


@dataclass
class ScriptAction(BaseAction):
    """Action for script execution.

    Attributes:
        script_path: Path to script file (optional if inline_code provided)
        script_type: Type of script (python, shell, etc.)
        inline_code: Inline code to execute (alternative to script_path)
        args: Script arguments
        env: Additional environment variables
        working_dir: Working directory for execution
        stdin: Input to send to script's stdin
        python_path: Additional Python path entries
    """
    script_path: Optional[str] = None
    script_type: ScriptType = ScriptType.PYTHON
    inline_code: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    stdin: Optional[str] = None
    python_path: List[str] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        if not self.script_path and not self.inline_code:
            raise ValueError("Either script_path or inline_code is required")
        if self.script_path and self.inline_code:
            raise ValueError("Cannot specify both script_path and inline_code")
        if self.script_path:
            path = Path(self.script_path)
            if not path.exists():
                raise ValueError(f"Script not found: {self.script_path}")


class ScriptExecutor(ExternalExecutor):
    """Executor for local scripts.

    Features:
    - Multiple script types (Python, shell, bash, node)
    - Inline code execution
    - Virtual environment support
    - Python path management
    - Secure temporary file handling for inline code

    Security considerations:
    - Scripts are executed with current user permissions
    - Use allowed_paths to restrict script locations
    - Inline code creates temporary files that are cleaned up
    """

    def __init__(
        self,
        name: str = "script",
        python_venv: Optional[str] = None,
        allowed_paths: Optional[List[str]] = None,
        default_working_dir: Optional[str] = None,
    ):
        """Initialize script executor.

        Args:
            name: Executor name for logging
            python_venv: Path to Python virtual environment
            allowed_paths: Optional whitelist of allowed script directories
            default_working_dir: Default working directory
        """
        super().__init__(name)
        self.python_venv = python_venv
        self.allowed_paths = allowed_paths
        self.default_working_dir = default_working_dir
        self._cli_executor = CLIExecutor(name=f"{name}_cli")

    def _get_interpreter(self, script_type: ScriptType) -> str:
        """Get interpreter for script type."""
        if script_type == ScriptType.PYTHON and self.python_venv:
            # Use venv python
            venv_python = Path(self.python_venv) / "bin" / "python"
            if venv_python.exists():
                return str(venv_python)

        candidates = INTERPRETERS.get(script_type, [])
        for candidate in candidates:
            import shutil
            if shutil.which(candidate):
                return candidate

        raise ExecutionError(f"No interpreter found for {script_type}")

    def _check_allowed_path(self, script_path: str) -> None:
        """Check if script path is in allowed list."""
        if self.allowed_paths is None:
            return

        script_path = Path(script_path).resolve()
        for allowed in self.allowed_paths:
            allowed_path = Path(allowed).resolve()
            try:
                script_path.relative_to(allowed_path)
                return  # Script is under an allowed path
            except ValueError:
                continue

        raise ExecutionError(
            f"Script '{script_path}' not in allowed paths: {self.allowed_paths}"
        )

    async def _execute_impl(self, action: BaseAction) -> ExecutionResult:
        """Execute script."""
        if not isinstance(action, ScriptAction):
            raise TypeError(f"Expected ScriptAction, got {type(action)}")

        # Get interpreter
        interpreter = self._get_interpreter(action.script_type)

        # Build environment
        env = action.env.copy()
        if action.python_path and action.script_type == ScriptType.PYTHON:
            existing = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
            new_path = os.pathsep.join(action.python_path)
            env["PYTHONPATH"] = f"{new_path}{os.pathsep}{existing}" if existing else new_path

        # Handle inline code vs script file
        temp_file = None
        try:
            if action.inline_code:
                # Write inline code to temporary file
                suffix = self._get_suffix(action.script_type)
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=suffix,
                    delete=False
                )
                temp_file.write(action.inline_code)
                temp_file.close()
                script_path = temp_file.name
            else:
                script_path = action.script_path
                self._check_allowed_path(script_path)

            # Build CLI action
            cli_action = CLIAction(
                name=action.name,
                command=interpreter,
                args=[script_path] + action.args,
                env=env,
                working_dir=action.working_dir or self.default_working_dir,
                stdin=action.stdin,
                timeout_seconds=action.timeout_seconds,
                retries=0,  # Retries handled at script level
            )

            # Execute
            result = await self._cli_executor._execute_impl(cli_action)

            # Add script metadata
            result.metadata["script_type"] = action.script_type.value
            result.metadata["interpreter"] = interpreter
            if action.inline_code:
                result.metadata["inline"] = True

            return result

        finally:
            # Clean up temporary file
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def _get_suffix(self, script_type: ScriptType) -> str:
        """Get file suffix for script type."""
        suffixes = {
            ScriptType.PYTHON: ".py",
            ScriptType.SHELL: ".sh",
            ScriptType.BASH: ".sh",
            ScriptType.POWERSHELL: ".ps1",
            ScriptType.NODE: ".js",
        }
        return suffixes.get(script_type, ".txt")


# Convenience functions
async def run_python(
    code_or_path: str,
    args: Optional[List[str]] = None,
    timeout: float = 30.0,
    venv: Optional[str] = None,
    **kwargs
) -> ExecutionResult:
    """Run Python code or script.

    Args:
        code_or_path: Python code string or path to .py file
        args: Script arguments
        timeout: Timeout in seconds
        venv: Path to virtual environment
        **kwargs: Additional ScriptAction parameters

    Returns:
        ExecutionResult
    """
    executor = ScriptExecutor(python_venv=venv)

    # Determine if code or path
    if code_or_path.endswith(".py") or Path(code_or_path).exists():
        action = ScriptAction(
            name="run_python",
            script_path=code_or_path,
            script_type=ScriptType.PYTHON,
            args=args or [],
            timeout_seconds=timeout,
            **kwargs
        )
    else:
        action = ScriptAction(
            name="run_python_inline",
            inline_code=code_or_path,
            script_type=ScriptType.PYTHON,
            args=args or [],
            timeout_seconds=timeout,
            **kwargs
        )

    return await executor.execute(action)


async def run_shell(
    code_or_path: str,
    args: Optional[List[str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ExecutionResult:
    """Run shell code or script.

    Args:
        code_or_path: Shell code string or path to script
        args: Script arguments
        timeout: Timeout in seconds
        **kwargs: Additional ScriptAction parameters

    Returns:
        ExecutionResult
    """
    executor = ScriptExecutor()

    is_file = code_or_path.endswith(".sh") or Path(code_or_path).exists()
    action = ScriptAction(
        name="run_shell",
        script_path=code_or_path if is_file else None,
        inline_code=None if is_file else code_or_path,
        script_type=ScriptType.BASH,
        args=args or [],
        timeout_seconds=timeout,
        **kwargs
    )

    return await executor.execute(action)
