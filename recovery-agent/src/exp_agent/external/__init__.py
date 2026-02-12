"""
External execution module for CLI tools, scripts, and API calls.

This module provides unified interfaces for:
- CLI tool execution (subprocess-based)
- Local script execution (Python, shell, etc.)
- External API calls (REST, GraphQL, etc.)

All executors follow the same pattern:
1. Define the action
2. Execute with timeout and retry
3. Return structured result
"""

from .base import ExternalExecutor, ExecutionResult, ExecutionError
from .cli import CLIExecutor, CLIAction
from .script import ScriptExecutor, ScriptAction
from .api import APIExecutor, APIAction

__all__ = [
    # Base
    "ExternalExecutor",
    "ExecutionResult",
    "ExecutionError",
    # CLI
    "CLIExecutor",
    "CLIAction",
    # Script
    "ScriptExecutor",
    "ScriptAction",
    # API
    "APIExecutor",
    "APIAction",
]
