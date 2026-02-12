from typing import Any, Protocol, Union
import time
import re
from .types import DeviceState

class Predicate(Protocol):
    def evaluate(self, state: DeviceState) -> bool: ...
    def describe(self) -> str: ...

class Comparison:
    def __init__(self, key: str, op: str, value: Any, tolerance: float = 0.0):
        self.key = key
        self.op = op
        self.value = value
        self.tolerance = tolerance

    def evaluate(self, state: DeviceState) -> bool:
        actual = self._get_value(state, self.key)
        if actual is None:
            return False
            
        try:
            val_num = float(self.value)
            act_num = float(actual)
            
            if self.op == "==":
                return abs(act_num - val_num) <= self.tolerance
            elif self.op == "~=": # Approximate
                return abs(act_num - val_num) <= (self.tolerance if self.tolerance > 0 else 1e-6)
            elif self.op == ">": return act_num > val_num
            elif self.op == ">=": return act_num >= val_num
            elif self.op == "<": return act_num < val_num
            elif self.op == "<=": return act_num <= val_num
        except (ValueError, TypeError):
            # String comparison
            if self.op == "==": return str(actual) == str(self.value)
            if self.op == "!=": return str(actual) != str(self.value)
            if self.op == "one_of": return str(actual) in [x.strip() for x in str(self.value).split("|")]
        return False

    def describe(self) -> str:
        tol = f" +/- {self.tolerance}" if self.tolerance > 0 else ""
        return f"{self.key} {self.op} {self.value}{tol}"

    def _get_value(self, state: DeviceState, path: str) -> Any:
        current = state
        for part in path.split('.'):
            if part == "status": current = current.status
            elif part == "telemetry": current = current.telemetry
            elif isinstance(current, dict): current = current.get(part)
            elif hasattr(current, part): current = getattr(current, part)
            else: return None
        return current

class ParsedPredicate:
    """Parsed version of a requirement string."""
    def __init__(self, requirement: str):
        self.raw = requirement
        self.condition = self._parse(requirement)
        self.timeout = 0
        if "within" in requirement:
            # check for "within X s"
            m = re.search(r"within\s+(\d+)\s*s?", requirement)
            if m:
                self.timeout = float(m.group(1))

    def _parse(self, req: str) -> Comparison:
        # Remove 'within' clause for parsing condition
        core_req = req.split("within")[0].strip()
        
        # Regex for "key operator value [+/- tolerance]"
        # Supports: key == val, key ~= val +/- tol
        pattern = r"([\w\.]+)\s*(==|!=|>=|<=|>|<|~=|one_of)\s*([^\s\+\-]+)(?:\s*\+/-\s*([\d\.]+))?"
        match = re.search(pattern, core_req)
        if not match:
            # Fallback for simple boolean existence or manual
            return Comparison("unknown", "==", "unknown")
            
        key, op, val, tol = match.groups()
        tolerance = float(tol) if tol else 0.0
        return Comparison(key, op, val, tolerance)

    def check(self, state: DeviceState) -> bool:
        return self.condition.evaluate(state)
