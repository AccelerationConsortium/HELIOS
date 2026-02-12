"""
Plugin Base Classes for Lab Automation

Defines the abstract base classes that all instrument plugins must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class OperationDef:
    """Definition of an operation supported by a plugin."""
    name: str
    action: str  # e.g., "liquid_handler.transfer", "potentiostat.run_eis"
    keywords: Dict[str, List[str]]  # {"en": ["transfer", "move"], "zh": ["转移"]}
    params_schema: Dict[str, Any] = field(default_factory=dict)  # Parameter definitions
    description: str = ""

    def matches(self, text: str, language: str = "en") -> Tuple[bool, float]:
        """Check if text matches this operation's keywords."""
        text_lower = text.lower()
        keywords = self.keywords.get(language, self.keywords.get("en", []))

        best_confidence = 0.0
        best_match = False

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                # Calculate confidence based on keyword match quality
                # Longer keywords that match = higher confidence
                # More of the text covered by keyword = higher confidence
                keyword_coverage = len(keyword_lower) / len(text_lower)

                if keyword_lower == text_lower:
                    # Exact match - highest confidence
                    confidence = 0.95
                elif text_lower.startswith(keyword_lower + " "):
                    # Keyword at start of text - very high confidence
                    confidence = 0.85 + (keyword_coverage * 0.1)
                else:
                    # Substring match - base + coverage bonus
                    confidence = 0.6 + (keyword_coverage * 0.2)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = True

        return best_match, best_confidence


class ParserBase(ABC):
    """Base class for natural language parsers."""

    @abstractmethod
    def parse(self, instruction: str) -> Dict[str, Any]:
        """
        Parse a natural language instruction.

        Args:
            instruction: Natural language text

        Returns:
            Dict with:
                - operation: str (operation name)
                - action: str (full action string)
                - params: Dict[str, Any]
                - confidence: float
                - language: str
        """
        pass

    def detect_language(self, text: str) -> str:
        """Detect if text is English or Chinese."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return "zh" if chinese_chars > len(text) * 0.1 else "en"

    def extract_number(self, text: str, patterns: List[str]) -> Optional[float]:
        """Extract a number from text using patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def extract_well(self, text: str) -> Optional[str]:
        """Extract well position like A1, B12, etc."""
        match = re.search(r'\b([A-Ha-h][1-9]|[A-Ha-h]1[0-2])\b', text)
        return match.group(1).upper() if match else None

    def extract_well_range(self, text: str) -> Optional[List[str]]:
        """Extract well range like A1-A12, B1-B4."""
        match = re.search(r'\b([A-Ha-h])(\d+)-([A-Ha-h])(\d+)\b', text, re.IGNORECASE)
        if match:
            row_start, col_start, row_end, col_end = match.groups()
            row_start, row_end = row_start.upper(), row_end.upper()
            col_start, col_end = int(col_start), int(col_end)

            wells = []
            if row_start == row_end:
                # Same row, different columns
                for col in range(col_start, col_end + 1):
                    wells.append(f"{row_start}{col}")
            else:
                # Different rows (assume same column)
                for row in range(ord(row_start), ord(row_end) + 1):
                    wells.append(f"{chr(row)}{col_start}")

            return wells
        return None


class PluginBase(ABC):
    """
    Abstract base class for all instrument plugins.

    Each plugin represents a category of lab equipment (e.g., liquid handlers,
    potentiostats, pump controllers) and provides:
    - Natural language parsing for that domain
    - Operation definitions
    - Hardware adapter management
    """

    # Plugin identity
    name: str = "base"
    device_type: str = "generic"
    version: str = "1.0.0"
    description: str = "Base plugin"

    def __init__(self):
        self._operations: Dict[str, OperationDef] = {}
        self._adapters: Dict[str, Any] = {}
        self._parser: Optional[ParserBase] = None
        self._register_operations()

    @abstractmethod
    def _register_operations(self):
        """Register all operations supported by this plugin."""
        pass

    @property
    def parser(self) -> ParserBase:
        """Get the parser for this plugin."""
        if self._parser is None:
            self._parser = self._create_parser()
        return self._parser

    @abstractmethod
    def _create_parser(self) -> ParserBase:
        """Create the parser instance for this plugin."""
        pass

    def register_operation(self, op: OperationDef):
        """Register an operation."""
        self._operations[op.name] = op

    def get_operations(self) -> Dict[str, OperationDef]:
        """Get all registered operations."""
        return self._operations

    def get_operation(self, name: str) -> Optional[OperationDef]:
        """Get a specific operation by name."""
        return self._operations.get(name)

    def parse(self, instruction: str) -> Dict[str, Any]:
        """Parse a natural language instruction using this plugin's parser."""
        return self.parser.parse(instruction)

    def can_handle(self, instruction: str) -> Tuple[bool, float]:
        """
        Check if this plugin can handle the given instruction.

        Returns:
            Tuple of (can_handle: bool, confidence: float)
        """
        result = self.parse(instruction)
        if result.get('operation'):
            return True, result.get('confidence', 0.5)
        return False, 0.0

    def register_adapter(self, name: str, adapter: Any):
        """Register a hardware adapter."""
        self._adapters[name] = adapter

    def get_adapter(self, name: str) -> Optional[Any]:
        """Get a registered adapter."""
        return self._adapters.get(name)

    def list_adapters(self) -> List[str]:
        """List all registered adapters."""
        return list(self._adapters.keys())

    def to_step_dict(self, parsed: Dict[str, Any], step_id: str) -> Dict[str, Any]:
        """
        Convert parsed instruction to workflow step dictionary.

        Args:
            parsed: Result from parse()
            step_id: Unique step identifier

        Returns:
            Step dictionary for workflow JSON
        """
        return {
            "step_id": step_id,
            "device_type": self.device_type,
            "action": parsed.get('action', f"{self.device_type}.unknown"),
            "params": parsed.get('params', {}),
            "description": parsed.get('description', ''),
        }
