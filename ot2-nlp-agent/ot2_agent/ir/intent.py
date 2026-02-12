"""
Intent layer - User's high-level goal representation.

This is the top layer of the IR, representing what the user wants to achieve
without specifying how to achieve it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MissingInfo:
    """
    Information needed from user to complete a workflow.

    Used by the Planner to indicate what parameters are missing
    and need to be provided by the user.
    """
    parameter: str                          # Parameter name/key
    question: str                           # Question in English
    question_zh: str                        # Question in Chinese
    options: Optional[List[str]] = None     # Suggested values (if applicable)
    default: Optional[Any] = None           # Default value
    required: bool = True                   # Whether this is required
    unit: Optional[str] = None              # Unit (e.g., "mA/cm²", "°C")
    value_type: str = "string"              # Expected type: string, number, boolean, list

    def get_question(self, language: str = "en") -> str:
        """Get question in specified language."""
        if language == "zh":
            return self.question_zh
        return self.question

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "parameter": self.parameter,
            "question": self.question,
            "question_zh": self.question_zh,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "unit": self.unit,
            "value_type": self.value_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissingInfo":
        """Create from dictionary."""
        return cls(
            parameter=data["parameter"],
            question=data["question"],
            question_zh=data.get("question_zh", data["question"]),
            options=data.get("options"),
            default=data.get("default"),
            required=data.get("required", True),
            unit=data.get("unit"),
            value_type=data.get("value_type", "string"),
        )


@dataclass
class PlanningContext:
    """
    Context for planning, containing known conditions and constraints.

    This includes information about available equipment, materials,
    and any constraints the user has specified.
    """
    # Available equipment
    devices: List[str] = field(default_factory=list)  # e.g., ["ot2", "potentiostat"]
    labware: List[str] = field(default_factory=list)  # e.g., ["96_wellplate", "reservoir"]

    # Materials and samples
    materials: Dict[str, Any] = field(default_factory=dict)  # e.g., {"electrode": "NiFe", "electrolyte": "1M KOH"}
    samples: List[str] = field(default_factory=list)  # Sample identifiers

    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)  # e.g., {"max_temperature": 60}

    # Previous workflow state (for multi-step experiments)
    previous_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "devices": self.devices,
            "labware": self.labware,
            "materials": self.materials,
            "samples": self.samples,
            "constraints": self.constraints,
            "previous_state": self.previous_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningContext":
        """Create from dictionary."""
        return cls(
            devices=data.get("devices", []),
            labware=data.get("labware", []),
            materials=data.get("materials", {}),
            samples=data.get("samples", []),
            constraints=data.get("constraints", {}),
            previous_state=data.get("previous_state", {}),
        )


@dataclass
class Intent:
    """
    User's high-level experimental intent.

    This is the top-level IR, representing WHAT the user wants
    without specifying HOW to achieve it.

    Example:
        Intent(
            goal="perform OER measurement",
            domain="electrochemistry",
            target_metrics=["overpotential", "Tafel slope"],
            known_conditions={"electrode": "NiFe catalyst"},
            original_text="我想做OER测量，用的是NiFe催化剂",
            language="zh"
        )
    """
    goal: str                                   # Extracted goal (normalized)
    domain: str                                 # Domain: electrochemistry, biology, etc.
    original_text: str                          # Original user input
    language: str = "en"                        # Detected language: "en" or "zh"

    # What the user wants to measure/achieve
    target_metrics: List[str] = field(default_factory=list)

    # Known conditions extracted from user input
    known_conditions: Dict[str, Any] = field(default_factory=dict)

    # Confidence in intent extraction (0.0-1.0)
    confidence: float = 0.0

    # Sub-intents for complex requests
    sub_intents: List["Intent"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "goal": self.goal,
            "domain": self.domain,
            "original_text": self.original_text,
            "language": self.language,
            "target_metrics": self.target_metrics,
            "known_conditions": self.known_conditions,
            "confidence": self.confidence,
            "sub_intents": [si.to_dict() for si in self.sub_intents],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intent":
        """Create from dictionary."""
        return cls(
            goal=data["goal"],
            domain=data["domain"],
            original_text=data["original_text"],
            language=data.get("language", "en"),
            target_metrics=data.get("target_metrics", []),
            known_conditions=data.get("known_conditions", {}),
            confidence=data.get("confidence", 0.0),
            sub_intents=[cls.from_dict(si) for si in data.get("sub_intents", [])],
        )
