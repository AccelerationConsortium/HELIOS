"""
Unit Operations (UO) layer - Domain semantic modules.

This is the middle layer of the IR, representing domain-specific
operations that are more concrete than intents but still
device-agnostic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .intent import MissingInfo


class UOType(Enum):
    """Types of Unit Operations."""
    # Preparation
    ELECTRODE_PREPARATION = "electrode_preparation"
    ELECTROLYTE_PREPARATION = "electrolyte_preparation"
    SAMPLE_PREPARATION = "sample_preparation"

    # Assembly
    CELL_ASSEMBLY = "cell_assembly"
    SYSTEM_SETUP = "system_setup"

    # Calibration
    CALIBRATION = "calibration"
    REFERENCE_CHECK = "reference_check"

    # Measurement
    MEASUREMENT = "measurement"
    CHARACTERIZATION = "characterization"
    STABILITY_TEST = "stability_test"

    # Data
    DATA_ANALYSIS = "data_analysis"
    DATA_LOGGING = "data_logging"

    # Cleanup
    CLEANUP = "cleanup"
    MAINTENANCE = "maintenance"

    # Control
    WAIT = "wait"
    USER_CHECKPOINT = "user_checkpoint"

    # General
    GENERAL = "general"


@dataclass
class Placeholder:
    """
    A parameter placeholder that needs to be filled by user.

    Similar to MissingInfo but tied to a specific UO parameter.
    """
    parameter: str                          # Parameter key
    question: str                           # English question
    question_zh: str                        # Chinese question
    default: Optional[Any] = None           # Default value
    required: bool = True                   # Is this required?
    options: Optional[List[str]] = None     # Allowed values
    unit: Optional[str] = None              # Unit string
    value_type: str = "string"              # Expected type
    validation: Optional[str] = None        # Validation rule (regex or range)

    def to_missing_info(self) -> MissingInfo:
        """Convert to MissingInfo for planner output."""
        return MissingInfo(
            parameter=self.parameter,
            question=self.question,
            question_zh=self.question_zh,
            options=self.options,
            default=self.default,
            required=self.required,
            unit=self.unit,
            value_type=self.value_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "question": self.question,
            "question_zh": self.question_zh,
            "default": self.default,
            "required": self.required,
            "options": self.options,
            "unit": self.unit,
            "value_type": self.value_type,
            "validation": self.validation,
        }


@dataclass
class UnitOperation:
    """
    A Unit Operation - a domain semantic module.

    UOs represent meaningful experiment steps at a higher level than
    device commands. They are domain-specific but device-agnostic.

    Example:
        UnitOperation(
            name="OERMeasurement",
            uo_type=UOType.MEASUREMENT,
            description="Perform LSV scan for OER characterization",
            parameters={
                "method": "LSV",
                "scan_rate_V_s": 0.005,
                "potential_range_V": [1.0, 1.8],
            },
            placeholders={
                "electrode_area_cm2": Placeholder(
                    parameter="electrode_area_cm2",
                    question="Electrode geometric area (cm²)?",
                    question_zh="电极几何面积(cm²)？",
                    required=True,
                    unit="cm²",
                    value_type="number"
                )
            },
            preconditions=["cell_assembled", "electrolyte_prepared"],
            postconditions=["lsv_data_collected"]
        )
    """
    name: str                                       # UO name (e.g., "OERMeasurement")
    uo_type: UOType                                 # UO category
    description: str = ""                           # Human-readable description
    description_zh: str = ""                        # Chinese description

    # Input/output definitions
    inputs: Dict[str, Any] = field(default_factory=dict)    # Required inputs
    outputs: Dict[str, Any] = field(default_factory=dict)   # Produced outputs

    # Fixed parameters (determined by template)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Parameters needing user input
    placeholders: Dict[str, Placeholder] = field(default_factory=dict)

    # Conditions
    preconditions: List[str] = field(default_factory=list)   # Must be true before execution
    postconditions: List[str] = field(default_factory=list)  # Will be true after execution

    # Timing
    estimated_duration_s: Optional[float] = None    # Estimated duration in seconds

    # Metadata
    domain: str = "general"                         # Domain this UO belongs to
    template_id: Optional[str] = None               # Template ID if from template

    def get_description(self, language: str = "en") -> str:
        """Get description in specified language."""
        if language == "zh" and self.description_zh:
            return self.description_zh
        return self.description

    def get_missing_info(self) -> List[MissingInfo]:
        """Get list of MissingInfo for unfilled placeholders."""
        return [ph.to_missing_info() for ph in self.placeholders.values()]

    def fill_placeholder(self, parameter: str, value: Any) -> bool:
        """Fill a placeholder with a value, moving it to parameters."""
        if parameter in self.placeholders:
            self.parameters[parameter] = value
            del self.placeholders[parameter]
            return True
        return False

    def fill_placeholders(self, values: Dict[str, Any]) -> List[str]:
        """Fill multiple placeholders. Returns list of unfilled required parameters."""
        unfilled = []
        for param, placeholder in list(self.placeholders.items()):
            if param in values:
                self.fill_placeholder(param, values[param])
            elif placeholder.required and placeholder.default is None:
                unfilled.append(param)
            elif placeholder.default is not None:
                self.fill_placeholder(param, placeholder.default)
        return unfilled

    def is_complete(self) -> bool:
        """Check if all required placeholders are filled."""
        for placeholder in self.placeholders.values():
            if placeholder.required and placeholder.default is None:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "uo_type": self.uo_type.value,
            "description": self.description,
            "description_zh": self.description_zh,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "placeholders": {k: v.to_dict() for k, v in self.placeholders.items()},
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "estimated_duration_s": self.estimated_duration_s,
            "domain": self.domain,
            "template_id": self.template_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnitOperation":
        """Create from dictionary."""
        placeholders = {}
        for k, v in data.get("placeholders", {}).items():
            placeholders[k] = Placeholder(
                parameter=v["parameter"],
                question=v["question"],
                question_zh=v.get("question_zh", v["question"]),
                default=v.get("default"),
                required=v.get("required", True),
                options=v.get("options"),
                unit=v.get("unit"),
                value_type=v.get("value_type", "string"),
                validation=v.get("validation"),
            )

        return cls(
            name=data["name"],
            uo_type=UOType(data["uo_type"]),
            description=data.get("description", ""),
            description_zh=data.get("description_zh", ""),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            parameters=data.get("parameters", {}),
            placeholders=placeholders,
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
            estimated_duration_s=data.get("estimated_duration_s"),
            domain=data.get("domain", "general"),
            template_id=data.get("template_id"),
        )

    def copy(self) -> "UnitOperation":
        """Create a deep copy of this UO."""
        import copy
        return UnitOperation(
            name=self.name,
            uo_type=self.uo_type,
            description=self.description,
            description_zh=self.description_zh,
            inputs=copy.deepcopy(self.inputs),
            outputs=copy.deepcopy(self.outputs),
            parameters=copy.deepcopy(self.parameters),
            placeholders={k: Placeholder(**v.to_dict()) for k, v in self.placeholders.items()},
            preconditions=self.preconditions.copy(),
            postconditions=self.postconditions.copy(),
            estimated_duration_s=self.estimated_duration_s,
            domain=self.domain,
            template_id=self.template_id,
        )
