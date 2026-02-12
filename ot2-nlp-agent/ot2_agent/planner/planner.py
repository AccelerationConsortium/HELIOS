"""
Main Planner class - Intent to candidate workflows.

The Planner is the main entry point for the planning phase.
It orchestrates intent parsing and workflow generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir import Intent, UnitOperation, MissingInfo, PlanningContext
from .intent_parser import IntentParser
from .workflow_generator import WorkflowGenerator
from .domain_knowledge import DomainKnowledge, OERDomainKnowledge


@dataclass
class WorkflowDraft:
    """
    A candidate workflow at UO level.

    This is what the Planner outputs for user review and selection.
    """
    name: str
    description: str
    description_zh: str = ""
    unit_operations: List[UnitOperation] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    missing_info: List[MissingInfo] = field(default_factory=list)
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)

    def get_description(self, language: str = "en") -> str:
        """Get description in specified language."""
        if language == "zh" and self.description_zh:
            return self.description_zh
        return self.description

    def get_required_missing_info(self) -> List[MissingInfo]:
        """Get only required missing info."""
        return [mi for mi in self.missing_info if mi.required]

    def get_optional_missing_info(self) -> List[MissingInfo]:
        """Get only optional missing info."""
        return [mi for mi in self.missing_info if not mi.required]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "description_zh": self.description_zh,
            "unit_operations": [uo.to_dict() for uo in self.unit_operations],
            "assumptions": self.assumptions,
            "missing_info": [mi.to_dict() for mi in self.missing_info],
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDraft":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            description_zh=data.get("description_zh", ""),
            unit_operations=[UnitOperation.from_dict(uo) for uo in data.get("unit_operations", [])],
            assumptions=data.get("assumptions", []),
            missing_info=[MissingInfo.from_dict(mi) for mi in data.get("missing_info", [])],
            confidence=data.get("confidence", 0.0),
            alternatives=data.get("alternatives", []),
        )


@dataclass
class ConfirmedWorkflow:
    """
    A user-confirmed workflow ready for compilation.

    This is created after the user selects a WorkflowDraft and
    fills in the missing parameters.
    """
    draft: WorkflowDraft
    filled_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "draft": self.draft.to_dict(),
            "filled_parameters": self.filled_parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfirmedWorkflow":
        """Create from dictionary."""
        return cls(
            draft=WorkflowDraft.from_dict(data["draft"]),
            filled_parameters=data.get("filled_parameters", {}),
        )

    def get_filled_unit_operations(self) -> List[UnitOperation]:
        """
        Get UOs with parameters filled in.

        Returns copies of UOs with placeholders replaced by filled values.
        """
        filled_uos = []

        for uo in self.draft.unit_operations:
            # Make a copy
            filled_uo = uo.copy()

            # Fill placeholders with provided values
            filled_uo.fill_placeholders(self.filled_parameters)

            filled_uos.append(filled_uo)

        return filled_uos


@dataclass
class PlannerOutput:
    """
    Output from the Planner.

    Contains the parsed intent and candidate workflows.
    """
    intent: Intent
    candidates: List[WorkflowDraft]
    context: PlanningContext = field(default_factory=PlanningContext)
    recommended_idx: int = 0

    def get_recommended(self) -> Optional[WorkflowDraft]:
        """Get the recommended workflow candidate."""
        if 0 <= self.recommended_idx < len(self.candidates):
            return self.candidates[self.recommended_idx]
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "context": self.context.to_dict(),
            "recommended_idx": self.recommended_idx,
        }


class Planner:
    """
    Main Planner class.

    Converts user intent to candidate workflow drafts.
    """

    def __init__(self, domain_knowledge: DomainKnowledge = None):
        """
        Initialize the Planner.

        Args:
            domain_knowledge: Optional domain-specific knowledge.
                            Defaults to OER if not specified.
        """
        self.intent_parser = IntentParser()
        self.workflow_generator = WorkflowGenerator()

        # Register domain knowledge if provided
        if domain_knowledge:
            self.workflow_generator._domain_instances[domain_knowledge.domain_name] = domain_knowledge

    def plan(
        self,
        user_input: str,
        context: Dict[str, Any] = None
    ) -> PlannerOutput:
        """
        Main planning method.

        Args:
            user_input: User's natural language intent
            context: Optional dict with known conditions (devices, materials, etc.)

        Returns:
            PlannerOutput with candidate workflows
        """
        # Parse context
        planning_context = self._parse_context(context)

        # Parse intent
        intent = self.intent_parser.parse(user_input, planning_context)

        # Generate candidate workflows
        candidates = self.workflow_generator.generate(intent, planning_context)

        # Determine recommended candidate (first one, highest confidence)
        recommended_idx = 0

        return PlannerOutput(
            intent=intent,
            candidates=candidates,
            context=planning_context,
            recommended_idx=recommended_idx,
        )

    def _parse_context(self, context: Dict[str, Any] = None) -> PlanningContext:
        """Parse context dictionary into PlanningContext."""
        if context is None:
            return PlanningContext()

        if isinstance(context, PlanningContext):
            return context

        return PlanningContext(
            devices=context.get("devices", []),
            labware=context.get("labware", []),
            materials=context.get("materials", {}),
            samples=context.get("samples", []),
            constraints=context.get("constraints", {}),
            previous_state=context.get("previous_state", {}),
        )

    def extract_intent(self, text: str) -> Intent:
        """
        Extract intent from text without generating workflows.

        Useful for understanding what the user wants before planning.
        """
        return self.intent_parser.parse(text)
