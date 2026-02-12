"""
Domain Knowledge - Rules and heuristics for specific experiment domains.

This module contains domain-specific knowledge for generating
appropriate workflow candidates.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

from ..ir import Intent, UnitOperation, MissingInfo
from ..templates import TemplateRegistry


class DomainKnowledge(ABC):
    """
    Base class for domain-specific knowledge.

    Subclasses implement rules for generating workflows in specific domains.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain name."""
        pass

    @abstractmethod
    def get_workflow_templates(self, intent: Intent) -> List[List[UnitOperation]]:
        """
        Get workflow templates based on intent.

        Returns a list of candidate workflows, each being a list of UOs.
        """
        pass

    @abstractmethod
    def get_default_assumptions(self, intent: Intent) -> List[str]:
        """Get default assumptions for this domain."""
        pass

    @abstractmethod
    def get_alternatives(self, intent: Intent) -> List[str]:
        """Get alternative approaches for this intent."""
        pass


class OERDomainKnowledge(DomainKnowledge):
    """
    Domain knowledge for OER (Oxygen Evolution Reaction) experiments.
    """

    @property
    def domain_name(self) -> str:
        return "electrochemistry"

    def get_workflow_templates(self, intent: Intent) -> List[List[UnitOperation]]:
        """
        Generate OER workflow candidates based on intent.

        Returns 2-3 candidates ranging from simple to comprehensive.
        """
        candidates = []

        # Candidate A: Fast characterization (screening)
        candidates.append(self._get_fast_characterization_workflow(intent))

        # Candidate B: Standard characterization (publication-ready)
        candidates.append(self._get_standard_characterization_workflow(intent))

        # Candidate C: Comprehensive (full study) - only if stability is mentioned
        if "stability" in intent.target_metrics or self._needs_comprehensive(intent):
            candidates.append(self._get_comprehensive_workflow(intent))

        return candidates

    def _get_fast_characterization_workflow(self, intent: Intent) -> List[UnitOperation]:
        """Quick OER screening workflow."""
        templates = []

        # 1. Electrode info
        templates.append(TemplateRegistry.get("oer", "ElectrodeInfo"))

        # 2. Electrolyte preparation
        templates.append(TemplateRegistry.get("oer", "ElectrolytePreparation"))

        # 3. Cell assembly
        templates.append(TemplateRegistry.get("oer", "CellAssembly"))

        # 4. Reference calibration
        templates.append(TemplateRegistry.get("oer", "ReferenceCalibration"))

        # 5. LSV measurement
        templates.append(TemplateRegistry.get("oer", "OER_LSV_Measurement"))

        # 6. Tafel analysis
        templates.append(TemplateRegistry.get("oer", "OER_Tafel_Analysis"))

        # 7. Save data
        templates.append(TemplateRegistry.get("oer", "DataSave"))

        # 8. Cleanup
        templates.append(TemplateRegistry.get("oer", "Cleanup"))

        # Pre-fill known conditions
        self._prefill_conditions(templates, intent)

        return [t for t in templates if t is not None]

    def _get_standard_characterization_workflow(self, intent: Intent) -> List[UnitOperation]:
        """Standard OER characterization workflow."""
        templates = []

        # Include all from fast, plus EIS
        templates.append(TemplateRegistry.get("oer", "ElectrodeInfo"))
        templates.append(TemplateRegistry.get("oer", "ElectrolytePreparation"))
        templates.append(TemplateRegistry.get("oer", "CellAssembly"))
        templates.append(TemplateRegistry.get("oer", "ReferenceCalibration"))
        templates.append(TemplateRegistry.get("oer", "OER_LSV_Measurement"))
        templates.append(TemplateRegistry.get("oer", "OER_EIS_Measurement"))  # Added
        templates.append(TemplateRegistry.get("oer", "OER_Tafel_Analysis"))
        templates.append(TemplateRegistry.get("oer", "OverpotentialAnalysis"))  # Added
        templates.append(TemplateRegistry.get("oer", "DataSave"))
        templates.append(TemplateRegistry.get("oer", "Cleanup"))

        self._prefill_conditions(templates, intent)

        return [t for t in templates if t is not None]

    def _get_comprehensive_workflow(self, intent: Intent) -> List[UnitOperation]:
        """Comprehensive OER study workflow."""
        templates = []

        # Full workflow including stability
        templates.append(TemplateRegistry.get("oer", "ElectrodeInfo"))
        templates.append(TemplateRegistry.get("oer", "ElectrolytePreparation"))
        templates.append(TemplateRegistry.get("oer", "CellAssembly"))
        templates.append(TemplateRegistry.get("oer", "ReferenceCalibration"))
        templates.append(TemplateRegistry.get("oer", "OER_LSV_Measurement"))
        templates.append(TemplateRegistry.get("oer", "OER_EIS_Measurement"))
        templates.append(TemplateRegistry.get("oer", "OER_Stability_Test"))  # Added
        templates.append(TemplateRegistry.get("oer", "OER_Tafel_Analysis"))
        templates.append(TemplateRegistry.get("oer", "OverpotentialAnalysis"))
        templates.append(TemplateRegistry.get("oer", "DataSave"))
        templates.append(TemplateRegistry.get("oer", "CellDisassembly"))  # Added
        templates.append(TemplateRegistry.get("oer", "Cleanup"))

        self._prefill_conditions(templates, intent)

        return [t for t in templates if t is not None]

    def _prefill_conditions(self, templates: List[UnitOperation], intent: Intent):
        """Pre-fill placeholders with known conditions from intent."""
        conditions = intent.known_conditions

        for template in templates:
            if template is None:
                continue

            # Map known conditions to placeholders
            condition_mapping = {
                "electrode": ["electrode_material"],
                "electrolyte": ["electrolyte_type"],
                "temperature": ["temperature_C"],
            }

            for condition_key, placeholder_keys in condition_mapping.items():
                if condition_key in conditions:
                    value = conditions[condition_key]
                    for pk in placeholder_keys:
                        if pk in template.placeholders:
                            template.fill_placeholder(pk, value)

    def _needs_comprehensive(self, intent: Intent) -> bool:
        """Check if intent suggests comprehensive workflow."""
        comprehensive_keywords = [
            "comprehensive", "complete", "full", "thorough",
            "publication", "paper", "study",
            "全面", "完整", "论文", "发表"
        ]
        text_lower = intent.original_text.lower()
        return any(kw in text_lower for kw in comprehensive_keywords)

    def get_default_assumptions(self, intent: Intent) -> List[str]:
        """Get default assumptions for OER experiments."""
        assumptions = [
            "Using three-electrode setup",
            "Reference electrode is properly calibrated",
            "Working electrode is clean and prepared",
        ]

        # Add electrolyte assumption
        if "electrolyte" not in intent.known_conditions:
            assumptions.append("Default electrolyte: 1M KOH (alkaline)")

        # Add gas assumption
        assumptions.append("Electrolyte is O2-saturated for OER")

        return assumptions

    def get_alternatives(self, intent: Intent) -> List[str]:
        """Get alternative approaches for OER."""
        alternatives = []

        if "stability" not in intent.target_metrics:
            alternatives.append("Add stability test for long-term performance")

        if "impedance" not in intent.target_metrics:
            alternatives.append("Add EIS for charge transfer analysis")

        alternatives.append("Use acidic electrolyte (H2SO4) for comparison")
        alternatives.append("Add rotating disk electrode (RDE) measurement")

        return alternatives[:3]  # Limit to 3


class GeneralDomainKnowledge(DomainKnowledge):
    """
    General domain knowledge for non-specialized experiments.
    """

    @property
    def domain_name(self) -> str:
        return "general"

    def get_workflow_templates(self, intent: Intent) -> List[List[UnitOperation]]:
        """Generate a basic workflow."""
        # For general domain, return empty and let user specify
        return [[]]

    def get_default_assumptions(self, intent: Intent) -> List[str]:
        return ["General experiment workflow"]

    def get_alternatives(self, intent: Intent) -> List[str]:
        return ["Specify experiment type for more specific workflow"]
