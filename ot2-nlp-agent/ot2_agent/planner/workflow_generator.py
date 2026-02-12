"""
Workflow Generator - Generate candidate workflow drafts.

This module generates WorkflowDraft candidates from an Intent
using domain knowledge.
"""

from typing import Dict, List, Optional

from ..ir import Intent, UnitOperation, MissingInfo, PlanningContext
from .domain_knowledge import DomainKnowledge, OERDomainKnowledge, GeneralDomainKnowledge


class WorkflowGenerator:
    """
    Generates candidate workflow drafts from user intent.
    """

    # Domain knowledge registry
    DOMAIN_KNOWLEDGE = {
        "electrochemistry": OERDomainKnowledge,
        "general": GeneralDomainKnowledge,
    }

    def __init__(self):
        """Initialize workflow generator."""
        self._domain_instances: Dict[str, DomainKnowledge] = {}

    def get_domain_knowledge(self, domain: str) -> DomainKnowledge:
        """Get or create domain knowledge instance."""
        if domain not in self._domain_instances:
            knowledge_class = self.DOMAIN_KNOWLEDGE.get(
                domain,
                GeneralDomainKnowledge
            )
            self._domain_instances[domain] = knowledge_class()
        return self._domain_instances[domain]

    def generate(
        self,
        intent: Intent,
        context: PlanningContext = None
    ) -> List["WorkflowDraft"]:
        """
        Generate candidate workflow drafts.

        Args:
            intent: Parsed user intent
            context: Planning context with known conditions

        Returns:
            List of WorkflowDraft candidates
        """
        # Import here to avoid circular import
        from .planner import WorkflowDraft

        # Get domain knowledge
        domain_knowledge = self.get_domain_knowledge(intent.domain)

        # Get workflow templates
        workflow_templates = domain_knowledge.get_workflow_templates(intent)

        # Get assumptions and alternatives
        default_assumptions = domain_knowledge.get_default_assumptions(intent)
        alternatives = domain_knowledge.get_alternatives(intent)

        # Generate drafts
        drafts = []
        workflow_names = self._get_workflow_names(intent, len(workflow_templates))

        for i, uo_list in enumerate(workflow_templates):
            # Collect all missing info from UOs
            missing_info = []
            for uo in uo_list:
                missing_info.extend(uo.get_missing_info())

            # Deduplicate by parameter name
            seen_params = set()
            unique_missing = []
            for mi in missing_info:
                if mi.parameter not in seen_params:
                    unique_missing.append(mi)
                    seen_params.add(mi.parameter)

            # Calculate confidence based on completeness
            confidence = self._calculate_draft_confidence(
                intent,
                uo_list,
                unique_missing
            )

            draft = WorkflowDraft(
                name=workflow_names[i]["name"],
                description=workflow_names[i]["description"],
                description_zh=workflow_names[i]["description_zh"],
                unit_operations=uo_list,
                assumptions=default_assumptions.copy(),
                missing_info=unique_missing,
                confidence=confidence,
                alternatives=alternatives if i == 0 else [],
            )

            drafts.append(draft)

        # Sort by confidence
        drafts.sort(key=lambda d: d.confidence, reverse=True)

        return drafts

    def _get_workflow_names(
        self,
        intent: Intent,
        num_workflows: int
    ) -> List[Dict[str, str]]:
        """Generate names and descriptions for workflows."""
        names = []

        if intent.domain == "electrochemistry":
            if num_workflows >= 1:
                names.append({
                    "name": "OER Fast Characterization",
                    "description": "Quick screening workflow with LSV and Tafel analysis",
                    "description_zh": "快速筛选流程：LSV和Tafel分析",
                })
            if num_workflows >= 2:
                names.append({
                    "name": "OER Standard Characterization",
                    "description": "Standard workflow with LSV, EIS, and overpotential analysis",
                    "description_zh": "标准表征流程：LSV、EIS和过电位分析",
                })
            if num_workflows >= 3:
                names.append({
                    "name": "OER Comprehensive Study",
                    "description": "Full characterization including stability test",
                    "description_zh": "全面表征：包含稳定性测试",
                })
        else:
            for i in range(num_workflows):
                names.append({
                    "name": f"Workflow Option {i + 1}",
                    "description": f"Generated workflow option {i + 1}",
                    "description_zh": f"生成的工作流程选项 {i + 1}",
                })

        return names

    def _calculate_draft_confidence(
        self,
        intent: Intent,
        uo_list: List[UnitOperation],
        missing_info: List[MissingInfo]
    ) -> float:
        """
        Calculate confidence score for a workflow draft.

        Higher confidence means:
        - Better match to user's stated metrics
        - Fewer missing required parameters
        - More complete workflow
        """
        base_confidence = intent.confidence

        # Penalize for missing required info
        required_missing = sum(1 for mi in missing_info if mi.required)
        if required_missing > 0:
            base_confidence -= min(required_missing * 0.05, 0.2)

        # Bonus for matching target metrics
        if intent.target_metrics:
            # This is a simplified check - could be more sophisticated
            metrics_covered = self._check_metrics_coverage(uo_list, intent.target_metrics)
            base_confidence += metrics_covered * 0.1

        return max(0.1, min(base_confidence, 0.95))

    def _check_metrics_coverage(
        self,
        uo_list: List[UnitOperation],
        target_metrics: List[str]
    ) -> float:
        """
        Check how well the workflow covers target metrics.

        Returns a score from 0 to 1.
        """
        if not target_metrics:
            return 0.5

        # Map metrics to UO types that produce them
        metric_to_uo = {
            "overpotential": ["OER_LSV_Measurement", "OverpotentialAnalysis"],
            "tafel_slope": ["OER_Tafel_Analysis"],
            "stability": ["OER_Stability_Test"],
            "impedance": ["OER_EIS_Measurement"],
            "current_density": ["OER_LSV_Measurement"],
        }

        uo_names = {uo.name for uo in uo_list}
        covered = 0

        for metric in target_metrics:
            required_uos = metric_to_uo.get(metric, [])
            if any(uo in uo_names for uo in required_uos):
                covered += 1

        return covered / len(target_metrics) if target_metrics else 0.5
