"""Inverse Design Agent -- L3 layer.

Given a performance objective (e.g. "HER catalyst with η10 < 50mV"),
recommends candidate element systems, precursors, and search dimensions
that can be directly consumed by DesignAgent (L2).

Design principle: built-in knowledge base is the fallback (no LLM required).
LLM is used as an *enhancement* when available.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O Models
# ---------------------------------------------------------------------------


class InverseDesignInput(BaseModel):
    """Input: what performance do you want?"""
    objective: str  # e.g. "HER catalyst with η10 < 50mV in 1M KOH"
    target_metrics: dict[str, dict]  # {"eta10_mv": {"direction": "minimize", "target": 50}}
    constraints: dict[str, Any] = Field(default_factory=dict)
    lab_capabilities: dict[str, Any] = Field(default_factory=dict)
    max_systems: int = 3
    search_mode: str = "literature"  # "literature" | "database" | "generative"


class CandidateSystem(BaseModel):
    """A recommended catalyst system."""
    system_name: str
    elements: list[str]
    rationale: str
    literature_refs: list[str]
    predicted_performance: dict[str, float]
    confidence: float  # 0-1
    recommended_precursors: list[dict]


class InverseDesignOutput(BaseModel):
    """Output: candidate systems + DesignAgent-ready parameters."""
    candidate_systems: list[CandidateSystem]
    recommended_stock_solutions: list[dict]
    suggested_dimensions: list[dict]  # SearchDimension-compatible
    suggested_protocol_template: dict
    search_summary: str


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class InverseDesignAgent(BaseAgent[InverseDesignInput, InverseDesignOutput]):
    """Inverse design: from performance target → candidate systems → search space."""

    name = "inverse_design_agent"
    description = "Recommend catalyst systems and search spaces from performance targets"
    layer = "L3"

    def __init__(self, llm_provider: Any | None = None) -> None:
        super().__init__()
        self._llm = llm_provider

    def validate_input(self, input_data: InverseDesignInput) -> list[str]:
        errors: list[str] = []
        if not input_data.objective:
            errors.append("objective is required")
        if not input_data.target_metrics:
            errors.append("target_metrics is required (at least one metric)")
        if input_data.max_systems < 1:
            errors.append("max_systems must be >= 1")
        if input_data.search_mode not in ("literature", "database", "generative"):
            errors.append(f"search_mode must be literature|database|generative, got {input_data.search_mode}")
        return errors

    async def process(self, input_data: InverseDesignInput) -> InverseDesignOutput:
        # Step 1: Try LLM-enhanced recommendation
        if self._llm and input_data.search_mode == "generative":
            try:
                return await self._process_with_llm(input_data)
            except Exception as exc:
                self.logger.warning("LLM recommendation failed, falling back to knowledge base: %s", exc)

        # Step 2: Knowledge base fallback (always works)
        result = self._process_with_knowledge_base(input_data)

        # Step 3: Optional Nexus enhancement
        result = self._enhance_with_nexus(result, input_data)

        return result

    def _process_with_knowledge_base(self, input_data: InverseDesignInput) -> InverseDesignOutput:
        """Use built-in electrocatalyst knowledge for recommendations."""
        from app.services.electrocatalyst_knowledge import (
            CatalystSystem,
            get_precursors_for_elements,
            infer_reaction_from_objective,
            recommend_systems,
        )

        reaction = infer_reaction_from_objective(input_data.objective)
        systems = recommend_systems(
            reaction=reaction,
            target_metrics=input_data.target_metrics,
            constraints=input_data.constraints,
            max_systems=input_data.max_systems,
        )

        if not systems:
            # Return empty but valid output
            return InverseDesignOutput(
                candidate_systems=[],
                recommended_stock_solutions=[],
                suggested_dimensions=[],
                suggested_protocol_template={},
                search_summary=f"No matching systems found for {reaction} with given constraints.",
            )

        # Build candidate systems
        candidates = []
        all_precursors: list[dict] = []
        all_elements: set[str] = set()

        for sys in systems:
            precursors = get_precursors_for_elements(sys.elements)
            # Predicted performance = typical values
            pred_perf = {
                metric: vals.get("typical", 0)
                for metric, vals in sys.typical_performance.items()
            }
            # Confidence: higher if typical is close to target
            confidence = self._compute_confidence(pred_perf, input_data.target_metrics)

            candidates.append(CandidateSystem(
                system_name=sys.name,
                elements=sys.elements,
                rationale=self._build_rationale(sys, reaction, input_data),
                literature_refs=sys.literature_refs,
                predicted_performance=pred_perf,
                confidence=confidence,
                recommended_precursors=precursors,
            ))
            all_precursors.extend(precursors)
            all_elements.update(sys.elements)

        # Deduplicate precursors by element
        seen_elements: set[str] = set()
        unique_precursors: list[dict] = []
        for p in all_precursors:
            if p["element"] not in seen_elements:
                seen_elements.add(p["element"])
                unique_precursors.append(p)

        # Build DesignAgent-compatible dimensions
        max_wells = input_data.lab_capabilities.get("max_wells", 24)
        dimensions = self._build_dimensions(list(all_elements), unique_precursors)

        # Build protocol template
        protocol_template = self._build_protocol_template(input_data)

        summary_parts = [
            f"Reaction: {reaction}",
            f"Found {len(candidates)} candidate system(s):",
        ]
        for c in candidates:
            summary_parts.append(f"  - {c.system_name} ({', '.join(c.elements)}): confidence {c.confidence:.2f}")

        return InverseDesignOutput(
            candidate_systems=candidates,
            recommended_stock_solutions=unique_precursors,
            suggested_dimensions=dimensions,
            suggested_protocol_template=protocol_template,
            search_summary="\n".join(summary_parts),
        )

    async def _process_with_llm(self, input_data: InverseDesignInput) -> InverseDesignOutput:
        """Use LLM for enhanced recommendations (generative mode)."""
        from app.services.llm_gateway import LLMMessage

        system_prompt = (
            "You are an expert electrocatalyst researcher. Given performance targets "
            "and constraints, recommend catalyst systems. Respond in JSON format with "
            "fields: systems (list of {name, elements, rationale, refs, performance, confidence})."
        )
        user_msg = (
            f"Objective: {input_data.objective}\n"
            f"Target metrics: {json.dumps(input_data.target_metrics)}\n"
            f"Constraints: {json.dumps(input_data.constraints)}\n"
            f"Lab capabilities: {json.dumps(input_data.lab_capabilities)}\n"
            f"Max systems: {input_data.max_systems}"
        )

        response = await self._llm.complete(
            messages=[LLMMessage(role="user", content=user_msg)],
            system=system_prompt,
        )

        # Parse LLM response and merge with knowledge base
        # For MVP, fall back to knowledge base and use LLM output as supplementary
        self.logger.info("LLM response received (%d chars), merging with knowledge base", len(response.content))
        kb_result = self._process_with_knowledge_base(input_data)

        # Append LLM summary
        kb_result.search_summary += f"\n\nLLM enhancement: {response.content[:500]}"
        return kb_result

    def _enhance_with_nexus(
        self,
        result: InverseDesignOutput,
        input_data: InverseDesignInput,
    ) -> InverseDesignOutput:
        """Optionally enhance results with Nexus causal discovery and hypothesis tracking.

        All Nexus calls are wrapped in try/except — if Nexus is unreachable,
        the original result is returned unchanged.
        """
        try:
            from app.services.nexus_advisor import NexusAdvisor

            nexus = NexusAdvisor()

            # Build historical experiment data from candidate predictions
            # (uses predicted performance as proxy for causal discovery)
            if not result.candidate_systems:
                return result

            # Prepare data matrix from candidate predicted performance
            all_metrics = set()
            for c in result.candidate_systems:
                all_metrics.update(c.predicted_performance.keys())
            metric_names = sorted(all_metrics)

            if not metric_names:
                return result

            causal_data: list[list[float]] = []
            for c in result.candidate_systems:
                row = [c.predicted_performance.get(m, 0.0) for m in metric_names]
                causal_data.append(row)

            # Causal discovery
            causal_edges = nexus.causal_discovery(
                data=causal_data,
                var_names=metric_names,
            )

            # Hypothesis status (build tracker from objectives)
            hypotheses_raw = [
                {
                    "id": f"hyp_{m}",
                    "statement": f"{m} is achievable at target level",
                    "status": "TESTING",
                    "evidence": [],
                    "tests_run": len(result.candidate_systems),
                }
                for m in input_data.target_metrics
            ]
            hyp_status = nexus.hypothesis_status({
                "hypotheses": hypotheses_raw,
            })

            # Re-rank candidates based on causal evidence
            if causal_edges:
                # Boost candidates whose elements align with strong causal drivers
                boosted = list(result.candidate_systems)
                for i, candidate in enumerate(boosted):
                    boost = sum(
                        e.strength * 0.1
                        for e in causal_edges
                        if e.strength > 0.5
                    )
                    boosted[i] = candidate.model_copy(update={
                        "confidence": min(1.0, candidate.confidence + boost),
                    })
                # Re-sort by confidence
                boosted.sort(key=lambda c: c.confidence, reverse=True)
            else:
                boosted = list(result.candidate_systems)

            # Build enriched summary
            nexus_notes: list[str] = []
            if causal_edges:
                nexus_notes.append(
                    f"Nexus causal: {len(causal_edges)} edge(s) discovered"
                )
            if hyp_status:
                supported = sum(1 for h in hyp_status if h.status == "SUPPORTED")
                refuted = sum(1 for h in hyp_status if h.status == "REFUTED")
                nexus_notes.append(
                    f"Hypotheses: {supported} supported, {refuted} refuted"
                )

            # Sync a mirror campaign (best-effort) and capture campaign_id
            nexus_campaign_id: str | None = None
            if causal_data:
                obs_dicts = [
                    {m: row[j] for j, m in enumerate(metric_names)}
                    for row in causal_data
                ]
                nexus_campaign_id = nexus.sync_campaign(
                    campaign_id="inverse-design-mirror",
                    observations=[{str(k): str(v) for k, v in d.items()} for d in obs_dicts],
                    name=f"InverseDesign: {input_data.objective[:60]}",
                )

            # Build updated output
            summary = result.search_summary
            if nexus_notes:
                summary += "\n\nNexus insights: " + "; ".join(nexus_notes)
            if nexus_campaign_id:
                summary += f"\nNexus campaign_id: {nexus_campaign_id}"

            # Rebuild protocol template with nexus_campaign_id in metadata
            proto = dict(result.suggested_protocol_template)
            if nexus_campaign_id:
                proto["nexus_campaign_id"] = nexus_campaign_id

            return InverseDesignOutput(
                candidate_systems=boosted[:input_data.max_systems],
                recommended_stock_solutions=result.recommended_stock_solutions,
                suggested_dimensions=result.suggested_dimensions,
                suggested_protocol_template=proto,
                search_summary=summary,
            )
        except Exception as exc:
            self.logger.debug("Nexus enhancement skipped: %s", exc)
            return result

    @staticmethod
    def _compute_confidence(
        predicted: dict[str, float],
        targets: dict[str, dict],
    ) -> float:
        """Compute confidence score (0-1) based on how well predicted meets targets."""
        if not targets:
            return 0.5
        scores = []
        for metric, spec in targets.items():
            pred = predicted.get(metric)
            target = spec.get("target")
            if pred is None or target is None:
                continue
            direction = spec.get("direction", "minimize")
            if direction == "minimize":
                score = min(target / max(pred, 1), 1.0)
            else:
                score = min(pred / max(target, 1), 1.0)
            scores.append(score)
        return round(sum(scores) / max(len(scores), 1), 2)

    @staticmethod
    def _build_rationale(
        sys: Any,
        reaction: str,
        input_data: InverseDesignInput,
    ) -> str:
        """Build a human-readable rationale string."""
        parts = [f"{sys.name} is a known {reaction} catalyst"]
        perf = sys.typical_performance
        for metric, vals in perf.items():
            parts.append(f"typical {metric}={vals.get('typical', '?')}")
        synth = input_data.constraints.get("synthesis_method")
        if synth and synth in sys.synthesis_methods:
            parts.append(f"compatible with {synth}")
        return "; ".join(parts) + "."

    @staticmethod
    def _build_dimensions(
        elements: list[str],
        precursors: list[dict],
    ) -> list[dict]:
        """Build SearchDimension-compatible dicts for DesignAgent."""
        dimensions = []
        for p in precursors:
            el = p["element"]
            dimensions.append({
                "param_name": f"{el}_fraction",
                "param_type": "number",
                "min_value": 0.0,
                "max_value": 1.0,
                "log_scale": False,
                "step_key": "composition",
                "primitive": "aspirate_dispense",
            })
        return dimensions

    @staticmethod
    def _build_protocol_template(input_data: InverseDesignInput) -> dict:
        """Build a protocol template based on lab capabilities."""
        platform = input_data.lab_capabilities.get("platform", "OT-2")
        techniques = input_data.lab_capabilities.get("available_techniques", [])
        synth = input_data.constraints.get("synthesis_method", "electrodeposition")

        return {
            "platform": platform,
            "synthesis_method": synth,
            "characterization_techniques": techniques,
            "steps": [
                {"name": "prepare_stock_solutions", "type": "liquid_handling"},
                {"name": "dispense_compositions", "type": "liquid_handling"},
                {"name": synth, "type": "synthesis"},
                {"name": "characterization", "type": "measurement", "techniques": techniques},
            ],
        }
