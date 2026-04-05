"""Design/Optimization Agent -- L2 planning layer.

Wraps existing candidate_gen.py and bayesian_opt.py.
Responsible for "what parameters to try next".

v3: Memory-enriched confidence calibration.
- After generating candidates, checks similarity to past experiments.
- Annotates each candidate with a confidence_hint and similar_kpi_estimate.
- Emits DecisionNode tracking the similarity assessment.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent, DecisionNode
from app.agents.pause import Granularity, PauseRequest

logger = logging.getLogger(__name__)


class DesignInput(BaseModel):
    """Input for parameter design."""
    dimensions: list[dict[str, Any]]
    protocol_template: dict[str, Any]
    strategy: str = "lhs"
    batch_size: int = 10
    seed: int | None = None
    campaign_id: str | None = None
    kpi_name: str = "overpotential_mv"
    store: bool = True  # False to skip DB persistence (e.g. dry_run orchestrator)


class DesignOutput(BaseModel):
    """Output from parameter design."""
    batch_id: str
    candidates: list[dict[str, Any]]
    strategy_used: str
    n_candidates: int
    # v3: similarity-based confidence annotations
    candidate_confidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-candidate confidence hints from similar experiment retrieval",
    )
    decision_nodes: list[dict[str, Any]] = Field(default_factory=list)


class DesignAgent(BaseAgent[DesignInput, DesignOutput]):
    name = "design_agent"
    description = "Parameter space exploration (BO/LHS/random)"
    layer = "L2"

    # v3: confidence thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.3
    PAUSE_ON_ALL_LOW_CONFIDENCE = True

    def validate_input(self, input_data: DesignInput) -> list[str]:
        errors: list[str] = []
        if not input_data.dimensions:
            errors.append("At least one search dimension required")
        if input_data.batch_size < 1:
            errors.append("batch_size must be >= 1")
        return errors

    async def assess_granularity(
        self,
        input_data: DesignInput,
        context: dict[str, Any] | None = None,
    ) -> Granularity:
        """Design agent uses ADAPTIVE — fine for novel regions, coarse for explored."""
        return Granularity.ADAPTIVE

    async def process(self, input_data: DesignInput) -> DesignOutput:
        from app.services.candidate_gen import (
            ParameterSpace,
            SearchDimension,
            generate_batch,
        )

        dims = []
        for d in input_data.dimensions:
            choices = d.get("choices")
            if choices is not None:
                choices = tuple(choices)
            dims.append(SearchDimension(
                param_name=d["param_name"],
                param_type=d.get("param_type", "number"),
                min_value=d.get("min_value"),
                max_value=d.get("max_value"),
                log_scale=d.get("log_scale", False),
                choices=choices,
                step_key=d.get("step_key"),
                primitive=d.get("primitive"),
            ))

        space = ParameterSpace(
            dimensions=tuple(dims),
            protocol_template=input_data.protocol_template,
        )

        batch = generate_batch(
            space,
            strategy=input_data.strategy,
            n_candidates=input_data.batch_size,
            seed=input_data.seed,
            campaign_id=input_data.campaign_id,
            kpi_name=input_data.kpi_name,
            store=input_data.store,
        )

        candidates = [c.params for c in batch.candidates]

        # ── v3: Similarity-based confidence calibration ───────────────
        candidate_confidence: list[dict[str, Any]] = []
        decision_nodes: list[dict[str, Any]] = []
        all_low_confidence = True

        try:
            from app.services.experiment_similarity import build_similarity_report

            for i, params in enumerate(candidates):
                report = build_similarity_report(
                    query_params=params,
                    campaign_id=input_data.campaign_id,
                    top_k=3,
                )
                conf_entry = {
                    "candidate_index": i,
                    "confidence": report.confidence_estimate,
                    "n_similar": len(report.matches),
                    "best_similarity": (
                        report.matches[0].similarity if report.matches else 0.0
                    ),
                    "expected_kpi": report.avg_kpi,
                    "kpi_uncertainty": report.kpi_stddev,
                    "explanation": report.explanation,
                }
                candidate_confidence.append(conf_entry)

                if report.confidence_estimate >= self.LOW_CONFIDENCE_THRESHOLD:
                    all_low_confidence = False

            # Emit decision node for similarity assessment
            high_conf = sum(
                1 for c in candidate_confidence
                if c["confidence"] >= self.LOW_CONFIDENCE_THRESHOLD
            )
            decision_nodes.append(DecisionNode(
                id="similarity_assessment",
                label="Similar experiment retrieval",
                options=["high_confidence", "mixed", "all_low_confidence", "no_data"],
                selected=(
                    "all_low_confidence" if all_low_confidence and candidate_confidence
                    else "no_data" if not candidate_confidence
                    else "high_confidence" if high_conf == len(candidate_confidence)
                    else "mixed"
                ),
                reason=(
                    f"{high_conf}/{len(candidate_confidence)} candidates have "
                    f"confidence >= {self.LOW_CONFIDENCE_THRESHOLD}"
                ),
            ).to_dict())

            # v3: If ALL candidates are in unexplored territory, pause
            if (
                self.PAUSE_ON_ALL_LOW_CONFIDENCE
                and all_low_confidence
                and candidate_confidence
            ):
                pause_result = await self.request_pause(PauseRequest(
                    reason=(
                        f"All {len(candidates)} candidates are in unexplored "
                        f"parameter territory — no similar past experiments found"
                    ),
                    risk_factors={
                        "all_low_confidence": 1.0,
                        "max_confidence": max(
                            (c["confidence"] for c in candidate_confidence),
                            default=0.0,
                        ),
                    },
                    suggested_action="approve",
                    checkpoint={"batch_id": batch.batch_id},
                ))
                # Don't block on rejection — just proceed with the batch.
                # The pause is informational, letting the operator know
                # these are novel experiments.

        except Exception:
            logger.debug("Similarity calibration failed (advisory)", exc_info=True)

        return DesignOutput(
            batch_id=batch.batch_id,
            candidates=candidates,
            strategy_used=batch.strategy,
            n_candidates=len(batch.candidates),
            candidate_confidence=candidate_confidence,
            decision_nodes=decision_nodes,
        )
