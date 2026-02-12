"""Design/Optimization Agent -- L2 planning layer.

Wraps existing candidate_gen.py and bayesian_opt.py.
Responsible for "what parameters to try next".
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent


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


class DesignAgent(BaseAgent[DesignInput, DesignOutput]):
    name = "design_agent"
    description = "Parameter space exploration (BO/LHS/random)"
    layer = "L2"

    def validate_input(self, input_data: DesignInput) -> list[str]:
        errors: list[str] = []
        if not input_data.dimensions:
            errors.append("At least one search dimension required")
        if input_data.batch_size < 1:
            errors.append("batch_size must be >= 1")
        return errors

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

        return DesignOutput(
            batch_id=batch.batch_id,
            candidates=[c.params for c in batch.candidates],
            strategy_used=batch.strategy,
            n_candidates=len(batch.candidates),
        )
