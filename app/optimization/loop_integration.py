"""Loop-facing seam for candidate-pool arbitration (feat: loop wiring).

The campaign loop calls ``arbitrate_round_if_enabled`` once it has the
authority's ``StrategyDecision``.  The seam is deliberately *fail-safe*:

* **flag off**  -> returns ``None`` immediately; the loop runs its existing
  generation path completely unchanged.
* **failure**   -> any exception from pool construction / arbitration is
  swallowed and ``None`` is returned, so a candidate-pool failure can neither
  downgrade the round nor break the campaign -- the loop falls back to its
  legacy path.

When enabled and successful it returns the ``OptimizationOutcome`` from
``arbitrate_next`` (which already degrades a Nexus outage to the local
baseline and records both the Δ2 backend-selection trace and the Δ1 scored
candidate-pool trace in provenance).

This module owns no campaign authority: it neither computes phase/weights nor
selects archetypes -- it forwards the authority ``StrategyDecision`` to
``arbitrate_next``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.optimization.schemas import OptimizationRequest
from app.optimization.service import OptimizationOutcome, arbitrate_next

if TYPE_CHECKING:  # pragma: no cover
    from app.optimization.arbitration_config import ArbitrationConfig
    from app.optimization.candidate_pool import CandidatePoolBuilder
    from app.optimization.decision_policy import OptimizationDecisionPolicy
    from app.optimization.provenance import ProvenanceLogger
    from app.optimization.schemas import OptimizationProvider
    from app.services.strategy_models import StrategyDecision

logger = logging.getLogger(__name__)


def build_optimization_request(
    *,
    campaign_id: str,
    dimensions: list[dict[str, Any]],
    protocol_template: dict[str, Any],
    all_params: list[dict[str, Any]],
    all_kpis: list[float],
    objective_name: str = "objective",
    direction: str = "maximize",
    n: int = 1,
    seed: int = 42,
    round_index: int = 0,
) -> OptimizationRequest:
    """Assemble an ``OptimizationRequest`` from campaign-loop context.

    Reconstructs the ``ParameterSpace`` from dimension dicts (same shape the
    DesignAgent uses) and zips the observation history into ``Observation``s.
    Pure and side-effect-free so the loop wiring stays testable.
    """
    from app.services.candidate_gen import ParameterSpace, SearchDimension
    from app.services.optimization_backends import Observation

    dims = []
    for d in dimensions:
        choices = d.get("choices")
        if choices is not None:
            choices = tuple(choices)
        dims.append(
            SearchDimension(
                param_name=d["param_name"],
                param_type=d.get("param_type", "number"),
                min_value=d.get("min_value"),
                max_value=d.get("max_value"),
                log_scale=d.get("log_scale", False),
                choices=choices,
                step_key=d.get("step_key"),
                primitive=d.get("primitive"),
            )
        )
    space = ParameterSpace(dimensions=tuple(dims), protocol_template=protocol_template)

    observations = tuple(
        Observation(params=dict(p), objective=float(k))
        for p, k in zip(all_params, all_kpis, strict=False)
    )

    return OptimizationRequest(
        campaign_id=campaign_id,
        space=space,
        observations=observations,
        objective_name=objective_name,
        direction=direction,
        n=n,
        seed=seed,
        round_index=round_index,
    )


def arbitrate_round_if_enabled(
    request: OptimizationRequest,
    decision: StrategyDecision,
    *,
    enabled: bool,
    provider: OptimizationProvider | None = None,
    fallback: OptimizationProvider | None = None,
    policy: OptimizationDecisionPolicy | None = None,
    provenance: ProvenanceLogger | None = None,
    pool_builder: CandidatePoolBuilder | None = None,
    config: ArbitrationConfig | None = None,
) -> OptimizationOutcome | None:
    """Run deep arbitration only when ``enabled``; never break the campaign.

    Returns ``None`` when disabled or when arbitration fails -- the caller then
    uses its existing generation path (no downgrade, no break).
    """
    if not enabled:
        return None
    try:
        return arbitrate_next(
            request,
            decision,
            provider=provider,
            fallback=fallback,
            policy=policy,
            provenance=provenance,
            pool_builder=pool_builder,
            config=config,
        )
    except Exception:
        logger.warning(
            "Candidate-pool arbitration failed; falling back to legacy "
            "generation for this round (no downgrade, campaign continues)",
            exc_info=True,
        )
        return None
