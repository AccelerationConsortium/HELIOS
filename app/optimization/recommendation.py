"""Map a Nexus problem fingerprint to a HELIOS backend recommendation (Δ2c).

This is HELIOS's *interpretation* of Nexus's advice -- a small, explicit rule
set that turns a fingerprint into an ordered ``nexus_*`` preference.  It is only
ever a recommendation: ``rank_backends`` applies it as a soft bias, and HELIOS's
strategy selector retains the final say (availability and phase compatibility
still veto).

``recommend_backends`` drives this from a HELIOS ``CampaignSnapshot`` through
the in-process Nexus profiler (the ``diagnose`` path).  It degrades to an empty
tuple -- meaning "no recommendation, behave exactly as before" -- whenever Nexus
is unavailable or there is no usable history.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.strategy_models import CampaignSnapshot

logger = logging.getLogger(__name__)

_HIGH_DIM = 10


def fingerprint_to_backends(fingerprint: dict[str, Any]) -> tuple[str, ...]:
    """Ordered ``nexus_*`` preference implied by a fingerprint dict."""
    if fingerprint.get("objective_form") == "multi_objective":
        return ("nexus_nsga2",)
    if fingerprint.get("noise_regime") == "high":
        return ("nexus_rf_bo", "nexus_random", "nexus_lhs")
    if fingerprint.get("data_scale") == "tiny":
        return ("nexus_lhs", "nexus_sobol")
    dim = fingerprint.get("effective_dimensionality", -1)
    if isinstance(dim, int) and dim >= _HIGH_DIM:
        return ("nexus_cmaes", "nexus_turbo")
    return ("nexus_gp_bo", "nexus_tpe")


def _infer_space(params: tuple[dict[str, Any], ...]):
    """Build a minimal HELIOS ParameterSpace from observed parameter history."""
    from app.services.candidate_gen import ParameterSpace, SearchDimension

    sample = params[0]
    dims: list[SearchDimension] = []
    for key in sorted(sample):
        values = [p.get(key) for p in params if key in p]
        first = sample[key]
        if isinstance(first, bool):
            dims.append(
                SearchDimension(
                    param_name=key, param_type="boolean", choices=tuple(sorted({bool(v) for v in values}))
                )
            )
        elif isinstance(first, str):
            dims.append(
                SearchDimension(
                    param_name=key, param_type="categorical", choices=tuple(sorted({str(v) for v in values}))
                )
            )
        else:
            numeric = [float(v) for v in values if isinstance(v, (int, float))]
            ptype = "integer" if all(isinstance(v, int) for v in values) else "number"
            dims.append(
                SearchDimension(
                    param_name=key,
                    param_type=ptype,
                    min_value=min(numeric) if numeric else None,
                    max_value=max(numeric) if numeric else None,
                )
            )
    return ParameterSpace(dimensions=tuple(dims), protocol_template={})


def recommend_backends(snapshot: CampaignSnapshot) -> tuple[str, ...]:
    """Return a fingerprint-driven ``nexus_*`` preference for *snapshot*.

    Empty tuple => no recommendation (Nexus unavailable / insufficient history).
    """
    if not snapshot.all_params or not snapshot.all_kpis:
        return ()

    try:
        from app.optimization.nexus_provider import NexusOptimizationProvider
        from app.optimization.schemas import OptimizationRequest
        from app.services.optimization_backends import Observation

        provider = NexusOptimizationProvider()
        if not provider.is_available():
            return ()

        space = _infer_space(snapshot.all_params)
        observations = tuple(
            Observation(params=dict(p), objective=float(k))
            for p, k in zip(snapshot.all_params, snapshot.all_kpis, strict=False)
        )
        request = OptimizationRequest(
            campaign_id=getattr(snapshot, "nexus_campaign_id", None) or "default",
            space=space,
            observations=observations,
            objective_name="kpi",
            direction=snapshot.direction,
        )
        fingerprint, _ = provider.diagnose(request)
        return fingerprint_to_backends(fingerprint)
    except Exception:
        logger.debug("Backend recommendation skipped (unavailable or error)", exc_info=True)
        return ()
