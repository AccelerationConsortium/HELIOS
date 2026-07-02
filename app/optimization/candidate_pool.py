"""Multi-source candidate pool with archetype-projected utility (Δ3/Δ7).

Instead of taking a single backend's batch verbatim, the pool gathers
candidates from several tagged *sources*, applies a *hard gate*, scores the
survivors with a utility whose component weights are *projected from the problem
archetype* (derived from the Nexus fingerprint), selects the top-N, and records
*scored-pool provenance* for every candidate considered.

Sources
-------
* ``nexus_top_k``     -- top-K from the fingerprint-selected optimizer backend.
* ``local_baseline``  -- built-in suggestions (always-available diversity/safety).
* ``replicate_best``  -- re-propose the current best point (noise reduction);
                         exempt from the duplicate gate by design.
* ``recovery_probe``  -- a safe-region probe near the best-known point.

Utility components: ``acquisition`` (optimizer value), ``diversity`` (novelty
vs history), ``robustness`` (proximity to known-good), ``replication``.

The module is pure-Python, deterministic under the request seed, and opt-in:
nothing in the live decision path calls it until it is explicitly wired in.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.optimization.arbitration_config import ArbitrationConfig
from app.optimization.decision_policy import _bounds_violation, _signature
from app.optimization.schemas import (
    CandidatePool as ArbitrationCandidatePool,
    CandidateSuggestion,
    OptimizationRequest,
    PooledCandidate as ArbitrationPooledCandidate,
)
from app.services.candidate_gen import sample_lhs
from app.services.optimization_backends import Observation, get_backend

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_models import StrategyDecision

COMPONENTS = ("acquisition", "diversity", "robustness", "replication")

DEFAULT_SOURCES = ("nexus_top_k", "local_baseline", "replicate_best", "recovery_probe")

# Per-source priors for the acquisition/robustness components (0-1).
_ACQ_PRIOR = {
    "nexus_top_k": 0.8, "local_baseline": 0.5, "replicate_best": 0.15, "recovery_probe": 0.3,
}
_ROB_PRIOR = {
    "nexus_top_k": 0.3, "local_baseline": 0.4, "replicate_best": 0.9, "recovery_probe": 0.8,
}

# Archetype -> component weights (each row sums to 1.0).
_ARCHETYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "standard":        {"acquisition": 0.6, "diversity": 0.20, "robustness": 0.10, "replication": 0.10},
    "high_noise":      {"acquisition": 0.3, "diversity": 0.15, "robustness": 0.25, "replication": 0.30},
    "tiny_data":       {"acquisition": 0.3, "diversity": 0.50, "robustness": 0.10, "replication": 0.10},
    "multi_objective": {"acquisition": 0.4, "diversity": 0.40, "robustness": 0.10, "replication": 0.10},
    "high_dim":        {"acquisition": 0.5, "diversity": 0.35, "robustness": 0.10, "replication": 0.05},
}

_HIGH_DIM = 10


@dataclass(frozen=True)
class PooledCandidate:
    params: dict[str, Any]
    source: str
    raw_scores: dict[str, float]
    archetype_weights: dict[str, float]
    final_score: float
    gate_status: str  # "passed" | "rejected:<reason>"
    selected: bool


@dataclass(frozen=True)
class PoolResult:
    selected: tuple[dict[str, Any], ...]
    candidates: tuple[PooledCandidate, ...]
    archetype: str
    archetype_weights: dict[str, float]
    source_distribution: dict[str, int]


# ---------------------------------------------------------------------------
# Archetype projection
# ---------------------------------------------------------------------------


def derive_archetype(fingerprint: dict[str, Any]) -> str:
    """Classify the problem into a coarse archetype from a Nexus fingerprint."""
    if fingerprint.get("objective_form") == "multi_objective":
        return "multi_objective"
    if fingerprint.get("noise_regime") == "high":
        return "high_noise"
    if fingerprint.get("data_scale") == "tiny":
        return "tiny_data"
    dim = fingerprint.get("effective_dimensionality", -1)
    if isinstance(dim, int) and dim >= _HIGH_DIM:
        return "high_dim"
    return "standard"


def archetype_weights(archetype: str) -> dict[str, float]:
    """Utility component weights for an archetype (defaults to ``standard``)."""
    return dict(_ARCHETYPE_WEIGHTS.get(archetype, _ARCHETYPE_WEIGHTS["standard"]))


# ---------------------------------------------------------------------------
# Distance / diversity helpers
# ---------------------------------------------------------------------------


def _distance(a: dict[str, Any], b: dict[str, Any], space: Any) -> float:
    dims = space.dimensions
    if not dims:
        return 1.0
    total = 0.0
    for d in dims:
        va, vb = a.get(d.param_name), b.get(d.param_name)
        if d.param_type in ("categorical", "boolean"):
            total += 0.0 if va == vb else 1.0
        else:
            lo, hi = d.min_value, d.max_value
            rng = (hi - lo) if (lo is not None and hi is not None and hi > lo) else 1.0
            try:
                total += abs(float(va) - float(vb)) / rng
            except (TypeError, ValueError):
                total += 1.0
    return min(1.0, total / len(dims))


def _diversity(params: dict[str, Any], observations: tuple[Observation, ...], space: Any) -> float:
    if not observations:
        return 1.0
    return min(_distance(params, o.params, space) for o in observations)


def _raw_scores(
    params: dict[str, Any],
    source: str,
    observations: tuple[Observation, ...],
    space: Any,
) -> dict[str, float]:
    return {
        "acquisition": _ACQ_PRIOR.get(source, 0.4),
        "diversity": _diversity(params, observations, space),
        "robustness": _ROB_PRIOR.get(source, 0.4),
        "replication": 1.0 if source == "replicate_best" else 0.0,
    }


# ---------------------------------------------------------------------------
# Source collectors
# ---------------------------------------------------------------------------


def _perturb_best(best: dict[str, Any], space: Any, rng: random.Random) -> dict[str, Any]:
    """Small safe-region probe around the best-known point."""
    probe = dict(best)
    for d in space.dimensions:
        if d.param_type in ("categorical", "boolean"):
            continue
        lo, hi = d.min_value, d.max_value
        if lo is None or hi is None:
            continue
        span = (hi - lo) * 0.1
        val = float(best.get(d.param_name, (lo + hi) / 2)) + rng.uniform(-span, span)
        probe[d.param_name] = max(lo, min(hi, val))
    return probe


def _collect_sources(
    request: OptimizationRequest,
    backend_name: str,
    k: int,
    sources: tuple[str, ...],
    rng: random.Random,
) -> list[tuple[dict[str, Any], str]]:
    obs = list(request.observations)
    space = request.space
    out: list[tuple[dict[str, Any], str]] = []

    if "nexus_top_k" in sources:
        backend = get_backend(backend_name)
        for c in backend.suggest(space, k, obs, seed=request.seed):
            out.append((c, "nexus_top_k"))
    if "local_baseline" in sources:
        backend = get_backend("built_in")
        for c in backend.suggest(space, k, obs, seed=request.seed + 1):
            out.append((c, "local_baseline"))
    if "replicate_best" in sources and obs:
        best = max(obs, key=lambda o: o.objective)
        out.append((dict(best.params), "replicate_best"))
    if "recovery_probe" in sources and obs:
        best = max(obs, key=lambda o: o.objective)
        out.append((_perturb_best(best.params, space, rng), "recovery_probe"))

    return out


# ---------------------------------------------------------------------------
# Pool assembly
# ---------------------------------------------------------------------------


def build_candidate_pool(
    request: OptimizationRequest,
    *,
    fingerprint: dict[str, Any] | None = None,
    backend_name: str = "built_in",
    k: int = 4,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    select_n: int | None = None,
    safety_check: Any | None = None,
) -> PoolResult:
    """Assemble, gate, score, and select a candidate pool with provenance."""
    fingerprint = fingerprint or {}
    select_n = request.n if select_n is None else select_n
    archetype = derive_archetype(fingerprint)
    weights = archetype_weights(archetype)
    rng = random.Random(request.seed)

    raw = _collect_sources(request, backend_name, k, sources, rng)

    seen = {_signature(o.params) for o in request.observations}
    scored: list[dict[str, Any]] = []
    for params, source in raw:
        violation = _bounds_violation(params, request.space)
        sig = _signature(params)
        if violation is not None:
            status = f"rejected:{violation}"
        elif sig in seen and source != "replicate_best":
            # replicate_best intentionally re-proposes a known point.
            status = "rejected:duplicate"
        elif safety_check is not None and not safety_check(params, request):
            status = "rejected:safety"
        else:
            status = "passed"
            seen.add(sig)

        rs = _raw_scores(params, source, request.observations, request.space)
        final = sum(weights[c] * rs[c] for c in COMPONENTS) if status == "passed" else 0.0
        scored.append({"params": params, "source": source, "raw": rs, "final": final, "status": status})

    passed = [p for p in scored if p["status"] == "passed"]
    # Deterministic: highest utility first, ties broken by source name then order.
    passed_sorted = sorted(
        passed,
        key=lambda p: (-p["final"], p["source"]),
    )
    top = passed_sorted[:select_n]
    top_ids = {id(p) for p in top}

    candidates = tuple(
        PooledCandidate(
            params=p["params"],
            source=p["source"],
            raw_scores={k_: round(v, 4) for k_, v in p["raw"].items()},
            archetype_weights=weights,
            final_score=round(p["final"], 4),
            gate_status=p["status"],
            selected=id(p) in top_ids,
        )
        for p in scored
    )
    selected = tuple(p["params"] for p in top)
    source_distribution = dict(Counter(c.source for c in candidates if c.selected))

    return PoolResult(
        selected=selected,
        candidates=candidates,
        archetype=archetype,
        archetype_weights=weights,
        source_distribution=source_distribution,
    )


# ---------------------------------------------------------------------------
# Authority-bound arbitration pool (main integration path)
# ---------------------------------------------------------------------------

_EXPLORATION_PHASES = {"exploration", "explore"}


def _authority_action(decision: StrategyDecision) -> str:
    """The archetype HELIOS authority selected this round."""
    if decision.actions_considered:
        return decision.actions_considered[0].name
    phase = decision.phase
    return {
        "exploration": "explore",
        "exploitation": "exploit",
        "refinement": "refine",
    }.get(phase, phase)


def _explicit_action(suggestion: CandidateSuggestion, index: int) -> str | None:
    """Read an explicit per-candidate archetype, if the provider supplied one."""
    if index < len(suggestion.per_candidate):
        meta = suggestion.per_candidate[index]
        action = meta.get("source_action") or meta.get("action") or meta.get("archetype")
        if action:
            return str(action)
    return None


class CandidatePoolBuilder:
    """Build a concrete arbitration pool from provider suggestions."""

    def __init__(self, config: ArbitrationConfig | None = None) -> None:
        self._config = config or ArbitrationConfig()

    def build(
        self,
        request: OptimizationRequest,
        decision: StrategyDecision,
        *,
        nexus_suggestion: CandidateSuggestion | None = None,
        local_suggestion: CandidateSuggestion | None = None,
        extra_suggestions: tuple[CandidateSuggestion, ...] = (),
    ) -> ArbitrationCandidatePool:
        cfg = self._config
        authority_action = _authority_action(decision)
        candidates: list[ArbitrationPooledCandidate] = []
        used: list[str] = []
        dropped: list[str] = []
        trace: list[str] = []

        if nexus_suggestion is not None and nexus_suggestion.candidates:
            for i, params in enumerate(nexus_suggestion.candidates):
                candidates.append(
                    ArbitrationPooledCandidate(
                        params=dict(params),
                        source="nexus",
                        source_action=_explicit_action(nexus_suggestion, i) or authority_action,
                        generator_backend=nexus_suggestion.algorithm,
                        rationale=nexus_suggestion.rationale,
                    )
                )
            used.append("nexus")
            trace.append(
                f"nexus: {len(nexus_suggestion.candidates)} candidate(s) via "
                f"{nexus_suggestion.algorithm}, archetype={authority_action}"
            )
        else:
            dropped.append("nexus")
            trace.append("nexus: no suggestion (dropped)")

        if cfg.include_local_baseline and local_suggestion is not None and local_suggestion.candidates:
            for params in local_suggestion.candidates:
                candidates.append(
                    ArbitrationPooledCandidate(
                        params=dict(params),
                        source="local",
                        source_action=authority_action,
                        generator_backend=local_suggestion.algorithm,
                        rationale=local_suggestion.rationale,
                    )
                )
            used.append("local")
            trace.append(f"local: {len(local_suggestion.candidates)} baseline(s)")
        elif cfg.include_local_baseline:
            dropped.append("local")
            trace.append("local: no baseline (dropped)")

        spec = decision.stabilize_spec
        if cfg.include_replicate_best and spec is not None and spec.points_to_replicate:
            for params in spec.points_to_replicate:
                candidates.append(
                    ArbitrationPooledCandidate(
                        params=dict(params),
                        source="replicate",
                        source_action="stabilize",
                        generator_backend="replicate",
                        rationale=spec.reason,
                    )
                )
            used.append("replicate")
            trace.append(
                f"replicate: {len(spec.points_to_replicate)} point(s) from stabilize_spec"
            )

        if cfg.include_sobol_in_exploration and decision.phase in _EXPLORATION_PHASES:
            diversity = sample_lhs(request.space, max(1, request.n), seed=request.seed)
            for params in diversity:
                candidates.append(
                    ArbitrationPooledCandidate(
                        params=dict(params),
                        source="sobol",
                        source_action="explore",
                        generator_backend="lhs",
                        rationale="space-filling diversity (exploration phase)",
                    )
                )
            used.append("sobol")
            trace.append(f"sobol/lhs: {len(diversity)} diversity candidate(s)")

        for sug in extra_suggestions:
            if sug is None or not sug.candidates:
                continue
            for i, params in enumerate(sug.candidates):
                candidates.append(
                    ArbitrationPooledCandidate(
                        params=dict(params),
                        source=sug.source,
                        source_action=_explicit_action(sug, i) or authority_action,
                        generator_backend=sug.algorithm,
                        rationale=sug.rationale,
                    )
                )
            used.append(sug.source)
            trace.append(f"{sug.source}: {len(sug.candidates)} candidate(s) via {sug.algorithm}")

        return ArbitrationCandidatePool(
            candidates=tuple(candidates),
            sources_used=tuple(used),
            sources_dropped=tuple(dropped),
            construction_trace=tuple(trace),
        )
