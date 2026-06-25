"""Replay / A-B comparison harness for the Nexus optimization integration.

Drives the *real* HELIOS decision path -- ``select_strategy`` -> backend bridge
-> ``OptimizationDecisionPolicy`` -- round by round on a surrogate SDL oracle,
under two configurations:

    BEFORE : existing HELIOS backend selection
             (classic backends only, no fingerprint soft bias)
    AFTER  : PhaseConfig + Nexus fingerprint soft bias
             (nexus_* backends available, recommendation enabled)

and reports the metrics requested for the comparison:

    * selected backend distribution
    * fallback rate
    * improvement over rounds (best-KPI trajectory + simple regret)
    * duplicate candidate rate
    * constraint violation rate
    * provenance completeness
    * phase stability  (did the fingerprint change the phase on the *same*
      snapshot?  it must not)

Real SDL1 zinc-deposition data is intentionally kept out of this repo, so the
default oracle is the chemistry-inspired surrogate from ``sdl_benchmark``.
Point ``--task`` at any registered BenchmarkTask; the harness is data-source
agnostic and will accept a recorded campaign oracle when one is available.

Usage::

    python -m benchmarks.nexus_replay --task her --rounds 12 --seeds 5
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.optimization.decision_policy import OptimizationDecisionPolicy
from app.optimization.recommendation import recommend_backends
from app.optimization.schemas import CandidateSuggestion, OptimizationRequest
from app.services.optimization_backends import Observation, get_backend
from app.services.strategy_models import CampaignSnapshot, PhaseConfig
from app.services.strategy_selector import select_strategy
from benchmarks.sdl_benchmark import (
    _approx_optimum,
    her_catalyst_synthetic,
    oer_catalyst_synthetic,
)

# Classic (pre-Nexus) availability and the enriched availability.
_CLASSIC_AVAILABLE = {
    "built_in": True, "lhs": True, "random_sampling": True,
    "optuna_tpe": False, "optuna_cmaes": False, "scipy_de": False, "pymoo_nsga2": False,
}
_NEXUS_BACKENDS = (
    "nexus_tpe", "nexus_gp_bo", "nexus_cmaes", "nexus_de", "nexus_nsga2",
    "nexus_turbo", "nexus_lhs", "nexus_sobol", "nexus_random", "nexus_rf_bo",
)
_NEXUS_AVAILABLE = {**_CLASSIC_AVAILABLE, **{b: True for b in _NEXUS_BACKENDS}}


@dataclass
class ReplayMetrics:
    label: str
    n_rounds: int
    backend_distribution: dict[str, int] = field(default_factory=dict)
    fallback_rate: float = 0.0
    best_kpi_trajectory: list[float] = field(default_factory=list)
    final_regret: float | None = None
    duplicate_rate: float = 0.0
    constraint_violation_rate: float = 0.0
    provenance_completeness: float = 0.0
    phase_instability: float = 0.0  # fraction of rounds where fingerprint flipped the phase

    def summary(self) -> str:
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(self.backend_distribution.items()))
        regret = "n/a" if self.final_regret is None else f"{self.final_regret:.4f}"
        traj = self.best_kpi_trajectory[-1] if self.best_kpi_trajectory else float("nan")
        return (
            f"[{self.label}] rounds={self.n_rounds}\n"
            f"  backends         : {dist}\n"
            f"  fallback_rate    : {self.fallback_rate:.3f}\n"
            f"  final_best_kpi   : {traj:.4f}\n"
            f"  final_regret     : {regret}\n"
            f"  duplicate_rate   : {self.duplicate_rate:.3f}\n"
            f"  constraint_viol  : {self.constraint_violation_rate:.3f}\n"
            f"  provenance_compl : {self.provenance_completeness:.3f}\n"
            f"  phase_instability: {self.phase_instability:.3f}"
        )


@contextmanager
def _fingerprint_only_advisor():
    """Isolate the feature under test: an advisor that supplies *only* the
    in-process Nexus fingerprint recommendation (no REST meta-learning)."""
    import app.services.optimization_intelligence as oi

    class _FingerprintAdvisor:
        def advise(self, snapshot, **_kw):
            return oi.OptimizationIntelligence(
                recommended_backends=recommend_backends(snapshot)
            )

    original = oi.OptimizationIntelligenceAdvisor
    oi.OptimizationIntelligenceAdvisor = _FingerprintAdvisor
    try:
        yield
    finally:
        oi.OptimizationIntelligenceAdvisor = original


def _provenance_complete(bs: Any) -> bool:
    return bool(
        bs is not None
        and bs.phase
        and bs.candidate_backends
        and bs.score_components
        and bs.selected_backend
        and bs.reason
    )


def _count_rejections(result: Any) -> tuple[int, int]:
    """Return (n_duplicate, n_constraint) from a DecisionResult."""
    dup = sum(1 for r in result.rejection_reasons if "duplicate" in r.lower())
    con = sum(
        1 for r in result.rejection_reasons
        if "bounds" in r.lower() or "safety" in r.lower() or "categor" in r.lower()
    )
    return dup, con


def _run_one_seed(
    task: Any,
    config: PhaseConfig,
    available: dict[str, bool],
    *,
    seed: int,
    rounds: int,
    batch_size: int,
    measure_phase_stability: bool,
) -> dict[str, Any]:
    space = task.parameter_space
    n_dims = len(space.dimensions)
    rng = np.random.default_rng(seed)
    policy = OptimizationDecisionPolicy()

    observations: list[Observation] = []
    all_params: list[dict[str, Any]] = []
    all_kpis: list[float] = []
    kpi_history: list[float] = []
    backend_failure_counts: dict[str, int] = {}
    last_kpis: list[float] = []
    last_params: list[dict[str, Any]] = []
    best = -float("inf")

    backends = Counter()
    fallbacks = 0
    prov_complete = 0
    phase_flips = 0
    total_candidates = 0
    duplicates = 0
    constraint_viol = 0
    trajectory: list[float] = []

    for r in range(1, rounds + 1):
        snapshot = CampaignSnapshot(
            round_number=r, max_rounds=rounds, n_observations=len(observations),
            n_dimensions=n_dims, has_categorical=False, has_log_scale=False,
            kpi_history=tuple(kpi_history), direction="maximize",
            available_backends=available,
            last_batch_kpis=tuple(last_kpis), last_batch_params=tuple(last_params),
            best_kpi_so_far=best if best > -float("inf") else None,
            all_params=tuple(all_params), all_kpis=tuple(all_kpis),
            backend_failure_counts=dict(backend_failure_counts),
        )

        decision = select_strategy(snapshot, config)
        backends[decision.backend_name] += 1
        bs = decision.backend_selection
        if bs is not None and bs.fallback:
            fallbacks += 1
        if _provenance_complete(bs):
            prov_complete += 1

        # Phase-stability check: same snapshot, fingerprint OFF -> phase must match.
        if measure_phase_stability:
            baseline = select_strategy(snapshot, PhaseConfig())
            if baseline.phase != decision.phase:
                phase_flips += 1

        backend = get_backend(decision.backend_name)
        cands = backend.suggest(space, batch_size, observations, seed=seed * 1000 + r)

        request = OptimizationRequest(
            campaign_id="replay", space=space, observations=tuple(observations),
            n=batch_size, seed=seed * 1000 + r,
        )
        suggestion = CandidateSuggestion(
            candidates=tuple(cands), algorithm=decision.backend_name, source="replay",
        )
        result = policy.evaluate(suggestion, request)
        total_candidates += len(cands)
        dup, con = _count_rejections(result)
        duplicates += dup
        constraint_viol += con

        accepted = list(result.final_candidates) or list(cands)
        round_kpis: list[float] = []
        for c in accepted:
            kpi = task.evaluate(c, rng)
            observations.append(Observation(params=c, objective=kpi))
            all_params.append(c)
            all_kpis.append(kpi)
            round_kpis.append(kpi)
            best = max(best, task.true_value(c))

        round_had_failure = len(accepted) == 0
        from app.optimization.failure_history import update_backend_failures
        backend_failure_counts = update_backend_failures(
            backend_failure_counts, decision.backend_name, round_had_failure=round_had_failure,
        )

        last_kpis = round_kpis
        last_params = accepted
        kpi_history.append(best)
        trajectory.append(best)

    return {
        "backends": backends,
        "fallbacks": fallbacks,
        "prov_complete": prov_complete,
        "phase_flips": phase_flips,
        "total_candidates": total_candidates,
        "duplicates": duplicates,
        "constraint_viol": constraint_viol,
        "trajectory": trajectory,
        "best": best,
    }


def run_replay(
    task: Any,
    config: PhaseConfig,
    available: dict[str, bool],
    *,
    label: str,
    seeds: int = 5,
    rounds: int = 12,
    batch_size: int = 4,
    fingerprint: bool = False,
    optimum: float | None = None,
) -> ReplayMetrics:
    """Run the replay over ``seeds`` and aggregate metrics."""
    agg_backends: Counter = Counter()
    fallbacks = prov = flips = total = dups = cons = 0
    rounds_total = 0
    trajectories: list[list[float]] = []
    bests: list[float] = []

    def _do(seed: int) -> dict[str, Any]:
        return _run_one_seed(
            task, config, available, seed=seed, rounds=rounds, batch_size=batch_size,
            measure_phase_stability=fingerprint,
        )

    for seed in range(seeds):
        out = _do(seed) if not fingerprint else None
        if fingerprint:
            with _fingerprint_only_advisor():
                out = _do(seed)
        agg_backends.update(out["backends"])
        fallbacks += out["fallbacks"]
        prov += out["prov_complete"]
        flips += out["phase_flips"]
        total += out["total_candidates"]
        dups += out["duplicates"]
        cons += out["constraint_viol"]
        rounds_total += rounds
        trajectories.append(out["trajectory"])
        bests.append(out["best"])

    mean_traj = [float(np.mean([t[i] for t in trajectories])) for i in range(rounds)]
    mean_best = float(np.mean(bests))
    regret = None if optimum is None else max(0.0, optimum - mean_best)

    return ReplayMetrics(
        label=label,
        n_rounds=rounds_total,
        backend_distribution=dict(agg_backends),
        fallback_rate=fallbacks / rounds_total if rounds_total else 0.0,
        best_kpi_trajectory=mean_traj,
        final_regret=regret,
        duplicate_rate=dups / total if total else 0.0,
        constraint_violation_rate=cons / total if total else 0.0,
        provenance_completeness=prov / rounds_total if rounds_total else 0.0,
        phase_instability=flips / rounds_total if rounds_total else 0.0,
    )


_TASKS = {"her": her_catalyst_synthetic, "oer": oer_catalyst_synthetic}


def compare_before_after(
    task_name: str = "her",
    *,
    seeds: int = 5,
    rounds: int = 12,
    batch_size: int = 4,
) -> dict[str, ReplayMetrics]:
    """Run BEFORE and AFTER and return both metric sets."""
    task = _TASKS[task_name]()
    optimum = task.optimal_value
    if optimum is None:
        optimum = _approx_optimum(task.objective_fn, task.parameter_space)

    before = run_replay(
        task, PhaseConfig(), _CLASSIC_AVAILABLE,
        label="BEFORE (classic selection)", seeds=seeds, rounds=rounds,
        batch_size=batch_size, fingerprint=False, optimum=optimum,
    )
    after = run_replay(
        task, PhaseConfig(enable_optimization_intelligence=True), _NEXUS_AVAILABLE,
        label="AFTER (nexus + fingerprint bias)", seeds=seeds, rounds=rounds,
        batch_size=batch_size, fingerprint=True, optimum=optimum,
    )
    return {"before": before, "after": after}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Nexus optimization replay comparison")
    parser.add_argument("--task", choices=sorted(_TASKS), default="her")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    results = compare_before_after(
        args.task, seeds=args.seeds, rounds=args.rounds, batch_size=args.batch_size,
    )
    print(f"\n=== Nexus optimization replay: task={args.task} ===\n")
    print(results["before"].summary())
    print()
    print(results["after"].summary())

    b, a = results["before"], results["after"]
    print("\n--- deltas (after - before) ---")
    if b.final_regret is not None and a.final_regret is not None:
        print(f"  regret           : {a.final_regret - b.final_regret:+.4f} (lower is better)")
    print(f"  fallback_rate    : {a.fallback_rate - b.fallback_rate:+.3f}")
    print(f"  duplicate_rate   : {a.duplicate_rate - b.duplicate_rate:+.3f}")
    print(f"  provenance_compl : {a.provenance_completeness - b.provenance_completeness:+.3f}")
    print(f"  phase_instability: {a.phase_instability:.3f} (must stay 0.000)")


if __name__ == "__main__":  # pragma: no cover
    main()
