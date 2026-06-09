"""End-to-end research study runner for HELIOS.

Unlike ``sdl_benchmark.py`` proxies, this module can evaluate the real
``OrchestratorAgent`` path against deterministic baselines on the same
electrochemistry surrogate. It is the evidence layer for paper-grade claims:
sample efficiency, trace completeness, safety/recovery counters, and ablations.
"""
from __future__ import annotations

import asyncio
import math
import statistics
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.services.electrochem_surrogate import (
    OPTIMAL_OVERPOTENTIAL_MV,
    ElectrochemSurrogateConfig,
    electrochem_protocol_template,
    electrochem_search_dimensions,
    evaluate_electrochem_candidate,
)

BaselineName = Literal[
    "helios_full",
    "random",
    "lhs",
    "single_bo",
    "no_adaptive",
]

SUPPORTED_BASELINES = {"helios_full", "random", "lhs", "single_bo", "no_adaptive"}


@dataclass(frozen=True)
class StudySpec:
    """Configuration for one reproducible research study."""

    name: str = "electrochem_closed_loop_e2e"
    seeds: tuple[int, ...] = (0, 1, 2)
    baselines: tuple[BaselineName, ...] = ("helios_full", "random", "lhs", "single_bo")
    max_rounds: int = 3
    batch_size: int = 3
    objective_kpi: str = "overpotential_mv"
    direction: Literal["minimize", "maximize"] = "minimize"
    strategy: str = "adaptive"
    noise_std_mv: float = 2.0
    fault_profile: str = "none"

    @property
    def budget(self) -> int:
        return self.max_rounds * self.batch_size

    def __post_init__(self) -> None:
        unknown = set(self.baselines) - SUPPORTED_BASELINES
        if unknown:
            raise ValueError(f"unsupported baseline(s): {sorted(unknown)}")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


@dataclass(frozen=True)
class TraceCompleteness:
    complete: bool
    score: float
    missing: tuple[str, ...] = ()
    event_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "score": self.score,
            "missing": list(self.missing),
            "event_count": self.event_count,
        }


@dataclass(frozen=True)
class StudyRunResult:
    study_name: str
    baseline: str
    seed: int
    trajectory: tuple[float, ...]
    final_best: float
    regret: float
    trace: TraceCompleteness
    safety_violations: int = 0
    recovery_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "baseline": self.baseline,
            "seed": self.seed,
            "trajectory": list(self.trajectory),
            "final_best": self.final_best,
            "regret": self.regret,
            "trace": self.trace.to_dict(),
            "safety_violations": self.safety_violations,
            "recovery_attempts": self.recovery_attempts,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class StudyAggregate:
    study_name: str
    results: tuple[StudyRunResult, ...]

    def by_baseline(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[StudyRunResult]] = {}
        for result in self.results:
            grouped.setdefault(result.baseline, []).append(result)

        out: dict[str, dict[str, Any]] = {}
        for baseline, rows in grouped.items():
            finals = [r.final_best for r in rows]
            regrets = [r.regret for r in rows]
            traces = [r.trace.score for r in rows]
            out[baseline] = {
                "n": len(rows),
                "mean_final_best": _mean(finals),
                "std_final_best": _std(finals),
                "mean_regret": _mean(regrets),
                "trace_completeness_rate": _mean(traces),
                "safety_violations": sum(r.safety_violations for r in rows),
                "recovery_attempts": sum(r.recovery_attempts for r in rows),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "summary": self.by_baseline(),
            "results": [r.to_dict() for r in self.results],
        }


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _regret(value: float, direction: str) -> float:
    if math.isnan(value):
        return float("nan")
    if direction == "minimize":
        return max(0.0, value - OPTIMAL_OVERPOTENTIAL_MV)
    return max(0.0, OPTIMAL_OVERPOTENTIAL_MV - value)


def _best_so_far(values: list[float], direction: str) -> tuple[float, ...]:
    out: list[float] = []
    best = math.inf if direction == "minimize" else -math.inf
    for value in values:
        best = min(best, value) if direction == "minimize" else max(best, value)
        out.append(best)
    return tuple(out)


def _policy_for_spec(spec: StudySpec, seed: int) -> dict[str, Any]:
    allowed_primitives = [
        "robot.home",
        "robot.load_pipettes",
        "robot.load_labware",
        "robot.pick_up_tip",
        "robot.drop_tip",
        "robot.dispense",
        "wait",
        "squidstat.run_experiment",
    ]
    return {
        "max_temp_c": 95.0,
        "max_volume_ul": 1000.0,
        "allowed_primitives": allowed_primitives,
        "require_human_approval": False,
        "simulation_oracle": "electrochem",
        "simulation_seed": seed,
        "simulation_noise_std_mv": spec.noise_std_mv,
        "simulation_fault_profile": spec.fault_profile,
    }


def _oracle_value(params: dict[str, Any], spec: StudySpec, seed: int, index: int) -> float:
    obs = evaluate_electrochem_candidate(
        params,
        config=ElectrochemSurrogateConfig(
            seed=seed + index,
            noise_std_mv=spec.noise_std_mv,
            fault_profile=spec.fault_profile,
        ),
    )
    return obs.overpotential_mv


def _parameter_space():
    from app.services.candidate_gen import ParameterSpace, SearchDimension

    dims = [
        SearchDimension(
            param_name=d["param_name"],
            param_type=d.get("param_type", "number"),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            primitive=d.get("primitive"),
        )
        for d in electrochem_search_dimensions()
    ]
    return ParameterSpace(dimensions=tuple(dims), protocol_template=electrochem_protocol_template())


def _run_sampling_baseline(spec: StudySpec, baseline: str, seed: int) -> StudyRunResult:
    from app.services.candidate_gen import sample_lhs, sample_random
    from benchmarks.sdl_benchmark import FixedBOBaseline

    if baseline not in {"random", "lhs", "single_bo"}:
        raise ValueError(f"{baseline!r} is not a sampling baseline")

    space = _parameter_space()
    observations: list[tuple[dict[str, Any], float]] = []
    values: list[float] = []

    if baseline == "single_bo":
        optimizer = FixedBOBaseline(k=5, pool=128, xi=0.02)
        import numpy as np

        rng = np.random.default_rng(seed)
        for step in range(spec.budget):
            params = optimizer.suggest(space, 1, observations, rng)[0]
            value = _oracle_value(params, spec, seed, step)
            observations.append((params, value))
            values.append(value)
    else:
        sampler = sample_lhs if baseline == "lhs" else sample_random
        candidates = sampler(space, spec.budget, seed=seed)
        for idx, params in enumerate(candidates):
            value = _oracle_value(params, spec, seed, idx)
            observations.append((params, value))
            values.append(value)

    trajectory = _best_so_far(values, spec.direction)
    final_best = trajectory[-1] if trajectory else float("nan")
    return StudyRunResult(
        study_name=spec.name,
        baseline=baseline,
        seed=seed,
        trajectory=trajectory,
        final_best=final_best,
        regret=_regret(final_best, spec.direction),
        trace=TraceCompleteness(complete=False, score=0.0, missing=("orchestrator_trace",)),
        metadata={"n_evaluations": len(values)},
    )


def assess_trace_completeness(events: list[dict[str, Any]]) -> TraceCompleteness:
    required_event_types = {
        "agent_stage_graph",
        "agent_stage_start",
        "agent_stage_end",
        "round_start",
    }
    required_agent_results = {
        "planner",
        "design",
        "safety",
        "executor",
        "monitor",
        "analyzer",
        "stop",
    }
    event_types = {event.get("event_type", "") for event in events}
    agents = {
        (event.get("payload") or {}).get("agent")
        for event in events
        if event.get("event_type") == "agent_result"
    }
    missing = sorted(required_event_types - event_types)
    missing += [f"agent_result:{agent}" for agent in sorted(required_agent_results - agents)]
    total = len(required_event_types) + len(required_agent_results)
    score = 1.0 - (len(missing) / total)
    return TraceCompleteness(
        complete=not missing,
        score=round(max(0.0, score), 4),
        missing=tuple(missing),
        event_count=len(events),
    )


def _count_safety_violations(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "agent_result" and payload.get("agent") == "safety":
            if payload.get("allowed") is False:
                count += 1
    return count


def _count_recovery_attempts(events: list[dict[str, Any]]) -> int:
    return sum(
        1
        for event in events
        if "recovery" in str(event.get("event_type", "")).lower()
        or "recovery" in str(event.get("payload", {})).lower()
    )


async def _run_helios_baseline(spec: StudySpec, baseline: str, seed: int) -> StudyRunResult:
    from app.agents.orchestrator import OrchestratorAgent, OrchestratorInput
    from app.core.db import init_db
    from app.services.campaign_events import replay_events
    from app.services.campaign_state import load_campaign

    init_db()
    campaign_id = f"study-{baseline}-{seed}-{uuid.uuid4().hex[:8]}"
    if baseline not in {"helios_full", "no_adaptive"}:
        raise ValueError(f"{baseline!r} is not a HELIOS orchestrator baseline")
    strategy = "lhs" if baseline == "no_adaptive" else spec.strategy

    orchestrator = OrchestratorAgent()
    result = await orchestrator.process(
        OrchestratorInput(
            contract_id=f"{spec.name}-{baseline}-{seed}",
            objective_kpi=spec.objective_kpi,
            direction=spec.direction,
            max_rounds=spec.max_rounds,
            batch_size=spec.batch_size,
            strategy=strategy,
            target_value=None,
            dimensions=electrochem_search_dimensions(),
            protocol_template=electrochem_protocol_template(),
            policy_snapshot=_policy_for_spec(spec, seed),
            dry_run=True,
            campaign_id=campaign_id,
        )
    )

    state = load_campaign(campaign_id) or {}
    values = [float(v) for v in state.get("all_kpis", [])]
    trajectory = _best_so_far(values, spec.direction)
    final_best = result.best_kpi if result.best_kpi is not None else (trajectory[-1] if trajectory else float("nan"))
    events = replay_events(campaign_id)

    return StudyRunResult(
        study_name=spec.name,
        baseline=baseline,
        seed=seed,
        trajectory=trajectory,
        final_best=float(final_best),
        regret=_regret(float(final_best), spec.direction),
        trace=assess_trace_completeness(events),
        safety_violations=_count_safety_violations(events),
        recovery_attempts=_count_recovery_attempts(events),
        metadata={
            "campaign_id": campaign_id,
            "status": result.status,
            "rounds_completed": result.rounds_completed,
            "n_events": len(events),
            "n_evaluations": len(values),
        },
    )


async def run_study(spec: StudySpec) -> StudyAggregate:
    """Run all requested baselines and seeds."""
    results: list[StudyRunResult] = []
    for baseline in spec.baselines:
        for seed in spec.seeds:
            if baseline in {"helios_full", "no_adaptive"}:
                results.append(await _run_helios_baseline(spec, baseline, seed))
            else:
                results.append(_run_sampling_baseline(spec, baseline, seed))
    return StudyAggregate(study_name=spec.name, results=tuple(results))


def run_study_sync(spec: StudySpec) -> StudyAggregate:
    return asyncio.run(run_study(spec))
