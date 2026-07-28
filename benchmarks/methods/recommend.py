"""Decision table: ``ProblemTags`` bucket -> ranked methods.

Aggregates a scoreboard into generalizable, explainable guidance.  Problems are
bucketed by structural tags (dimensionality, modality, noise) and within each
bucket the methods are ranked by a weighted, normalized score over the project
criteria (accuracy / sample-efficiency / robustness / cost).  No ML -- just a
transparent weighted sum, lower-is-better metrics normalized per bucket.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from benchmarks.methods.scoreboard import MethodScore


@dataclass(frozen=True)
class ScoreWeights:
    """Relative weights over the (lower-is-better) criteria. Sum need not be 1."""

    regret: float = 0.5
    evals_to_target: float = 0.25
    robustness: float = 0.15
    cost: float = 0.1


@dataclass(frozen=True)
class RankedMethod:
    """A method's standing within a tag bucket (lower ``score`` = better)."""

    backend: str
    score: float
    mean_regret: float
    mean_evals_to_target: float | None
    target_hit_rate: float


@dataclass(frozen=True)
class Recommendation:
    """Ranked methods for one structural bucket."""

    bucket: str
    n_dims_class: str   # "low" | "high"
    modality: str       # "unimodal" | "multimodal"
    noise: str          # "low" | "high"
    ranked: list[RankedMethod] = field(default_factory=list)

    @property
    def best(self) -> str | None:
        return self.ranked[0].backend if self.ranked else None


def _dims_class(n_dims: int) -> str:
    return "low" if n_dims <= 3 else "high"


def _noise_class(noise_std: float) -> str:
    return "low" if noise_std <= 0.0 else "high"


def bucket_key(score: MethodScore) -> tuple[str, str, str]:
    """Map a MethodScore's tags to a (dims, modality, noise) bucket."""
    t = score.tags
    return (
        _dims_class(t.n_dims),
        "multimodal" if t.multimodal else "unimodal",
        _noise_class(t.noise_std),
    )


def _norm(value: float, lo: float, hi: float) -> float:
    """Min-max normalize a lower-is-better value into [0, 1]."""
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _aggregate_by_backend(
    scores: list[MethodScore],
) -> dict[str, dict[str, float]]:
    """Average each backend's metrics across the problems in a bucket."""
    by_backend: dict[str, list[MethodScore]] = defaultdict(list)
    for s in scores:
        by_backend[s.backend].append(s)

    agg: dict[str, dict[str, float]] = {}
    for backend, items in by_backend.items():
        n = len(items)
        regret = sum(s.mean_regret for s in items) / n
        robustness = sum(s.regret_robustness for s in items) / n
        cost = sum(s.mean_cost_s for s in items) / n
        hit = sum(s.target_hit_rate for s in items) / n
        etts = [
            s.mean_evals_to_target
            for s in items
            if s.mean_evals_to_target is not None
        ]
        # Penalize "never reached target" with a large surrogate eval count.
        ett = sum(etts) / len(etts) if etts else float("inf")
        agg[backend] = {
            "regret": regret,
            "evals_to_target": ett,
            "robustness": robustness,
            "cost": cost,
            "hit": hit,
        }
    return agg


def recommend(
    scores: list[MethodScore],
    *,
    weights: ScoreWeights | None = None,
) -> list[Recommendation]:
    """Build the decision table: one ranked Recommendation per tag bucket."""
    weights = weights or ScoreWeights()
    by_bucket: dict[tuple[str, str, str], list[MethodScore]] = defaultdict(list)
    for s in scores:
        by_bucket[bucket_key(s)].append(s)

    recs: list[Recommendation] = []
    for key, bucket_scores in sorted(by_bucket.items()):
        agg = _aggregate_by_backend(bucket_scores)

        def _finite(metric: str, agg=agg) -> list[float]:
            return [m[metric] for m in agg.values() if m[metric] != float("inf")]

        # Per-bucket normalization ranges (ignore inf for the range).
        ranges: dict[str, tuple[float, float]] = {}
        for metric in ("regret", "evals_to_target", "robustness", "cost"):
            vals = _finite(metric)
            ranges[metric] = (min(vals), max(vals)) if vals else (0.0, 1.0)

        ranked: list[RankedMethod] = []
        for backend, m in agg.items():
            parts = []
            for metric, w in (
                ("regret", weights.regret),
                ("evals_to_target", weights.evals_to_target),
                ("robustness", weights.robustness),
                ("cost", weights.cost),
            ):
                lo, hi = ranges[metric]
                val = m[metric]
                norm = 1.0 if val == float("inf") else _norm(val, lo, hi)
                parts.append(w * norm)
            score = sum(parts)
            ett_items = [
                s.mean_evals_to_target
                for s in bucket_scores
                if s.backend == backend and s.mean_evals_to_target is not None
            ]
            ranked.append(
                RankedMethod(
                    backend=backend,
                    score=score,
                    mean_regret=m["regret"],
                    mean_evals_to_target=(
                        sum(ett_items) / len(ett_items) if ett_items else None
                    ),
                    target_hit_rate=m["hit"],
                )
            )

        ranked.sort(key=lambda r: (r.score, r.mean_regret))
        dims, modality, noise = key
        recs.append(
            Recommendation(
                bucket=f"{dims}-dim/{modality}/{noise}-noise",
                n_dims_class=dims,
                modality=modality,
                noise=noise,
                ranked=ranked,
            )
        )
    return recs
