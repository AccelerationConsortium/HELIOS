"""Tests for the tag->method recommender on synthetic score tables."""
from __future__ import annotations

from benchmarks.methods.problems import ProblemTags
from benchmarks.methods.recommend import bucket_key, recommend
from benchmarks.methods.scoreboard import MethodScore


def _score(backend: str, regret: float, ett: float | None, **kw) -> MethodScore:
    tags = kw.pop("tags", None) or ProblemTags(
        n_dims=2,
        multimodal=True,
        noise_std=0.0,
        separable=False,
        has_categorical=False,
        surface_class="multimodal",
    )
    return MethodScore(
        problem_id=kw.get("problem_id", "p"),
        backend=backend,
        n_seeds=3,
        mean_regret=regret,
        mean_auc=kw.get("auc", regret * 2),
        mean_evals_to_target=ett,
        target_hit_rate=kw.get("hit", 1.0 if ett is not None else 0.0),
        regret_robustness=kw.get("rob", 0.0),
        mean_cost_s=kw.get("cost", 0.01),
        tags=tags,
        errors=0,
    )


def test_better_method_ranked_first():
    scores = [
        _score("good_method", regret=0.01, ett=5),
        _score("bad_method", regret=2.0, ett=None),
    ]
    recs = recommend(scores)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.best == "good_method"
    assert [m.backend for m in rec.ranked] == ["good_method", "bad_method"]
    assert rec.ranked[0].score <= rec.ranked[1].score


def test_buckets_separated_by_tags():
    low_dim_tags = ProblemTags(2, False, 0.0, True, False, "convex")
    high_dim_tags = ProblemTags(6, True, 0.0, False, False, "multimodal")
    scores = [
        _score("m_a", 0.1, 5, tags=low_dim_tags, problem_id="lo"),
        _score("m_b", 0.2, 8, tags=high_dim_tags, problem_id="hi"),
    ]
    recs = recommend(scores)
    keys = {(r.n_dims_class, r.modality) for r in recs}
    assert ("low", "unimodal") in keys
    assert ("high", "multimodal") in keys


def test_bucket_key_classes():
    lo = ProblemTags(3, False, 0.0, True, False, "convex")
    hi = ProblemTags(4, True, 0.5, False, False, "multimodal")
    assert bucket_key(_score("x", 0.0, 1, tags=lo)) == ("low", "unimodal", "low")
    assert bucket_key(_score("y", 0.0, 1, tags=hi)) == ("high", "multimodal", "high")


def test_weighting_can_favor_sample_efficiency():
    # Two methods: one slightly better regret but far worse sample efficiency.
    scores = [
        _score("low_regret_slow", regret=0.05, ett=50),
        _score("fast_converger", regret=0.10, ett=3),
    ]
    from benchmarks.methods.recommend import ScoreWeights

    # Heavily weight evals-to-target -> the fast converger should win.
    recs = recommend(
        scores,
        weights=ScoreWeights(regret=0.1, evals_to_target=0.9, robustness=0.0, cost=0.0),
    )
    assert recs[0].best == "fast_converger"
