from __future__ import annotations

import json
from datetime import UTC, datetime

from app.services.candidate_gen import space_from_dimensions
from app.services.llm_candidate_proposer import (
    LLMCandidateProposer,
    LLMProposerShadow,
    count_point_overlaps,
    validate_proposal,
)
from app.services.llm_gateway import MockProvider
from app.services.llm_proposer_evidence import (
    LLMProposerEvidence,
    build_llm_proposer_evidence,
)

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)


def _space():
    return space_from_dimensions(
        [
            {"param_name": "x", "param_type": "number", "min_value": 0.0, "max_value": 1.0},
            {"param_name": "c", "param_type": "categorical", "choices": ["a", "b"]},
        ]
    )


async def _shadow(points, *, failure_zones=None, extra_rejectors=None):
    provider = MockProvider(responses=[json.dumps({"proposals": points})])
    proposal = await LLMCandidateProposer(provider=provider).propose(
        campaign_id="camp-1",
        round_index=0,
        space=_space(),
        objective_kpi="conductivity",
        direction="maximize",
        trigger_reason="plateau",
        now=_NOW,
    )
    validation = validate_proposal(
        proposal,
        space=_space(),
        failure_zones=failure_zones,
        extra_rejectors=extra_rejectors,
        now=_NOW,
    )
    return LLMProposerShadow(proposal=proposal, validation=validation, created_at=_NOW)


def test_count_point_overlaps():
    space = _space()
    points = [{"x": 0.5, "c": "a"}, {"x": 0.9, "c": "b"}]
    candidates = [{"x": 0.5, "c": "a"}, {"x": 0.1, "c": "b"}]

    assert count_point_overlaps(points, candidates, space=space) == 1


async def test_build_llm_proposer_evidence_aggregates_and_baselines():
    s1 = await _shadow(
        [
            {"params": {"x": 0.5, "c": "a"}, "reason": "matches selection"},
            {"params": {"x": 2.0, "c": "a"}, "reason": "schema reject"},
            {"params": {"x": 0.9, "c": "b"}, "reason": "novel valid"},
        ]
    )
    s2 = await _shadow(
        [{"params": {"x": 0.5, "c": "a"}, "reason": "hits failure zone"}],
        failure_zones=[{"x": 0.5, "c": "a"}],
    )
    s3 = await _shadow(
        [{"params": {"x": 0.5, "c": "a"}, "reason": "unsafe"}],
        extra_rejectors=[lambda p: "safety: blocked for test"],
    )

    evidence = build_llm_proposer_evidence(
        shadows=[s1, s2, s3],
        selections_by_round=[
            [{"x": 0.5, "c": "a"}, {"x": 0.1, "c": "b"}],
            [{"x": 0.5, "c": "a"}],
            [{"x": 0.5, "c": "a"}],
        ],
        space=_space(),
        random_points_by_round=[
            [{"x": 0.5, "c": "a"}, {"x": 0.7, "c": "a"}],
            [{"x": 0.7, "c": "a"}],
            [{"x": 0.7, "c": "a"}],
        ],
        now=_NOW,
    )

    assert isinstance(evidence, LLMProposerEvidence)
    assert evidence.rounds == 3
    assert evidence.proposed == 5
    assert evidence.accepted == 2  # only round-1's two in-bounds points survive
    assert evidence.validity_rate == 0.4
    # Q4: rejection reasons by category
    assert evidence.rejection_histogram == {"schema": 1, "failure_zone": 1, "safety": 1}
    # Q5: rejectors demonstrably fired
    assert evidence.rejector_fired["failure_zone"] == 1
    assert evidence.rejector_fired["safety"] == 1
    # Q2: overlap with actual selection
    assert evidence.overlap_count == 1
    # Q3: novel-but-valid candidates the selection did not include
    assert evidence.novel_valid == 1
    # Q6: random baseline overlap for comparison
    assert evidence.random_overlap_rate == 0.25  # 1 of 4 random points matched


async def test_evidence_is_json_safe():
    s1 = await _shadow([{"params": {"x": 0.5, "c": "a"}, "reason": "ok"}])
    evidence = build_llm_proposer_evidence(
        shadows=[s1],
        selections_by_round=[[{"x": 0.5, "c": "a"}]],
        space=_space(),
        now=_NOW,
    )

    json.dumps(evidence.model_dump(mode="json"))
    assert evidence.created_at == _NOW


def test_import_smoke():
    import app.services.llm_proposer_evidence  # noqa: F401
