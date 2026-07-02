from __future__ import annotations

import json
from datetime import UTC, datetime

from app.services.candidate_gen import space_from_dimensions
from app.services.llm_candidate_proposer import (
    LLMCandidateProposer,
    LLMProposerShadow,
    ValidatedProposal,
    compare_llm_proposal_to_selection,
    make_safety_bounds_rejector,
    parse_llm_proposer_shadow_log_line,
    should_invoke_llm_proposer,
    space_centroid,
    validate_proposal,
)
from app.services.llm_gateway import MockProvider

_NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)


def _space():
    return space_from_dimensions(
        [
            {"param_name": "x", "param_type": "number", "min_value": 0.0, "max_value": 1.0},
            {"param_name": "c", "param_type": "categorical", "choices": ["a", "b"]},
        ]
    )


def _proposal_json(points):
    return json.dumps({"proposals": points})


# --- trigger gate ---------------------------------------------------------


def test_should_invoke_on_plateau_or_high_uncertainty():
    assert should_invoke_llm_proposer(plateau=True, epistemic_uncertainty=0.1) is True
    assert should_invoke_llm_proposer(plateau=False, epistemic_uncertainty=0.9) is True
    assert should_invoke_llm_proposer(plateau=False, epistemic_uncertainty=0.1) is False
    assert should_invoke_llm_proposer(plateau=False, epistemic_uncertainty=None) is False


# --- propose (async, mock provider) --------------------------------------


async def test_propose_parses_mock_json_into_proposal():
    provider = MockProvider(
        responses=[
            _proposal_json(
                [
                    {"params": {"x": 0.3, "c": "a"}, "reason": "explore low x"},
                    {"params": {"x": 0.8, "c": "b"}, "reason": "near best"},
                ]
            )
        ]
    )
    proposer = LLMCandidateProposer(provider=provider)

    proposal = await proposer.propose(
        campaign_id="camp-1",
        round_index=3,
        space=_space(),
        objective_kpi="conductivity",
        direction="maximize",
        trigger_reason="plateau",
        now=_NOW,
    )

    assert proposal.shadow_only is True
    assert [p.params for p in proposal.points] == [
        {"x": 0.3, "c": "a"},
        {"x": 0.8, "c": "b"},
    ]
    assert proposal.points[0].reason == "explore low x"
    assert proposal.trigger_reason == "plateau"
    assert proposal.created_at == _NOW


async def test_propose_fail_open_on_provider_error():
    provider = MockProvider(responses=[])  # complete() raises LLMError

    proposal = await LLMCandidateProposer(provider=provider).propose(
        campaign_id="camp-1",
        round_index=0,
        space=_space(),
        objective_kpi="conductivity",
        direction="maximize",
        trigger_reason="plateau",
        now=_NOW,
    )

    assert proposal.points == []
    assert proposal.metadata.get("failed") is True


async def test_propose_fail_open_on_bad_json():
    provider = MockProvider(responses=["not valid json at all"])

    proposal = await LLMCandidateProposer(provider=provider).propose(
        campaign_id="camp-1",
        round_index=0,
        space=_space(),
        objective_kpi="conductivity",
        direction="maximize",
        trigger_reason="uncertainty",
        now=_NOW,
    )

    assert proposal.points == []
    assert proposal.metadata.get("failed") is True


# --- validation gate ------------------------------------------------------


async def _proposal_with(points):
    provider = MockProvider(responses=[_proposal_json(points)])
    return await LLMCandidateProposer(provider=provider).propose(
        campaign_id="camp-1",
        round_index=0,
        space=_space(),
        objective_kpi="conductivity",
        direction="maximize",
        trigger_reason="plateau",
        now=_NOW,
    )


async def test_validate_schema_and_space_rules():
    proposal = await _proposal_with(
        [
            {"params": {"x": 0.5, "c": "a"}, "reason": "ok"},
            {"params": {"x": 2.0, "c": "a"}, "reason": "x out of bounds"},
            {"params": {"x": 0.5, "c": "z"}, "reason": "bad category"},
            {"params": {"x": 0.5}, "reason": "missing c"},
            {"params": {"x": 0.5, "c": "a", "y": 9}, "reason": "unknown y"},
        ]
    )

    result = validate_proposal(proposal, space=_space(), now=_NOW)

    accepted = [v.params for v in result.validations if v.accepted]
    assert accepted == [{"x": 0.5, "c": "a"}]
    assert result.accepted_points == [{"x": 0.5, "c": "a"}]
    # every rejected point carries at least one reason
    assert all(v.rejections for v in result.validations if not v.accepted)


async def test_validate_rejects_points_in_failure_zone():
    proposal = await _proposal_with([{"params": {"x": 0.5, "c": "a"}, "reason": "ok"}])

    result = validate_proposal(
        proposal,
        space=_space(),
        failure_zones=[{"x": 0.5, "c": "a"}],
        now=_NOW,
    )

    assert result.accepted_points == []
    assert any("failure_zone" in r for v in result.validations for r in v.rejections)


async def test_validate_extra_rejector_is_pluggable():
    proposal = await _proposal_with([{"params": {"x": 0.5, "c": "a"}, "reason": "ok"}])

    def _reject_all(params):
        return "safety: blocked for test"

    result = validate_proposal(
        proposal, space=_space(), extra_rejectors=[_reject_all], now=_NOW
    )

    assert result.accepted_points == []
    assert any("safety" in r for v in result.validations for r in v.rejections)


async def test_validated_proposal_is_json_safe_and_deterministic():
    proposal = await _proposal_with([{"params": {"x": 0.5, "c": "a"}, "reason": "ok"}])

    result = validate_proposal(proposal, space=_space(), now=_NOW)

    dumped = result.model_dump(mode="json")
    json.dumps(dumped)
    assert result.created_at == _NOW
    assert isinstance(result, ValidatedProposal)


def test_config_flag_default_false(monkeypatch):
    monkeypatch.delenv("LLM_PROPOSER_SHADOW_ENABLED", raising=False)
    from app.core.config import Settings

    assert Settings().llm_proposer_shadow_enabled is False


async def test_compare_llm_proposal_to_selection_overlap():
    proposal = await _proposal_with(
        [
            {"params": {"x": 0.5, "c": "a"}, "reason": "matches a selected candidate"},
            {"params": {"x": 2.0, "c": "a"}, "reason": "invalid, rejected"},
        ]
    )
    validated = validate_proposal(proposal, space=_space(), now=_NOW)

    comparison = compare_llm_proposal_to_selection(
        validated,
        selected_candidates=[{"x": 0.5, "c": "a"}, {"x": 0.9, "c": "b"}],
        space=_space(),
        now=_NOW,
    )

    assert comparison.n_proposed == 2
    assert comparison.n_accepted == 1
    assert comparison.validity_rate == 0.5
    assert comparison.overlap_count == 1  # the accepted point matches a selection
    assert comparison.created_at == _NOW


async def test_llm_proposer_shadow_log_round_trips():
    proposal = await _proposal_with([{"params": {"x": 0.5, "c": "a"}, "reason": "ok"}])
    validated = validate_proposal(proposal, space=_space(), now=_NOW)
    shadow = LLMProposerShadow(proposal=proposal, validation=validated)

    line = "llm_proposer_shadow " + json.dumps(shadow.model_dump(mode="json"), sort_keys=True)
    parsed = parse_llm_proposer_shadow_log_line(line)

    assert parsed is not None
    assert parsed.validation.accepted_points == [{"x": 0.5, "c": "a"}]
    assert parse_llm_proposer_shadow_log_line("unrelated line") is None


def test_space_centroid():
    assert space_centroid(_space()) == {"x": 0.5, "c": "a"}


def test_make_safety_bounds_rejector_uses_policy_limits():
    reject = make_safety_bounds_rejector({"max_temp_c": 50.0, "max_volume_ul": 200.0})

    assert reject({"temp_c": 99.0}) is not None
    assert "safety" in reject({"temp_c": 99.0})
    assert reject({"volume_ul": 500.0}) is not None
    assert reject({"temp_c": 25.0, "volume_ul": 100.0}) is None
    assert reject({"x": 0.5}) is None  # non-safety params are ignored


def test_import_smoke():
    import app.services.llm_candidate_proposer  # noqa: F401
