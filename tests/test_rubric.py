"""B1: phase-aware, versioned rubric applied over verifiable-reward signals.

The rubric reweights *existing* verifications, so a decision can be re-scored
under a new rubric without rerunning the campaign — the seed of the evolving
evaluation loop.
"""
from __future__ import annotations

import pytest

from app.services.campaign_mode import CampaignMode
from app.services.rubric import (
    STATIC_RUBRIC,
    Rubric,
    WeightedReward,
    get_rubric,
    rescore,
    rubric_for_mode,
)
from app.services.verifiable_reward import (
    split_reward,
    verify_execution,
    verify_objective,
    verify_proxy_gap,
    verify_safety,
    verify_validation,
)


def _verifications():
    return [
        verify_execution(True),
        verify_objective(0.5),
        verify_proxy_gap(-0.4),
        verify_validation(True),
        verify_safety(1),
    ]


# --- v0.1_static is the identity rubric (regression: matches Phase A) ----


def test_static_rubric_is_identity():
    v = _verifications()
    weighted = rescore(v, STATIC_RUBRIC)
    process, outcome = split_reward(v)
    assert weighted.process_reward == process
    assert weighted.outcome_reward == outcome
    assert weighted.total == round(process + outcome, 10)
    assert weighted.rubric_version == "v0.1_static"
    assert isinstance(weighted, WeightedReward)


def test_missing_weight_defaults_to_one():
    r = Rubric(version="v-partial", weights={"objective": 2.0})
    assert r.weight("objective") == 2.0
    assert r.weight("execution") == 1.0  # unspecified → identity


# --- phase-aware reweighting changes the score in the expected direction -


def test_safety_mode_amplifies_safety_penalty():
    v = _verifications()  # safety incident → negative safety score
    static = rescore(v, STATIC_RUBRIC)
    safety_mode = rescore(v, rubric_for_mode(CampaignMode.SAFETY_CONSTRAINT_TIGHTENING))
    # safety is weighted up, so a safety incident hurts more (lower total)
    assert safety_mode.total < static.total
    assert safety_mode.rubric_version != "v0.1_static"


def test_optimization_mode_amplifies_objective():
    v = [verify_objective(0.5)]
    static = rescore(v, STATIC_RUBRIC)
    opt = rescore(v, rubric_for_mode(CampaignMode.BO_OPTIMIZATION))
    assert opt.total > static.total  # objective improvement weighted up


def test_every_mode_maps_to_a_rubric():
    for mode in CampaignMode:
        r = rubric_for_mode(mode)
        assert isinstance(r, Rubric)
        assert r.version  # non-empty version tag


# --- registry lookup -----------------------------------------------------


def test_get_rubric_by_version_roundtrips():
    assert get_rubric("v0.1_static") is STATIC_RUBRIC


def test_get_rubric_unknown_raises():
    with pytest.raises(KeyError):
        get_rubric("v-does-not-exist")
