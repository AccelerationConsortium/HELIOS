"""Pathology layer: wrapper injection, ground-truth event ledger, determinism."""
from __future__ import annotations

import dataclasses

import pytest

from benchmarks.methods.pathologies import (
    Censoring,
    NoiseDrift,
    ObjectiveShift,
    ProxyGapShift,
    SpatialFailure,
    apply_pathology,
    in_failure_region,
)
from benchmarks.methods.problems import get_problem

ORIGIN = {"x1": 0.0, "x2": 0.0}
CORNER = {"x1": -5.0, "x2": -5.0}


def _sphere():
    return get_problem("sphere_2d")


def test_wrapper_returns_new_problem_and_leaves_base_untouched():
    base = _sphere()
    wrapped, state = apply_pathology(base, [NoiseDrift(start_eval=0, rate=1.0)], seed=7)

    assert wrapped is not base
    assert wrapped.id == "sphere_2d@noise_drift"
    assert base.evaluate(ORIGIN).observed_value == 0.0  # base evaluator untouched
    assert state.eval_count == 0
    assert state.events == []


def test_noise_drift_bias_grows_linearly_after_onset():
    wrapped, state = apply_pathology(
        _sphere(), [NoiseDrift(start_eval=3, rate=0.5, mode="linear")], seed=0
    )

    observed = [wrapped.evaluate(ORIGIN).observed_value for _ in range(6)]

    # Evals 1-3 untouched; bias = rate * (eval_index - start_eval) afterwards.
    assert observed[:3] == [0.0, 0.0, 0.0]
    assert observed[3] == pytest.approx(0.5)
    assert observed[4] == pytest.approx(1.0)
    assert observed[5] == pytest.approx(1.5)
    # raw ruler never corrupted
    assert all(e.raw_value == 0.0 for e in [wrapped.evaluate(ORIGIN)])
    # exactly one onset event, at the first affected eval
    drift_events = [e for e in state.events if e.event_type == "noise_drift"]
    assert len(drift_events) == 1
    assert drift_events[0].eval_index == 4
    assert drift_events[0].duration is None


def test_noise_drift_step_mode_applies_constant_bias():
    wrapped, _ = apply_pathology(
        _sphere(), [NoiseDrift(start_eval=2, rate=2.0, mode="step")], seed=0
    )
    observed = [wrapped.evaluate(ORIGIN).observed_value for _ in range(4)]
    assert observed == [0.0, 0.0, 2.0, 2.0]


def test_spatial_failure_fires_only_inside_region():
    spec = SpatialFailure(
        center={"x1": 0.5, "x2": 0.5},
        radius=0.2,
        p_max=1.0,
        p_base=0.0,
        observed_penalty=10.0,
    )
    wrapped, state = apply_pathology(_sphere(), [spec], seed=1)

    at_center = wrapped.evaluate(ORIGIN)  # origin = normalized (0.5, 0.5)
    assert at_center.execution_success is False
    assert at_center.failure_type == "pathology_execution_failure"
    assert at_center.observed_value == pytest.approx(at_center.raw_value + 10.0)
    assert at_center.raw_value == 0.0  # ruler untouched

    far = wrapped.evaluate(CORNER)
    assert far.execution_success is True
    assert far.failure_type is None

    failures = [e for e in state.events if e.event_type == "spatial_failure"]
    assert len(failures) == 1
    assert failures[0].eval_index == 1
    assert failures[0].region == {"center": {"x1": 0.5, "x2": 0.5}, "radius": 0.2}
    assert failures[0].params == ORIGIN


def test_spatial_failure_is_deterministic_per_seed():
    spec = SpatialFailure(
        center={"x1": 0.5, "x2": 0.5}, radius=0.4, p_max=0.6, p_base=0.1
    )
    points = [
        {"x1": (i % 5) - 2.0, "x2": (i % 3) - 1.0} for i in range(30)
    ]

    def run(seed):
        wrapped, state = apply_pathology(_sphere(), [spec], seed=seed)
        evals = [wrapped.evaluate(p) for p in points]
        return (
            [e.execution_success for e in evals],
            [e.observed_value for e in evals],
            [dataclasses.asdict(ev) for ev in state.events],
        )

    assert run(42) == run(42)
    successes_a, _, _ = run(42)
    successes_b, _, _ = run(43)
    assert successes_a != successes_b  # seed actually matters


def test_censoring_clips_observed_and_flags_metadata():
    wrapped, state = apply_pathology(_sphere(), [Censoring(lod=1.0)], seed=0)

    censored = wrapped.evaluate({"x1": 2.0, "x2": 0.0})  # raw = 4.0 > lod
    clear = wrapped.evaluate({"x1": 0.5, "x2": 0.0})  # raw = 0.25 <= lod

    assert censored.observed_value == 1.0
    assert censored.raw_value == 4.0
    assert censored.metadata["censored"] is True
    assert clear.observed_value == 0.25
    assert "censored" not in clear.metadata

    events = [e for e in state.events if e.event_type == "censoring"]
    assert len(events) == 1  # only the first censored eval is ledgered
    assert events[0].eval_index == 1
    wrapped.evaluate({"x1": 3.0, "x2": 0.0})
    assert len([e for e in state.events if e.event_type == "censoring"]) == 1


def test_proxy_gap_shift_changes_mapping_at_shift_eval():
    wrapped, state = apply_pathology(
        _sphere(), [ProxyGapShift(shift_eval=2, bias=3.0, scale=2.0)], seed=0
    )
    point = {"x1": 1.0, "x2": 0.0}  # raw = 1.0

    before = [wrapped.evaluate(point) for _ in range(2)]
    after = wrapped.evaluate(point)

    assert all(e.observed_value == 1.0 for e in before)
    assert after.observed_value == pytest.approx(2.0 * 1.0 + 3.0)
    assert after.raw_value == 1.0

    events = [e for e in state.events if e.event_type == "proxy_gap_shift"]
    assert len(events) == 1
    assert events[0].eval_index == 3


def test_objective_shift_translates_argmin_but_preserves_optimum_value():
    # delta is in normalized units: x1 range is [-5, 5] so 0.1 -> +1.0 shift.
    wrapped, state = apply_pathology(
        _sphere(), [ObjectiveShift(shift_eval=0, delta={"x1": 0.1, "x2": 0.0})], seed=0
    )

    at_old_argmin = wrapped.evaluate(ORIGIN)
    at_new_argmin = wrapped.evaluate({"x1": 1.0, "x2": 0.0})

    assert at_old_argmin.raw_value == pytest.approx(1.0)  # (0 - 1)^2
    assert at_new_argmin.raw_value == pytest.approx(0.0)  # optimum value preserved

    events = [e for e in state.events if e.event_type == "objective_shift"]
    assert len(events) == 1
    assert events[0].eval_index == 1


def test_composed_pathologies_apply_in_order():
    wrapped, _ = apply_pathology(
        _sphere(),
        [
            ObjectiveShift(shift_eval=0, delta={"x1": 0.1, "x2": 0.0}),
            ProxyGapShift(shift_eval=0, bias=5.0, scale=1.0),
        ],
        seed=0,
    )
    evaluation = wrapped.evaluate(ORIGIN)
    assert evaluation.raw_value == pytest.approx(1.0)  # shifted landscape
    assert evaluation.observed_value == pytest.approx(6.0)  # then proxy bias
    assert wrapped.id == "sphere_2d@objective_shift+proxy_gap_shift"


def test_event_fields_are_complete_and_ids_unique():
    spec = SpatialFailure(
        center={"x1": 0.5, "x2": 0.5}, radius=1.0, p_max=1.0, p_base=1.0
    )
    wrapped, state = apply_pathology(_sphere(), [spec], seed=0)
    wrapped.evaluate(ORIGIN)
    wrapped.evaluate(CORNER)

    assert len(state.events) == 2
    ids = [e.event_id for e in state.events]
    assert len(set(ids)) == 2
    for event in state.events:
        assert event.event_type == "spatial_failure"
        assert event.eval_index in (1, 2)
        assert event.severity > 0.0
        assert event.region is not None
        assert isinstance(event.params, dict)
        assert isinstance(event.details, dict)


def test_in_failure_region_helper():
    space = _sphere().space
    region = {"center": {"x1": 0.5, "x2": 0.5}, "radius": 0.2}
    assert in_failure_region(ORIGIN, space, region) is True
    assert in_failure_region(CORNER, space, region) is False
