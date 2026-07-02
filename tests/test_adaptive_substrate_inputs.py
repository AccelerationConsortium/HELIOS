from __future__ import annotations

from datetime import UTC, datetime

from app.services.action_contracts import (
    ActionContract,
    Precondition,
    SafetyClass,
    TimeoutConfig,
)
from app.services.adaptive_substrate_inputs import (
    action_specs_from_registry,
    available_capabilities,
    campaign_instruments_from_protocol,
    failure_attribution_from_events,
    objective_state_from_input,
)
from app.services.failure_attribution import FailureAttributionCategory
from app.services.primitives_registry import PrimitiveSpec

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _spec(
    name: str,
    *,
    instrument: str | None,
    safety_class: SafetyClass = SafetyClass.CAREFUL,
    contract: ActionContract | None = None,
) -> PrimitiveSpec:
    return PrimitiveSpec(
        name=name,
        description="",
        error_class="CRITICAL",
        instrument=instrument,
        resource_id=None,
        skill_name="skill",
        params=(),
        safety_class=safety_class,
        contract=contract,
    )


class _FakeRegistry:
    def __init__(self, specs: list[PrimitiveSpec]) -> None:
        self._specs = {s.name: s for s in specs}

    def list_primitives(self) -> list[PrimitiveSpec]:
        return list(self._specs.values())

    def get_primitive(self, name: str) -> PrimitiveSpec | None:
        return self._specs.get(name)

    def list_instruments(self) -> list[str]:
        return sorted({s.instrument for s in self._specs.values() if s.instrument})


# --- objective_state_from_input ------------------------------------------


def test_objective_state_from_input_builds_snapshot_state():
    state = objective_state_from_input(
        campaign_id="camp-1",
        objective_kpi="conductivity",
        max_rounds=10,
        created_at=_NOW,
    )

    assert state.campaign_id == "camp-1"
    assert state.primary_objective == "conductivity"
    assert state.objective_confidence == 0.5
    assert state.stopping_criteria is not None
    assert state.stopping_criteria.max_rounds == 10
    assert state.proxy_gap is None
    assert state.revision == 0


# --- failure_attribution_from_events -------------------------------------


def test_failure_attribution_none_when_no_events():
    assert failure_attribution_from_events([], now=_NOW) is None


def test_failure_attribution_maps_instrument_event():
    events = [
        {"failure_type": "hardware", "primitive": "heat", "error": "temp overshoot exceeded", "step": "s1"}
    ]

    dist = failure_attribution_from_events(events, now=_NOW)

    assert dist is not None
    assert dist.dominant_category == FailureAttributionCategory.INSTRUMENT


def test_failure_attribution_picks_most_severe_event():
    events = [
        {"failure_type": "x", "primitive": "robot.aspirate", "error": "volume deviation", "step": "s1"},
        {"failure_type": "x", "primitive": "heat", "error": "temp overshoot exceeded", "step": "s2"},
    ]

    dist = failure_attribution_from_events(events, now=_NOW)

    # temp overshoot is CRITICAL (vs HIGH for volume) -> instrument dominates.
    assert dist is not None
    assert dist.source_failure_type == "temperature_overshoot"


# --- campaign_instruments_from_protocol ----------------------------------


def test_campaign_instruments_from_protocol():
    registry = _FakeRegistry(
        [
            _spec("heat", instrument="heater"),
            _spec("squidstat.measure", instrument="squidstat"),
            _spec("log", instrument=None),
        ]
    )
    protocol = {"steps": [{"primitive": "heat"}, {"primitive": "log"}]}

    instruments = campaign_instruments_from_protocol(protocol, registry)

    assert instruments == {"heater"}


# --- action_specs_from_registry ------------------------------------------


def test_action_specs_filter_by_instrument_but_keep_instrumentless():
    registry = _FakeRegistry(
        [
            _spec("heat", instrument="heater"),
            _spec("squidstat.measure", instrument="squidstat"),
            _spec("log", instrument=None, safety_class=SafetyClass.INFORMATIONAL),
        ]
    )

    specs = action_specs_from_registry(registry, instruments={"heater"})
    names = {s.name for s in specs}

    # heater kept (campaign instrument); squidstat dropped; instrument-less kept.
    assert names == {"heat", "log"}


def test_action_spec_field_mapping():
    contract = ActionContract(
        preconditions=(Precondition("robot_homed"),),
        timeout=TimeoutConfig(seconds=120.0),
    )
    registry = _FakeRegistry(
        [
            _spec(
                "heat",
                instrument="heater",
                safety_class=SafetyClass.REVERSIBLE,
                contract=contract,
            ),
            _spec("calibrate.run", instrument="heater"),
            _spec("log", instrument=None, safety_class=SafetyClass.INFORMATIONAL),
        ]
    )

    specs = {s.name: s for s in action_specs_from_registry(registry, instruments={"heater"})}

    heat = specs["heat"]
    assert heat.safety_class == "reversible"
    assert heat.required_capabilities == ["heater"]
    assert heat.latency == 120.0
    assert heat.context_dependencies == ["robot_homed"]
    assert heat.kind == "experiment"
    assert specs["calibrate.run"].kind == "calibration"
    assert specs["log"].kind == "report"


def test_action_specs_cap_limits_experiment_actions_but_keeps_specials():
    registry = _FakeRegistry(
        [
            _spec("exp_a", instrument="heater"),
            _spec("exp_b", instrument="heater"),
            _spec("exp_c", instrument="heater"),
            _spec("log", instrument=None, safety_class=SafetyClass.INFORMATIONAL),
        ]
    )

    specs = action_specs_from_registry(registry, instruments={"heater"}, cap=2)
    names = [s.name for s in specs]

    # The instrument-less "log" is always kept; experiments capped to 2.
    assert "log" in names
    assert len([n for n in names if n.startswith("exp_")]) == 2


# --- available_capabilities ----------------------------------------------


def test_available_capabilities_prefers_campaign_subset():
    registry = _FakeRegistry([_spec("heat", instrument="heater")])

    caps, source = available_capabilities(
        campaign_instruments={"heater", "squidstat"}, registry=registry
    )

    assert caps == ["heater", "squidstat"]
    assert source == "campaign_deck"


def test_available_capabilities_falls_back_to_registry():
    registry = _FakeRegistry(
        [_spec("heat", instrument="heater"), _spec("m", instrument="squidstat")]
    )

    caps, source = available_capabilities(campaign_instruments=set(), registry=registry)

    assert caps == ["heater", "squidstat"]
    assert source == "registry_fallback"


# --- config flag ----------------------------------------------------------


def test_config_has_adaptive_substrate_flag_default_false(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_SUBSTRATE_SHADOW_ENABLED", raising=False)
    from app.core.config import Settings

    settings = Settings()
    assert settings.adaptive_substrate_shadow_enabled is False


def test_kind_mapper_reclassifies_utility_primitives():
    registry = _FakeRegistry(
        [
            _spec("cleanup.run_full", instrument=None, safety_class=SafetyClass.CAREFUL),
            _spec("sample.prepare_from_csv", instrument=None, safety_class=SafetyClass.HAZARDOUS),
            _spec("heat", instrument=None, safety_class=SafetyClass.REVERSIBLE),
            _spec("robot.aspirate", instrument="ot2-robot", safety_class=SafetyClass.HAZARDOUS),
            _spec("mystery_op", instrument=None),
        ]
    )

    specs = {s.name: s for s in action_specs_from_registry(registry, instruments=None)}

    assert specs["cleanup.run_full"].kind == "cleanup"
    assert specs["sample.prepare_from_csv"].kind == "preparation"
    # heat has no instrument but is a known single-instrument op -> experiment
    # (preserves its metadata-gap true positive downstream).
    assert specs["heat"].kind == "experiment"
    # instrument-bearing unknown -> experiment; instrument-less unknown -> workflow.
    assert specs["robot.aspirate"].kind == "experiment"
    assert specs["mystery_op"].kind == "workflow"


def test_import_smoke():
    import app.services.adaptive_substrate_inputs  # noqa: F401
