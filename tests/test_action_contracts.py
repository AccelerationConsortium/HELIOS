"""Tests for app.services.action_contracts — Action Contract DSL."""
from __future__ import annotations

import pytest

from app.services.action_contracts import (
    ActionContract,
    Effect,
    LEGACY_SAFETY_MAP,
    Precondition,
    SafetyClass,
    TimeoutConfig,
    _apply_effect_dict,
    _coerce_value,
    _evaluate_predicate_dict,
    _render_template,
    build_action_contract,
    parse_contract,
    parse_safety_class,
    parse_timeout,
)


# ---------------------------------------------------------------------------
# SafetyClass enum
# ---------------------------------------------------------------------------


class TestSafetyClass:
    def test_ordering(self) -> None:
        assert SafetyClass.INFORMATIONAL < SafetyClass.REVERSIBLE
        assert SafetyClass.REVERSIBLE < SafetyClass.CAREFUL
        assert SafetyClass.CAREFUL < SafetyClass.HAZARDOUS

    def test_values(self) -> None:
        assert SafetyClass.INFORMATIONAL == 0
        assert SafetyClass.REVERSIBLE == 1
        assert SafetyClass.CAREFUL == 2
        assert SafetyClass.HAZARDOUS == 3

    def test_from_string(self) -> None:
        assert SafetyClass.from_string("INFORMATIONAL") == SafetyClass.INFORMATIONAL
        assert SafetyClass.from_string("hazardous") == SafetyClass.HAZARDOUS
        assert SafetyClass.from_string("  Careful  ") == SafetyClass.CAREFUL

    def test_from_string_invalid(self) -> None:
        with pytest.raises(KeyError):
            SafetyClass.from_string("UNKNOWN")

    def test_to_legacy_bypass(self) -> None:
        assert SafetyClass.INFORMATIONAL.to_legacy() == "BYPASS"
        assert SafetyClass.REVERSIBLE.to_legacy() == "BYPASS"

    def test_to_legacy_critical(self) -> None:
        assert SafetyClass.CAREFUL.to_legacy() == "CRITICAL"
        assert SafetyClass.HAZARDOUS.to_legacy() == "CRITICAL"


# ---------------------------------------------------------------------------
# Legacy mapping
# ---------------------------------------------------------------------------


class TestLegacySafetyMap:
    def test_informational_primitives(self) -> None:
        for name in ["robot.home", "wait", "log", "upload_artifact",
                      "ssh.start_stream", "squidstat.get_data"]:
            assert LEGACY_SAFETY_MAP[name] == SafetyClass.INFORMATIONAL, name

    def test_reversible_primitives(self) -> None:
        for name in ["robot.blowout", "heat", "plc.dispense_ml",
                      "relay.set_channel", "relay.turn_on"]:
            assert LEGACY_SAFETY_MAP[name] == SafetyClass.REVERSIBLE, name

    def test_careful_primitives(self) -> None:
        for name in ["robot.load_labware", "robot.pick_up_tip",
                      "robot.drop_tip", "cleanup.run_full"]:
            assert LEGACY_SAFETY_MAP[name] == SafetyClass.CAREFUL, name

    def test_hazardous_primitives(self) -> None:
        for name in ["robot.aspirate", "robot.dispense",
                      "squidstat.run_experiment", "sample.prepare_from_csv"]:
            assert LEGACY_SAFETY_MAP[name] == SafetyClass.HAZARDOUS, name

    def test_map_covers_all_expected(self) -> None:
        # We expect at least 30 entries (some originals like aspirate, eis too)
        assert len(LEGACY_SAFETY_MAP) >= 30


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_single_substitution(self) -> None:
        result = _render_template("labware_loaded:{labware}", {"labware": "plate1"})
        assert result == "labware_loaded:plate1"

    def test_multiple_substitutions(self) -> None:
        result = _render_template(
            "decrease:well_volume:{labware}:{well}:{volume}",
            {"labware": "plate1", "well": "A1", "volume": "50"},
        )
        assert result == "decrease:well_volume:plate1:A1:50"

    def test_missing_param_kept(self) -> None:
        result = _render_template("tip_on:{pipette}", {})
        assert result == "tip_on:{pipette}"

    def test_no_placeholders(self) -> None:
        result = _render_template("pipettes_loaded", {"foo": "bar"})
        assert result == "pipettes_loaded"


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


class TestPrecondition:
    def test_render(self) -> None:
        p = Precondition(predicate="labware_loaded:{labware}")
        assert p.render({"labware": "plate1"}) == "labware_loaded:plate1"

    def test_evaluate_labware_loaded_true(self) -> None:
        state = {"labware_loaded": {"plate1": True}}
        p = Precondition(predicate="labware_loaded:{labware}")
        assert p.evaluate(state, {"labware": "plate1"}) is True

    def test_evaluate_labware_loaded_false(self) -> None:
        state = {"labware_loaded": {}}
        p = Precondition(predicate="labware_loaded:{labware}")
        assert p.evaluate(state, {"labware": "plate1"}) is False

    def test_evaluate_tip_on(self) -> None:
        state = {"tip_state": {"left": "on"}}
        p = Precondition(predicate="tip_on:{pipette}")
        assert p.evaluate(state, {"pipette": "left"}) is True

    def test_evaluate_tip_on_off(self) -> None:
        state = {"tip_state": {"left": "off"}}
        p = Precondition(predicate="tip_on:{pipette}")
        assert p.evaluate(state, {"pipette": "left"}) is False

    def test_evaluate_tip_off(self) -> None:
        state = {"tip_state": {"left": "off"}}
        p = Precondition(predicate="tip_off:{pipette}")
        assert p.evaluate(state, {"pipette": "left"}) is True

    def test_evaluate_tip_off_when_on(self) -> None:
        state = {"tip_state": {"left": "on"}}
        p = Precondition(predicate="tip_off:{pipette}")
        assert p.evaluate(state, {"pipette": "left"}) is False

    def test_evaluate_pipettes_loaded(self) -> None:
        state = {"pipettes_loaded": True}
        p = Precondition(predicate="pipettes_loaded")
        assert p.evaluate(state, {}) is True

    def test_evaluate_pipettes_not_loaded(self) -> None:
        state = {"pipettes_loaded": False}
        p = Precondition(predicate="pipettes_loaded")
        assert p.evaluate(state, {}) is False

    def test_evaluate_robot_homed(self) -> None:
        state = {"robot_homed": True}
        p = Precondition(predicate="robot_homed")
        assert p.evaluate(state, {}) is True

    def test_evaluate_experiment_idle(self) -> None:
        state = {"experiment_running": {"0": False}}
        p = Precondition(predicate="experiment_idle:{channel}")
        assert p.evaluate(state, {"channel": "0"}) is True

    def test_evaluate_experiment_running(self) -> None:
        state = {"experiment_running": {"0": True}}
        p = Precondition(predicate="experiment_idle:{channel}")
        assert p.evaluate(state, {"channel": "0"}) is False

    def test_evaluate_unknown_predicate(self) -> None:
        state = {}
        p = Precondition(predicate="unknown_predicate")
        assert p.evaluate(state, {}) is False


# ---------------------------------------------------------------------------
# _evaluate_predicate_dict (standalone)
# ---------------------------------------------------------------------------


class TestEvaluatePredicateDict:
    def test_labware_loaded(self) -> None:
        assert _evaluate_predicate_dict("labware_loaded:plate1", {"labware_loaded": {"plate1": True}}) is True

    def test_labware_not_loaded(self) -> None:
        assert _evaluate_predicate_dict("labware_loaded:plate1", {"labware_loaded": {}}) is False

    def test_tip_on(self) -> None:
        assert _evaluate_predicate_dict("tip_on:left", {"tip_state": {"left": "on"}}) is True

    def test_tip_off(self) -> None:
        assert _evaluate_predicate_dict("tip_off:left", {"tip_state": {"left": "off"}}) is True

    def test_experiment_idle_default(self) -> None:
        # Channel not in dict → idle (default)
        assert _evaluate_predicate_dict("experiment_idle:1", {"experiment_running": {}}) is True

    def test_unknown(self) -> None:
        assert _evaluate_predicate_dict("foo_bar", {}) is False


# ---------------------------------------------------------------------------
# _coerce_value
# ---------------------------------------------------------------------------


class TestCoerceValue:
    def test_true(self) -> None:
        assert _coerce_value("true") is True
        assert _coerce_value("True") is True

    def test_false(self) -> None:
        assert _coerce_value("false") is False

    def test_float(self) -> None:
        assert _coerce_value("3.14") == pytest.approx(3.14)

    def test_int_as_float(self) -> None:
        assert _coerce_value("42") == pytest.approx(42.0)

    def test_string(self) -> None:
        assert _coerce_value("hello") == "hello"


# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------


class TestEffect:
    def test_render(self) -> None:
        e = Effect(operation="increase:pipette_volume:{pipette}:{volume}")
        result = e.render({"pipette": "left", "volume": "100"})
        assert result == "increase:pipette_volume:left:100"

    def test_apply_set_simple(self) -> None:
        state: dict = {}
        e = Effect(operation="set:robot_homed:true")
        e.apply(state, {})
        assert state["robot_homed"] is True

    def test_apply_set_nested(self) -> None:
        state: dict = {}
        e = Effect(operation="set:tip_state:{pipette}:on")
        e.apply(state, {"pipette": "left"})
        assert state["tip_state"]["left"] == "on"

    def test_apply_increase(self) -> None:
        state: dict = {"pipette_volume": {"left": 50.0}}
        e = Effect(operation="increase:pipette_volume:{pipette}:{volume}")
        e.apply(state, {"pipette": "left", "volume": "100"})
        assert state["pipette_volume"]["left"] == pytest.approx(150.0)

    def test_apply_decrease(self) -> None:
        state: dict = {"pipette_volume": {"left": 150.0}}
        e = Effect(operation="decrease:pipette_volume:{pipette}:{volume}")
        e.apply(state, {"pipette": "left", "volume": "50"})
        assert state["pipette_volume"]["left"] == pytest.approx(100.0)

    def test_apply_increase_from_zero(self) -> None:
        state: dict = {}
        e = Effect(operation="increase:pipette_volume:{pipette}:{volume}")
        e.apply(state, {"pipette": "left", "volume": "100"})
        assert state["pipette_volume"]["left"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _apply_effect_dict (standalone)
# ---------------------------------------------------------------------------


class TestApplyEffectDict:
    def test_set_simple(self) -> None:
        state: dict = {}
        _apply_effect_dict("set:robot_homed:true", state)
        assert state["robot_homed"] is True

    def test_set_nested(self) -> None:
        state: dict = {}
        _apply_effect_dict("set:labware_loaded:plate1:true", state)
        assert state["labware_loaded"]["plate1"] is True

    def test_set_deeply_nested(self) -> None:
        state: dict = {}
        _apply_effect_dict("set:well_volume:plate1:A1:100", state)
        assert state["well_volume"]["plate1"]["A1"] == pytest.approx(100.0)

    def test_increase(self) -> None:
        state: dict = {"pipette_volume": {"left": 50.0}}
        _apply_effect_dict("increase:pipette_volume:left:25", state)
        assert state["pipette_volume"]["left"] == pytest.approx(75.0)

    def test_decrease(self) -> None:
        state: dict = {"pipette_volume": {"left": 100.0}}
        _apply_effect_dict("decrease:pipette_volume:left:30", state)
        assert state["pipette_volume"]["left"] == pytest.approx(70.0)

    def test_too_short(self) -> None:
        state: dict = {}
        _apply_effect_dict("set:x", state)  # len < 3 → no-op
        assert state == {}

    def test_unknown_op(self) -> None:
        state: dict = {}
        _apply_effect_dict("multiply:x:y:5", state)
        assert state == {}  # unrecognized → no-op


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------


class TestTimeoutConfig:
    def test_defaults(self) -> None:
        tc = TimeoutConfig()
        assert tc.seconds == 300.0
        assert tc.retries == 0

    def test_custom(self) -> None:
        tc = TimeoutConfig(seconds=30.0, retries=3)
        assert tc.seconds == 30.0
        assert tc.retries == 3


# ---------------------------------------------------------------------------
# ActionContract
# ---------------------------------------------------------------------------


class TestActionContract:
    def test_empty_contract(self) -> None:
        c = ActionContract()
        assert c.preconditions == ()
        assert c.effects == ()
        assert c.timeout.seconds == 300.0
        assert c.safety_class == SafetyClass.CAREFUL

    def test_full_contract(self) -> None:
        c = ActionContract(
            preconditions=(Precondition("tip_on:{pipette}"),),
            effects=(Effect("increase:pipette_volume:{pipette}:{volume}"),),
            timeout=TimeoutConfig(seconds=30.0, retries=0),
            safety_class=SafetyClass.HAZARDOUS,
        )
        assert len(c.preconditions) == 1
        assert len(c.effects) == 1
        assert c.safety_class == SafetyClass.HAZARDOUS


# ---------------------------------------------------------------------------
# parse_contract
# ---------------------------------------------------------------------------


class TestParseContract:
    def test_none(self) -> None:
        assert parse_contract(None) is None

    def test_empty_dict(self) -> None:
        assert parse_contract({}) is None

    def test_not_dict(self) -> None:
        assert parse_contract("invalid") is None  # type: ignore[arg-type]

    def test_with_preconditions_and_effects(self) -> None:
        raw = {
            "preconditions": ["labware_loaded:{labware}", "tip_on:{pipette}"],
            "effects": ["increase:pipette_volume:{pipette}:{volume}"],
        }
        result = parse_contract(raw)
        assert result is not None
        preconditions, effects = result
        assert len(preconditions) == 2
        assert len(effects) == 1
        assert preconditions[0].predicate == "labware_loaded:{labware}"
        assert effects[0].operation == "increase:pipette_volume:{pipette}:{volume}"

    def test_preconditions_only(self) -> None:
        raw = {"preconditions": ["pipettes_loaded"]}
        result = parse_contract(raw)
        assert result is not None
        preconditions, effects = result
        assert len(preconditions) == 1
        assert len(effects) == 0

    def test_skips_falsy_entries(self) -> None:
        raw = {"preconditions": ["tip_on:{pipette}", None, "", "robot_homed"]}
        result = parse_contract(raw)
        assert result is not None
        preconditions, _ = result
        assert len(preconditions) == 2


# ---------------------------------------------------------------------------
# parse_timeout
# ---------------------------------------------------------------------------


class TestParseTimeout:
    def test_none(self) -> None:
        tc = parse_timeout(None)
        assert tc.seconds == 300.0
        assert tc.retries == 0

    def test_empty_dict(self) -> None:
        tc = parse_timeout({})
        assert tc.seconds == 300.0

    def test_custom_values(self) -> None:
        tc = parse_timeout({"seconds": 60, "retries": 3})
        assert tc.seconds == 60.0
        assert tc.retries == 3

    def test_partial(self) -> None:
        tc = parse_timeout({"seconds": 10})
        assert tc.seconds == 10.0
        assert tc.retries == 0


# ---------------------------------------------------------------------------
# parse_safety_class
# ---------------------------------------------------------------------------


class TestParseSafetyClass:
    def test_explicit_string(self) -> None:
        assert parse_safety_class("HAZARDOUS") == SafetyClass.HAZARDOUS

    def test_explicit_case_insensitive(self) -> None:
        assert parse_safety_class("informational") == SafetyClass.INFORMATIONAL

    def test_invalid_string_fallback_to_map(self) -> None:
        result = parse_safety_class("BOGUS", primitive_name="robot.home")
        assert result == SafetyClass.INFORMATIONAL  # from LEGACY_SAFETY_MAP

    def test_none_fallback_to_map(self) -> None:
        result = parse_safety_class(None, primitive_name="robot.aspirate")
        assert result == SafetyClass.HAZARDOUS

    def test_none_no_map_bypass(self) -> None:
        result = parse_safety_class(None, error_class="BYPASS", primitive_name="unknown")
        assert result == SafetyClass.REVERSIBLE

    def test_none_no_map_critical(self) -> None:
        result = parse_safety_class(None, error_class="CRITICAL", primitive_name="unknown")
        assert result == SafetyClass.CAREFUL


# ---------------------------------------------------------------------------
# build_action_contract
# ---------------------------------------------------------------------------


class TestBuildActionContract:
    def test_no_contract_no_timeout(self) -> None:
        c = build_action_contract(None, None, SafetyClass.INFORMATIONAL)
        assert c.preconditions == ()
        assert c.effects == ()
        assert c.timeout.seconds == 300.0
        assert c.safety_class == SafetyClass.INFORMATIONAL

    def test_full_build(self) -> None:
        raw_contract = {
            "preconditions": ["labware_loaded:{labware}", "tip_on:{pipette}"],
            "effects": ["increase:pipette_volume:{pipette}:{volume}"],
        }
        raw_timeout = {"seconds": 30, "retries": 0}
        c = build_action_contract(raw_contract, raw_timeout, SafetyClass.HAZARDOUS)

        assert len(c.preconditions) == 2
        assert len(c.effects) == 1
        assert c.timeout.seconds == 30.0
        assert c.timeout.retries == 0
        assert c.safety_class == SafetyClass.HAZARDOUS


# ---------------------------------------------------------------------------
# Integration: aspirate contract scenario
# ---------------------------------------------------------------------------


class TestAspirateSmokeTest:
    """End-to-end test simulating robot.aspirate precondition + effect flow."""

    def test_aspirate_preconditions_met(self) -> None:
        state = {
            "labware_loaded": {"corning_96": True},
            "tip_state": {"left": "on"},
            "pipettes_loaded": True,
            "pipette_volume": {"left": 0.0},
        }
        params = {"labware": "corning_96", "well": "A1", "pipette": "left", "volume": "100"}

        # Check preconditions
        for pred_str in ["labware_loaded:{labware}", "tip_on:{pipette}", "pipettes_loaded"]:
            p = Precondition(predicate=pred_str)
            assert p.evaluate(state, params) is True, f"Failed: {pred_str}"

        # Apply effect
        effect = Effect(operation="increase:pipette_volume:{pipette}:{volume}")
        effect.apply(state, params)
        assert state["pipette_volume"]["left"] == pytest.approx(100.0)

    def test_aspirate_precondition_fails(self) -> None:
        state = {
            "labware_loaded": {},
            "tip_state": {"left": "off"},
            "pipettes_loaded": False,
        }
        params = {"labware": "corning_96", "pipette": "left"}

        p = Precondition(predicate="labware_loaded:{labware}")
        assert p.evaluate(state, params) is False

        p2 = Precondition(predicate="tip_on:{pipette}")
        assert p2.evaluate(state, params) is False
